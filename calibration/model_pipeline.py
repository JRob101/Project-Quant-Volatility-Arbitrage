"""Model calibration pipeline for volatility arbitrage."""

import numpy as np
from pricing.pricer import american_trinomial_option_price_and_greeks, find_strike_for_target_delta


class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits=5, purge_window=5, embargo_window=5):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_window
            test_end = test_start + fold_size
            if test_end > n_samples:
                break
            yield list(range(0, train_end)), list(range(test_start, test_end))


def backtest_cross_asset_strategy(market_data_test, asset_signals, position_sizes=None, holding_period=5, delta_tolerance=0.08):
    """
    Backtest multi-actifs (Cross-Asset) avec rééquilibrage Delta et MTM multi-échéances.
    asset_signals est une liste de tuples : (strat_name, asset_name)
    """
    r = 0.04
    pnl_history = []
    
    # Récupérer la taille de l'horizon de test à partir du premier actif
    first_asset = list(market_data_test.keys())[0]
    n_days = len(market_data_test[first_asset])
    
    if position_sizes is None:
        position_sizes = np.ones(n_days)
    
    t = 0
    while t < n_days - 1:
        strat, asset_curr = asset_signals[t]
        size = position_sizes[t]
        h = min(holding_period, n_days - 1 - t)
        
        if strat == 'NO_TRADE' or size <= 0 or h <= 0 or asset_curr is None:
            pnl_history.append(0.0)
            t += 1
            continue
            
        df_asset = market_data_test[asset_curr]
        
        S0 = df_asset['Close'].iloc[t]
        sigma0 = df_asset['IV_ATM'].iloc[t]
        dynamic_spread = df_asset['BidAskSpread'].iloc[t] if 'BidAskSpread' in df_asset.columns else 0.0002
        
        T_short = 30 / 252
        T_long = 60 / 252
        div_schedule_t0 = [(12 / 252, 1.50)]

        if strat == 'SHORT_STRADDLE':
            legs = [('call', S0, -1, T_short), ('put', S0, -1, T_short)]
        elif strat == 'SHORT_STRANGLE':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T_short, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T_short, r, div_schedule_t0, sigma0, 'put')
            legs = [('call', kc_25, -1, T_short), ('put', kp_25, -1, T_short)]
        elif strat == 'IRON_CONDOR':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T_short, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T_short, r, div_schedule_t0, sigma0, 'put')
            kc_05 = find_strike_for_target_delta(S0, 0.05, T_short, r, div_schedule_t0, sigma0, 'call')
            kp_05 = find_strike_for_target_delta(S0, -0.05, T_short, r, div_schedule_t0, sigma0, 'put')
            legs = [
                ('call', kc_25, -1, T_short), ('put', kp_25, -1, T_short),
                ('call', kc_05, +1, T_short), ('put', kp_05, +1, T_short)
            ]
        elif strat == 'LONG_STRANGLE':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T_short, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T_short, r, div_schedule_t0, sigma0, 'put')
            legs = [('call', kc_25, +1, T_short), ('put', kp_25, +1, T_short)]
        elif strat == 'RISK_REVERSAL':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T_short, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T_short, r, div_schedule_t0, sigma0, 'put')
            legs = [('put', kp_25, +1, T_short), ('call', kc_25, -1, T_short)]
        elif strat == 'CALENDAR_SPREAD':
            legs = [('call', S0, -1, T_short), ('call', S0, +1, T_long)]
        elif strat == 'DIAGONAL_SPREAD':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T_short, r, div_schedule_t0, sigma0, 'call')
            kc_70_long = find_strike_for_target_delta(S0, 0.70, T_long, r, div_schedule_t0, sigma0, 'call')
            legs = [('call', kc_25, -1, T_short), ('call', kc_70_long, +1, T_long)]
        else:
            pnl_history.append(0.0)
            t += 1
            continue

        for step in range(h):
            idx_curr = t + step
            idx_next = t + step + 1
            
            S_curr, S_next = df_asset['Close'].iloc[idx_curr], df_asset['Close'].iloc[idx_next]
            sig_curr, sig_next = df_asset['IV_ATM'].iloc[idx_curr], df_asset['IV_ATM'].iloc[idx_next]
            
            div_curr = [(td - step/252, D) for td, D in div_schedule_t0 if td > step/252]
            div_next = [(td - (step+1)/252, D) for td, D in div_schedule_t0 if td > (step+1)/252]

            val_curr, net_delta_curr, open_gross_premium = 0.0, 0.0, 0.0
            for opt_type, K, direction, mat in legs:
                t_rem_curr = mat - (step / 252)
                res = american_trinomial_option_price_and_greeks(S_curr, K, t_rem_curr, r, div_curr, sig_curr, opt_type, N=60)
                val_curr += direction * res['price']
                net_delta_curr += direction * res['delta']
                open_gross_premium += abs(direction) * res['price']

            val_next = 0.0
            close_gross_premium = 0.0
            for opt_type, K, direction, mat in legs:
                t_rem_next = mat - ((step + 1) / 252)
                res = american_trinomial_option_price_and_greeks(S_next, K, t_rem_next, r, div_next, sig_next, opt_type, N=60)
                val_next += direction * res['price']
                
                if step == h - 1 and abs(direction) == 1 and res['price'] < 0.05:
                    continue
                close_gross_premium += abs(direction) * res['price']

            daily_opt_pnl = (val_next - val_curr) * size
            
            if abs(net_delta_curr) > delta_tolerance:
                hedge_pos = -net_delta_curr
            else:
                hedge_pos = 0.0
                
            daily_hedge_pnl = hedge_pos * (S_next - S_curr) * size
            
            traded_premium = open_gross_premium if step == 0 else (close_gross_premium if step == h - 1 else 0.0)
            tc_day = (abs(hedge_pos) * S_curr + traded_premium) * dynamic_spread * size
            
            pnl_history.append(daily_opt_pnl + daily_hedge_pnl - tc_day)
            
        t += h
        
    return np.array(pnl_history)