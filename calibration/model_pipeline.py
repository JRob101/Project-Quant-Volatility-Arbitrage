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


def backtest_multi_structure_strategy(
    df_test, 
    strategy_signals, 
    position_sizes=None, 
    transaction_cost_bps=0.001,
    delta_tolerance=0.08,    # Seuil de déclenchement du Delta-Hedging
    take_profit_pct=0.50,    # Clôture à 50% du gain max
    stop_loss_pct=1.00       # Clôture à 100% de perte sur la prime
):
    r = 0.04
    pnl_history = []
    if position_sizes is None:
        position_sizes = np.ones(len(df_test))
    
    current_hedge_pos = 0.0  # Position de couverture en cours d'accumulation
    
    for t in range(len(df_test) - 1):
        strat = strategy_signals[t]
        size = position_sizes[t]
        
        if strat == 'NO_TRADE' or size <= 0:
            current_hedge_pos = 0.0
            pnl_history.append(0.0)
            continue
            
        S0, S1 = df_test['Close'].iloc[t], df_test['Close'].iloc[t+1]
        sigma0, sigma1 = df_test['IV_ATM'].iloc[t], df_test['IV_ATM'].iloc[t+1]
        
        # Récupération du spread réel si disponible, sinon valeur par défaut
        spread = df_test['BidAskSpread'].iloc[t] if 'BidAskSpread' in df_test.columns else transaction_cost_bps
        
        T, dt = 30 / 252, 1 / 252
        div_schedule_t0 = [(12 / 252, 1.50)]
        div_schedule_t1 = [(t_div - dt, D) for t_div, D in div_schedule_t0 if t_div > dt]

        # 1. Sélection des jambes
        if strat == 'SHORT_STRADDLE':
            legs = [('call', S0, -1), ('put', S0, -1)]
        elif strat == 'SHORT_STRANGLE':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T, r, div_schedule_t0, sigma0, 'put')
            legs = [('call', kc_25, -1), ('put', kp_25, -1)]
        elif strat == 'IRON_CONDOR':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T, r, div_schedule_t0, sigma0, 'put')
            kc_10 = find_strike_for_target_delta(S0, 0.10, T, r, div_schedule_t0, sigma0, 'call')
            kp_10 = find_strike_for_target_delta(S0, -0.10, T, r, div_schedule_t0, sigma0, 'put')
            legs = [('call', kc_25, -1), ('put', kp_25, -1), ('call', kc_10, +1), ('put', kp_10, +1)]
        elif strat == 'LONG_STRANGLE':
            kc_25 = find_strike_for_target_delta(S0, 0.25, T, r, div_schedule_t0, sigma0, 'call')
            kp_25 = find_strike_for_target_delta(S0, -0.25, T, r, div_schedule_t0, sigma0, 'put')
            legs = [('call', kc_25, +1), ('put', kp_25, +1)]
        else:
            pnl_history.append(0.0)
            continue

        # 2. Pricing à t0 et calcul du Delta théorique
        val_t0, net_delta_t0 = 0.0, 0.0
        for opt_type, K, direction in legs:
            res = american_trinomial_option_price_and_greeks(S0, K, T, r, div_schedule_t0, sigma0, opt_type, N=60)
            val_t0 += direction * res['price']
            net_delta_t0 += direction * res['delta']

        # 3. Pricing à t1 (Mark-to-Market)
        val_t1 = 0.0
        for opt_type, K, direction in legs:
            res = american_trinomial_option_price_and_greeks(S1, K, T - dt, r, div_schedule_t1, sigma1, opt_type, N=60)
            val_t1 += direction * res['price']

        is_short = strat in ['SHORT_STRADDLE', 'SHORT_STRANGLE', 'IRON_CONDOR']
        raw_structure_pnl = (val_t0 - val_t1) if is_short else (val_t1 - val_t0)
        
        # 4. Application du Take-Profit / Stop-Loss
        premium_base = abs(val_t0)
        if is_short:
            if raw_structure_pnl >= take_profit_pct * premium_base:
                structure_pnl = take_profit_pct * premium_base * size
            elif raw_structure_pnl <= -stop_loss_pct * premium_base:
                structure_pnl = -stop_loss_pct * premium_base * size
            else:
                structure_pnl = raw_structure_pnl * size
        else:
            structure_pnl = raw_structure_pnl * size

        # 5. Delta-Hedging dynamique sous bande de tolérance
        target_hedge_pos = -net_delta_t0
        delta_gap = target_hedge_pos - current_hedge_pos
        
        if abs(delta_gap) > delta_tolerance:
            trade_hedge = delta_gap
            current_hedge_pos = target_hedge_pos
        else:
            trade_hedge = 0.0
            
        hedge_pnl = current_hedge_pos * (S1 - S0) * size
        
        # Frais de transaction
        tc = (abs(trade_hedge) * S0 * spread + abs(val_t0) * spread) * size
        
        pnl_history.append(structure_pnl + hedge_pnl - tc)
        
    return np.array(pnl_history)