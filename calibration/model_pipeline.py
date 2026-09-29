"""Model calibration pipeline for volatility arbitrage."""

import numpy as np
from pricing.pricer import american_trinomial_option_price_and_greeks

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

def backtest_delta_neutral_strategy(df_test, predictions, transaction_cost_bps=0.001):
    r = 0.04
    pnl_history = []
    
    for t in range(len(df_test) - 1):
        signal = predictions[t]
        if signal == 0:
            pnl_history.append(0.0)
            continue
            
        S0 = df_test['Close'].iloc[t]
        S1 = df_test['Close'].iloc[t+1]
        sigma0 = df_test['IV_ATM'].iloc[t]
        sigma1 = df_test['IV_ATM'].iloc[t+1]
        
        K = S0
        T = 30 / 252
        
        div_schedule_t0 = [(12 / 252, 1.50)]
        
        call_t0 = american_trinomial_option_price_and_greeks(S0, K, T, r, div_schedule_t0, sigma0, 'call', N=60)
        put_t0  = american_trinomial_option_price_and_greeks(S0, K, T, r, div_schedule_t0, sigma0, 'put', N=60)
        
        straddle_price_t0 = call_t0['price'] + put_t0['price']
        net_delta_t0 = call_t0['delta'] + put_t0['delta']
        hedge_stock_position = -net_delta_t0
        
        dt = 1 / 252
        div_schedule_t1 = [(t_div - dt, D) for t_div, D in div_schedule_t0 if t_div > dt]
        
        call_t1 = american_trinomial_option_price_and_greeks(S1, K, T - dt, r, div_schedule_t1, sigma1, 'call', N=60)
        put_t1  = american_trinomial_option_price_and_greeks(S1, K, T - dt, r, div_schedule_t1, sigma1, 'put', N=60)
        
        straddle_price_t1 = call_t1['price'] + put_t1['price']
        
        option_pnl = straddle_price_t0 - straddle_price_t1
        hedge_pnl = hedge_stock_position * (S1 - S0)
        tc = (abs(hedge_stock_position) * S0 + straddle_price_t0) * transaction_cost_bps
        
        pnl_history.append(option_pnl + hedge_pnl - tc)
        
    return np.array(pnl_history)