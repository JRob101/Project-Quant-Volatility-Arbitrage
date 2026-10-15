"""Data ingestion and preparation for volatility arbitrage."""

import numpy as np
import pandas as pd
from pricing.pricer import american_trinomial_option_price_and_greeks, find_strike_for_target_delta


def generate_synthetic_market_data(n_days=1500):
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    mu, kappa, theta, xi, rho, dt = 0.04, 4.0, 0.04, 0.4, -0.7, 1 / 252
    
    S = np.zeros(n_days)
    v = np.zeros(n_days)
    S[0], v[0] = 400, theta
    
    W1 = np.random.normal(0, 1, n_days)
    W2 = rho * W1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_days)
    
    high, low = np.zeros(n_days), np.zeros(n_days)
    
    for t in range(1, n_days):
        v_prev = max(v[t-1], 1e-6)
        S[t] = S[t-1] * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1[t])
        v[t] = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * W2[t]
        
        intra_vol = np.sqrt(max(v[t], 1e-6) * dt)
        high[t] = S[t] * (1 + np.abs(np.random.normal(0, intra_vol * 0.4)))
        low[t] = S[t-1] * (1 - np.abs(np.random.normal(0, intra_vol * 0.4)))

    open_p = np.roll(S, 1)
    open_p[0] = S[0]
    
    iv_atm = np.sqrt(np.maximum(v, 1e-6)) + 0.015 + np.random.normal(0, 0.005, n_days)
    iv_skew = 0.05 - (rho * np.sqrt(np.maximum(v, 1e-6))) + np.random.normal(0, 0.01, n_days)
    term_structure = 0.02 + kappa * (theta - v) * 0.1 + np.random.normal(0, 0.005, n_days)
    
    bid_ask_spread = 0.0001 + (iv_atm * 0.0005)
    
    df = pd.DataFrame({
        'Date': dates, 'Open': open_p, 'High': high, 'Low': low, 'Close': S,
        'IV_ATM': iv_atm, 'IV_Skew': iv_skew, 'IV_Term_Structure': term_structure,
        'GEX': np.random.normal(1e6, 5e5, n_days),
        'BidAskSpread': bid_ask_spread
    }).set_index('Date')
    
    return df


def compute_advanced_features(df):
    data = df.copy()
    
    # 1. Features
    log_hl = np.log(data['High'] / data['Low']) ** 2
    log_co = np.log(data['Close'] / data['Open']) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    data['RV_GarmanKlass'] = np.sqrt(np.maximum(0, gk_var.rolling(20).mean() * 252))
    
    data['Vol_Premium'] = data['IV_ATM'] - data['RV_GarmanKlass']
    spread_mean = data['Vol_Premium'].rolling(60).mean()
    spread_std = data['Vol_Premium'].rolling(60).std()
    data['Spread_ZScore'] = (data['Vol_Premium'] - spread_mean) / (spread_std + 1e-8)
    
    data['GEX_Normalized'] = (data['GEX'] - data['GEX'].rolling(60).mean()) / data['GEX'].rolling(60).std()
    data['Skew_Momentum'] = data['IV_Skew'].diff(5)
    
    data['IV_EWMA_10'] = data['IV_ATM'].ewm(span=10, adjust=False).mean()
    data['IV_EWMA_Diff'] = data['IV_ATM'] - data['IV_EWMA_10']
    
    # 2. CALCUL DES PNL MTM À J+5 ET CREATION DE LA TARGET MULTI-CLASSES
    r = 0.04
    T0 = 30 / 252
    dt5 = 5 / 252
    n_days = len(data)
    
    pnl_ic = np.zeros(n_days)
    pnl_strangle = np.zeros(n_days)
    pnl_rr = np.zeros(n_days)
    
    closes = data['Close'].values
    ivs = data['IV_ATM'].values
    spreads = data['BidAskSpread'].values
    
    for t in range(n_days - 5):
        S0, S5 = closes[t], closes[t+5]
        sig0, sig5 = ivs[t], ivs[t+5]
        sp0 = spreads[t]
        
        div_t0 = [(12 / 252, 1.50)]
        div_t5 = [(td - dt5, D) for td, D in div_t0 if td > dt5]
        
        kc_25 = find_strike_for_target_delta(S0, 0.25, T0, r, div_t0, sig0, 'call')
        kp_25 = find_strike_for_target_delta(S0, -0.25, T0, r, div_t0, sig0, 'put')
        kc_05 = find_strike_for_target_delta(S0, 0.05, T0, r, div_t0, sig0, 'call')
        kp_05 = find_strike_for_target_delta(S0, -0.05, T0, r, div_t0, sig0, 'put')
        
        legs_ic = [('call', kc_25, -1), ('put', kp_25, -1), ('call', kc_05, +1), ('put', kp_05, +1)]
        legs_strangle = [('call', kc_25, +1), ('put', kp_25, +1)]
        legs_rr = [('put', kp_25, +1), ('call', kc_25, -1)]
        
        def calc_mtm(legs):
            v0, g0, v5 = 0.0, 0.0, 0.0
            for opt_type, K, direction in legs:
                res0 = american_trinomial_option_price_and_greeks(S0, K, T0, r, div_t0, sig0, opt_type, N=40)
                v0 += direction * res0['price']
                g0 += abs(direction) * res0['price']
                res5 = american_trinomial_option_price_and_greeks(S5, K, T0 - dt5, r, div_t5, sig5, opt_type, N=40)
                v5 += direction * res5['price']
            return (v5 - v0) - (g0 * sp0)
            
        pnl_ic[t] = calc_mtm(legs_ic)
        pnl_strangle[t] = calc_mtm(legs_strangle)
        pnl_rr[t] = calc_mtm(legs_rr)
        
    # Attributions des Labels de Classes
    # 0 = NO_TRADE, 1 = IRON_CONDOR, 2 = LONG_STRANGLE, 3 = RISK_REVERSAL
    labels = np.zeros(n_days, dtype=int)
    min_profit_threshold = 0.05
    
    for t in range(n_days - 5):
        pnls = [0.0, pnl_ic[t], pnl_strangle[t], pnl_rr[t]]
        best_class = int(np.argmax(pnls))
        if pnls[best_class] > min_profit_threshold:
            labels[t] = best_class
        else:
            labels[t] = 0
            
    data['Target'] = labels
    return data.iloc[:-5].dropna()