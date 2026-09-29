"""Data ingestion and preparation for volatility arbitrage."""

import numpy as np
import pandas as pd

def generate_synthetic_market_data(n_days=1000):
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    mu = 0.04
    kappa = 4.0
    theta = 0.04
    xi = 0.4
    rho = -0.7
    dt = 1/252
    
    S = np.zeros(n_days)
    v = np.zeros(n_days)
    S[0] = 400
    v[0] = theta
    
    W1 = np.random.normal(0, 1, n_days)
    W2 = rho * W1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_days)
    
    high = np.zeros(n_days)
    low = np.zeros(n_days)
    
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
    
    df = pd.DataFrame({
        'Date': dates, 'Open': open_p, 'High': high, 'Low': low, 'Close': S,
        'IV_ATM': iv_atm, 'IV_Skew': iv_skew, 'IV_Term_Structure': term_structure,
        'GEX': np.random.normal(1e6, 5e5, n_days)
    }).set_index('Date')
    
    return df

def compute_advanced_features(df):
    data = df.copy()
    
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
    
    data['Target'] = data['Vol_Premium'].shift(-5) - data['Vol_Premium']
    
    return data.dropna()