"""Data ingestion and preparation for volatility arbitrage."""

import numpy as np
import pandas as pd
from pricing.pricer import american_trinomial_option_price_and_greeks, find_strike_for_target_delta


REGIME_PARAMS = {
    'BULL_STABLE': {
        'SPY': {'S0': 400, 'mu': 0.08, 'kappa': 5.0, 'theta': 0.020, 'xi': 0.25, 'rho': -0.60},
        'QQQ': {'S0': 350, 'mu': 0.09, 'kappa': 4.5, 'theta': 0.030, 'xi': 0.30, 'rho': -0.55},
        'IWM': {'S0': 200, 'mu': 0.06, 'kappa': 5.5, 'theta': 0.025, 'xi': 0.30, 'rho': -0.65}
    },
    'BEAR_STRESS': {
        'SPY': {'S0': 400, 'mu': -0.12, 'kappa': 3.0, 'theta': 0.080, 'xi': 0.60, 'rho': -0.80},
        'QQQ': {'S0': 350, 'mu': -0.15, 'kappa': 2.5, 'theta': 0.100, 'xi': 0.70, 'rho': -0.75},
        'IWM': {'S0': 200, 'mu': -0.10, 'kappa': 3.5, 'theta': 0.090, 'xi': 0.65, 'rho': -0.85}
    },
    'HIGH_VOL_OF_VOL': {
        'SPY': {'S0': 400, 'mu': 0.02, 'kappa': 2.0, 'theta': 0.040, 'xi': 0.80, 'rho': -0.70},
        'QQQ': {'S0': 350, 'mu': 0.03, 'kappa': 1.8, 'theta': 0.050, 'xi': 0.90, 'rho': -0.65},
        'IWM': {'S0': 200, 'mu': 0.01, 'kappa': 2.2, 'theta': 0.045, 'xi': 0.85, 'rho': -0.75}
    },
    'JUMP_CRASH': {
        'SPY': {'S0': 400, 'mu': -0.05, 'kappa': 3.5, 'theta': 0.050, 'xi': 0.45, 'rho': -0.75},
        'QQQ': {'S0': 350, 'mu': -0.06, 'kappa': 3.0, 'theta': 0.065, 'xi': 0.50, 'rho': -0.70},
        'IWM': {'S0': 200, 'mu': -0.04, 'kappa': 4.0, 'theta': 0.060, 'xi': 0.45, 'rho': -0.80}
    }
}


def generate_multi_asset_market_data(n_days=1500, assets=['SPY', 'QQQ', 'IWM'], seed=None, market_regime='BULL_STABLE'):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
        
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    n_assets = len(assets)
    dt = 1 / 252
    
    regime = REGIME_PARAMS.get(market_regime, REGIME_PARAMS['BULL_STABLE'])
    
    corr_matrix_S = np.array([
        [1.00, 0.85, 0.75],
        [0.85, 1.00, 0.70],
        [0.75, 0.70, 1.00]
    ])
    
    L_S = np.linalg.cholesky(corr_matrix_S)
    
    S = {asset: np.zeros(n_days) for asset in assets}
    v = {asset: np.zeros(n_days) for asset in assets}
    high = {asset: np.zeros(n_days) for asset in assets}
    low = {asset: np.zeros(n_days) for asset in assets}
    
    for asset in assets:
        S[asset][0] = regime[asset]['S0']
        v[asset][0] = regime[asset]['theta']
        
    lambda_j = 4.0 if market_regime == 'JUMP_CRASH' else 0.0
    mu_j, sigma_j = -0.06, 0.08
    
    for t in range(1, n_days):
        Z_S_indep = np.random.normal(0, 1, n_assets)
        Z_v_indep = np.random.normal(0, 1, n_assets)
        W1 = L_S @ Z_S_indep
        
        jumps_count = np.random.poisson(lambda_j * dt, n_assets) if lambda_j > 0 else np.zeros(n_assets)
        jumps_size = np.random.normal(mu_j, sigma_j, n_assets) * jumps_count
        
        for i, asset in enumerate(assets):
            p = regime[asset]
            v_prev = max(v[asset][t-1], 1e-6)
            W2 = p['rho'] * W1[i] + np.sqrt(1 - p['rho']**2) * Z_v_indep[i]
            
            S[asset][t] = S[asset][t-1] * np.exp((p['mu'] - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1[i] + jumps_size[i])
            v[asset][t] = v_prev + p['kappa'] * (p['theta'] - v_prev) * dt + p['xi'] * np.sqrt(v_prev * dt) * W2
            
            intra_vol = np.sqrt(max(v[asset][t], 1e-6) * dt)
            jump_impact = abs(jumps_size[i]) if jumps_size[i] != 0 else 0
            high[asset][t] = S[asset][t] * (1 + np.abs(np.random.normal(0, intra_vol * 0.4)) + jump_impact)
            low[asset][t] = S[asset][t-1] * (1 - np.abs(np.random.normal(0, intra_vol * 0.4)) - jump_impact)

    market_data = {}
    for asset in assets:
        p = regime[asset]
        open_p = np.roll(S[asset], 1)
        open_p[0] = S[asset][0]
        
        iv_atm = np.sqrt(np.maximum(v[asset], 1e-6)) + 0.015 + np.random.normal(0, 0.005, n_days)
        iv_skew = 0.05 - (p['rho'] * np.sqrt(np.maximum(v[asset], 1e-6))) + np.random.normal(0, 0.01, n_days)
        term_structure = 0.02 + p['kappa'] * (p['theta'] - v[asset]) * 0.1 + np.random.normal(0, 0.005, n_days)
        bid_ask_spread = 0.0001 + (iv_atm * 0.0005)
        
        df = pd.DataFrame({
            'Date': dates, 'Open': open_p, 'High': high[asset], 'Low': low[asset], 'Close': S[asset],
            'IV_ATM': iv_atm, 'IV_Skew': iv_skew, 'IV_Term_Structure': term_structure,
            'GEX': np.random.normal(1e6, 5e5, n_days),
            'BidAskSpread': bid_ask_spread
        }).set_index('Date')
        
        market_data[asset] = df
        
    return market_data


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
    
    ts_mean = data['IV_Term_Structure'].rolling(60).mean()
    ts_std = data['IV_Term_Structure'].rolling(60).std()
    data['Term_Structure_ZScore'] = (data['IV_Term_Structure'] - ts_mean) / (ts_std + 1e-8)
    data['Slope_Momentum'] = data['IV_Term_Structure'].diff(5)
    
    data['GEX_Normalized'] = (data['GEX'] - data['GEX'].rolling(60).mean()) / data['GEX'].rolling(60).std()
    data['Skew_Momentum'] = data['IV_Skew'].diff(5)
    
    data['IV_EWMA_10'] = data['IV_ATM'].ewm(span=10, adjust=False).mean()
    data['IV_EWMA_Diff'] = data['IV_ATM'] - data['IV_EWMA_10']
    
    r = 0.04
    T_short = 30 / 252
    T_long = 60 / 252
    dt5 = 5 / 252
    n_days = len(data)
    
    pnl_ic = np.zeros(n_days)
    pnl_strangle = np.zeros(n_days)
    pnl_rr = np.zeros(n_days)
    pnl_calendar = np.zeros(n_days)
    pnl_diagonal = np.zeros(n_days)
    
    closes = data['Close'].values
    ivs = data['IV_ATM'].values
    spreads = data['BidAskSpread'].values
    
    for t in range(n_days - 5):
        S0, S5 = closes[t], closes[t+5]
        sig0, sig5 = ivs[t], ivs[t+5]
        sp0 = spreads[t]
        
        div_t0 = [(12 / 252, 1.50)]
        div_t5 = [(td - dt5, D) for td, D in div_t0 if td > dt5]
        
        kc_25 = find_strike_for_target_delta(S0, 0.25, T_short, r, div_t0, sig0, 'call')
        kp_25 = find_strike_for_target_delta(S0, -0.25, T_short, r, div_t0, sig0, 'put')
        kc_05 = find_strike_for_target_delta(S0, 0.05, T_short, r, div_t0, sig0, 'call')
        kp_05 = find_strike_for_target_delta(S0, -0.05, T_short, r, div_t0, sig0, 'put')
        kc_70_long = find_strike_for_target_delta(S0, 0.70, T_long, r, div_t0, sig0, 'call')
        
        legs_ic = [('call', kc_25, -1, T_short), ('put', kp_25, -1, T_short), ('call', kc_05, +1, T_short), ('put', kp_05, +1, T_short)]
        legs_strangle = [('call', kc_25, +1, T_short), ('put', kp_25, +1, T_short)]
        legs_rr = [('put', kp_25, +1, T_short), ('call', kc_25, -1, T_short)]
        legs_calendar = [('call', S0, -1, T_short), ('call', S0, +1, T_long)]
        legs_diagonal = [('call', kc_25, -1, T_short), ('call', kc_70_long, +1, T_long)]
        
        def calc_mtm_multi_maturity(legs):
            v0, g0, v5 = 0.0, 0.0, 0.0
            for opt_type, K, direction, mat in legs:
                res0 = american_trinomial_option_price_and_greeks(S0, K, mat, r, div_t0, sig0, opt_type, N=40)
                v0 += direction * res0['price']
                g0 += abs(direction) * res0['price']
                res5 = american_trinomial_option_price_and_greeks(S5, K, mat - dt5, r, div_t5, sig5, opt_type, N=40)
                v5 += direction * res5['price']
            return (v5 - v0) - (g0 * sp0)
            
        pnl_ic[t] = calc_mtm_multi_maturity(legs_ic)
        pnl_strangle[t] = calc_mtm_multi_maturity(legs_strangle)
        pnl_rr[t] = calc_mtm_multi_maturity(legs_rr)
        pnl_calendar[t] = calc_mtm_multi_maturity(legs_calendar)
        pnl_diagonal[t] = calc_mtm_multi_maturity(legs_diagonal)
        
    labels = np.zeros(n_days, dtype=int)
    min_profit_threshold = 0.01
    
    for t in range(n_days - 5):
        pnls = [0.0, pnl_ic[t], pnl_strangle[t], pnl_rr[t], pnl_calendar[t], pnl_diagonal[t]]
        best_class = int(np.argmax(pnls))
        if pnls[best_class] > min_profit_threshold:
            labels[t] = best_class
        else:
            labels[t] = 0
            
    data['Target'] = labels
    return data.iloc[:-5].dropna()