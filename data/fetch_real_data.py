import os
import numpy as np
import pandas as pd
import yfinance as yf
from data.data_engineering import compute_advanced_features

import os
import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

def generate_micro_noise(n_samples, rho=0.6, sigma=0.002, seed=42):
    """
    Génère un bruit autoregressif AR(1) pour simuler la mémoire courte 
    des imperfections du carnet d'ordres (Bruit Rouge).
    """
    np.random.seed(seed)
    noise = np.zeros(n_samples)
    e = np.random.normal(0, sigma, n_samples)
    for t in range(1, n_samples):
        noise[t] = rho * noise[t-1] + e[t]
    return noise

def fetch_real_cross_asset_data(
    assets=['SPY', 'QQQ', 'IWM'], 
    start_year=2014, 
    end_year=2019, 
    use_cache=True
):
    """
    Télécharge les historiques réels et génère un proxy avancé de surface d'options
    avec micro-bruit stochastique AR(1) et spreads bid-ask non-linéaires en cas de crise.
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    market_data = {}
    
    for idx, asset in enumerate(assets):
        cache_path = os.path.join(CACHE_DIR, f"{asset}_advanced_proxy_{start_year}_{end_year}.csv")
        
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(asset, start=start_date, end=end_date, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(asset, level=1, axis=1)
                
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            n_obs = len(df)
            
            # 1. Volatilité Réalisée Garman-Klass (base 20j)
            log_hl = np.log(df['High'] / df['Low']) ** 2
            log_co = np.log(df['Close'] / df['Open']) ** 2
            gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
            df['RV_GarmanKlass'] = np.sqrt(np.maximum(0, gk_var.rolling(20).mean() * 252))
            
            # 2. Proxy IV_ATM avec Prime de Risque de Volatilité (VRP) dynamique
            # L'IV réagit plus vite aux baisses du marché (effet de levier)
            returns_1d = df['Close'].pct_change()
            leverage_effect = np.where(returns_1d < -0.01, np.abs(returns_1d) * 1.5, 0)
            
            ewma_vol = df['RV_GarmanKlass'].ewm(span=10, adjust=False).mean()
            ar_noise = generate_micro_noise(n_obs, rho=0.6, sigma=0.003, seed=42 + idx)
            
            # Construction de l'IV_ATM avec comportement assymétrique
            df['IV_ATM'] = np.maximum(0.08, ewma_vol + 0.015 + leverage_effect + ar_noise)
            
            # 3. Dynamic Skew & Term Structure avec Sauts de Panique
            returns_5d = df['Close'].pct_change(5)
            skew_noise = generate_micro_noise(n_obs, rho=0.4, sigma=0.004, seed=100 + idx)
            df['IV_Skew'] = 0.035 - 0.20 * np.minimum(0, returns_5d) + skew_noise
            
            # Choc de volatilité : si l'IV dépasse sa moyenne mobile 60j, la structure passe en Backwardation
            vol_ma60 = df['IV_ATM'].rolling(60).mean()
            vol_shock = (df['IV_ATM'] - vol_ma60) / (vol_ma60 + 1e-8)
            
            term_noise = generate_micro_noise(n_obs, rho=0.5, sigma=0.003, seed=200 + idx)
            df['IV_Term_Structure'] = 0.025 - 0.12 * np.maximum(-0.2, vol_shock) + term_noise
            
            # 4. Microstructure : Élargissement Exponentiel du Bid-Ask Spread en Crise
            # Le spread de base est de 1-2 bps, mais explose lors des chocs de vol (> 25% IV)
            base_spread = 0.0002
            vol_stress_multiplier = np.exp(np.maximum(0, df['IV_ATM'] - 0.20) * 12)
            
            # Ajout de chocs de liquidité stochastiques (Sauts de Poisson / Fat Tails)
            liquidity_jumps = np.random.poisson(lam=0.02, size=n_obs) * np.random.uniform(0.002, 0.008, size=n_obs)
            
            df['BidAskSpread'] = base_spread + (df['IV_ATM'] * 0.0005 * vol_stress_multiplier) + liquidity_jumps
            df['GEX'] = returns_5d.rolling(20).mean() * 1e7 + np.random.normal(1e6, 3e5, n_obs)
            
            df = df.dropna()
            df.to_csv(cache_path)
            
        market_data[asset] = df
        
    return market_data