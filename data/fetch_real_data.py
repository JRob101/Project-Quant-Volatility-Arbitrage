import os
import numpy as np
import pandas as pd
import yfinance as yf
from data.data_engineering import compute_advanced_features

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
    Télécharge les historiques réels pour SPY, QQQ et IWM via yfinance pour une plage d'années définie.
    Permet de sélectionner une année unique si start_year == end_year (ex: 2020 à 2020 pour l'année complète 2020).
    """
    start_date = f"{start_year}-01-01"
    
    # Si la même année est choisie pour le début et la fin, on s'arrête au 1er janvier de l'année suivante (exclusif)
    if start_year == end_year:
        end_date = f"{end_year + 1}-01-01"
    else:
        end_date = f"{end_year}-01-01"
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    market_data = {}
    
    for idx, asset in enumerate(assets):
        cache_path = os.path.join(CACHE_DIR, f"{asset}_advanced_proxy_{start_year}_{end_year}.csv")
        
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            print(f"Téléchargement des cours réels yfinance pour {asset} [{start_date} -> {end_date}]...")
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
            returns_1d = df['Close'].pct_change()
            leverage_effect = np.where(returns_1d < -0.01, np.abs(returns_1d) * 1.5, 0)
            
            ewma_vol = df['RV_GarmanKlass'].ewm(span=10, adjust=False).mean()
            ar_noise = generate_micro_noise(n_obs, rho=0.6, sigma=0.003, seed=42 + idx)
            
            df['IV_ATM'] = np.maximum(0.08, ewma_vol + 0.015 + leverage_effect + ar_noise)
            
            # 3. Dynamic Skew & Term Structure avec Sauts de Panique
            returns_5d = df['Close'].pct_change(5)
            skew_noise = generate_micro_noise(n_obs, rho=0.4, sigma=0.004, seed=100 + idx)
            df['IV_Skew'] = 0.035 - 0.20 * np.minimum(0, returns_5d) + skew_noise
            
            vol_ma60 = df['IV_ATM'].rolling(60).mean()
            vol_shock = (df['IV_ATM'] - vol_ma60) / (vol_ma60 + 1e-8)
            
            term_noise = generate_micro_noise(n_obs, rho=0.5, sigma=0.003, seed=200 + idx)
            df['IV_Term_Structure'] = 0.025 - 0.12 * np.maximum(-0.2, vol_shock) + term_noise
            
            # 4. Microstructure : Élargissement Exponentiel du Bid-Ask Spread en Crise
            base_spread = 0.0002
            vol_stress_multiplier = np.exp(np.maximum(0, df['IV_ATM'] - 0.20) * 12)
            liquidity_jumps = np.random.poisson(lam=0.02, size=n_obs) * np.random.uniform(0.002, 0.008, size=n_obs)
            
            df['BidAskSpread'] = base_spread + (df['IV_ATM'] * 0.0005 * vol_stress_multiplier) + liquidity_jumps
            df['GEX'] = returns_5d.rolling(20).mean() * 1e7 + np.random.normal(1e6, 3e5, n_obs)
            
            df = df.dropna()
            df.to_csv(cache_path)
            
        market_data[asset] = df
        
    return market_data