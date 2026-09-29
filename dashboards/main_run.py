"""Application entry point for dashboard workflows."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings

from data.data_engineering import generate_synthetic_market_data, compute_advanced_features
from calibration.model_pipeline import PurgedGroupTimeSeriesSplit, backtest_delta_neutral_strategy

warnings.filterwarnings('ignore')

def main():
    print("1. Génération des données synthétiques (Modèle de Heston)...")
    raw_df = generate_synthetic_market_data(n_days=1500)
    
    print("2. Calcul des features et Target de régression...")
    df_features = compute_advanced_features(raw_df)
    
    features = ['RV_GarmanKlass', 'Vol_Premium', 'Spread_ZScore', 'GEX_Normalized', 'Skew_Momentum', 'IV_Term_Structure']
    X = df_features[features]
    y = df_features['Target']
    
    print("3. Entraînement ML via Purged CV (Régression)...")
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=5, embargo_window=5)
    
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    all_daily_pnls = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        signals = (preds < -0.01).astype(int)
        
        mse = mean_squared_error(y_test, preds)
        print(f"   Fold {fold_idx+1} - MSE: {mse:.6f} | Trades déclenchés : {np.sum(signals)}")
        
        df_test_period = df_features.iloc[test_idx]
        daily_pnls = backtest_delta_neutral_strategy(df_test_period, signals)
        all_daily_pnls.extend(daily_pnls)
    
    print("\n4. Agrégation des résultats du backtest...")
    all_daily_pnls = np.array(all_daily_pnls)
    total_pnl = np.sum(all_daily_pnls)
    sharpe_ratio = (np.mean(all_daily_pnls) / (np.std(all_daily_pnls) + 1e-8)) * np.sqrt(252)
    max_drawdown = np.max(np.maximum.accumulate(np.cumsum(all_daily_pnls)) - np.cumsum(all_daily_pnls))
    
    print("\n" + "="*50)
    print("   RÉSULTATS DE LA STRATÉGIE (PRICER NUMBA)")
    print("="*50)
    print(f"   PnL Cumulé Net :         ${total_pnl:.2f}")
    print(f"   Ratio de Sharpe Annuel :  {sharpe_ratio:.2f}")
    print(f"   Max Drawdown ($) :        ${max_drawdown:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()