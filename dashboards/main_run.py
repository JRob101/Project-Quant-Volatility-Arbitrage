import numpy as np
import xgboost as xgb
import pandas as pd
import warnings

from data.data_engineering import generate_synthetic_market_data, compute_advanced_features
from calibration.model_pipeline import PurgedGroupTimeSeriesSplit, backtest_multi_structure_strategy

warnings.filterwarnings('ignore')

CLASS_MAP = {
    0: 'NO_TRADE',
    1: 'IRON_CONDOR',
    2: 'LONG_STRANGLE',
    3: 'RISK_REVERSAL'
}

def main():
    print("1. Génération des données synthétiques...")
    raw_df = generate_synthetic_market_data(n_days=1500)
    
    print("2. Calcul des features et Target Multi-Classes...")
    df_features = compute_advanced_features(raw_df)
    
    features = [
        'RV_GarmanKlass', 'Vol_Premium', 'Spread_ZScore', 
        'GEX_Normalized', 'Skew_Momentum', 'IV_Term_Structure', 'IV_EWMA_Diff'
    ]
    X = df_features[features]
    y = df_features['Target']
    
    print("3. Entraînement XGBClassifier Multi-Classes (Configuration Référence)...")
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=5, embargo_window=5)
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=4, random_state=42
    )
    
    prob_grid = [0.30, 0.35, 0.40, 0.45, 0.50]
    results_table = []
    
    for prob_thresh in prob_grid:
        all_daily_pnls = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            proba_matrix = model.predict_proba(X_test)
            
            df_test_period = df_features.iloc[test_idx]
            
            strategy_signals = []
            position_sizes = []
            cooldown_counter = 0
            
            for i in range(len(proba_matrix)):
                if cooldown_counter > 0:
                    strategy_signals.append('NO_TRADE')
                    position_sizes.append(0.0)
                    cooldown_counter -= 1
                    continue
                    
                probs = proba_matrix[i]
                best_class = int(np.argmax(probs))
                max_p = probs[best_class]
                
                if best_class != 0 and max_p >= prob_thresh:
                    strat = CLASS_MAP[best_class]
                    size = np.clip(max_p * 2.5, 1.0, 2.0)
                    cooldown_counter = 3
                else:
                    strat = 'NO_TRADE'
                    size = 0.0
                    
                strategy_signals.append(strat)
                position_sizes.append(size)
                
            daily_pnls = backtest_multi_structure_strategy(
                df_test_period, strategy_signals, position_sizes=position_sizes, holding_period=5
            )
            all_daily_pnls.extend(daily_pnls)
        
        all_daily_pnls = np.array(all_daily_pnls)
        total_pnl = np.sum(all_daily_pnls)
        sharpe_ratio = (np.mean(all_daily_pnls) / (np.std(all_daily_pnls) + 1e-8)) * np.sqrt(252)
        max_drawdown = np.max(np.maximum.accumulate(np.cumsum(all_daily_pnls)) - np.cumsum(all_daily_pnls))
        
        results_table.append({
            'Seuil Confiance (p*)': f"Prob >= {prob_thresh}",
            'PnL Cumulé ($)': round(total_pnl, 2),
            'Ratio de Sharpe': round(sharpe_ratio, 2),
            'Max Drawdown ($)': round(max_drawdown, 2)
        })

    print("\n" + "="*65)
    print("   RÉSULTATS DE RÉFÉRENCE - CLASSIFICATION MULTI-CLASSES (PURE)")
    print("="*65)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))
    print("="*65)

if __name__ == "__main__":
    main()