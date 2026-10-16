import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

from data.data_engineering import generate_multi_asset_market_data, compute_advanced_features
from calibration.model_pipeline import PurgedGroupTimeSeriesSplit, backtest_cross_asset_strategy

warnings.filterwarnings('ignore')

CLASS_MAP = {
    0: 'NO_TRADE',
    1: 'IRON_CONDOR',
    2: 'LONG_STRANGLE',
    3: 'RISK_REVERSAL',
    4: 'CALENDAR_SPREAD',
    5: 'DIAGONAL_SPREAD'
}

def run_single_experiment(seed=42, market_regime='BULL_STABLE', verbose=False):
    assets = ['SPY', 'QQQ', 'IWM']
    raw_market_data = generate_multi_asset_market_data(n_days=1500, assets=assets, seed=seed, market_regime=market_regime)
    
    features_by_asset = {}
    for asset in assets:
        features_by_asset[asset] = compute_advanced_features(raw_market_data[asset])
        
    features_list = [
        'RV_GarmanKlass', 'Vol_Premium', 'Spread_ZScore', 
        'Term_Structure_ZScore', 'Slope_Momentum',
        'GEX_Normalized', 'Skew_Momentum', 'IV_Term_Structure', 'IV_EWMA_Diff'
    ]
    
    df_global_list = []
    for asset in assets:
        df_a = features_by_asset[asset].copy()
        df_a['Asset'] = asset
        df_global_list.append(df_a)
    df_global = pd.concat(df_global_list, axis=0)
    
    ref_asset_df = features_by_asset['SPY']
    X_ref = ref_asset_df[features_list]
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=5, embargo_window=5)
    
    prob_thresh = 0.45
    K_max = 3
    all_daily_pnls = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_ref)):
        test_dates = ref_asset_df.index[test_idx]
        train_dates = ref_asset_df.index[train_idx]
        
        df_train_fold = df_global[df_global.index.isin(train_dates)]
        X_train = df_train_fold[features_list]
        y_train = df_train_fold['Target']
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        n_classes_fold = len(le.classes_)
        
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=n_classes_fold, random_state=42
        )
        model.fit(X_train, y_train_encoded)
        
        proba_by_asset = {}
        market_data_test = {}
        for asset in assets:
            df_test_asset = features_by_asset[asset].reindex(test_dates).dropna()
            X_test_asset = df_test_asset[features_list]
            proba_by_asset[asset] = model.predict_proba(X_test_asset)
            market_data_test[asset] = df_test_asset
            
        n_test_days = len(test_dates)
        fold_daily_pnls = np.zeros(n_test_days)
        
        for t in range(n_test_days):
            candidates = []
            for asset in assets:
                if t < len(proba_by_asset[asset]):
                    probs = proba_by_asset[asset][t]
                    best_class_enc = int(np.argmax(probs))
                    p_val = probs[best_class_enc]
                    real_class = le.inverse_transform([best_class_enc])[0]
                    
                    if real_class != 0 and p_val >= prob_thresh:
                        candidates.append({'asset': asset, 'prob': p_val, 'strat': CLASS_MAP[real_class]})
            
            candidates = sorted(candidates, key=lambda x: x['prob'], reverse=True)[:K_max]
            
            if len(candidates) > 0:
                size_per_trade = 3.5 / len(candidates)
                day_pnl_sum = 0.0
                for cand in candidates:
                    single_position_sizes = [0.0] * n_test_days
                    single_position_sizes[t] = size_per_trade
                    pnl_arr = backtest_cross_asset_strategy(
                        market_data_test, 
                        [(cand['strat'], cand['asset']) if i == t else ('NO_TRADE', None) for i in range(n_test_days)],
                        position_sizes=single_position_sizes, 
                        holding_period=5
                    )
                    day_pnl_sum += pnl_arr[t] if t < len(pnl_arr) else 0.0
                fold_daily_pnls[t] = day_pnl_sum
            else:
                fold_daily_pnls[t] = 0.0
                
        all_daily_pnls.extend(fold_daily_pnls)
        
    all_daily_pnls = np.array(all_daily_pnls)
    total_pnl = np.sum(all_daily_pnls)
    sharpe_ratio = (np.mean(all_daily_pnls) / (np.std(all_daily_pnls) + 1e-8)) * np.sqrt(252)
    max_drawdown = np.max(np.maximum.accumulate(np.cumsum(all_daily_pnls)) - np.cumsum(all_daily_pnls))
    
    if verbose:
        print("\n" + "="*65)
        print(f"   EVALUATION UNIQUE | Regime: {market_regime} | Seed: {seed}")
        print("="*65)
        print(f"   PnL Cumulé Net :         ${total_pnl:.2f}")
        print(f"   Ratio de Sharpe Annuel :  {sharpe_ratio:.2f}")
        print(f"   Max Drawdown ($) :        ${max_drawdown:.2f}")
        print("="*65)
        
    return total_pnl, sharpe_ratio, max_drawdown

def main():
    run_single_experiment(seed=42, market_regime='BULL_STABLE', verbose=True)

if __name__ == "__main__":
    main()