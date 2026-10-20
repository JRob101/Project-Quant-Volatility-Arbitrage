import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings

from data.data_engineering import (
    generate_multi_asset_market_data, 
    generate_dynamic_market_data, 
    compute_advanced_features
)
from data.fetch_real_data import fetch_real_cross_asset_data
from calibration.model_pipeline import PurgedGroupTimeSeriesSplit, backtest_multi_structure_strategy

warnings.filterwarnings('ignore')


def route_strategy(pred_target, z_score, term_struct, iv_atm, min_iv_thresh=0.12):
    """
    Routage conditionnel avec Filtre Macro (IV Min & Term Structure).
    """
    if iv_atm < min_iv_thresh or term_struct < -0.05:
        return 'NO_TRADE'

    if pred_target < -0.015 and z_score > 1.2:
        return 'SHORT_STRADDLE'
    elif -0.015 <= pred_target < -0.005 and 0.4 <= z_score <= 1.2:
        return 'IRON_CONDOR'
    elif pred_target > 0.01 and z_score < -0.8:
        return 'LONG_STRANGLE'
    elif pred_target < -0.005:
        return 'SHORT_STRANGLE'
    else:
        return 'NO_TRADE'


def run_single_experiment(seed=42, data_mode='REAL', market_regime='BEAR_STRESS', start_year=2015, end_year=2015, use_simple_split=False, verbose=True):
    assets = ['SPY', 'QQQ', 'IWM']
    
    # 1. Chargement des données
    if data_mode == 'REAL':
        if verbose:
            print(f"\n>>> Chargement des DONNÉES RÉELLES de marché (SPY, QQQ, IWM via yfinance {start_year}-{end_year})...")
        raw_market_data = fetch_real_cross_asset_data(
            assets=assets, 
            start_year=start_year, 
            end_year=end_year, 
            use_cache=False
        )
    elif data_mode == 'SYNTHETIC_DYNAMIC':
        if verbose:
            print("\n>>> Génération des DONNÉES SYNTHÉTIQUES DYNAMIQUES (Markov-Switching)...")
        raw_market_data = generate_dynamic_market_data(n_days=1260, assets=assets, seed=seed, verbose=verbose)
    else:
        if verbose:
            print(f"\n>>> Génération des DONNÉES SYNTHÉTIQUES STATIQUES (Régime: {market_regime})...")
        raw_market_data = generate_multi_asset_market_data(n_days=1500, assets=assets, seed=seed, market_regime=market_regime)

    # 2. Feature Engineering
    df_features = compute_advanced_features(raw_market_data['SPY'])
    
    features = ['RV_GarmanKlass', 'Vol_Premium', 'Spread_ZScore', 'GEX_Normalized', 'Skew_Momentum', 'IV_Term_Structure']
    X = df_features[features]
    y = df_features['Target']

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=seed
    )

    all_daily_pnls = []

    # ------------------------------------------------------------------
    # OPTION A : SIMPLE TRAIN/TEST SPLIT (80/20 CHRONOLOGIQUE)
    # ------------------------------------------------------------------
    if use_simple_split:
        if verbose:
            print(">>> Mode d'évaluation : SIMPLE TRAIN/TEST SPLIT (80% Train / 20% Test)...")
        
        split_idx = int(len(X) * 0.8)
        train_idx = list(range(0, split_idx))
        test_idx = list(range(split_idx, len(X)))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        df_test_period = df_features.iloc[test_idx]
        train_std = y_train.std() + 1e-8

        strategy_signals = []
        position_sizes = []

        for i in range(len(preds)):
            z_sc = df_test_period['Spread_ZScore'].iloc[i]
            t_struct = df_test_period['IV_Term_Structure'].iloc[i]
            iv_atm = df_test_period['IV_ATM'].iloc[i]

            strat = route_strategy(preds[i], z_sc, t_struct, iv_atm)
            strategy_signals.append(strat)

            if strat != 'NO_TRADE':
                size = np.clip(np.abs(preds[i]) / train_std, 1.0, 5.0)
            else:
                size = 0.0
            position_sizes.append(size)

        all_daily_pnls = backtest_multi_structure_strategy(
            df_test_period, strategy_signals, position_sizes=position_sizes
        )

        if verbose:
            mse = mean_squared_error(y_test, preds)
            n_trades = sum(1 for s in strategy_signals if s != 'NO_TRADE')
            print(f"   Simple Split - MSE: {mse:.6f} | Trades: {n_trades} sur {len(preds)} jours de test")

    # ------------------------------------------------------------------
    # OPTION B : PURGED GROUP TIME SERIES SPLIT (DEFAUT)
    # ------------------------------------------------------------------
    else:
        if verbose:
            print(">>> Mode d'évaluation : PURGED GROUP TIME SERIES SPLIT...")
        
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=5, embargo_window=5)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            df_test_period = df_features.iloc[test_idx]
            train_std = y_train.std() + 1e-8

            strategy_signals = []
            position_sizes = []

            for i in range(len(preds)):
                z_sc = df_test_period['Spread_ZScore'].iloc[i]
                t_struct = df_test_period['IV_Term_Structure'].iloc[i]
                iv_atm = df_test_period['IV_ATM'].iloc[i]

                strat = route_strategy(preds[i], z_sc, t_struct, iv_atm)
                strategy_signals.append(strat)

                if strat != 'NO_TRADE':
                    size = np.clip(np.abs(preds[i]) / train_std, 1.0, 5.0)
                else:
                    size = 0.0
                position_sizes.append(size)

            daily_pnls = backtest_multi_structure_strategy(
                df_test_period, strategy_signals, position_sizes=position_sizes
            )
            all_daily_pnls.extend(daily_pnls)

            if verbose:
                mse = mean_squared_error(y_test, preds)
                n_trades = sum(1 for s in strategy_signals if s != 'NO_TRADE')
                print(f"   Fold {fold_idx+1} - MSE: {mse:.6f} | Trades: {n_trades}")

    # 4. Performance
    all_daily_pnls = np.array(all_daily_pnls)
    total_pnl = np.sum(all_daily_pnls)
    sharpe_ratio = (np.mean(all_daily_pnls) / (np.std(all_daily_pnls) + 1e-8)) * np.sqrt(252)
    running_max = np.maximum.accumulate(np.cumsum(all_daily_pnls)) if len(all_daily_pnls) > 0 else np.array([0])
    max_drawdown = np.max(running_max - np.cumsum(all_daily_pnls)) if len(all_daily_pnls) > 0 else 0.0

    if verbose:
        print("\n" + "="*50)
        print(f"   RÉSULTATS DE LA STRATÉGIE ({data_mode})")
        print("="*50)
        print(f"   PnL Cumulé Net :         ${total_pnl:.2f}")
        print(f"   Ratio de Sharpe Annuel :  {sharpe_ratio:.2f}")
        print(f"   Max Drawdown ($) :        ${max_drawdown:.2f}")
        print("="*50 + "\n")

    return total_pnl, sharpe_ratio, max_drawdown


def main():
    print("="*65)
    print("      LANCEMENT DU BACKTEST D'ARBITRAGE DE VOLATILITÉ")
    print("="*65)
    print("\nChoisissez le type de données à utiliser :")
    print(" 1. Données Réelles de Marché (SPY, QQQ, IWM via yfinance avec plage au choix)")
    print(" 2. Données Synthétiques Dynamiques (Simulation Markov-Switching sur 5 ans)")
    print(" 3. Données Synthétiques Statiques (Régime de Marché Fixe)")

    choice = input("\nVotre choix (1-3) [Défaut: 1]: ").strip()

    if choice == '2':
        run_single_experiment(data_mode='SYNTHETIC_DYNAMIC')
    elif choice == '3':
        print("\nChoix du régime statique :")
        print(" a. BULL_STABLE")
        print(" b. BEAR_STRESS")
        print(" c. HIGH_VOL_OF_VOL")
        print(" d. JUMP_CRASH")
        reg_choice = input("Sélectionnez le régime (a-d) [Défaut: b]: ").strip().lower()
        reg_map = {'a': 'BULL_STABLE', 'b': 'BEAR_STRESS', 'c': 'HIGH_VOL_OF_VOL', 'd': 'JUMP_CRASH'}
        selected_regime = reg_map.get(reg_choice, 'BEAR_STRESS')
        run_single_experiment(data_mode='SYNTHETIC_STATIC', market_regime=selected_regime)
    else:
        print("\n--- Configuration de la plage de dates pour les Données Réelles ---")
        try:
            start_yr_in = input("Entrez l'année de début (ex: 2015) [Défaut: 2015]: ").strip()
            start_year = int(start_yr_in) if start_yr_in else 2015
            
            end_yr_in = input("Entrez l'année de fin (ex: 2015) [Défaut: 2015]: ").strip()
            end_year = int(end_yr_in) if end_yr_in else 2015
        except ValueError:
            print("Entrée invalide, utilisation de l'année par défaut (2015).")
            start_year, end_year = 2015, 2015

        split_choice = input("\nUtiliser un Simple Split 80/20 à la place du Purged Group Split (Idéal pour une stratégie courte sur 1 ou 2 ans) ? (y/N) [Défaut: N]: ").strip().lower()
        use_simple = True if split_choice == 'y' else False

        run_single_experiment(
            data_mode='REAL', 
            start_year=start_year, 
            end_year=end_year, 
            use_simple_split=use_simple
        )


if __name__ == "__main__":
    main()