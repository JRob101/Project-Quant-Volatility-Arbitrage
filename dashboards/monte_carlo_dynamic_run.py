import numpy as np
import pandas as pd
import warnings

from data.data_engineering import generate_dynamic_market_data, compute_advanced_features, REGIME_NAMES
from dashboards.main_run import run_single_experiment

warnings.filterwarnings('ignore')


def run_dynamic_monte_carlo(n_simulations=10, n_days=1260, min_iv_thresh=0.12):
    """
    Simule N trajectoires dynamiques sur 5 ans (Markov-Switching) 
    et évalue la stratégie d'arbitrage de volatilité sur chaque chemin.
    """
    print("=" * 65)
    print("   ANALYSE MONTE CARLO - MODÈLE À CHANGEMENT DE RÉGIME DYNAMIQUE")
    print("=" * 65)
    print(f" Nombre de trajectoires simulées : {n_simulations}")
    print(f" Horizon par simulation          : {n_days} jours (~5 ans)")
    print(f" Filtre IV_ATM minimum          : {min_iv_thresh * 100:.1f}%")
    print("=" * 65 + "\n")

    pnl_results = []
    sharpe_results = []
    max_dd_results = []
    regime_distributions = {reg: [] for reg in REGIME_NAMES}

    for i in range(n_simulations):
        current_seed = 2000 + i
        
        # 1. Génération des données dynamiques pour statistiques de régimes
        market_data = generate_dynamic_market_data(
            n_days=n_days, 
            seed=current_seed, 
            verbose=False
        )
        regime_series = market_data['SPY']['Regime']
        counts = pd.Series(regime_series).value_counts()
        
        for reg in REGIME_NAMES:
            pct = (counts.get(reg, 0) / n_days) * 100
            regime_distributions[reg].append(pct)

        # 2. Backtest de la stratégie sur ce chemin unique
        pnl, sharpe, max_dd = run_single_experiment(
            seed=current_seed, 
            market_regime='DYNAMIC',  
            verbose=False
        )

        pnl_results.append(pnl)
        sharpe_results.append(sharpe)
        max_dd_results.append(max_dd)

        bull_pct = (counts.get('BULL_STABLE', 0) / n_days) * 100
        bear_pct = (counts.get('BEAR_STRESS', 0) / n_days) * 100
        vol_pct = (counts.get('HIGH_VOL_OF_VOL', 0) / n_days) * 100
        jump_pct = (counts.get('JUMP_CRASH', 0) / n_days) * 100

        print(
            f" Sim {i+1:02d}/{n_simulations:02d} (Seed {current_seed}) | "
            f"PnL: ${pnl:7.2f} | Sharpe: {sharpe:5.2f} | MaxDD: ${max_dd:6.2f} | "
            f"[BULL:{bull_pct:.0f}% BEAR:{bear_pct:.0f}% VOL:{vol_pct:.0f}% JUMP:{jump_pct:.0f}%]"
        )

    # 3. Synthèse statistique globale
    print("\n" + "=" * 65)
    print("   SYNTHÈSE STATISTIQUE MONTE CARLO DYNAMIQUE (MARKOV)")
    print("=" * 65)
    print(f" Ratio de Sharpe Moyen :   {np.mean(sharpe_results):.2f}  (Std: {np.std(sharpe_results):.2f})")
    print(f" Ratio de Sharpe Médian :  {np.median(sharpe_results):.2f}")
    print(f" Sharpe Min / Max :        {np.min(sharpe_results):.2f} / {np.max(sharpe_results):.2f}")
    print(f" PnL Moyen Net ($) :       ${np.mean(pnl_results):.2f}")
    print(f" Pire Max Drawdown ($) :   ${np.max(max_dd_results):.2f}")
    print("-" * 65)
    print(" Répartition moyenne des régimes sur l'ensemble des tirages :")
    for reg in REGIME_NAMES:
        mean_pct = np.mean(regime_distributions[reg])
        print(f"   • {reg:<16} : {mean_pct:5.1f}% du temps")
    print("=" * 65)


def main():
    n_sims_input = input("Nombre de trajectoires Monte Carlo à simuler [Défaut: 10]: ").strip()
    try:
        n_simulations = int(n_sims_input)
    except ValueError:
        n_simulations = 10

    iv_input = input("Seuil minimum d'IV_ATM (%) [Défaut: 12]: ").strip()
    try:
        min_iv_thresh = float(iv_input) / 100.0
    except ValueError:
        min_iv_thresh = 0.12

    run_dynamic_monte_carlo(n_simulations=n_simulations, n_days=1260, min_iv_thresh=min_iv_thresh)


if __name__ == "__main__":
    main()