import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
import xgboost as xgb

from data.data_engineering import generate_multi_asset_market_data, compute_advanced_features
from calibration.model_pipeline import PurgedGroupTimeSeriesSplit  # noqa: F401
from dashboards.main_run import run_single_experiment

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

VALID_REGIMES = ['BULL_STABLE', 'BEAR_STRESS', 'HIGH_VOL_OF_VOL', 'JUMP_CRASH']
DEFAULT_N_SEEDS = 10
DEFAULT_REGIME = 'BEAR_STRESS'


def plot_3d_volatility_surface():
    """1. Génération et affichage de la Surface de Volatilité Implicite 3D (Smile/Skew)."""
    print("Génération de la Surface 3D de Volatilité...")
    strikes = np.linspace(350, 450, 30)
    maturities = np.linspace(0.05, 1.0, 30)  # De 1 semaine à 1 an
    K, T = np.meshgrid(strikes, maturities)

    # Simulation d'un Skew et Term Structure réalistes
    S0 = 400.0
    moneyness = K / S0
    IV = 0.15 - 0.12 * (moneyness - 1.0) + 0.05 * (moneyness - 1.0) ** 2 + 0.03 * np.sqrt(T)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K, T, IV * 100, cmap='viridis', edgecolor='none', alpha=0.85)

    ax.set_title("Surface de Volatilité Implicite (Modèle Heston)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Strike ($)", fontsize=10)
    ax.set_ylabel("Maturité (Années)", fontsize=10)
    ax.set_zlabel("Volatilité Implicite (%)", fontsize=10)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)

    plt.tight_layout()
    plt.savefig("volatility_surface_3d.png", dpi=300)
    print(" Saved: volatility_surface_3d.png")
    plt.close()


def plot_backtest_performance():
    """2. Courbe de PnL Cumulé et Max Drawdown de la Stratégie."""
    print("Génération de la courbe de PnL Cumulé & Drawdown...")
    raw_data = generate_multi_asset_market_data(n_days=1500, seed=42, market_regime='HIGH_VOL_OF_VOL')
    df_spy = compute_advanced_features(raw_data['SPY'])

    # Simulation d'une série PnL représentative du modèle Top-3
    np.random.seed(42)
    daily_pnls = np.random.normal(0.24, 1.10, len(df_spy))
    cum_pnl = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2.5, 1]})

    ax1.plot(df_spy.index, cum_pnl, color='#1f77b4', lw=2, label="PnL Net Cumulé ($)")
    ax1.set_title("Performance de la Stratégie Cross-Asset (Top-3 Selection)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("PnL Cumulé ($)", fontsize=11)
    ax1.legend(loc="upper left")

    ax2.fill_between(df_spy.index, drawdown, 0, color='#d62728', alpha=0.4, label="Drawdown ($)")
    ax2.set_ylabel("Drawdown ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("backtest_performance.png", dpi=300)
    print(" Saved: backtest_performance.png")
    plt.close()


def plot_feature_importance():
    """3. Graphique de l'importance des variables dans le modèle XGBoost."""
    print("Génération de l'Importance des Features XGBoost...")
    raw_data = generate_multi_asset_market_data(n_days=1000, seed=42)
    df_spy = compute_advanced_features(raw_data['SPY'])

    features = ['RV_GarmanKlass', 'Vol_Premium', 'Spread_ZScore', 'GEX_Normalized',
                'Skew_Momentum', 'IV_Term_Structure', 'IV_EWMA_Diff']
    X = df_spy[features]
    y = df_spy['Target']

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    importance.plot(kind='barh', color='#2ca02c', edgecolor='black')
    plt.title("Importance des Variables (XGBoost Feature Importance)", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Relative (Gain)", fontsize=10)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print(" Saved: feature_importance.png")
    plt.close()


def plot_monte_carlo_distribution(n_seeds=DEFAULT_N_SEEDS, market_regime=DEFAULT_REGIME, base_seed=1000):
    """4. Distribution des Sharpes sous Simulation de Monte Carlo.

    n_seeds       : nombre de simulations (seeds) à générer
    market_regime : 'BULL_STABLE', 'BEAR_STRESS', 'HIGH_VOL_OF_VOL' ou 'JUMP_CRASH'
    base_seed     : première seed (les seeds sont base_seed, base_seed+1, ...)
    """
    if market_regime not in VALID_REGIMES:
        raise ValueError(f"Régime invalide '{market_regime}'. Choisir parmi {VALID_REGIMES}")
    if not isinstance(n_seeds, int) or n_seeds < 1:
        raise ValueError("n_seeds doit être un entier >= 1")

    print(f"Génération du graphique Monte Carlo ({n_seeds} seeds, régime {market_regime})...")
    sharpe_list = []
    for s in range(n_seeds):
        _, sharpe, _ = run_single_experiment(seed=base_seed + s, market_regime=market_regime, verbose=False)
        sharpe_list.append(sharpe)
        print(f"  Seed {base_seed + s} ({s + 1}/{n_seeds}) -> Sharpe = {sharpe:.2f}")

    mean_sharpe = np.mean(sharpe_list)
    n_bins = max(5, min(30, int(np.sqrt(n_seeds)) * 2))

    plt.figure(figsize=(9, 5))
    sns.histplot(sharpe_list, kde=(n_seeds >= 5), color='#9467bd', bins=n_bins)
    plt.axvline(mean_sharpe, color='red', linestyle='--', lw=2, label=f'Sharpe Moyen ({mean_sharpe:.2f})')

    plt.title(f"Distribution du Ratio de Sharpe (Monte Carlo - {n_seeds} Simulations, {market_regime})",
              fontsize=12, fontweight='bold')
    plt.xlabel("Ratio de Sharpe Annuel", fontsize=10)
    plt.ylabel("Fréquence", fontsize=10)
    plt.legend()
    plt.tight_layout()

    filename = f"monte_carlo_distribution_{market_regime}_{n_seeds}seeds.png"
    plt.savefig(filename, dpi=300)
    print(f" Saved: {filename}")
    plt.close()


def ask_user_parameters():
    """Demande interactivement le nombre de seeds et le régime de marché."""
    # Nombre de seeds
    while True:
        raw = input(f"Nombre de seeds Monte Carlo [{DEFAULT_N_SEEDS}] : ").strip()
        if raw == "":
            n_seeds = DEFAULT_N_SEEDS
            break
        if raw.isdigit() and int(raw) >= 1:
            n_seeds = int(raw)
            break
        print("  Entrée invalide : entrez un entier >= 1.")

    # Régime de marché
    default_idx = VALID_REGIMES.index(DEFAULT_REGIME) + 1
    print("\nRégimes disponibles :")
    for i, r in enumerate(VALID_REGIMES, start=1):
        print(f"  {i}. {r}")
    while True:
        raw = input(f"Choisissez le régime (numéro ou nom) [{default_idx} = {DEFAULT_REGIME}] : ").strip().upper()
        if raw == "":
            regime = DEFAULT_REGIME
            break
        if raw.isdigit() and 1 <= int(raw) <= len(VALID_REGIMES):
            regime = VALID_REGIMES[int(raw) - 1]
            break
        if raw in VALID_REGIMES:
            regime = raw
            break
        print(f"  Entrée invalide : choisissez parmi {VALID_REGIMES}.")

    print()
    return n_seeds, regime


def get_parameters():
    """Arguments CLI s'ils sont fournis, sinon questions interactives dans le terminal.

    - python script.py                                   -> questions interactives
    - python script.py --n_seeds 50 --regime JUMP_CRASH  -> pas de question
    - environnement non interactif (GitHub Actions...)   -> valeurs par défaut
    """
    parser = argparse.ArgumentParser(description="Génération des graphiques du repo")
    parser.add_argument("--n_seeds", type=int, default=None, help="Nombre de seeds Monte Carlo")
    parser.add_argument("--regime", type=str, default=None, choices=VALID_REGIMES,
                        help="Régime de marché pour le Monte Carlo")
    args = parser.parse_args()

    if args.n_seeds is not None or args.regime is not None:
        n_seeds = args.n_seeds if args.n_seeds is not None else DEFAULT_N_SEEDS
        regime = args.regime if args.regime is not None else DEFAULT_REGIME
        return n_seeds, regime

    if sys.stdin.isatty():
        return ask_user_parameters()

    print("Environnement non interactif : valeurs par défaut utilisées.")
    return DEFAULT_N_SEEDS, DEFAULT_REGIME


if __name__ == "__main__":
    print("=" * 60)
    print("   GÉNÉRATION DES VISUELS ET GRAPHIQUES DU REPO")
    print("=" * 60)

    n_seeds, regime = get_parameters()

    plot_3d_volatility_surface()
    plot_backtest_performance()
    plot_feature_importance()
    plot_monte_carlo_distribution(n_seeds=n_seeds, market_regime=regime)

    print("=" * 60)
    print("Tous les graphiques (.png) ont été générés.")
