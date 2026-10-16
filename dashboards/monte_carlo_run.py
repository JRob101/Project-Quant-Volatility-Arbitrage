import numpy as np
import pandas as pd
import warnings
from dashboards.main_run import run_single_experiment

warnings.filterwarnings('ignore')

AVAILABLE_REGIMES = ['BULL_STABLE', 'BEAR_STRESS', 'HIGH_VOL_OF_VOL', 'JUMP_CRASH']

def main():
    print("="*65)
    print("      MOTEUR D'ANALYSE MONTE CARLO ET STRESS-TESTING MULTI-RÉGIMES")
    print("="*65)
    
    print("\nChoix des Régimes de Marché :")
    print(" 1. BULL_STABLE       (Marché Haussier Calme)")
    print(" 2. BEAR_STRESS       (Marché Baissier à Forte Volatilité)")
    print(" 3. HIGH_VOL_OF_VOL   (Incertitude et Volatilité Eratique)")
    print(" 4. JUMP_CRASH        (Risque de Krach et Sauts de Poisson)")
    
    regime_choice = input("\nChoisissez un régime (1-4) [Defaut: 1]: ").strip()
    regime_map = {'1': 'BULL_STABLE', '2': 'BEAR_STRESS', '3': 'HIGH_VOL_OF_VOL', '4': 'JUMP_CRASH'}
    selected_regime = regime_map.get(regime_choice, 'BULL_STABLE')
    
    n_seeds_input = input("Nombre de seeds/simulations à exécuter [Defaut: 10]: ").strip()
    try:
        n_simulations = int(n_seeds_input)
    except ValueError:
        n_simulations = 10
        
    print(f"\nLancement de {n_simulations} simulations sur {selected_regime}...\n")
    
    pnl_results = []
    sharpe_results = []
    max_dd_results = []
    
    for i in range(n_simulations):
        current_seed = 1000 + i
        pnl, sharpe, max_dd = run_single_experiment(
            seed=current_seed, 
            market_regime=selected_regime, 
            verbose=False
        )
        
        pnl_results.append(pnl)
        sharpe_results.append(sharpe)
        max_dd_results.append(max_dd)
        
        print(f" Sim {i+1:02d}/{n_simulations:02d} (Seed {current_seed}) | PnL: ${pnl:6.2f} | Sharpe: {sharpe:4.2f} | MaxDD: ${max_dd:5.2f}")
        
    print("\n" + "="*65)
    print(f"   SYNTHÈSE STATISTIQUE MONTE CARLO ({selected_regime})")
    print("="*65)
    print(f"   Ratio de Sharpe Moyen :   {np.mean(sharpe_results):.2f}  (Std: {np.std(sharpe_results):.2f})")
    print(f"   Ratio de Sharpe Médian :  {np.median(sharpe_results):.2f}")
    print(f"   Sharpe Min / Max :        {np.min(sharpe_results):.2f} / {np.max(sharpe_results):.2f}")
    print(f"   PnL Moyen Net ($) :       ${np.mean(pnl_results):.2f}")
    print(f"   Pire Max Drawdown ($) :   ${np.max(max_dd_results):.2f}")
    print("="*65)

if __name__ == "__main__":
    main()