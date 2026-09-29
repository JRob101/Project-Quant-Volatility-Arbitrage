"""Option pricing models for volatility arbitrage."""

import numpy as np
from numba import njit

@njit
def _trinomial_backward_induction(N, dt, r, pu, pm, pd_prob, df, S_clean, u, div_times, div_amounts, T, K, is_call, V):
    delta_val = 0.0
    gamma_val = 0.0
    
    for i in range(N - 1, -1, -1):
        t_current = i * dt
        n_nodes = 2 * i + 1
        V_new = np.empty(n_nodes)
        S_true_nodes = np.empty(n_nodes)
        
        pv_divs_current = 0.0
        for k in range(len(div_times)):
            if t_current < div_times[k] <= T:
                pv_divs_current += div_amounts[k] * np.exp(-r * (div_times[k] - t_current))
                
        for j in range(n_nodes):
            j_exp = i - j
            S_true = S_clean * (u ** j_exp) + pv_divs_current
            S_true_nodes[j] = S_true
            
            v_cont = df * (pu * V[j] + pm * V[j+1] + pd_prob * V[j+2])
            
            if is_call == 1:
                v_ex = max(0.0, S_true - K)
            else:
                v_ex = max(0.0, K - S_true)
                
            V_new[j] = max(v_cont, v_ex)
        
        V = V_new
        
        if i == 1:
            dS = S_true_nodes[0] - S_true_nodes[2]
            if dS != 0:
                delta_val = (V[0] - V[2]) / dS
            h1 = S_true_nodes[0] - S_true_nodes[1]
            h2 = S_true_nodes[1] - S_true_nodes[2]
            if h1 > 0 and h2 > 0:
                gamma_val = 2 * ((V[0] - V[1]) / h1 - (V[1] - V[2]) / h2) / (h1 + h2)
                
    return V[0], delta_val, gamma_val

def american_trinomial_option_price_and_greeks(S, K, T, r, div_schedule, sigma, option_type='call', N=100):
    if T <= 0 or sigma <= 0:
        val = max(0, S - K) if option_type == 'call' else max(0, K - S)
        return {'price': val, 'delta': 0, 'gamma': 0, 'vega': 0}

    pv_divs = sum([D * np.exp(-r * t_div) for t_div, D in div_schedule if 0 < t_div <= T])
    S_clean = max(S - pv_divs, 1e-4)

    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    
    nu = r - 0.5 * sigma**2
    pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / (dx**2) + (nu * dt) / dx)
    pd_prob = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / (dx**2) - (nu * dt) / dx)
    pm = 1.0 - pu - pd_prob
    df_factor = np.exp(-r * dt)

    j_values = np.arange(N, -N - 1, -1)
    ST_clean = S_clean * (u ** j_values)
    ST = ST_clean + sum([D for t_div, D in div_schedule if t_div > T])

    is_call = 1 if option_type == 'call' else 0
    if is_call == 1:
        V = np.maximum(0, ST - K)
    else:
        V = np.maximum(0, K - ST)

    div_times = np.array([d[0] for d in div_schedule]) if div_schedule else np.array([])
    div_amounts = np.array([d[1] for d in div_schedule]) if div_schedule else np.array([])

    price, delta_val, gamma_val = _trinomial_backward_induction(
        N, dt, r, pu, pm, pd_prob, df_factor, S_clean, u, div_times, div_amounts, T, K, is_call, V
    )

    v_up = american_trinomial_option_price_and_greeks(S, K, T, r, div_schedule, sigma + 0.01, option_type, N)['price']
    vega_val = v_up - price

    return {'price': price, 'delta': delta_val, 'gamma': gamma_val, 'vega': vega_val}