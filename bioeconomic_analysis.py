# =============================================================================
# COMPLETE BIOECONOMIC MODEL - GOOGLE COLAB VERSION
# 9 TABLES: 3 (Price) + 3 (Cost) + 3 (Catchability) Sensitivity Analysis
# Blackspot Seabream (N) vs Portuguese Dogfish (P)
# =============================================================================

# Install required packages
!pip install -q plotly kaleido openpyxl

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.linalg import inv
import pandas as pd
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Create output directory
output_dir = "bioeconomic_complete_results"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# PLOT STYLE SETTINGS
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 2,
})

COLORS = {
    'prey': '#2E86AB',
    'predator': '#D64933',
    'profit': '#009E73',
    'surface': '#56B4E9',
    'optimal': '#F0E442',
    'boundary': '#CC79A7',
    'interior': '#2E86AB',
    'b1': '#E69F00',
    'b2': '#D64933'
}

# =============================================================================
# MODEL PARAMETERS - BASELINE
# =============================================================================

ecological_params = {
    'r': 1.2,           # Prey intrinsic growth rate (yr^-1)
    'K': 100.0,         # Prey carrying capacity (tonnes)
    'gamma': 0.15,      # Prey natural mortality (yr^-1)
    'alpha': 0.025,     # Predation rate (tonnes^-1 yr^-1)
    'm': 0.3,           # Refuge proportion
    'f': 0.008,         # Fear intensity (tonnes^-1)
    'e': 0.25,          # Conversion efficiency
    'beta': 0.45,       # Alternative prey growth rate (yr^-1)
    'Kp': 60.0,         # Predator carrying capacity (tonnes)
    'd': 0.25,          # Predator natural mortality (yr^-1)
}

economic_params = {
    'q_N': 0.12,        # Catchability for prey (vessel^-1 yr^-1)
    'q_P': 0.09,        # Catchability for predator (vessel^-1 yr^-1)
    'p_N': 12.0,        # Price of Seabream (USD/tonne)
    'p_P': 18.0,        # Price of Dogfish (USD/tonne)
    'c_N': 8.0,         # Cost per unit effort for prey (USD/vessel)
    'c_P': 10.0,        # Cost per unit effort for predator (USD/vessel)
}

params = {**ecological_params, **economic_params}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def find_coexistence_equilibrium(params):
    r = params['r']; K = params['K']; f = params['f']
    alpha = params['alpha']; m = params['m']; gamma = params['gamma']
    e = params['e']; beta = params['beta']; Kp = params['Kp']; d = params['d']
    A = alpha * (1 - m)

    def predator_nullcline(P): return max(0, (d - beta + (beta/Kp) * P) / (e * A))
    def prey_residual(P):
        N = predator_nullcline(P)
        if N <= 0: return 1e6
        fear = 1 / (1 + f * P)
        return r * fear * (1 - N/K) - A * P - gamma

    try:
        P_scan = np.linspace(0.01, Kp, 300)
        residuals = [prey_residual(P) for P in P_scan]
        for i in range(len(P_scan)-1):
            if residuals[i] * residuals[i+1] < 0:
                P_star = brentq(prey_residual, P_scan[i], P_scan[i+1])
                N_star = predator_nullcline(P_star)
                if 0 < N_star < K * 1.2 and 0 < P_star < Kp * 1.2:
                    return N_star, P_star
    except: pass
    return np.nan, np.nan

def compute_jacobian(N, P, params):
    r = params['r']; K = params['K']; f = params['f']
    alpha = params['alpha']; m = params['m']; gamma = params['gamma']
    e = params['e']; beta = params['beta']; Kp = params['Kp']; d = params['d']
    A = alpha * (1 - m)
    fear_factor = 1 + f * P
    J11 = r / fear_factor * (1 - 2*N/K) - A * P - gamma
    J12 = -r * f * N / (fear_factor**2) * (1 - N/K) - A * N
    J21 = e * A * P
    J22 = e * A * N + beta * (1 - 2*P/Kp) - d
    return np.array([[J11, J12], [J21, J22]])

def compute_bioeconomic_coefficients(N, P, params):
    J = compute_jacobian(N, P, params)
    delta_star = np.linalg.det(J)
    J_inv = inv(J)
    q_N, q_P = params['q_N'], params['q_P']
    a_NN = J_inv[0, 0] * q_N * N
    a_NP = J_inv[0, 1] * q_P * P
    a_PN = J_inv[1, 0] * q_N * N
    a_PP = J_inv[1, 1] * q_P * P
    return a_NN, a_NP, a_PN, a_PP, delta_star

def interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, params):
    s_N = params['p_N'] * params['q_N']
    s_P = params['p_P'] * params['q_P']
    c_N, c_P = params['c_N'], params['c_P']
    H11 = 2 * s_N * a_NN
    H12 = s_N * a_NP + s_P * a_PN
    H22 = 2 * s_P * a_PP
    b1 = c_N - s_N * N_star
    b2 = c_P - s_P * P_star
    det_H = H11 * H22 - H12 * H12
    if det_H > 0:
        E_N = (b1 * H22 - b2 * H12) / det_H
        E_P = (b2 * H11 - b1 * H12) / det_H
        return E_N, E_P, det_H
    return np.nan, np.nan, det_H

def boundary_B1(N_star, P_star, a_NN, a_PN, params):
    s_N = params['p_N'] * params['q_N']
    c_N = params['c_N']
    E_N_hat = (c_N - s_N * N_star) / (2 * s_N * a_NN)
    cap1 = N_star / (-a_NN) if a_NN < 0 else np.inf
    cap2 = P_star / (-a_PN) if a_PN < 0 else np.inf
    return max(0, min(E_N_hat, cap1, cap2))

def boundary_B2(N_star, P_star, a_PP, params):
    s_P = params['p_P'] * params['q_P']
    c_P = params['c_P']
    E_P_hat = (c_P - s_P * P_star) / (2 * s_P * a_PP)
    cap = P_star / (-a_PP) if a_PP < 0 else np.inf
    return max(0, min(E_P_hat, cap))

def compute_equilibrium_with_effort(E_N, E_P, params):
    params_copy = params.copy()
    params_copy['E_N'] = E_N
    params_copy['E_P'] = E_P

    def eq_system(x):
        N, P = x
        if N <= 0 or P <= 0: return [1e6, 1e6]
        r = params['r']; K = params['K']; f = params['f']
        alpha = params['alpha']; m = params['m']; gamma = params['gamma']
        e = params['e']; beta = params['beta']; Kp = params['Kp']; d = params['d']
        q_N, q_P = params['q_N'], params['q_P']
        A = alpha * (1 - m)
        fear = 1 / (1 + f * P)
        dN = r * N * fear * (1 - N/K) - A * N * P - gamma * N - q_N * E_N * N
        dP = e * A * N * P + beta * P * (1 - P/Kp) - d * P - q_P * E_P * P
        return [dN, dP]

    try:
        N_eq, P_eq = fsolve(eq_system, [params['K']*0.5, params['Kp']*0.5])
        if N_eq > 0 and P_eq > 0:
            H_N = params['q_N'] * E_N * N_eq
            H_P = params['q_P'] * E_P * P_eq
            profit = params['p_N'] * H_N + params['p_P'] * H_P - params['c_N'] * E_N - params['c_P'] * E_P
            return N_eq, P_eq, H_N, H_P, profit
    except: pass
    return np.nan, np.nan, 0, 0, -1e12

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("COMPLETE BIOECONOMIC MODEL - GOOGLE COLAB VERSION")
    print("9 TABLES: 3 (Price) + 3 (Cost) + 3 (Catchability) Sensitivity")
    print("Blackspot Seabream (N) vs Portuguese Dogfish (P)")
    print("=" * 80)

    # Get ecological equilibrium
    N_star, P_star = find_coexistence_equilibrium(params)
    N0 = params['K'] * (1 - params['gamma']/params['r'])
    P0 = params['Kp'] * (1 - params['d']/params['beta'])

    print(f"\nEcological Equilibria:")
    print(f"  Prey-only EN:     ({N0:.2f}, 0)")
    print(f"  Predator-only EP: (0, {P0:.2f})")
    print(f"  Coexistence ENP:  ({N_star:.2f}, {P_star:.2f})")

    # Bioeconomic coefficients
    a_NN, a_NP, a_PN, a_PP, delta_star = compute_bioeconomic_coefficients(N_star, P_star, params)
    s_N = params['p_N'] * params['q_N']
    s_P = params['p_P'] * params['q_P']

    print(f"\nBioeconomic Coefficients:")
    print(f"  a_NN = {a_NN:.6f} (dN/dE_N)")
    print(f"  a_NP = {a_NP:.6f} (dN/dE_P)")
    print(f"  a_PN = {a_PN:.6f} (dP/dE_N)")
    print(f"  a_PP = {a_PP:.6f} (dP/dE_P)")
    print(f"  Delta* = {delta_star:.6f}")

    # =========================================================================
    # TABLE 1-3: PRICE SENSITIVITY (3 tables: Interior, B1, B2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLES 1-3: PRICE SENSITIVITY ANALYSIS")
    print("Fixed: c_N=8, c_P=10, q_N=0.12, q_P=0.09")
    print("=" * 80)

    price_scenarios = [(5,6), (9,11), (12,20), (20,25), (30,35), (45,60), (75,90)]

    # Table 1: Price Sensitivity - Interior Solution
    price_interior_results = []
    # Table 2: Price Sensitivity - B1 (Prey only)
    price_b1_results = []
    # Table 3: Price Sensitivity - B2 (Predator only)
    price_b2_results = []

    for p_N_val, p_P_val in price_scenarios:
        test_params = params.copy()
        test_params['p_N'] = p_N_val
        test_params['p_P'] = p_P_val

        # Interior
        E_N_int, E_P_int, _ = interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, test_params)
        if E_N_int > 0 and E_P_int > 0:
            N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(E_N_int, E_P_int, test_params)
            price_interior_results.append({
                'p_N': p_N_val, 'p_P': p_P_val,
                'E_N': f"{E_N_int:.4f}", 'E_P': f"{E_P_int:.4f}",
                'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
                'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
                'Profit': f"${profit:.2f}"
            })
        else:
            price_interior_results.append({
                'p_N': p_N_val, 'p_P': p_P_val,
                'E_N': "N/A", 'E_P': "N/A",
                'N': "N/A", 'P': "N/A",
                'H_N': "N/A", 'H_P': "N/A",
                'Profit': "N/A"
            })

        # B1 (Prey only)
        E_N_b1 = boundary_B1(N_star, P_star, a_NN, a_PN, test_params)
        N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(E_N_b1, 0, test_params)
        price_b1_results.append({
            'p_N': p_N_val, 'p_P': p_P_val,
            'E_N': f"{E_N_b1:.4f}", 'E_P': "0.0000",
            'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
            'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
            'Profit': f"${profit:.2f}"
        })

        # B2 (Predator only)
        E_P_b2 = boundary_B2(N_star, P_star, a_PP, test_params)
        N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(0, E_P_b2, test_params)
        price_b2_results.append({
            'p_N': p_N_val, 'p_P': p_P_val,
            'E_N': "0.0000", 'E_P': f"{E_P_b2:.4f}",
            'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
            'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
            'Profit': f"${profit:.2f}"
        })

    print("\n" + "-" * 90)
    print("TABLE 1: PRICE SENSITIVITY - INTERIOR SOLUTION (Joint Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(price_interior_results).to_string(index=False))

    print("\n" + "-" * 90)
    print("TABLE 2: PRICE SENSITIVITY - B1 (Prey Only Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(price_b1_results).to_string(index=False))

    print("\n" + "-" * 90)
    print("TABLE 3: PRICE SENSITIVITY - B2 (Predator Only Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(price_b2_results).to_string(index=False))

    # =========================================================================
    # TABLES 4-6: COST SENSITIVITY (3 tables: Interior, B1, B2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLES 4-6: COST SENSITIVITY ANALYSIS")
    print("Fixed: p_N=12, p_P=18, q_N=0.12, q_P=0.09")
    print("=" * 80)

    cost_scenarios = [(2,2.5), (4,5), (6,7.5), (8,10), (10,12.5), (12,15), (15,18.75), (20,25)]

    cost_interior_results = []
    cost_b1_results = []
    cost_b2_results = []

    for c_N_val, c_P_val in cost_scenarios:
        test_params = params.copy()
        test_params['c_N'] = c_N_val
        test_params['c_P'] = c_P_val

        # Interior
        E_N_int, E_P_int, _ = interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, test_params)
        if E_N_int > 0 and E_P_int > 0:
            N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(E_N_int, E_P_int, test_params)
            cost_interior_results.append({
                'c_N': c_N_val, 'c_P': c_P_val,
                'E_N': f"{E_N_int:.4f}", 'E_P': f"{E_P_int:.4f}",
                'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
                'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
                'Profit': f"${profit:.2f}"
            })
        else:
            cost_interior_results.append({
                'c_N': c_N_val, 'c_P': c_P_val,
                'E_N': "N/A", 'E_P': "N/A",
                'N': "N/A", 'P': "N/A",
                'H_N': "N/A", 'H_P': "N/A",
                'Profit': "N/A"
            })

        # B1 (Prey only)
        E_N_b1 = boundary_B1(N_star, P_star, a_NN, a_PN, test_params)
        N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(E_N_b1, 0, test_params)
        cost_b1_results.append({
            'c_N': c_N_val, 'c_P': c_P_val,
            'E_N': f"{E_N_b1:.4f}", 'E_P': "0.0000",
            'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
            'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
            'Profit': f"${profit:.2f}"
        })

        # B2 (Predator only)
        E_P_b2 = boundary_B2(N_star, P_star, a_PP, test_params)
        N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(0, E_P_b2, test_params)
        cost_b2_results.append({
            'c_N': c_N_val, 'c_P': c_P_val,
            'E_N': "0.0000", 'E_P': f"{E_P_b2:.4f}",
            'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
            'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
            'Profit': f"${profit:.2f}"
        })

    print("\n" + "-" * 90)
    print("TABLE 4: COST SENSITIVITY - INTERIOR SOLUTION (Joint Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(cost_interior_results).to_string(index=False))

    print("\n" + "-" * 90)
    print("TABLE 5: COST SENSITIVITY - B1 (Prey Only Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(cost_b1_results).to_string(index=False))

    print("\n" + "-" * 90)
    print("TABLE 6: COST SENSITIVITY - B2 (Predator Only Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(cost_b2_results).to_string(index=False))

    # =========================================================================
    # TABLES 7-9: CATCHABILITY SENSITIVITY (3 tables: Interior, B1, B2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLES 7-9: CATCHABILITY SENSITIVITY ANALYSIS")
    print("Fixed: p_N=12, p_P=18, c_N=8, c_P=10")
    print("=" * 80)

    catch_scenarios = [(0.04,0.03), (0.06,0.045), (0.08,0.06), (0.10,0.075),
                       (0.12,0.09), (0.15,0.1125), (0.20,0.15), (0.25,0.1875), (0.30,0.225)]

    catch_interior_results = []
    catch_b1_results = []
    catch_b2_results = []

    for q_N_val, q_P_val in catch_scenarios:
        test_params = params.copy()
        test_params['q_N'] = q_N_val
        test_params['q_P'] = q_P_val

        # Recompute coefficients for new q values
        a_NN_q, a_NP_q, a_PN_q, a_PP_q, _ = compute_bioeconomic_coefficients(N_star, P_star, test_params)

        # Interior
        E_N_int, E_P_int, _ = interior_solution(N_star, P_star, a_NN_q, a_NP_q, a_PN_q, a_PP_q, test_params)
        if E_N_int > 0 and E_P_int > 0:
            N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(E_N_int, E_P_int, test_params)
            catch_interior_results.append({
                'q_N': q_N_val, 'q_P': q_P_val,
                'E_N': f"{E_N_int:.4f}", 'E_P': f"{E_P_int:.4f}",
                'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
                'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
                'Profit': f"${profit:.2f}"
            })
        else:
            catch_interior_results.append({
                'q_N': q_N_val, 'q_P': q_P_val,
                'E_N': "N/A", 'E_P': "N/A",
                'N': "N/A", 'P': "N/A",
                'H_N': "N/A", 'H_P': "N/A",
                'Profit': "N/A"
            })

        # B1 (Prey only)
        E_N_b1 = boundary_B1(N_star, P_star, a_NN_q, a_PN_q, test_params)
        N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(E_N_b1, 0, test_params)
        catch_b1_results.append({
            'q_N': q_N_val, 'q_P': q_P_val,
            'E_N': f"{E_N_b1:.4f}", 'E_P': "0.0000",
            'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
            'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
            'Profit': f"${profit:.2f}"
        })

        # B2 (Predator only)
        E_P_b2 = boundary_B2(N_star, P_star, a_PP_q, test_params)
        N_eq, P_eq, H_N, H_P, profit = compute_equilibrium_with_effort(0, E_P_b2, test_params)
        catch_b2_results.append({
            'q_N': q_N_val, 'q_P': q_P_val,
            'E_N': "0.0000", 'E_P': f"{E_P_b2:.4f}",
            'N': f"{N_eq:.2f}", 'P': f"{P_eq:.2f}",
            'H_N': f"{H_N:.2f}", 'H_P': f"{H_P:.2f}",
            'Profit': f"${profit:.2f}"
        })

    print("\n" + "-" * 90)
    print("TABLE 7: CATCHABILITY SENSITIVITY - INTERIOR SOLUTION (Joint Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(catch_interior_results).to_string(index=False))

    print("\n" + "-" * 90)
    print("TABLE 8: CATCHABILITY SENSITIVITY - B1 (Prey Only Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(catch_b1_results).to_string(index=False))

    print("\n" + "-" * 90)
    print("TABLE 9: CATCHABILITY SENSITIVITY - B2 (Predator Only Harvesting)")
    print("-" * 90)
    print(pd.DataFrame(catch_b2_results).to_string(index=False))

    # =========================================================================
    # BASELINE RESULTS FOR FIGURES
    # =========================================================================
    E_N_int_base, E_P_int_base, _ = interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, params)
    E_N_b1_base = boundary_B1(N_star, P_star, a_NN, a_PN, params)
    E_P_b2_base = boundary_B2(N_star, P_star, a_PP, params)

    _, _, _, _, profit_int_base = compute_equilibrium_with_effort(E_N_int_base if E_N_int_base > 0 else 0,
                                                                   E_P_int_base if E_P_int_base > 0 else 0, params)
    _, _, _, _, profit_b1_base = compute_equilibrium_with_effort(E_N_b1_base, 0, params)
    _, _, _, _, profit_b2_base = compute_equilibrium_with_effort(0, E_P_b2_base, params)

    # =========================================================================
    # FIGURE 1: 3D Profit Surface for Prices
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 1: 3D Profit Surface for Prices")
    print("=" * 60)

    p_N_range = np.linspace(2, 80, 50)
    p_P_range = np.linspace(3, 95, 50)
    P_N, P_P = np.meshgrid(p_N_range, p_P_range)
    profit_grid = np.zeros_like(P_N)

    for i in range(len(p_N_range)):
        for j in range(len(p_P_range)):
            test_params = params.copy()
            test_params['p_N'] = P_N[i, j]
            test_params['p_P'] = P_P[i, j]
            E_N, E_P, _ = interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, test_params)
            _, _, _, _, profit = compute_equilibrium_with_effort(E_N if E_N > 0 else 0, E_P if E_P > 0 else 0, test_params)
            profit_grid[i, j] = profit if profit > -1e10 else 0

    fig1 = plt.figure(figsize=(14, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(P_N, P_P, profit_grid, cmap=cm.viridis, alpha=0.9, linewidth=0, antialiased=True)
    ax1.set_xlabel('Prey Price p_N (USD/tonne)', fontsize=12)
    ax1.set_ylabel('Predator Price p_P (USD/tonne)', fontsize=12)
    ax1.set_zlabel('Profit (USD)', fontsize=12)
    ax1.set_title('Figure 1: 3D Profit Surface vs Fish Prices', fontsize=14)
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Profit (USD)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_profit_surface_prices.png", dpi=300)
    plt.show()
    print("  Figure 1 saved: profit_surface_prices.png")

    # =========================================================================
    # FIGURE 2: 3D Profit Surface for Costs
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 2: 3D Profit Surface for Costs")
    print("=" * 60)

    c_N_range = np.linspace(2, 25, 50)
    c_P_range = np.linspace(2.5, 30, 50)
    C_N, C_P = np.meshgrid(c_N_range, c_P_range)
    profit_cost_grid = np.zeros_like(C_N)

    for i in range(len(c_N_range)):
        for j in range(len(c_P_range)):
            test_params = params.copy()
            test_params['c_N'] = C_N[i, j]
            test_params['c_P'] = C_P[i, j]
            E_N, E_P, _ = interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, test_params)
            _, _, _, _, profit = compute_equilibrium_with_effort(E_N if E_N > 0 else 0, E_P if E_P > 0 else 0, test_params)
            profit_cost_grid[i, j] = profit if profit > -1e10 else 0

    fig2 = plt.figure(figsize=(14, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(C_N, C_P, profit_cost_grid, cmap=cm.plasma, alpha=0.9)
    ax2.set_xlabel('Prey Cost c_N (USD/vessel)', fontsize=12)
    ax2.set_ylabel('Predator Cost c_P (USD/vessel)', fontsize=12)
    ax2.set_zlabel('Profit (USD)', fontsize=12)
    ax2.set_title('Figure 2: Profit Surface vs Fishing Costs', fontsize=14)
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_profit_vs_costs.png", dpi=300)
    plt.show()
    print("  Figure 2 saved: profit_vs_costs.png")

    # =========================================================================
    # FIGURE 3: 3D Profit Surface for Catchability
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 3: 3D Profit Surface for Catchability")
    print("=" * 60)

    q_N_range = np.linspace(0.04, 0.30, 50)
    q_P_range = np.linspace(0.03, 0.23, 50)
    Q_N, Q_P = np.meshgrid(q_N_range, q_P_range)
    profit_catch_grid = np.zeros_like(Q_N)

    for i in range(len(q_N_range)):
        for j in range(len(q_P_range)):
            test_params = params.copy()
            test_params['q_N'] = Q_N[i, j]
            test_params['q_P'] = Q_P[i, j]
            a_NN_q, a_NP_q, a_PN_q, a_PP_q, _ = compute_bioeconomic_coefficients(N_star, P_star, test_params)
            E_N, E_P, _ = interior_solution(N_star, P_star, a_NN_q, a_NP_q, a_PN_q, a_PP_q, test_params)
            _, _, _, _, profit = compute_equilibrium_with_effort(E_N if E_N > 0 else 0, E_P if E_P > 0 else 0, test_params)
            profit_catch_grid[i, j] = profit if profit > -1e10 else 0

    fig3 = plt.figure(figsize=(14, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(Q_N, Q_P, profit_catch_grid, cmap=cm.inferno, alpha=0.9)
    ax3.set_xlabel('Prey Catchability q_N', fontsize=12)
    ax3.set_ylabel('Predator Catchability q_P', fontsize=12)
    ax3.set_zlabel('Profit (USD)', fontsize=12)
    ax3.set_title('Figure 3: Profit Surface vs Catchability', fontsize=14)
    fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_profit_vs_catchability.png", dpi=300)
    plt.show()
    print("  Figure 3 saved: profit_vs_catchability.png")

    # =========================================================================
    # FIGURE 4: Strategy Comparison Bar Chart
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 4: Strategy Comparison")
    print("=" * 60)

    fig4 = plt.figure(figsize=(12, 7))
    ax4 = fig4.add_subplot(111)

    strategies = ['Interior', 'B1 (Prey Only)', 'B2 (Predator Only)']
    profits_compare = [profit_int_base if profit_int_base > -1e10 else 0,
                       profit_b1_base if profit_b1_base > -1e10 else 0,
                       profit_b2_base if profit_b2_base > -1e10 else 0]
    colors_bar = [COLORS['interior'], COLORS['b1'], COLORS['b2']]

    bars = ax4.bar(strategies, profits_compare, color=colors_bar, edgecolor='black', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, profits_compare):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'${val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax4.set_ylabel('Maximum Profit (USD)', fontsize=12)
    ax4.set_title('Figure 4: Profit Comparison Across Harvesting Strategies\n(Baseline: p_N=12, p_P=18, c_N=8, c_P=10, q_N=0.12, q_P=0.09)', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_strategy_comparison.png", dpi=300)
    plt.show()
    print("  Figure 4 saved: strategy_comparison.png")

    # =========================================================================
    # FIGURE 5: Optimal Efforts vs Price (Line plot)
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 5: Optimal Efforts vs Price")
    print("=" * 60)

    fig5 = plt.figure(figsize=(12, 7))
    ax5 = fig5.add_subplot(111)

    p_vals = np.linspace(2, 80, 40)
    E_N_list = []
    E_P_list = []

    for p in p_vals:
        test_params = params.copy()
        test_params['p_N'] = p
        test_params['p_P'] = p * 1.2
        E_N, E_P, _ = interior_solution(N_star, P_star, a_NN, a_NP, a_PN, a_PP, test_params)
        E_N_list.append(E_N if E_N > 0 else 0)
        E_P_list.append(E_P if E_P > 0 else 0)

    ax5.plot(p_vals, E_N_list, 'b-', linewidth=2.5, label='Prey Effort E_N*')
    ax5.plot(p_vals, E_P_list, 'r--', linewidth=2.5, label='Predator Effort E_P*')
    ax5.axvline(params['p_N'], color='k', linestyle=':', alpha=0.7, linewidth=2, label='Baseline (p_N=12)')
    ax5.set_xlabel('Prey Price p_N (USD/tonne)', fontsize=12)
    ax5.set_ylabel('Optimal Fishing Effort', fontsize=12)
    ax5.set_title('Figure 5: Optimal Fishing Efforts vs Prey Price', fontsize=13)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure5_effort_vs_price.png", dpi=300)
    plt.show()
    print("  Figure 5 saved: effort_vs_price.png")

    # =========================================================================
    # FIGURE 6: Phase Plane with Vector Field
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 6: Phase Plane Analysis")
    print("=" * 60)

    fig6 = plt.figure(figsize=(12, 10))
    ax6 = fig6.add_subplot(111)

    N_grid = np.linspace(0.1, params['K'] * 1.2, 20)
    P_grid = np.linspace(0.1, params['Kp'] * 1.2, 20)
    NN, PP = np.meshgrid(N_grid, P_grid)
    dN = np.zeros_like(NN)
    dP = np.zeros_like(PP)

    def model_no_harvest(t, state, params):
        N, P = state
        N = max(N, 1e-6); P = max(P, 1e-6)
        r = params['r']; K = params['K']; f = params['f']
        alpha = params['alpha']; m = params['m']; gamma = params['gamma']
        e = params['e']; beta = params['beta']; Kp = params['Kp']; d = params['d']
        A = alpha * (1 - m)
        fear = 1 / (1 + f * P)
        dN = r * N * fear * (1 - N/K) - A * N * P - gamma * N
        dP = e * A * N * P + beta * P * (1 - P/Kp) - d * P
        return [dN, dP]

    for i in range(len(N_grid)):
        for j in range(len(P_grid)):
            dN[i,j], dP[i,j] = model_no_harvest(0, [NN[i,j], PP[i,j]], params)

    magnitude = np.sqrt(dN**2 + dP**2)
    magnitude[magnitude == 0] = 1
    dN_norm = dN / magnitude
    dP_norm = dP / magnitude

    ax6.quiver(NN, PP, dN_norm, dP_norm, magnitude, alpha=0.6, cmap='viridis', scale=30)
    ax6.scatter(N_star, P_star, color='red', s=250, zorder=5, marker='*',
               label=f'Coexistence Equilibrium\nN*={N_star:.1f}, P*={P_star:.1f}')
    ax6.set_xlabel('Prey Biomass N (tonnes)', fontsize=12)
    ax6.set_ylabel('Predator Biomass P (tonnes)', fontsize=12)
    ax6.set_title('Figure 6: Phase Plane with Vector Field (No Harvesting)', fontsize=14)
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure6_phase_plane.png", dpi=300)
    plt.show()
    print("  Figure 6 saved: phase_plane.png")

    # =========================================================================
    # FIGURE 7: Interactive Plotly 3D Surface
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURE 7: Interactive 3D Plotly Surface")
    print("=" * 60)

    fig7 = go.Figure(data=[go.Surface(z=profit_grid, x=p_N_range, y=p_P_range, colorscale='Viridis')])
    fig7.update_layout(title='Interactive 3D Profit Surface vs Fish Prices',
                       scene=dict(xaxis_title='Prey Price p_N (USD/tonne)',
                                 yaxis_title='Predator Price p_P (USD/tonne)',
                                 zaxis_title='Profit (USD)'),
                       width=1000, height=800)
    fig7.write_html(f"{output_dir}/figure7_interactive_profit_surface.html")
    fig7.show()
    print("  Figure 7 saved: interactive_profit_surface.html")

    # =========================================================================
    # EXPORT ALL TABLES TO EXCEL
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPORTING ALL 9 TABLES TO EXCEL")
    print("=" * 60)

    with pd.ExcelWriter(f"{output_dir}/bioeconomic_9_tables_complete.xlsx", engine='openpyxl') as writer:
        # Tables 1-3: Price Sensitivity
        pd.DataFrame(price_interior_results).to_excel(writer, sheet_name='Table1_Price_Interior', index=False)
        pd.DataFrame(price_b1_results).to_excel(writer, sheet_name='Table2_Price_B1', index=False)
        pd.DataFrame(price_b2_results).to_excel(writer, sheet_name='Table3_Price_B2', index=False)

        # Tables 4-6: Cost Sensitivity
        pd.DataFrame(cost_interior_results).to_excel(writer, sheet_name='Table4_Cost_Interior', index=False)
        pd.DataFrame(cost_b1_results).to_excel(writer, sheet_name='Table5_Cost_B1', index=False)
        pd.DataFrame(cost_b2_results).to_excel(writer, sheet_name='Table6_Cost_B2', index=False)

        # Tables 7-9: Catchability Sensitivity
        pd.DataFrame(catch_interior_results).to_excel(writer, sheet_name='Table7_Catch_Interior', index=False)
        pd.DataFrame(catch_b1_results).to_excel(writer, sheet_name='Table8_Catch_B1', index=False)
        pd.DataFrame(catch_b2_results).to_excel(writer, sheet_name='Table9_Catch_B2', index=False)

        # Summary Sheet
        summary_df = pd.DataFrame([
            ('Ecological Parameters', '', ''),
            ('r (prey growth rate)', params['r'], 'yr^-1'),
            ('K (prey carrying capacity)', params['K'], 'tonnes'),
            ('gamma (prey mortality)', params['gamma'], 'yr^-1'),
            ('alpha (predation rate)', params['alpha'], 'tonnes^-1 yr^-1'),
            ('m (refuge proportion)', params['m'], ''),
            ('f (fear intensity)', params['f'], 'tonnes^-1'),
            ('e (conversion efficiency)', params['e'], ''),
            ('beta (alt prey growth)', params['beta'], 'yr^-1'),
            ('Kp (predator capacity)', params['Kp'], 'tonnes'),
            ('d (predator mortality)', params['d'], 'yr^-1'),
            ('', '', ''),
            ('Economic Parameters', '', ''),
            ('q_N (prey catchability)', params['q_N'], 'vessel^-1 yr^-1'),
            ('q_P (predator catchability)', params['q_P'], 'vessel^-1 yr^-1'),
            ('p_N (prey price)', params['p_N'], 'USD/tonne'),
            ('p_P (predator price)', params['p_P'], 'USD/tonne'),
            ('c_N (prey cost)', params['c_N'], 'USD/vessel'),
            ('c_P (predator cost)', params['c_P'], 'USD/vessel'),
            ('', '', ''),
            ('Bioeconomic Coefficients', '', ''),
            ('a_NN', a_NN, 'dN/dE_N'),
            ('a_NP', a_NP, 'dN/dE_P'),
            ('a_PN', a_PN, 'dP/dE_N'),
            ('a_PP', a_PP, 'dP/dE_P'),
            ('Delta* (det J)', delta_star, ''),
            ('', '', ''),
            ('Baseline Optimal Results', '', ''),
            ('Interior E_N*', E_N_int_base if E_N_int_base > 0 else 0, ''),
            ('Interior E_P*', E_P_int_base if E_P_int_base > 0 else 0, ''),
            ('Interior Profit', f"${profit_int_base:.2f}", ''),
            ('B1 E_N*', E_N_b1_base, ''),
            ('B1 Profit', f"${profit_b1_base:.2f}", ''),
            ('B2 E_P*', E_P_b2_base, ''),
            ('B2 Profit', f"${profit_b2_base:.2f}", ''),
        ], columns=['Parameter', 'Value', 'Units'])

        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"  Excel file saved: {output_dir}/bioeconomic_9_tables_complete.xlsx")

    # =========================================================================
    # CREATE ZIP FILE FOR DOWNLOAD (COLAB)
    # =========================================================================
    try:
        from google.colab import files
        import zipfile

        zip_path = f"{output_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files_in_dir in os.walk(output_dir):
                for file in files_in_dir:
                    zipf.write(os.path.join(root, file),
                              os.path.relpath(os.path.join(root, file), os.path.dirname(output_dir)))

        print(f"\n" + "=" * 60)
        print("DOWNLOAD ALL RESULTS")
        print("=" * 60)
        print(f"ZIP file created: {zip_path}")
        files.download(zip_path)
        print("  Download started for all results!")
    except:
        print("\n  Not running in Colab - files saved locally")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPLETE BIOECONOMIC ANALYSIS - FINAL SUMMARY")
    print("=" * 80)
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│              9 TABLES + 7 FIGURES COMPLETED SUCCESSFULLY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ TABLES (9 Total):                                                           │
│   Tables 1-3: Price Sensitivity (Interior, B1, B2)                          │
│   Tables 4-6: Cost Sensitivity (Interior, B1, B2)                           │
│   Tables 7-9: Catchability Sensitivity (Interior, B1, B2)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ EACH TABLE INCLUDES:                                                        │
│   • Optimal Efforts (E_N, E_P)                                              │
│   • Equilibrium Biomass (N, P)                                              │
│   • Harvest (H_N, H_P)                                                      │
│   • Maximum Profit (π*)                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ FIGURES (7 Total):                                                          │
│   1. 3D Profit Surface vs Prices                                            │
│   2. 3D Profit Surface vs Costs                                             │
│   3. 3D Profit Surface vs Catchability                                      │
│   4. Strategy Comparison Bar Chart                                          │
│   5. Optimal Efforts vs Price                                               │
│   6. Phase Plane with Vector Field                                          │
│   7. Interactive 3D Plotly Surface (HTML)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ BASELINE OPTIMAL RESULTS:                                                   │
│   Interior Strategy:  E_N = {E_N_int_base:.4f}, E_P = {E_P_int_base:.4f}    │
│   Profit: ${profit_int_base:.2f}                                            │
│   B1 Strategy:        E_N = {E_N_b1_base:.4f}, Profit: ${profit_b1_base:.2f}│
│   B2 Strategy:        E_P = {E_P_b2_base:.4f}, Profit: ${profit_b2_base:.2f}│
├─────────────────────────────────────────────────────────────────────────────┤
│ OUTPUT FILES:                                                               │
│   • 7 PNG figures + 1 HTML interactive plot                                 │
│   • 1 Excel file with 10 sheets (9 tables + summary)                        │
└─────────────────────────────────────────────────────────────────────────────┘
""".format(E_N_int_base=E_N_int_base, E_P_int_base=E_P_int_base, profit_int_base=profit_int_base,
           E_N_b1_base=E_N_b1_base, profit_b1_base=profit_b1_base,
           E_P_b2_base=E_P_b2_base, profit_b2_base=profit_b2_base))

    print(f"\n All files saved to: {os.path.abspath(output_dir)}/")
    print("\n" + "=" * 80)
    print("ALL BIOECONOMIC ANALYSES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
