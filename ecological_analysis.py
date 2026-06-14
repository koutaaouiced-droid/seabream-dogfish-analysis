# =============================================================================
# COMPLETE ECOLOGICAL DYNAMICS AND STABILITY ANALYSIS
# Blackspot Seabream (N) vs Portuguese Dogfish (P)
# With FEAR effect, PREY REFUGE, and ALTERNATIVE RESOURCES
#
# This code calculates:
#   1. All 4 equilibria (E0, EN, EP, ENP)
#   2. Jacobian and stability for each equilibrium
#   3. Time series curves for EACH equilibrium point (E0, EN, EP, ENP)
#   4. Sensitivity tables for ALL parameters (Low, Medium, High values)
#   5. Phase plane and nullcline figures
#   6. Sensitivity of coexistence equilibrium to key parameters
#   7. Trace & determinant sensitivity analysis
#   8. Exports all results to Excel
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PLOT STYLE SETTINGS
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Color schemes
COLORS = {
    'prey': '#1f77b4',
    'predator': '#d62728',
    'null_prey': '#2ca02c',
    'null_pred': '#ff7f0e',
    'E0': '#9467bd',
    'EN': '#1f77b4',
    'EP': '#ff7f0e',
    'ENP': '#d62728',
    'trajectory': 'black'
}

print("=" * 80)
print("COMPLETE ECOLOGICAL ANALYSIS")
print("Blackspot Seabream (N) vs Portuguese Dogfish (P)")
print("=" * 80)

# =============================================================================
# SECTION 1: MODEL DEFINITION
# =============================================================================
def model_equations(t, state, params):
    """
    dN/dt = rN/(1+fP)*(1-N/K) - α(1-m)NP - γN
    dP/dt = eα(1-m)NP + βP(1-P/Kp) - dP
    """
    N, P = state
    N = max(N, 1e-8)
    P = max(P, 1e-8)

    r = params['r']; K = params['K']; f = params['f']
    alpha = params['alpha']; m = params['m']; gamma = params['gamma']
    e = params['e']; beta = params['beta']; Kp = params['Kp']; d = params['d']

    A = alpha * (1 - m)
    fear_term = 1 / (1 + f * P)

    dNdt = r * N * fear_term * (1 - N/K) - A * N * P - gamma * N
    dPdt = e * A * N * P + beta * P * (1 - P/Kp) - d * P

    return [dNdt, dPdt]

# =============================================================================
# SECTION 2: BASELINE PARAMETERS (YOUR VALUES)
# =============================================================================
baseline_params = {
    'r': 1.2,           # Prey growth rate (yr⁻¹)
    'K': 100.0,         # Prey carrying capacity (tonnes)
    'gamma': 0.15,      # Prey background mortality (yr⁻¹)
    'alpha': 0.025,     # Predation rate (tonnes⁻¹ yr⁻¹)
    'm': 0.3,           # Refuge proportion
    'f': 0.008,         # Fear intensity (tonnes⁻¹)
    'e': 0.25,          # Conversion efficiency
    'beta': 0.45,       # Alternative prey growth rate (yr⁻¹)
    'Kp': 60.0,         # Alternative prey carrying capacity (tonnes)
    'd': 0.25           # Predator mortality (yr⁻¹)
}

print("\n" + "-" * 50)
print("BASELINE PARAMETERS (YOUR VALUES)")
print("-" * 50)
for key, val in baseline_params.items():
    print(f"   {key:8s} = {val}")

# =============================================================================
# SECTION 3: PARAMETER SETS (LOW, MEDIUM, HIGH) - USING YOUR VALUES AS MEDIUM
# =============================================================================
parameter_sets = {
    'r': {'Low': 0.9, 'Medium': 1.2, 'High': 1.6, 'label': 'Prey Growth Rate r'},
    'K': {'Low': 70, 'Medium': 100, 'High': 130, 'label': 'Prey Carrying Capacity K'},
    'gamma': {'Low': 0.08, 'Medium': 0.15, 'High': 0.25, 'label': 'Prey Mortality γ'},
    'alpha': {'Low': 0.015, 'Medium': 0.025, 'High': 0.035, 'label': 'Predation Rate α'},
    'm': {'Low': 0.15, 'Medium': 0.30, 'High': 0.50, 'label': 'Refuge Proportion m'},
    'f': {'Low': 0.003, 'Medium': 0.008, 'High': 0.015, 'label': 'Fear Intensity f'},
    'e': {'Low': 0.15, 'Medium': 0.25, 'High': 0.40, 'label': 'Conversion Efficiency e'},
    'beta': {'Low': 0.30, 'Medium': 0.45, 'High': 0.60, 'label': 'Alt Prey Growth β'},
    'Kp': {'Low': 40, 'Medium': 60, 'High': 80, 'label': 'Alt Prey Carrying Capacity Kp'},
    'd': {'Low': 0.18, 'Medium': 0.25, 'High': 0.35, 'label': 'Predator Mortality d'}
}

# =============================================================================
# SECTION 4: EQUILIBRIUM CALCULATIONS
# =============================================================================
def compute_E0():
    """Trivial equilibrium (0,0)"""
    return np.array([0.0, 0.0])

def compute_EN(params):
    """Prey-only equilibrium (N0, 0)"""
    r = params['r']; K = params['K']; gamma = params['gamma']
    if r > gamma:
        N0 = K * (1 - gamma/r)
        return np.array([N0, 0.0])
    return None

def compute_EP(params):
    """Predator-only equilibrium (0, P0)"""
    beta = params['beta']; Kp = params['Kp']; d = params['d']
    if beta > d:
        P0 = Kp * (1 - d/beta)
        return np.array([0.0, P0])
    return None

def compute_ENP(params):
    """
    Coexistence equilibrium (N*, P*)
    Solves: A P^2 + (B+D) P + C = 0
    """
    r = params['r']; K = params['K']; f = params['f']
    alpha = params['alpha']; m = params['m']; gamma = params['gamma']
    e = params['e']; beta = params['beta']; Kp = params['Kp']; d = params['d']

    A_coef = e * K * f * alpha**2 * (1-m)**2
    B_coef = e * K * alpha * (1-m) * (alpha*(1-m) + f*gamma)
    C_coef = r * (d - beta) - e * alpha * (1-m) * K * (r - gamma)
    D_coef = r * beta / Kp

    a_quad = A_coef
    b_quad = B_coef + D_coef
    c_quad = C_coef

    if a_quad == 0 or np.isnan(a_quad):
        return None

    discriminant = b_quad**2 - 4 * a_quad * c_quad

    if discriminant < 0:
        return None

    P_star = (-b_quad + np.sqrt(discriminant)) / (2 * a_quad)
    if P_star <= 0:
        P_star = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)

    if P_star <= 0:
        return None

    denominator = e * alpha * (1-m)
    if denominator <= 0:
        return None

    N_star = (d - beta + (beta/Kp) * P_star) / denominator

    if N_star <= 0 or N_star > K * 1.2:
        return None

    return np.array([N_star, P_star])

# =============================================================================
# SECTION 5: JACOBIAN AND STABILITY ANALYSIS
# =============================================================================
def jacobian_matrix(N, P, params):
    """Full Jacobian matrix"""
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

def stability_analysis(N, P, params, label=""):
    """Compute and return stability properties"""
    J = jacobian_matrix(N, P, params)
    trace = np.trace(J)
    det = np.linalg.det(J)
    eigenvalues = np.linalg.eigvals(J)

    if trace < 0 and det > 0:
        if trace**2 - 4*det < 0:
            classification = "Stable Focus"
        else:
            classification = "Stable Node"
    elif trace > 0 and det > 0:
        classification = "Unstable Node"
    elif det < 0:
        classification = "Saddle (Unstable)"
    else:
        classification = "Center/Bifurcation"

    return {
        'equilibrium': label,
        'N': N,
        'P': P,
        'trace': trace,
        'det': det,
        'λ1': eigenvalues[0],
        'λ2': eigenvalues[1],
        'classification': classification,
        'stable': (trace < 0 and det > 0)
    }

# =============================================================================
# SECTION 6: COMPUTE ALL EQUILIBRIA FOR BASELINE
# =============================================================================
E0 = compute_E0()
EN = compute_EN(baseline_params)
EP = compute_EP(baseline_params)
ENP = compute_ENP(baseline_params)

print("\n" + "=" * 60)
print("EQUILIBRIUM POINTS (YOUR PARAMETER VALUES)")
print("=" * 60)
print(f"E0 (Trivial):              ({E0[0]:.4f}, {E0[1]:.4f})")
if EN is not None:
    print(f"EN (Prey-only):            ({EN[0]:.4f}, {EN[1]:.4f})")
else:
    print("EN: Not feasible (r <= gamma)")
if EP is not None:
    print(f"EP (Predator-only):        ({EP[0]:.4f}, {EP[1]:.4f})")
else:
    print("EP: Not feasible (beta <= d)")
if ENP is not None:
    print(f"ENP (Coexistence):         ({ENP[0]:.4f}, {ENP[1]:.4f})")
else:
    print("ENP: Not feasible")

# =============================================================================
# SECTION 7: STABILITY ANALYSIS FOR EACH EQUILIBRIUM
# =============================================================================
print("\n" + "=" * 60)
print("STABILITY ANALYSIS (YOUR PARAMETER VALUES)")
print("=" * 60)

stability_results = []

if E0 is not None:
    result = stability_analysis(E0[0], E0[1], baseline_params, "E0")
    stability_results.append(result)
    print(f"\nE0 (0, 0):")
    print(f"  Trace = {result['trace']:.6f}, Det = {result['det']:.6f}")
    print(f"  Eigenvalues: λ1 = {result['λ1']:.6f}, λ2 = {result['λ2']:.6f}")
    print(f"  Classification: {result['classification']}")

if EN is not None:
    result = stability_analysis(EN[0], EN[1], baseline_params, "EN")
    stability_results.append(result)
    print(f"\nEN ({EN[0]:.4f}, 0):")
    print(f"  Trace = {result['trace']:.6f}, Det = {result['det']:.6f}")
    print(f"  Eigenvalues: λ1 = {result['λ1']:.6f}, λ2 = {result['λ2']:.6f}")
    print(f"  Classification: {result['classification']}")

if EP is not None:
    result = stability_analysis(EP[0], EP[1], baseline_params, "EP")
    stability_results.append(result)
    print(f"\nEP (0, {EP[1]:.4f}):")
    print(f"  Trace = {result['trace']:.6f}, Det = {result['det']:.6f}")
    print(f"  Eigenvalues: λ1 = {result['λ1']:.6f}, λ2 = {result['λ2']:.6f}")
    print(f"  Classification: {result['classification']}")

if ENP is not None:
    result = stability_analysis(ENP[0], ENP[1], baseline_params, "ENP")
    stability_results.append(result)
    print(f"\nENP ({ENP[0]:.4f}, {ENP[1]:.4f}):")
    print(f"  Trace = {result['trace']:.6f}, Det = {result['det']:.6f}")
    print(f"  Eigenvalues: λ1 = {result['λ1']:.6f}, λ2 = {result['λ2']:.6f}")
    print(f"  Classification: {result['classification']}")

# =============================================================================
# SECTION 8: SIMULATION FUNCTION
# =============================================================================
def simulate(params, t_span, initial_state, t_eval):
    """Run numerical integration"""
    sol = solve_ivp(
        lambda t, y: model_equations(t, y, params),
        t_span, initial_state, t_eval=t_eval,
        method='RK45', rtol=1e-8, atol=1e-10
    )
    return sol.t, sol.y[0], sol.y[1]

# =============================================================================
# FIGURE 1A: E0 (TRIVIAL EQUILIBRIUM) TIME SERIES CURVES (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 1A: Simulating E0 (Trivial Equilibrium) Dynamics")
print("=" * 60)

e0_initial_conditions = [
    {'name': 'Case 1: Very close', 'N0': 0.01, 'P0': 0.005, 'color': '#9467bd'},
    {'name': 'Case 2: Close', 'N0': 0.05, 'P0': 0.02, 'color': '#1f77b4'},
    {'name': 'Case 3: Moderate', 'N0': 0.1, 'P0': 0.05, 'color': '#d62728'},
    {'name': 'Case 4: Further', 'N0': 0.2, 'P0': 0.1, 'color': '#2ca02c'},
    {'name': 'Case 5: Far', 'N0': 0.5, 'P0': 0.2, 'color': '#ff7f0e'},
]

t_span_e0 = (0, 100)
t_eval_e0 = np.linspace(0, 100, 2000)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax1 = axes[0, 0]
for ic in e0_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_e0, [ic['N0'], ic['P0']], t_eval_e0)
    ax1.plot(t_sim, N_sim, color=ic['color'], linewidth=2, label=f"{ic['name']} (N₀={ic['N0']}, P₀={ic['P0']})")
ax1.set_xlabel('Time (years)')
ax1.set_ylabel('Prey Biomass N (tonnes)')
ax1.set_title('(a) Prey Population Dynamics')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

ax2 = axes[0, 1]
for ic in e0_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_e0, [ic['N0'], ic['P0']], t_eval_e0)
    ax2.plot(t_sim, P_sim, color=ic['color'], linewidth=2, label=ic['name'])
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Predator Biomass P (tonnes)')
ax2.set_title('(b) Predator Population Dynamics')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

ax3 = axes[1, 0]
for ic in e0_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_e0, [ic['N0'], ic['P0']], t_eval_e0)
    ax3.plot(N_sim, P_sim, color=ic['color'], linewidth=2, alpha=0.8)
    ax3.scatter(ic['N0'], ic['P0'], color=ic['color'], s=80, zorder=5, edgecolors='black')
ax3.scatter(0, 0, color='red', s=200, marker='*', zorder=10, label='E0 (0,0)', edgecolors='black')
ax3.set_xlabel('Prey Biomass N (tonnes)')
ax3.set_ylabel('Predator Biomass P (tonnes)')
ax3.set_title('(c) Phase Plane Trajectories')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
for ic in e0_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_e0, [ic['N0'], ic['P0']], t_eval_e0)
    distance = np.sqrt(N_sim**2 + P_sim**2)
    ax4.plot(t_sim, distance, color=ic['color'], linewidth=2, label=ic['name'])
ax4.set_xlabel('Time (years)')
ax4.set_ylabel('Distance from E0')
ax4.set_title('(d) Distance from Equilibrium')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig("figure1a_e0_timeseries.png", dpi=300)
plt.show()
print("✅ Figure 1A saved: figure1a_e0_timeseries.png")

# =============================================================================
# FIGURE 1B: EN (PREY-ONLY EQUILIBRIUM) TIME SERIES CURVES (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 1B: Simulating EN (Prey-Only Equilibrium) Dynamics")
print("=" * 60)

N0_val = EN[0] if EN is not None else 87.5

en_initial_conditions = [
    {'name': 'Case 1: Very close', 'N0': N0_val * 0.99, 'P0': 0.01, 'color': '#1f77b4'},
    {'name': 'Case 2: Close', 'N0': N0_val * 0.95, 'P0': 0.05, 'color': '#d62728'},
    {'name': 'Case 3: Moderate', 'N0': N0_val * 0.90, 'P0': 0.10, 'color': '#2ca02c'},
    {'name': 'Case 4: Further', 'N0': N0_val * 0.85, 'P0': 0.50, 'color': '#ff7f0e'},
    {'name': 'Case 5: Far', 'N0': N0_val * 0.80, 'P0': 1.00, 'color': '#9467bd'},
]

t_span_en = (0, 150)
t_eval_en = np.linspace(0, 150, 2000)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax1 = axes[0, 0]
for ic in en_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_en, [ic['N0'], ic['P0']], t_eval_en)
    ax1.plot(t_sim, N_sim, color=ic['color'], linewidth=2, label=f"{ic['name']} (N₀={ic['N0']:.1f}, P₀={ic['P0']})")
ax1.axhline(N0_val, color='blue', linestyle=':', alpha=0.7, label=f'EN N* = {N0_val:.1f}')
ax1.set_xlabel('Time (years)')
ax1.set_ylabel('Prey Biomass N (tonnes)')
ax1.set_title('(a) Prey Population Dynamics')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
for ic in en_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_en, [ic['N0'], ic['P0']], t_eval_en)
    ax2.plot(t_sim, P_sim, color=ic['color'], linewidth=2, label=ic['name'])
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Predator Biomass P (tonnes)')
ax2.set_title('(b) Predator Population Dynamics')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
for ic in en_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_en, [ic['N0'], ic['P0']], t_eval_en)
    ax3.plot(N_sim, P_sim, color=ic['color'], linewidth=2, alpha=0.8)
    ax3.scatter(ic['N0'], ic['P0'], color=ic['color'], s=80, zorder=5, edgecolors='black')
ax3.scatter(N0_val, 0, color='red', s=200, marker='*', zorder=10, label=f'EN ({N0_val:.1f}, 0)')
if ENP is not None:
    ax3.scatter(ENP[0], ENP[1], color='green', s=100, marker='s', label=f'ENP ({ENP[0]:.1f}, {ENP[1]:.1f})')
ax3.set_xlabel('Prey Biomass N (tonnes)')
ax3.set_ylabel('Predator Biomass P (tonnes)')
ax3.set_title('(c) Phase Plane Trajectories')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
for ic in en_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_en, [ic['N0'], ic['P0']], t_eval_en)
    distance = np.sqrt((N_sim - N0_val)**2 + P_sim**2)
    ax4.plot(t_sim, distance, color=ic['color'], linewidth=2, label=ic['name'])
ax4.set_xlabel('Time (years)')
ax4.set_ylabel('Distance from EN')
ax4.set_title('(d) Distance from Equilibrium')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig("figure1b_en_timeseries.png", dpi=300)
plt.show()
print("✅ Figure 1B saved: figure1b_en_timeseries.png")

# =============================================================================
# FIGURE 1C: EP (PREDATOR-ONLY EQUILIBRIUM) TIME SERIES CURVES (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 1C: Simulating EP (Predator-Only Equilibrium) Dynamics")
print("=" * 60)

P0_val = EP[1] if EP is not None else 26.67

ep_initial_conditions = [
    {'name': 'Case 1: Very close', 'N0': 0.01, 'P0': P0_val * 0.99, 'color': '#ff7f0e'},
    {'name': 'Case 2: Close', 'N0': 0.05, 'P0': P0_val * 0.95, 'color': '#1f77b4'},
    {'name': 'Case 3: Moderate', 'N0': 0.10, 'P0': P0_val * 0.90, 'color': '#d62728'},
    {'name': 'Case 4: Further', 'N0': 0.50, 'P0': P0_val * 0.85, 'color': '#2ca02c'},
    {'name': 'Case 5: Far', 'N0': 1.00, 'P0': P0_val * 0.80, 'color': '#9467bd'},
]

t_span_ep = (0, 150)
t_eval_ep = np.linspace(0, 150, 2000)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax1 = axes[0, 0]
for ic in ep_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_ep, [ic['N0'], ic['P0']], t_eval_ep)
    ax1.plot(t_sim, N_sim, color=ic['color'], linewidth=2, label=f"{ic['name']} (N₀={ic['N0']}, P₀={ic['P0']:.1f})")
ax1.set_xlabel('Time (years)')
ax1.set_ylabel('Prey Biomass N (tonnes)')
ax1.set_title('(a) Prey Population Dynamics')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
for ic in ep_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_ep, [ic['N0'], ic['P0']], t_eval_ep)
    ax2.plot(t_sim, P_sim, color=ic['color'], linewidth=2, label=ic['name'])
ax2.axhline(P0_val, color='orange', linestyle=':', alpha=0.7, label=f'EP P* = {P0_val:.1f}')
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Predator Biomass P (tonnes)')
ax2.set_title('(b) Predator Population Dynamics')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
for ic in ep_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_ep, [ic['N0'], ic['P0']], t_eval_ep)
    ax3.plot(N_sim, P_sim, color=ic['color'], linewidth=2, alpha=0.8)
    ax3.scatter(ic['N0'], ic['P0'], color=ic['color'], s=80, zorder=5, edgecolors='black')
ax3.scatter(0, P0_val, color='red', s=200, marker='*', zorder=10, label=f'EP (0, {P0_val:.1f})')
if ENP is not None:
    ax3.scatter(ENP[0], ENP[1], color='green', s=100, marker='s', label=f'ENP ({ENP[0]:.1f}, {ENP[1]:.1f})')
ax3.set_xlabel('Prey Biomass N (tonnes)')
ax3.set_ylabel('Predator Biomass P (tonnes)')
ax3.set_title('(c) Phase Plane Trajectories')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
for ic in ep_initial_conditions:
    t_sim, N_sim, P_sim = simulate(baseline_params, t_span_ep, [ic['N0'], ic['P0']], t_eval_ep)
    distance = np.sqrt(N_sim**2 + (P_sim - P0_val)**2)
    ax4.plot(t_sim, distance, color=ic['color'], linewidth=2, label=ic['name'])
ax4.set_xlabel('Time (years)')
ax4.set_ylabel('Distance from EP')
ax4.set_title('(d) Distance from Equilibrium')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig("figure1c_ep_timeseries.png", dpi=300)
plt.show()
print("✅ Figure 1C saved: figure1c_ep_timeseries.png")

# =============================================================================
# FIGURE 1D: ENP (COEXISTENCE EQUILIBRIUM) TIME SERIES CURVES (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 1D: Simulating ENP (Coexistence Equilibrium) Dynamics")
print("=" * 60)

if ENP is not None:
    N_star_val, P_star_val = ENP[0], ENP[1]

    enp_initial_conditions = [
        {'name': 'Case 1: Very close', 'N0': N_star_val * 0.99, 'P0': P_star_val * 0.99, 'color': '#d62728'},
        {'name': 'Case 2: Close', 'N0': N_star_val * 0.95, 'P0': P_star_val * 0.95, 'color': '#1f77b4'},
        {'name': 'Case 3: Moderate', 'N0': N_star_val * 0.90, 'P0': P_star_val * 0.90, 'color': '#2ca02c'},
        {'name': 'Case 4: Further', 'N0': N_star_val * 0.80, 'P0': P_star_val * 0.80, 'color': '#ff7f0e'},
        {'name': 'Case 5: Far', 'N0': N_star_val * 0.70, 'P0': P_star_val * 0.70, 'color': '#9467bd'},
    ]

    t_span_enp = (0, 100)
    t_eval_enp = np.linspace(0, 100, 2000)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    for ic in enp_initial_conditions:
        t_sim, N_sim, P_sim = simulate(baseline_params, t_span_enp, [ic['N0'], ic['P0']], t_eval_enp)
        ax1.plot(t_sim, N_sim, color=ic['color'], linewidth=2, label=f"{ic['name']} (N₀={ic['N0']:.1f}, P₀={ic['P0']:.1f})")
    ax1.axhline(N_star_val, color='red', linestyle=':', alpha=0.7, label=f'ENP N* = {N_star_val:.1f}')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Prey Biomass N (tonnes)')
    ax1.set_title('(a) Prey Population Dynamics')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for ic in enp_initial_conditions:
        t_sim, N_sim, P_sim = simulate(baseline_params, t_span_enp, [ic['N0'], ic['P0']], t_eval_enp)
        ax2.plot(t_sim, P_sim, color=ic['color'], linewidth=2, label=ic['name'])
    ax2.axhline(P_star_val, color='red', linestyle=':', alpha=0.7, label=f'ENP P* = {P_star_val:.1f}')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Predator Biomass P (tonnes)')
    ax2.set_title('(b) Predator Population Dynamics')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for ic in enp_initial_conditions:
        t_sim, N_sim, P_sim = simulate(baseline_params, t_span_enp, [ic['N0'], ic['P0']], t_eval_enp)
        ax3.plot(N_sim, P_sim, color=ic['color'], linewidth=2, alpha=0.8)
        ax3.scatter(ic['N0'], ic['P0'], color=ic['color'], s=80, zorder=5, edgecolors='black')
    ax3.scatter(N_star_val, P_star_val, color='red', s=200, marker='*', zorder=10, label=f'ENP ({N_star_val:.1f}, {P_star_val:.1f})')
    ax3.set_xlabel('Prey Biomass N (tonnes)')
    ax3.set_ylabel('Predator Biomass P (tonnes)')
    ax3.set_title('(c) Phase Plane Trajectories')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    for ic in enp_initial_conditions:
        t_sim, N_sim, P_sim = simulate(baseline_params, t_span_enp, [ic['N0'], ic['P0']], t_eval_enp)
        distance = np.sqrt((N_sim - N_star_val)**2 + (P_sim - P_star_val)**2)
        ax4.plot(t_sim, distance, color=ic['color'], linewidth=2, label=ic['name'])
    ax4.set_xlabel('Time (years)')
    ax4.set_ylabel('Distance from ENP')
    ax4.set_title('(d) Distance from Equilibrium')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig("figure1d_enp_timeseries.png", dpi=300)
    plt.show()
    print("✅ Figure 1D saved: figure1d_enp_timeseries.png")

# =============================================================================
# SECTION 9: BASELINE SIMULATION AND FIGURE 2 (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("BASELINE SIMULATION")
print("=" * 60)

t_span = (0, 300)
t_eval = np.linspace(0, 300, 3000)
initial_state = [15.0, 8.0]

sol = solve_ivp(
    lambda t, y: model_equations(t, y, baseline_params),
    t_span, initial_state, t_eval=t_eval,
    method='RK45', rtol=1e-8, atol=1e-10
)

t, N, P = sol.t, sol.y[0], sol.y[1]
print(f"Simulation completed: t ∈ [{t_span[0]}, {t_span[1]}] years")
print(f"Final state: N = {N[-1]:.2f}, P = {P[-1]:.2f}")

# Figure 2: Baseline time series (NO MAIN TITLE)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t, N, 'b-', linewidth=2.5, label='Prey N(t)')
ax.plot(t, P, 'r--', linewidth=2.5, label='Predator P(t)')
if ENP is not None:
    ax.axhline(ENP[0], color='b', linestyle=':', alpha=0.7, label=f'N* = {ENP[0]:.2f}')
    ax.axhline(ENP[1], color='r', linestyle=':', alpha=0.7, label=f'P* = {ENP[1]:.2f}')
ax.set_xlabel("Time (years)")
ax.set_ylabel("Population Biomass (tonnes)")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure2_timeseries.png", dpi=300)
plt.show()
print("✅ Figure 2 saved: figure2_timeseries.png")

# =============================================================================
# FIGURE 3: PHASE PLANE WITH ALL EQUILIBRIA (NO MAIN TITLE)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(N, P, 'k-', linewidth=2, alpha=0.8, label='Trajectory')

if ENP is not None:
    ax.scatter(ENP[0], ENP[1], color='red', s=200, marker='*', zorder=5,
               label=f'ENP ({ENP[0]:.1f}, {ENP[1]:.1f})')
if EN is not None:
    ax.scatter(EN[0], EN[1], color='blue', s=150, marker='s', zorder=5,
               label=f'EN ({EN[0]:.1f}, 0)')
if EP is not None:
    ax.scatter(EP[0], EP[1], color='orange', s=150, marker='^', zorder=5,
               label=f'EP (0, {EP[1]:.1f})')
ax.scatter(E0[0], E0[1], color='purple', s=150, marker='o', zorder=5,
           label='E0 (0,0)', edgecolors='black', linewidths=2)

ax.set_xlabel("Prey Biomass N (tonnes)")
ax.set_ylabel("Predator Biomass P (tonnes)")
ax.legend(frameon=False, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure3_phase_plane.png", dpi=300)
plt.show()
print("✅ Figure 3 saved: figure3_phase_plane.png")

# =============================================================================
# FIGURE 4: SENSITIVITY OF COEXISTENCE EQUILIBRIUM (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 4: Sensitivity of Coexistence Equilibrium to Key Parameters")
print("=" * 60)

def sensitivity_analysis_2d(param_name, values, params):
    """Compute N* and P* for range of parameter values"""
    N_vals = []
    P_vals = []
    for val in values:
        p = params.copy()
        p[param_name] = val
        ENP_val = compute_ENP(p)
        if ENP_val is not None:
            N_vals.append(ENP_val[0])
            P_vals.append(ENP_val[1])
        else:
            N_vals.append(np.nan)
            P_vals.append(np.nan)
    return N_vals, P_vals

# Define ranges for sensitivity plots
sensitivity_ranges = {
    'r': np.linspace(0.8, 1.8, 20),
    'f': np.linspace(0.002, 0.018, 20),
    'm': np.linspace(0.1, 0.6, 20),
    'beta': np.linspace(0.3, 0.7, 20),
    'd': np.linspace(0.15, 0.4, 20),
    'alpha': np.linspace(0.015, 0.04, 20)
}

sensitivity_labels = {
    'r': 'Prey Growth Rate r (yr⁻¹)',
    'f': 'Fear Intensity f (tonnes⁻¹)',
    'm': 'Refuge Proportion m',
    'beta': 'Alternative Prey Growth β (yr⁻¹)',
    'd': 'Predator Mortality d (yr⁻¹)',
    'alpha': 'Predation Rate α (tonnes⁻¹ yr⁻¹)'
}

key_params = ['r', 'f', 'm', 'beta', 'd', 'alpha']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, param_name in enumerate(key_params):
    values = sensitivity_ranges[param_name]
    N_vals, P_vals = sensitivity_analysis_2d(param_name, values, baseline_params)
    ax = axes[idx]
    ax.plot(values, N_vals, 'b-o', linewidth=2, markersize=4, label='Prey N*')
    ax.plot(values, P_vals, 'r-s', linewidth=2, markersize=4, label='Predator P*')
    ax.axvline(x=baseline_params[param_name], color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel(sensitivity_labels[param_name])
    ax.set_ylabel('Equilibrium Biomass (tonnes)')
    ax.set_title(f'Sensitivity to {param_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure4_sensitivity_coexistence.png", dpi=300)
plt.show()
print("✅ Figure 4 saved: figure4_sensitivity_coexistence.png")

# =============================================================================
# FIGURE 5: TRACE AND DETERMINANT SENSITIVITY (NO MAIN TITLE)
# =============================================================================
print("\n" + "=" * 60)
print("FIGURE 5: Trace and Determinant Sensitivity Analysis")
print("=" * 60)

def compute_stability_metrics(param_name, values, params):
    """Compute trace and determinant for range of parameter values"""
    trace_vals = []
    det_vals = []
    for val in values:
        p = params.copy()
        p[param_name] = val
        ENP_val = compute_ENP(p)
        if ENP_val is not None:
            N_s, P_s = ENP_val
            J = jacobian_matrix(N_s, P_s, p)
            trace_vals.append(np.trace(J))
            det_vals.append(np.linalg.det(J))
        else:
            trace_vals.append(np.nan)
            det_vals.append(np.nan)
    return trace_vals, det_vals

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, param_name in enumerate(key_params):
    values = sensitivity_ranges[param_name]
    trace_vals, det_vals = compute_stability_metrics(param_name, values, baseline_params)
    ax = axes[idx]
    ax.plot(values, trace_vals, 'g-^', linewidth=2, markersize=4, label='Trace tr(J)')
    ax.plot(values, det_vals, 'm-v', linewidth=2, markersize=4, label='Determinant det(J)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=baseline_params[param_name], color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel(sensitivity_labels[param_name])
    ax.set_ylabel('Jacobian Properties')
    ax.set_title(f'Stability vs {param_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure5_trace_det_sensitivity.png", dpi=300)
plt.show()
print("✅ Figure 5 saved: figure5_trace_det_sensitivity.png")

# =============================================================================
# SECTION 10: COMPREHENSIVE RESULTS TABLE (LOW, MEDIUM, HIGH VALUES)
# =============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS TABLE (LOW, MEDIUM, HIGH PARAMETER VALUES)")
print("=" * 80)

all_results = []

for param_name, param_range in parameter_sets.items():
    for level, value in param_range.items():
        if level != 'label':
            test_params = baseline_params.copy()
            test_params[param_name] = value

            EN_test = compute_EN(test_params)
            EP_test = compute_EP(test_params)
            ENP_test = compute_ENP(test_params)

            if ENP_test is not None:
                N_star, P_star = ENP_test
                J = jacobian_matrix(N_star, P_star, test_params)
                trace = np.trace(J)
                det = np.linalg.det(J)
                eigvals = np.linalg.eigvals(J)
                stable = (trace < 0 and det > 0)

                if trace < 0 and det > 0:
                    if trace**2 - 4*det < 0:
                        stability_type = "Stable Focus"
                    else:
                        stability_type = "Stable Node"
                elif det < 0:
                    stability_type = "Saddle"
                else:
                    stability_type = "Unstable"
            else:
                N_star = np.nan
                P_star = np.nan
                trace = np.nan
                det = np.nan
                eigvals = [np.nan, np.nan]
                stable = False
                stability_type = "No coexistence"

            all_results.append({
                'Parameter': param_range['label'],
                'Level': level,
                'Value': value,
                'EN_N*': EN_test[0] if EN_test is not None else np.nan,
                'EN_P*': 0,
                'EP_N*': 0,
                'EP_P*': EP_test[1] if EP_test is not None else np.nan,
                'ENP_N*': N_star,
                'ENP_P*': P_star,
                'Trace': trace,
                'Determinant': det,
                'λ1': eigvals[0],
                'λ2': eigvals[1],
                'Stable': 'Yes' if stable else 'No',
                'Stability Type': stability_type
            })

df_comprehensive = pd.DataFrame(all_results)
print("\n" + df_comprehensive.to_string(index=False))
df_comprehensive.to_excel("comprehensive_results_all_parameters.xlsx", index=False)
print("\n✅ Saved: comprehensive_results_all_parameters.xlsx")

# =============================================================================
# SECTION 11: SUMMARY TABLE FOR BASELINE AND VARIATIONS
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY TABLE (Selected Scenarios)")
print("=" * 60)

scenarios = [
    {'name': 'BASELINE (Your Values)', 'params': baseline_params.copy()},
    {'name': 'Low Fear (f=0.003)', 'params': {**baseline_params, 'f': 0.003}},
    {'name': 'High Fear (f=0.015)', 'params': {**baseline_params, 'f': 0.015}},
    {'name': 'Low Refuge (m=0.15)', 'params': {**baseline_params, 'm': 0.15}},
    {'name': 'High Refuge (m=0.50)', 'params': {**baseline_params, 'm': 0.50}},
    {'name': 'Low Prey Growth (r=0.9)', 'params': {**baseline_params, 'r': 0.9}},
    {'name': 'High Prey Growth (r=1.6)', 'params': {**baseline_params, 'r': 1.6}},
    {'name': 'Low Predation (α=0.015)', 'params': {**baseline_params, 'alpha': 0.015}},
    {'name': 'High Predation (α=0.035)', 'params': {**baseline_params, 'alpha': 0.035}},
    {'name': 'Low Alt Prey (β=0.30)', 'params': {**baseline_params, 'beta': 0.30}},
    {'name': 'High Alt Prey (β=0.60)', 'params': {**baseline_params, 'beta': 0.60}},
    {'name': 'Low Pred Mortality (d=0.18)', 'params': {**baseline_params, 'd': 0.18}},
    {'name': 'High Pred Mortality (d=0.35)', 'params': {**baseline_params, 'd': 0.35}},
]

scenario_results = []
for sc in scenarios:
    p = sc['params']
    ENP_val = compute_ENP(p)

    if ENP_val is not None:
        N_star, P_star = ENP_val
        J = jacobian_matrix(N_star, P_star, p)
        trace = np.trace(J)
        det = np.linalg.det(J)
        eigvals = np.linalg.eigvals(J)

        if trace < 0 and det > 0:
            if trace**2 - 4*det < 0:
                stype = "Stable Focus"
            else:
                stype = "Stable Node"
        else:
            stype = "Unstable"

        scenario_results.append({
            'Scenario': sc['name'],
            'N*': round(N_star, 2),
            'P*': round(P_star, 2),
            'Trace': round(trace, 4),
            'Det': round(det, 4),
            'λ1': f"{eigvals[0]:.4f}",
            'λ2': f"{eigvals[1]:.4f}",
            'Stability': stype
        })
    else:
        scenario_results.append({
            'Scenario': sc['name'],
            'N*': 'NaN',
            'P*': 'NaN',
            'Trace': 'NaN',
            'Det': 'NaN',
            'λ1': 'NaN',
            'λ2': 'NaN',
            'Stability': 'No coexistence'
        })

df_scenarios = pd.DataFrame(scenario_results)
print("\n" + df_scenarios.to_string(index=False))
df_scenarios.to_excel("scenarios_summary.xlsx", index=False)
print("\n✅ Saved: scenarios_summary.xlsx")

# =============================================================================
# SECTION 12: SAVE ALL RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("SAVING ALL RESULTS")
print("=" * 60)

df_stability = pd.DataFrame(stability_results)
df_stability.to_excel("stability_analysis_baseline.xlsx", index=False)
print("✅ Saved: stability_analysis_baseline.xlsx")

df_simulation = pd.DataFrame({'Time': t, 'Prey_N': N, 'Predator_P': P})
df_simulation.to_excel("simulation_data.xlsx", index=False)
print("✅ Saved: simulation_data.xlsx")

df_params = pd.DataFrame({
    'Parameter': list(baseline_params.keys()),
    'Your_Value': list(baseline_params.values()),
    'Description': [
        'Prey growth rate (yr⁻¹)',
        'Prey carrying capacity (tonnes)',
        'Prey background mortality (yr⁻¹)',
        'Predation rate (tonnes⁻¹ yr⁻¹)',
        'Refuge proportion',
        'Fear intensity (tonnes⁻¹)',
        'Conversion efficiency',
        'Alternative prey growth rate (yr⁻¹)',
        'Alternative prey carrying capacity (tonnes)',
        'Predator mortality (yr⁻¹)'
    ]
})
df_params.to_excel("parameters_your_values.xlsx", index=False)
print("✅ Saved: parameters_your_values.xlsx")

df_ranges = pd.DataFrame([{
    'Parameter': info['label'],
    'Low': info['Low'],
    'Medium': info['Medium'],
    'High': info['High']
} for param_name, info in parameter_sets.items()])
df_ranges.to_excel("parameter_ranges_low_medium_high.xlsx", index=False)
print("✅ Saved: parameter_ranges_low_medium_high.xlsx")

# =============================================================================
# SECTION 13: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 80)

e0_str = f"({E0[0]:.2f}, {E0[1]:.2f})"
en_str = f"({EN[0]:.2f}, 0)" if EN is not None else "Not feasible"
ep_str = f"(0, {EP[1]:.2f})" if EP is not None else "Not feasible"
enp_str = f"({ENP[0]:.2f}, {ENP[1]:.2f})" if ENP is not None else "Not feasible"

if ENP is not None and len(stability_results) > 0:
    final_result = stability_results[-1]
    trace_val = final_result['trace']
    det_val = final_result['det']
    class_val = final_result['classification']
    stable_val = final_result['stable']
else:
    trace_val = det_val = class_val = stable_val = "N/A"

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE RESULTS SUMMARY (YOUR VALUES)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ EQUILIBRIUM POINTS (YOUR PARAMETER VALUES)                                   │
│   • E0 (Trivial):                    {e0_str}                                │
│   • EN (Prey-only):                  {en_str}                                │
│   • EP (Predator-only):              {ep_str}                                │
│   • ENP (Coexistence):               {enp_str}                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ STABILITY AT COEXISTENCE (ENP)                                               │
│   • Trace tr(J):                     {trace_val}                             │
│   • Determinant det(J):              {det_val}                               │
│   • Classification:                  {class_val}                             │
│   • Stable:                          {stable_val}                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ GENERATED FIGURES (NO MAIN TITLES - ADD YOUR OWN IN LATEX):                  │
│   • figure1a_e0_timeseries.png       - E0 (Trivial) time series curves      │
│   • figure1b_en_timeseries.png       - EN (Prey-only) time series curves    │
│   • figure1c_ep_timeseries.png       - EP (Predator-only) time series curves│
│   • figure1d_enp_timeseries.png      - ENP (Coexistence) time series curves │
│   • figure2_timeseries.png           - Baseline time series                 │
│   • figure3_phase_plane.png          - Phase plane with all equilibria      │
│   • figure4_sensitivity_coexistence.png - Sensitivity of ENP to parameters  │
│   • figure5_trace_det_sensitivity.png - Trace & determinant sensitivity     │
├─────────────────────────────────────────────────────────────────────────────┤
│ GENERATED EXCEL FILES:                                                       │
│   • comprehensive_results_all_parameters.xlsx - ALL parameters (L/M/H)      │
│   • scenarios_summary.xlsx           - 13 scenarios summary                 │
│   • stability_analysis_baseline.xlsx - Jacobian and eigenvalues             │
│   • simulation_data.xlsx             - Time series data                     │
│   • parameters_your_values.xlsx      - Your parameter values                │
│   • parameter_ranges_low_medium_high.xlsx - L/M/H ranges for all parameters │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
print("All parameter values have been replaced with your provided values!")
print("All figure main titles have been removed - you can add your own in LaTeX!")
print("Time series curves for E0, EN, EP, and ENP have been generated!")
print("=" * 80)
