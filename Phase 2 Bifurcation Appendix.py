"""
Phase 2 — Analysis of Equilibria

Uses the posterior means mu = 0.11, gamma_1 = 0.80, gamma_2 = 0.06 and
N = 317,031 with a constant transmission rate beta.

Produces the bifurcation diagram (Figure 4): steady-state endemic prevalence
versus R0 for four delta values, showing the transcritical
bifurcation at R0 = 1.

Each point on a colored curve is an independent steady state, the long-run endemic
prevalence a system with constant beta and the given R0 and delta converges to. 
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -- Fixed parameters
MU      = 0.11
GAMMA_1 = 0.80
GAMMA_2 = 0.06
BETA_0  = 0.16
N       = 317031
T_LONG  = 10000


# -- Bifurcation parameter grid
R0_grid   = np.linspace(0.5, 4.5, 600)
beta_grid = R0_grid * MU

# -- Delta values
delta_values = [0.001, 0.01, 0.05, 0.10]
delta_labels = ['δ=0.001', 'δ=0.01', 'δ=0.05', 'δ=0.10']
colors       = ['darkblue', 'orange', 'red', 'purple']


# -- Analytical endemic equilibrium
def endemic_equilibrium(R0_values, delta, mu, gamma1, gamma2, N):
    """
    Returns (I* + I_D*) / N at the endemic equilibrium for each R0.
    Returns 0 for R0 <= 1 (disease-free equilibrium).
    Derived analytically for constant beta and delta > 0.
    """
    I_star  = np.where(
        R0_values > 1,
        N * (1 - 1/R0_values) / (1 + gamma1/gamma2 + mu/delta),
        0.0
    )
    ID_star = np.where(R0_values > 1, gamma1/gamma2 * I_star, 0.0)
    return (I_star + ID_star) / N


# -- Numerical verification
def siidr_extended(t, y, beta_0, beta_max, mu, g1, g2, N, k, t_half, delta):
    S, I, ID, R = y
    beta_t = beta_0 + (beta_max - beta_0) / (1 + np.exp(-k * (t - t_half)))
    dS  = -beta_t * S * I / N  +  delta * R
    dI  =  beta_t * S * I / N  -  (g1 + mu) * I  +  g2 * ID
    dID =  g1 * I  -  g2 * ID
    dR  =  mu * I  -  delta * R
    return [dS, dI, dID, dR]


def numerical_equilibrium(beta_val, delta):
    """
    Runs the ODE with constant beta = beta_val to T_LONG and returns
    the final (I + I_D) / N as a numerical check of the analytical formula.
    beta_0 = beta_max = beta_val ensures beta is constant throughout.
    t_half = -T_LONG places the logistic midpoint far in the past so
    beta(t) ≈ beta_val from t=0 onward.
    """
    t_eval = np.linspace(T_LONG * 0.9, T_LONG, 100)
    sol = solve_ivp(
        siidr_extended,
        (0, T_LONG), [N-1, 1, 0, 0],
        t_eval=t_eval,
        args=(beta_val, beta_val, MU, GAMMA_1, GAMMA_2, N,
              0.02, -T_LONG, delta),
        method='RK45', rtol=1e-9, atol=1e-9
    )
    I_v, ID_v = sol.y[1], sol.y[2]
    return (I_v + ID_v)[-1] / N


print("Numerical verification (all delta values):")
overall_max_dev = 0.0
for delta_v in delta_values:
    print(f"\n── δ = {delta_v} ──────────────────────────────────────────")
    print(f"{'R0':>6}  {'Analytical':>12}  {'Numerical':>12}  {'Difference':>12}")
    delta_max_dev = 0.0
    for r0 in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        analytical = endemic_equilibrium(np.array([r0]), delta_v,
                                         MU, GAMMA_1, GAMMA_2, N)[0]
        numerical  = numerical_equilibrium(r0 * MU, delta_v)
        diff = abs(analytical - numerical)
        delta_max_dev = max(delta_max_dev, diff)
        overall_max_dev = max(overall_max_dev, diff)
        print(f"{r0:>6.1f}  {analytical:>12.6f}  {numerical:>12.6f}  "
              f"{diff:>12.2e}")
    print(f"  Max deviation (δ={delta_v}): {delta_max_dev:.2e}")
print(f"\nOverall max deviation across all (R₀, δ) combinations: {overall_max_dev:.2e}")
print()


# -- LINEAR STABILITY ANALYSIS — endemic equilibrium
# Posterior mean parameters, constant beta, all 2,100 grid points with R0 > 1.

def endemic_jacobian(beta, delta, mu, g1, g2):
    """Jacobian of the reduced system (I, I_D, R) at the endemic equilibrium."""
    R0 = beta / mu
    S  = 1 / R0
    I  = (1 - 1/R0) / (1 + g1/g2 + mu/delta)
    return np.array([
        [beta * (S - I) - (g1 + mu), -beta * I + g2, -beta * I],
        [g1,                         -g2,            0.0      ],
        [mu,                          0.0,           -delta   ],
    ])


max_real_part = -np.inf
n_points      = 0
for delta_v in delta_values:
    for r0 in R0_grid[R0_grid > 1]:
        eigenvalues = np.linalg.eigvals(
            endemic_jacobian(r0 * MU, delta_v, MU, GAMMA_1, GAMMA_2))
        max_real_part = max(max_real_part, eigenvalues.real.max())
        n_points += 1

print("Linear stability of the endemic equilibrium:")
print(f"  Grid points evaluated (R0 > 1, all δ): {n_points}")
print(f"  Largest real part of any eigenvalue:   {max_real_part:+.2e}")
print(f"  Result: {'all stable' if max_real_part < 0 else 'UNSTABLE POINT FOUND'}")
print()


# -- Reference R0 values
R0_start = BETA_0 / MU               # ≈ 1.45 — baseline operating point
R0_1_5   = (1.5 * BETA_0) / MU      # ≈ 2.18 — β_max = 1.5β₀
R0_2_0   = (2.0 * BETA_0) / MU      # ≈ 2.91 — β_max = 2.0β₀
R0_3_0   = (3.0 * BETA_0) / MU      # ≈ 4.36 — β_max = 3.0β₀

scenario_R0s        = [R0_1_5, R0_2_0, R0_3_0]
scenario_labels_bif = ['β_max=1.5β₀', 'β_max=2.0β₀', 'β_max=3.0β₀']
scenario_colors_bif = ['darkblue', 'orange', 'red']

# -- Maximum endemic prevalence asymptotes
# -- Equilibrium predictions vs observed medians (Appendix E)
# Predicted endemic equilibrium for constant beta = beta_max at the posterior
# means, compared with the observed median peak infected counts. The observed
# medians are taken from the outputs of the scenario group scripts (Table 4).
G_RATIO = GAMMA_1 / GAMMA_2

def delta_factor(delta):
    return (1 + G_RATIO) / (1 + G_RATIO + MU / delta)

prediction_configs = [
    # label,                       beta_max/beta_0, delta, observed median
    ('β_max = 1.5β₀, δ = 0.01',    1.5, 0.01,  128812),
    ('β_max = 2.0β₀, δ = 0.01',    2.0, 0.01,  151434),
    ('β_max = 3.0β₀, δ = 0.01',    3.0, 0.01,  161847),
    ('β_max = 2.0β₀, δ = 0.001',   2.0, 0.001,  72603),
    ('β_max = 2.0β₀, δ = 0.05',    2.0, 0.05,  199892),
    ('Combined scenario',          2.0, 0.01,  151302),
]

print("Equilibrium predictions vs observed medians:")
print(f"{'Configuration':<27} {'R0':>5} {'Ceiling':>8} {'δ factor':>9} "
      f"{'Predicted':>20} {'Observed':>20} {'Obs/Pred':>9}")
for label, m, delta_v, observed in prediction_configs:
    r0   = m * BETA_0 / MU
    pred = endemic_equilibrium(np.array([r0]), delta_v,
                               MU, GAMMA_1, GAMMA_2, N)[0]
    pred_str = f"{100*pred:.1f}% ({pred*N:,.0f})"
    obs_str  = f"{100*observed/N:.1f}% ({observed:,})"
    print(f"{label:<27} {r0:>5.2f} {100*(1 - 1/r0):>7.1f}% "
          f"{delta_factor(delta_v):>9.3f} {pred_str:>20} {obs_str:>20} "
          f"{observed/(pred*N):>9.2f}")
print(f"  δ factor rises ×{delta_factor(0.05)/delta_factor(0.001):.1f} "
      f"from δ = 0.001 to δ = 0.05")
print(f"  Ceiling rises ×{(1 - MU/(3.0*BETA_0))/(1 - MU/(1.5*BETA_0)):.1f} "
      f"from β_max = 1.5β₀ to 3.0β₀")

print("Plateaus of the combined scenario (predicted equilibrium, δ = 0.01):")
for label, beta_val in [('β₀ (first plateau)', BETA_0),
                        ('2.0β₀ (second plateau)', 2.0 * BETA_0)]:
    r0   = beta_val / MU
    pred = endemic_equilibrium(np.array([r0]), 0.01,
                               MU, GAMMA_1, GAMMA_2, N)[0]
    print(f"  {label:<24} R0 = {r0:.2f}: {100*pred:.1f}% of N ({pred*N:,.0f} nodes)")
print()


# -- Figure setup
fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))


# -- Bifurcation diagram
# Collection of steady states — one per (R0, delta) combination.

# -- Disease-free equilibrium branch 
ax1.plot(R0_grid[R0_grid <= 1], np.zeros(np.sum(R0_grid <= 1)),
         color='black', linewidth=2.5, linestyle='-',
         label='Disease-free equilibrium (stable)')
ax1.plot(R0_grid[R0_grid > 1], np.zeros(np.sum(R0_grid > 1)),
         color='black', linewidth=2.5, linestyle='--',
         label='Disease-free equilibrium (unstable)')

# -- Endemic equilibrium branches 
for delta, label, color in zip(delta_values, delta_labels, colors):
    eq   = endemic_equilibrium(R0_grid, delta, MU, GAMMA_1, GAMMA_2, N)
    mask = R0_grid > 1
    ax1.plot(R0_grid[mask], eq[mask], color=color, linewidth=2,
             label=f'Endemic steady state ({label})')

# -- Bifurcation point
ax1.axvline(x=1.0, color='grey', linewidth=1.2, linestyle=':',
            label='Bifurcation point R₀=1')

# -- Baseline operating point
ax1.axvline(x=R0_start, color='black', linewidth=1.5,
            linestyle='-.', alpha=0.7,
            label=f'Baseline: R₀≈{R0_start:.2f}')

# -- beta_max scenario reference lines
# Each vertical line marks the steady-state collection point for that
# beta_max scenario — the R0 value the limiting constant-beta system sits at.
for r0_end, slabel, scolor in zip(
        scenario_R0s, scenario_labels_bif, scenario_colors_bif):
    ax1.axvline(x=r0_end, color=scolor, linewidth=1.5,
                linestyle=':', alpha=0.8,
                label=f'{slabel}: R₀≈{r0_end:.2f}')

ax1.set_xlabel('Basic Reproduction Number R₀ = β/μ', fontsize=14)
ax1.set_ylabel('Endemic steady-state prevalence (I* + I_D*) / N', fontsize=14)
ax1.legend(fontsize=12, loc='upper center',
           bbox_to_anchor=(0.5, -0.15),
           ncol=2, borderaxespad=0,)
fig.tight_layout(rect=[0, 0.12, 1, 0.90])
ax1.set_xlim(0.9, 4.5)
ax1.set_ylim(-0.02, 0.85)



fig.suptitle(
    'Asymptotic Bifurcation Diagram - Sensitivity Analysis for β_max\n'
    'Endemic steady-state prevalence for varying R₀ and δ',
    fontsize=14
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('Figures_Presentation/Results/Bifurcation_Diagram.svg',
            bbox_inches='tight')
plt.show()