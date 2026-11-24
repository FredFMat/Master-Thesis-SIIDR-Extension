"""
Phase 1 Baseline

Runs the unmodified SIIDR ODE system 50 times per trace, drawing parameters
from normal distributions using the posterior mean and standard deviation
from Table 5 of the companion paper (Chernikova et al., 2023). Both mean
and standard deviation come from the same ABC-SMC-MNN estimation
procedure.

Sources:
  Table 5 (means and stds): Chernikova et al. (2023), Applied Network Science paper
  N per trace:              Chernikova et al. (2023), Applied Network Science paper Table 3
  7-trace selection:        Chernikova et al. (2022), ESORICS Table 2
"""

import numpy as np
from scipy.integrate import solve_ivp

# -- Random seed for reproducibility
np.random.seed(42)

# -- WannaCry Traces 
# N = number of contacted IPs per trace.
#
# Format per trace:
#   (beta_mean, mu_mean, gamma1_mean, gamma2_mean, N,
#    beta_std,  mu_std,  gamma1_std,  gamma2_std)

traces = {
    'wc_1_500ms': (0.16, 0.11, 0.79, 0.06, 37,  0.10, 0.11, 0.15, 0.07),
    'wc_1_1s':    (0.16, 0.11, 0.80, 0.06, 37,  0.11, 0.10, 0.15, 0.06),
    'wc_1_5s':    (0.05, 0.04, 0.82, 0.02, 37,  0.03, 0.03, 0.12, 0.01),
    'wc_1_10s':   (0.13, 0.08, 0.80, 0.05, 34,  0.08, 0.07, 0.15, 0.04),
    'wc_4_1s':    (0.17, 0.11, 0.79, 0.07, 35,  0.13, 0.13, 0.17, 0.07),
    'wc_4_5s':    (0.14, 0.09, 0.76, 0.07, 36,  0.10, 0.08, 0.18, 0.07),
    'wc_8_20s':   (0.14, 0.09, 0.80, 0.07, 35,  0.10, 0.09, 0.15, 0.06)
}

# -- Simulation parameters
T_STEPS = 1000
N_RUNS  = 50


# -- ODE System

def siidr_original(t, y, beta, mu, gamma_1, gamma_2, N):
    """
    Original unmodified SIIDR ODE system from Chernikova et al. (2022).
    No extensions — constant beta, delta = 0.
    """
    S, I, ID, R = y
    dS_dt  = -beta * S * I / N
    dI_dt  =  beta * S * I / N  -  (gamma_1 + mu) * I  +  gamma_2 * ID
    dID_dt =  gamma_1 * I       -  gamma_2 * ID
    dR_dt  =  mu * I
    return [dS_dt, dI_dt, dID_dt, dR_dt]


t_eval = np.linspace(0, T_STEPS, T_STEPS)


for idx, (trace_name, params) in enumerate(traces.items()):

    b_mean, mu_mean, g1_mean, g2_mean, N = params[:5]
    b_std,  mu_std,  g1_std,  g2_std     = params[5:]

    R0_val     = b_mean / mu_mean
    valid_runs = []

    for run in range(N_RUNS):

        # -- Sample parameters from ABC-SMC-MNN posterior (Table 5) 
        # Parameters are per-step probabilities so must stay in [0, 1].

        beta    = np.clip(np.random.normal(b_mean,  b_std),  0, 1)
        mu      = np.clip(np.random.normal(mu_mean, mu_std), 0, 1)
        gamma_1 = np.clip(np.random.normal(g1_mean, g1_std), 0, 1)
        gamma_2 = np.clip(np.random.normal(g2_mean, g2_std), 0, 1)

        # -- Initial conditions
        y0 = [N - 1, 1, 0, 0]

        # -- Solve ODE
        solution = solve_ivp(
            fun          = siidr_original,
            t_span       = (0, T_STEPS),
            y0           = y0,
            t_eval       = t_eval,
            args         = (beta, mu, gamma_1, gamma_2, N),
            method       = 'RK45',
            dense_output = False
        )

        S_v, I_v, ID_v, R_v = solution.y

        # -- Population conservation check
        max_dev = np.max(np.abs(S_v + I_v + ID_v + R_v - N))
        if max_dev > 1e-6:
            continue

        cumulative = I_v + ID_v + R_v
        valid_runs.append(cumulative)

    if len(valid_runs) == 0:
        print(f"WARNING: {trace_name} — no valid runs.")
        continue

    # -- Print summary
    print(f"Trace: {trace_name}")
    print(f"  N={N} | R₀={R0_val:.3f} | Valid runs: {len(valid_runs)}/{N_RUNS}")
    print()
