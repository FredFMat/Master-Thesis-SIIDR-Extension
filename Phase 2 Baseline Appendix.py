""" 
Phase 2 Baseline Standalone

Runs the original unmodified SIIDR ODE system 500 times,
sampling mu, gamma_1 and gamma_2 from the posterior
distribution (companion paper Table 5, wc_1_1s). beta_0 is fixed at its
Table 5 mean value and is not sampled.

Only outbreak runs are included.

Output metric: I + I_D.

Population conservation check uses a relative threshold (max_dev/N > 1e-9),
consistent with Scenario Groups 2 through 5 at N = 317,031.

Sources:
  Table 5 (means and stds): Chernikova et al. (2023), Applied Network Science
  N = 317,031:              Chernikova et al. (2022), ESORICS Table 4, port 445
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(42)

# -- Fixed parameters
# beta_0 = beta, mu, gamma_1, gamma_2 are sampled from the posterior distribution.

BETA_0   = 0.16
MU_MEAN  = 0.11;  MU_STD  = 0.10
G1_MEAN  = 0.80;  G1_STD  = 0.15
G2_MEAN  = 0.06;  G2_STD  = 0.06
N        = 317031

T_STEPS  = 10000
N_RUNS   = 500

t_eval = np.linspace(0, T_STEPS, T_STEPS)


# -- Original unmodified SIIDR ODE system

def siidr_original(t, y, beta, mu, gamma_1, gamma_2, N):
    """
    Original unmodified SIIDR ODE system from Chernikova et al. (2022).
    """
    S, I, ID, R = y
    dS_dt  = -beta * S * I / N
    dI_dt  =  beta * S * I / N  -  (gamma_1 + mu) * I  +  gamma_2 * ID
    dID_dt =  gamma_1 * I       -  gamma_2 * ID
    dR_dt  =  mu * I
    return [dS_dt, dI_dt, dID_dt, dR_dt]


# -- Running the simulation (only outbreak runs)

outbreak_runs = []
peaks         = []
n_contained   = 0

for _ in range(N_RUNS):
    mu      = np.clip(np.random.normal(MU_MEAN, MU_STD), 0, 1)
    gamma_1 = np.clip(np.random.normal(G1_MEAN, G1_STD), 0, 1)
    gamma_2 = np.clip(np.random.normal(G2_MEAN, G2_STD), 0, 1)

    # Only include outbreak runs: R0 = beta_0/mu > 1
    if BETA_0 / max(mu, 1e-10) <= 1.0:
        n_contained += 1
        continue

    y0 = [N - 1, 1, 0, 0]

    solution = solve_ivp(
        fun          = siidr_original,
        t_span       = (0, T_STEPS),
        y0           = y0,
        t_eval       = t_eval,
        args         = (BETA_0, mu, gamma_1, gamma_2, N),
        method       = 'RK45',
        dense_output = False
    )

    S_v, I_v, ID_v, R_v = solution.y

    # Check to see if relative population is conserved 
    max_dev = np.max(np.abs(S_v + I_v + ID_v + R_v - N)) / N
    if max_dev > 1e-9:
        print(f"WARNING: population conservation violated (rel dev={max_dev:.2e}). Run excluded.")
        continue


    instant = I_v + ID_v
    outbreak_runs.append(instant)
    peaks.append(max(instant))

print(f"Outbreak runs (R0 > 1): {len(outbreak_runs)}/{N_RUNS}")
print(f"Contained runs (R0 ≤ 1): {n_contained}/{N_RUNS}")
print(f"Median peak I+I_D: {np.median(peaks):.0f} ({100*np.median(peaks)/N:.2f}% N)")
print(f"Mean peak I+I_D:   {np.mean(peaks):.0f} ({100*np.mean(peaks)/N:.2f}% N)")

if not outbreak_runs:
    print("No valid outbreak runs. Exiting.")
    exit()


# -- PLOT 2: Violin + boxplot overlay with text summary 
peaks_arr = np.array(peaks)

fig2, ax2 = plt.subplots(figsize=(10, 7))

# -- Violin 
parts = ax2.violinplot([peaks_arr], showmeans=False, showmedians=False,
                       showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('darkblue')
    pc.set_alpha(0.5)


# -- Boxplot overlay 
ax2.boxplot([peaks_arr], positions=[1], widths=0.1,
            patch_artist=True,
            boxprops=dict(facecolor='white', color='black', linewidth=1.5),
            medianprops=dict(color='orange', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.5),
            flierprops=dict(marker='', markersize=0))  # hide default outlier dots


# -- Individual run markers
# Add a small random horizontal jitter so overlapping points are visible.
# Jitter is purely cosmetic.

np.random.seed(42)   # separate seed so jitter is reproducible independently
jitter   = np.random.uniform(-0.005, 0.005, size=len(peaks_arr))
x_jitter = 1 + jitter

ax2.scatter(x_jitter, peaks_arr, color='darkblue', edgecolors='white',
            linewidths=0.5, s=30, alpha=0.8, zorder=3,
            label=f'Individual runs (n={len(peaks_arr)})')

ax2.set_xticks([1])
ax2.set_xticklabels(['wc_1_1s baseline'], fontsize=13)
ax2.set_ylabel('Infected machines (I + I_D)', fontsize=14)
ax2.set_title(
    f'Highest infection count reached in each run - Phase 2 Baseline\n'
    f'Outbreak runs only (n={len(outbreak_runs)})', fontsize=14
)
fig2.tight_layout()
plt.savefig('Figures_Presentation/Results/Phase_2_Baseline_Violin.svg',
            bbox_inches='tight')
plt.show()

# -- Text summary 
q1  = np.percentile(peaks_arr, 25)
q3  = np.percentile(peaks_arr, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

n_below_q1      = int(np.sum(peaks_arr < q1))
n_iqr           = int(np.sum((peaks_arr >= q1) & (peaks_arr <= q3)))
n_above_q3      = int(np.sum((peaks_arr > q3) & (peaks_arr <= upper_fence)))
n_outliers_high = int(np.sum(peaks_arr > upper_fence))
n_outliers_low  = int(np.sum(peaks_arr < lower_fence))

print("\n—— Violin / Boxplot Summary —————————————————————————————————————————")
print(f"  Total outbreak runs:     {len(peaks_arr)}")
print(f"  Min:                     {peaks_arr.min():.0f}")
print(f"  Q1  (25th percentile):   {q1:.0f}")
print(f"  Median:                  {np.median(peaks_arr):.0f}")
print(f"  Q3  (75th percentile):   {q3:.0f}")
print(f"  Max:                     {peaks_arr.max():.0f}")
print(f"  Mean:                    {peaks_arr.mean():.0f}")
print(f"  IQR:                     {iqr:.0f}")
print(f"  Lower fence (Q1-1.5×IQR): {max(lower_fence, 0):.0f}")
print(f"  Upper fence (Q3+1.5×IQR): {upper_fence:.0f}")
print()
print(f"  Below Q1  (lower whisker region): {n_below_q1} runs")
print(f"  Within IQR (box):                 {n_iqr} runs")
print(f"  Above Q3  (upper whisker region): {n_above_q3} runs")
print(f"  Upper outliers (above fence):     {n_outliers_high} runs")
print(f"  Lower outliers (below fence):     {n_outliers_low} runs")
thr = 0.01 * N
print(f"  <1% N:    {int(np.sum(peaks_arr < thr))}/{len(peaks_arr)} "
      f"({100*np.sum(peaks_arr < thr)/len(peaks_arr):.1f}%)")