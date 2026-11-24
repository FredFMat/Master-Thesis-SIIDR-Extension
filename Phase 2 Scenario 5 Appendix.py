"""
Phase 2 — Scenario 5: Combined Scenario

Left panel:  I + I_D — combined scenario vs baseline

Combined: beta_max=2*beta_0, k=0.02, t_half=5000, delta=0.01
Baseline: beta constant=beta_0, delta=0
Outbreak runs only (R0 > 1). 95% band only.

Sources:
  Table 5 (means and stds): Chernikova et al. (2023), Applied Network Science
  N = 317,031:              Chernikova et al. (2022), ESORICS Table 4, port 445
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(42)

BETA_0   = 0.16
MU_MEAN  = 0.11;  MU_STD  = 0.10
G1_MEAN  = 0.80;  G1_STD  = 0.15
G2_MEAN  = 0.06;  G2_STD  = 0.06
N        = 317031
BETA_MAX = 2.0 * BETA_0
K        = 0.02
T_HALF   = 5000
DELTA    = 0.01
T_STEPS  = 10000
N_RUNS   = 500

t_eval      = np.linspace(0, T_STEPS, T_STEPS)


def siidr_extended(t, y, beta_0, beta_max, mu, g1, g2, N, k, t_half, delta):
    S, I, ID, R = y
    beta_t = beta_0 + (beta_max - beta_0) / (1 + np.exp(-k * (t - t_half)))
    dS  = -beta_t * S * I / N  +  delta * R
    dI  =  beta_t * S * I / N  -  (g1 + mu) * I  +  g2 * ID
    dID =  g1 * I  -  g2 * ID
    dR  =  mu * I  -  delta * R
    return [dS, dI, dID, dR]


def run_scenario(beta_max, k, t_half, delta):
    instant_runs = []

    for _ in range(N_RUNS):
        mu  = np.clip(np.random.normal(MU_MEAN, MU_STD), 0, 1)
        g1  = np.clip(np.random.normal(G1_MEAN, G1_STD), 0, 1)
        g2  = np.clip(np.random.normal(G2_MEAN, G2_STD), 0, 1)

        if BETA_0 / max(mu, 1e-10) <= 1.0:
            continue

        sol = solve_ivp(siidr_extended, (0, T_STEPS), [N-1, 1, 0, 0],
                        t_eval=t_eval,
                        args=(BETA_0, beta_max, mu, g1, g2, N,
                              k, t_half, delta),
                        method='RK45')
        S_v, I_v, ID_v, R_v = sol.y
        if np.max(np.abs(S_v + I_v + ID_v + R_v - N)) / N > 1e-9:
            continue

        instant_runs.append(I_v + ID_v)

    return instant_runs


# -- Run both scenarios
print("Computing Phase 2 baseline...")
base_instant = run_scenario(BETA_0, 0.02, T_STEPS*2, 0)
print(f"  Baseline outbreak runs: {len(base_instant)}/{N_RUNS}")

print("Computing combined scenario...")
comb_instant = run_scenario(BETA_MAX, K, T_HALF, DELTA)
print(f"  Combined outbreak runs: {len(comb_instant)}/{N_RUNS}")


base_i_median = np.median(np.array(base_instant), axis=0)
comb_i_median = np.median(np.array(comb_instant), axis=0)

base_med_peak = np.median(np.max(np.array(base_instant), axis=1))
comb_med_peak = np.median(np.max(np.array(comb_instant), axis=1))
peak_abs_diff = comb_med_peak - base_med_peak

print(f"\nBaseline median peak: {base_med_peak:.0f} ({100*base_med_peak/N:.2f}% N)")
print(f"Combined median peak: {comb_med_peak:.0f} ({100*comb_med_peak/N:.2f}% N)")
print(f"Difference:           +{peak_abs_diff:.0f} nodes ({100*peak_abs_diff/N:.2f}% N)")


# -- PLOT 1: Infection prevalence curves — combined vs baseline

fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(t_eval, base_i_median, color='grey', linewidth=1.5,
         linestyle='--',
         label=f'Baseline median (n={len(base_instant)})')

ax1.plot(t_eval, comb_i_median, color='darkblue', linewidth=2,
         label=f'Combined median (n={len(comb_instant)})')

ax1.set_title('')
ax1.set_xlabel('Simulation Steps (t)', fontsize=14)
ax1.set_ylabel('Infected machines (I + I_D)', fontsize=14)
ax1.legend(fontsize=12)

fig1.suptitle(
    'Total infection prevalence over time | Extended model vs Baseline\n'
    f'β_max=2β₀ | δ={DELTA}',
    fontsize=14
)
fig1.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig('Figures_Presentation/Results/Phase_5_Infections.svg',
            bbox_inches='tight')
plt.show()


# -- PLOT 2: Violin + boxplot + individual runs — baseline vs combined

base_peaks = list(np.max(np.array(base_instant), axis=1))
comb_peaks = list(np.max(np.array(comb_instant), axis=1))

fig2, ax2 = plt.subplots(figsize=(10, 7))
np.random.seed(99)

for pos, (peaks_list, color, label, count) in enumerate(
        zip([base_peaks, comb_peaks],
            ['grey', 'darkblue'],
            ['Baseline', 'Combined'],
            [len(base_instant), len(comb_instant)]), start=1):

    peaks_arr = np.array(peaks_list)

    parts = ax2.violinplot([peaks_arr], positions=[pos],
                           showmeans=False, showmedians=False,
                           showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.5)

    ax2.boxplot([peaks_arr], positions=[pos], widths=0.1,
                patch_artist=True,
                boxprops=dict(facecolor='white', color='black', linewidth=1.5),
                medianprops=dict(color='orange', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.2),
                capprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='', markersize=0))

    jitter = np.random.uniform(-0.01, 0.01, size=len(peaks_arr))
    ax2.scatter(pos + jitter, peaks_arr, color=color, edgecolors='white',
                linewidths=0.5, s=20, alpha=0.8, zorder=3)

    q1          = np.percentile(peaks_arr, 25)
    q3          = np.percentile(peaks_arr, 75)
    iqr         = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    print(f"\n—— {label} ——————————————————————————————————————————")
    print(f"  n={len(peaks_arr)}  |  Min={peaks_arr.min():.0f}  |  "
          f"Q1={q1:.0f}  |  Median={np.median(peaks_arr):.0f}  |  "
          f"Q3={q3:.0f}  |  Max={peaks_arr.max():.0f}")
    print(f"  Mean={peaks_arr.mean():.0f}  |  IQR={iqr:.0f}  |  "
          f"Lower fence={max(lower_fence,0):.0f}  |  Upper fence={upper_fence:.0f}")
    print(f"  Below Q1: {int(np.sum(peaks_arr < q1))}  |  "
          f"IQR box: {int(np.sum((peaks_arr >= q1) & (peaks_arr <= q3)))}  |  "
          f"Above Q3 (whisker): {int(np.sum((peaks_arr > q3) & (peaks_arr <= upper_fence)))}  |  "
          f"Upper outliers: {int(np.sum(peaks_arr > upper_fence))}")

ax2.set_xticks([1, 2])
ax2.set_xticklabels([f'Baseline - {len(base_instant)} outbreak runs',
                     f'Combined - {len(comb_instant)} outbreak runs'])
ax2.set_xlabel('Scenario')
ax2.set_ylabel('Infected machines (I + I_D)', fontsize=14)
ax2.set_title('Highest infection count reached in each run - Combined Scenario',
              fontsize=14)
fig2.tight_layout()
plt.savefig('Figures_Presentation/Results/Phase_5_Violin.svg',
            bbox_inches='tight')
plt.show()