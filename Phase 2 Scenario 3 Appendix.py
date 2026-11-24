"""
Phase 2 — Scenario Group 3: Sensitivity to t_half and k.

Runs the extended SIIDR model for a 4 x 3 grid of the efficiency gain
midpoint t_half in {1000, 3000, 5000, 7000} and steepness k in
{0.08, 0.02, 0.008}, with beta_max = 2.0 * beta_0, delta = 0.01,
T = 10,000 steps and 500 parameter draws per cell. 
Only outbreak runs are included.

Outputs:
  - per cell: outbreak runs, median peak, IQR and median peak time
    (Table 3)
  - pooled by t_half and by k: summary statistics (Table 4) and
    violin plots (Figures 6 and 7)

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
DELTA    = 0.01
T_STEPS  = 10000
N_RUNS   = 500

t_half_values = [1000, 3000, 5000, 7000]
t_half_labels = ['t_half=1000', 't_half=3000', 't_half=5000', 't_half=7000']
k_values      = [0.08, 0.02, 0.008]
k_labels      = ['k=0.08 (rapid)',
                  'k=0.02 (moderate)',
                  'k=0.008 (slow)']
k_colors      = ['darkblue', 'orange', 'red']
t_half_colors = ['darkblue', 'orange', 'red', 'purple']

t_eval = np.linspace(0, T_STEPS, T_STEPS)


def siidr_extended(t, y, beta_0, beta_max, mu, g1, g2, N, k, t_half, delta):
    S, I, ID, R = y
    beta_t = beta_0 + (beta_max - beta_0) / (1 + np.exp(-k * (t - t_half)))
    dS  = -beta_t * S * I / N  +  delta * R
    dI  =  beta_t * S * I / N  -  (g1 + mu) * I  +  g2 * ID
    dID =  g1 * I  -  g2 * ID
    dR  =  mu * I  -  delta * R
    return [dS, dI, dID, dR]


# -- Pre-compute Phase 2 baseline
print("Computing Phase 2 baseline...")
baseline_runs = []
for _ in range(N_RUNS):
    mu  = np.clip(np.random.normal(MU_MEAN, MU_STD), 0, 1)
    g1  = np.clip(np.random.normal(G1_MEAN, G1_STD), 0, 1)
    g2  = np.clip(np.random.normal(G2_MEAN, G2_STD), 0, 1)
    if BETA_0 / max(mu, 1e-10) <= 1.0:
        continue
    sol = solve_ivp(siidr_extended, (0, T_STEPS), [N-1, 1, 0, 0],
                    t_eval=t_eval,
                    args=(BETA_0, BETA_0, mu, g1, g2, N, 0.02, T_STEPS*2, 0),
                    method='RK45')
    S_v, I_v, ID_v, R_v = sol.y
    if np.max(np.abs(S_v + I_v + ID_v + R_v - N)) / N > 1e-9:
        continue
    baseline_runs.append(I_v + ID_v)

baseline_arr    = np.array(baseline_runs)
baseline_med_peak = np.median(np.max(baseline_arr, axis=1))
print(f"  Baseline outbreak runs: {len(baseline_runs)}/{N_RUNS}")
print(f"  Baseline median peak: {baseline_med_peak:.0f} "
      f"({100*baseline_med_peak/N:.2f}% N)\n")


# -- Storage for violin plots
# peaks_by_thalf[row] = list of peak values across all k for that t_half
# peaks_by_k[col]     = list of peak values across all t_half for that k
peaks_by_thalf = {row: [] for row in range(len(t_half_values))}
peaks_by_k     = {col: [] for col in range(len(k_values))}


# -- Main loop over the 4x3 grid of (t_half, k) cells
for row, t_half in enumerate(t_half_values):
    for col, (k, k_label) in enumerate(zip(k_values, k_labels)):

        outbreak_runs = []
        peaks         = []
        t_peaks       = []

        for _ in range(N_RUNS):
            mu  = np.clip(np.random.normal(MU_MEAN, MU_STD), 0, 1)
            g1  = np.clip(np.random.normal(G1_MEAN, G1_STD), 0, 1)
            g2  = np.clip(np.random.normal(G2_MEAN, G2_STD), 0, 1)
            if BETA_0 / max(mu, 1e-10) <= 1.0:
                continue
            sol = solve_ivp(siidr_extended, (0, T_STEPS), [N-1, 1, 0, 0],
                            t_eval=t_eval,
                            args=(BETA_0, BETA_MAX, mu, g1, g2, N,
                                  k, t_half, DELTA),
                            method='RK45')
            S_v, I_v, ID_v, R_v = sol.y
            if np.max(np.abs(S_v + I_v + ID_v + R_v - N)) / N > 1e-9:
                continue
            instant = I_v + ID_v
            outbreak_runs.append(instant)
            peaks.append(max(instant))
            t_peaks.append(t_eval[np.argmax(instant)])

        if not outbreak_runs:
            continue

        # Store peaks for violin plots
        peaks_by_thalf[row].extend(peaks)
        peaks_by_k[col].extend(peaks)

        peak_abs_diff = np.median(peaks) - baseline_med_peak

        print(f"t_half={t_half:4d}, k={k:.3f}: "
              f"n_outbreak={len(outbreak_runs)}, "
              f"median peak={np.median(peaks):.0f} "
              f"({100*np.median(peaks)/N:.2f}% N), "
              f"vs baseline +{peak_abs_diff:.0f} nodes, "
              f"median t_peak={np.median(t_peaks):.0f}, "
              f"IQR={np.percentile(peaks, 25):.0f}–{np.percentile(peaks, 75):.0f}")


# -- Run count summaries for violin tick labels
thalf_counts = [len(peaks_by_thalf[row]) for row in range(len(t_half_values))]
k_counts     = [len(peaks_by_k[col])     for col in range(len(k_values))]


# -- Violin helper
def draw_violin_panel(ax, pos, peaks_list, color):
    """Draws violin + boxplot + individual runs for one group."""
    peaks_arr = np.array(peaks_list)
    if len(peaks_arr) == 0:
        return peaks_arr

    parts = ax.violinplot([peaks_arr], positions=[pos],
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

    ax.boxplot([peaks_arr], positions=[pos], widths=0.15,
               patch_artist=True,
               boxprops=dict(facecolor='white', color='black', linewidth=1.5),
               medianprops=dict(color='orange', linewidth=2),
               whiskerprops=dict(color='black', linewidth=1.2),
               capprops=dict(color='black', linewidth=1.5),
               flierprops=dict(marker='', markersize=0))

    np.random.seed(42)
    jitter   = np.random.uniform(-0.01, 0.01, size=len(peaks_arr))
    ax.scatter(pos + jitter, peaks_arr, color=color, edgecolors='white',
               linewidths=0.5, s=20, alpha=0.8, zorder=3)

    return peaks_arr


def print_violin_summary(label, peaks_arr):
    q1  = np.percentile(peaks_arr, 25)
    q3  = np.percentile(peaks_arr, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    print(f"\n—— {label} ——————————————————————————————")
    print(f"  n={len(peaks_arr)}  |  Min={peaks_arr.min():.0f}  |  "
          f"Q1={q1:.0f}  |  Median={np.median(peaks_arr):.0f}  |  "
          f"Q3={q3:.0f}  |  Max={peaks_arr.max():.0f}")
    print(f"  IQR={iqr:.0f}  |  Lower fence={max(lower_fence,0):.0f}  |  "
          f"Upper fence={upper_fence:.0f}")
    print(f"  Below Q1: {int(np.sum(peaks_arr < q1))}  |  "
          f"IQR box: {int(np.sum((peaks_arr >= q1) & (peaks_arr <= q3)))}  |  "
          f"Above Q3: {int(np.sum((peaks_arr > q3) & (peaks_arr <= upper_fence)))}  |  "
          f"Outliers: {int(np.sum(peaks_arr > upper_fence))}")


# -- Violin Plot A: Peak I+I_D by t_half (aggregated across k values)
print("\n\n=== VIOLIN A: Peak I+I_D by t_half (all k values pooled) ===")
fig_va, ax_va = plt.subplots(figsize=(12, 7))

for pos, (row, t_half, label, color) in enumerate(
        zip(range(len(t_half_values)), t_half_values,
            t_half_labels, t_half_colors), start=1):
    peaks_arr = draw_violin_panel(ax_va, pos, peaks_by_thalf[row], color)
    if len(peaks_arr) > 0:
        print_violin_summary(label, peaks_arr)

ax_va.set_xticks(range(1, len(t_half_values) + 1))
ax_va.set_xticklabels([f't_half={t} - {n} runs'
                       for t, n in zip(t_half_values, thalf_counts)])
ax_va.set_xlabel('Efficiency gain midpoint t_half', fontsize=14)
ax_va.set_ylabel('Infected machines (I + I_D)', fontsize=14)
ax_va.set_title('Highest infection count reached in each run - Sensitivity to t_half',
                fontsize=14)
fig_va.tight_layout()
plt.savefig('Figures_Presentation/Results/Phase_3_thalf_Sensitivity_Violin.svg',
            bbox_inches='tight')
plt.show()


# -- Violin Plot B: Peak I+I_D by k (aggregated across t_half values)
print("\n\n=== VIOLIN B: Peak I+I_D by k (all t_half values pooled) ===")
fig_vb, ax_vb = plt.subplots(figsize=(12, 7))

k_tick_labels = ['k=0.08\n(rapid)', 'k=0.02\n(moderate)', 'k=0.008\n(slow)']

for pos, (col, k, label, color) in enumerate(
        zip(range(len(k_values)), k_values,
            k_tick_labels, k_colors), start=1):
    peaks_arr = draw_violin_panel(ax_vb, pos, peaks_by_k[col], color)
    if len(peaks_arr) > 0:
        print_violin_summary(f'k={k}', peaks_arr)

ax_vb.set_xticks(range(1, len(k_values) + 1))
ax_vb.set_xticklabels([f'k={k:.3f} - {n} runs'
                       for k, n in zip(k_values, k_counts)])
ax_vb.set_xlabel('Steepness parameter k', fontsize=14)
ax_vb.set_ylabel('Infected machines (I + I_D)', fontsize=14)
ax_vb.set_title('Highest infection count reached in each run - Sensitivity to k',
                fontsize=14)
fig_vb.tight_layout()
plt.savefig('Figures_Presentation/Results/Phase_3_k_Sensitivity_Violin.svg',
            bbox_inches='tight')
plt.show()