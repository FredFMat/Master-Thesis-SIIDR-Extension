"""
Phase 2 — Scenario Group 4: Sensitivity to delta

Runs the extended SIIDR ODE system per delta value, drawing
the inherited parameters (mu, gamma_1, gamma_2) from normal distributions
using the posterior mean and standard deviation from Table 5 of the
companion paper (Chernikova et al., 2023).

delta is varied across three values:
  delta in {0.001, 0.01, 0.05}

All other extension parameters are fixed at their thesis baseline values:
  beta_max = 2.0 * beta_0
  k        = 0.02 per step  (200-step transition width)
  t_half   = 5000 steps     (midpoint of simulation window)

Sources:
  Table 5 (means and stds): Chernikova et al. (2023), Applied Network Science
  N = 317,031:              Chernikova et al. (2022), ESORICS Table 4, port 445
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd

np.random.seed(42)

BETA_0   = 0.16
MU_MEAN  = 0.11;  MU_STD  = 0.10
G1_MEAN  = 0.80;  G1_STD  = 0.15
G2_MEAN  = 0.06;  G2_STD  = 0.06
N        = 317031
BETA_MAX = 2.0 * BETA_0
K        = 0.02
T_HALF   = 5000
T_STEPS  = 10000
N_RUNS   = 500

delta_values = {
    'δ=0.001': 0.001,
    'δ=0.01':  0.01,
    'δ=0.05':  0.05,
}

colors = ['darkblue', 'orange', 'red']

t_eval = np.linspace(0, T_STEPS, T_STEPS)


def siidr_extended(t, y, beta_0, beta_max, mu, gamma_1, gamma_2, N,
                   k, t_half, delta):
    S, I, ID, R = y
    beta_t = beta_0 + (beta_max - beta_0) / (1 + np.exp(-k * (t - t_half)))
    dS_dt  = -beta_t * S * I / N  +  delta * R
    dI_dt  =  beta_t * S * I / N  -  (gamma_1 + mu) * I  +  gamma_2 * ID
    dID_dt =  gamma_1 * I         -  gamma_2 * ID
    dR_dt  =  mu * I              -  delta * R
    return [dS_dt, dI_dt, dID_dt, dR_dt]


# -- Pre-compute Phase 2 baseline (beta constant = beta_0, delta = 0)
print("Computing Phase 2 baseline...")
baseline_runs = []
for _ in range(N_RUNS):
    mu      = np.clip(np.random.normal(MU_MEAN, MU_STD), 0, 1)
    g1      = np.clip(np.random.normal(G1_MEAN, G1_STD), 0, 1)
    g2      = np.clip(np.random.normal(G2_MEAN, G2_STD), 0, 1)
    if BETA_0 / max(mu, 1e-10) <= 1.0:
        continue
    sol = solve_ivp(siidr_extended, (0, T_STEPS), [N-1, 1, 0, 0],
                    t_eval=t_eval,
                    args=(BETA_0, BETA_0, mu, g1, g2, N,
                          0.02, T_STEPS*2, 0),
                    method='RK45')
    S_v, I_v, ID_v, R_v = sol.y
    if np.max(np.abs(S_v + I_v + ID_v + R_v - N)) / N > 1e-9:
        continue
    baseline_runs.append(I_v + ID_v)

baseline_arr      = np.array(baseline_runs)
baseline_med_peak = np.median(np.max(baseline_arr, axis=1))
print(f"  Baseline outbreak runs: {len(baseline_runs)}/{N_RUNS}")
print(f"  Baseline median peak:   {baseline_med_peak:.0f} "
      f"({100*baseline_med_peak/N:.2f}% N)\n")


# -- Storage for violin
all_peaks_instant = []
scenario_labels   = []
valid_run_counts  = []


for color, (label, delta) in zip(colors, delta_values.items()):

    valid_instant = []
    peaks_instant = []

    for run in range(N_RUNS):
        mu      = np.clip(np.random.normal(MU_MEAN, MU_STD), 0, 1)
        gamma_1 = np.clip(np.random.normal(G1_MEAN, G1_STD), 0, 1)
        gamma_2 = np.clip(np.random.normal(G2_MEAN, G2_STD), 0, 1)

        if BETA_0 / max(mu, 1e-10) <= 1.0:
            continue

        solution = solve_ivp(
            fun          = siidr_extended,
            t_span       = (0, T_STEPS),
            y0           = [N - 1, 1, 0, 0],
            t_eval       = t_eval,
            args         = (BETA_0, BETA_MAX, mu, gamma_1, gamma_2, N,
                            K, T_HALF, delta),
            method       = 'RK45',
            dense_output = False
        )

        S_v, I_v, ID_v, R_v = solution.y

        max_dev = np.max(np.abs(S_v + I_v + ID_v + R_v - N)) / N
        if max_dev > 1e-9:
            continue

        instant = I_v + ID_v
        valid_instant.append(instant)
        peaks_instant.append(max(instant))

    if len(valid_instant) == 0:
        print(f"WARNING: {label} — no valid runs.")
        continue

    peak_abs_diff = np.median(peaks_instant) - baseline_med_peak

    all_peaks_instant.append(peaks_instant)
    scenario_labels.append(label)
    valid_run_counts.append(len(valid_instant))

    print(f"δ = {label}  (delta = {delta:.3f})")
    print(f"  Valid runs:        {len(valid_instant)}/{N_RUNS}")
    print(f"  Median peak I+I_D: {np.median(peaks_instant):.0f} "
          f"({100*np.median(peaks_instant)/N:.4f}% of N)")
    print(f"  vs baseline:       +{peak_abs_diff:.0f} nodes")
    print()


# -- PLOT 2: Violin + boxplot + individual runs
fig2, ax2 = plt.subplots(figsize=(12, 7))
np.random.seed(99)

for pos, (peaks_list, color, label) in enumerate(
        zip(all_peaks_instant, colors, scenario_labels), start=1):

    peaks_arr = np.array(peaks_list)

    parts = ax2.violinplot([peaks_arr], positions=[pos],
                           showmeans=False, showmedians=False,
                           showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

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

ax2.set_xticks(range(1, len(scenario_labels) + 1))
ax2.set_xticklabels([f'{label} - {count} outbreak runs'
                     for label, count in zip(scenario_labels, valid_run_counts)])
ax2.set_xlabel('Waning immunity rate δ', fontsize=14)
ax2.set_ylabel('Infected machines (I + I_D)', fontsize=14)
ax2.set_title('Highest infection count reached in each run - Sensitivity to δ',
              fontsize=14)
fig2.tight_layout()
plt.savefig('Figures_Presentation/Results/Phase_4_Delta_Sensitivity_Violin.svg',
            bbox_inches='tight')
plt.show()

"""
Statistical tests: Kruskal-Wallis + Dunn's post-hoc

Kruskal-Wallis because the distributions are heavily skewed and bimodal.
H statistic and p-value test whether at least one group differs.
"""

print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

# -- Kruskal-Wallis
kw_stat, kw_p = kruskal(*[np.array(p) for p in all_peaks_instant])
print(f"\nKruskal-Wallis test across all {len(scenario_labels)} groups:")
print(f"  H = {kw_stat:.4f}  |  p = {kw_p:.6f}")
if kw_p < 0.05:
    print("  Result: At least one group differs significantly (p < 0.05)")
    print("  Proceeding to Dunn's pairwise post-hoc test...")
else:
    print("  Result: No significant difference detected (p >= 0.05)")
    print("  Dunn's test not warranted.")

# -- Dunn's test
if kw_p < 0.05:
    data = pd.DataFrame({
        'peak':  np.concatenate([np.array(p) for p in all_peaks_instant]),
        'group': np.concatenate([[label]*len(p)
                                 for label, p in zip(scenario_labels,
                                                     all_peaks_instant)])
    })

    dunn = sp.posthoc_dunn(data, val_col='peak',
                           group_col='group',
                           p_adjust='bonferroni')

    print(f"\nDunn's test (Bonferroni-corrected p-values):")
    print(f"  Significance threshold: 0.05")
    print()

    labels_list = list(dunn.columns)
    for i in range(len(labels_list)):
        for j in range(i+1, len(labels_list)):
            a, b = labels_list[i], labels_list[j]
            p_val = dunn.loc[a, b]
            sig = "significant" if p_val < 0.05 else "not significant"
            print(f"  {a} vs {b}:")
            print(f"    p = {p_val:.6f}  —  {sig}")