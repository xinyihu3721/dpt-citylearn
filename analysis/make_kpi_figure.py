"""
Report-quality KPI comparison figure: DPT-CITYLEARN (r3) vs CHESCA vs RBC, 8 official CityLearn KPIs +
average_score. Values are read directly from results/gate6b_group1_results.json's Phase 3 (8-KPI
breakdown) entries, normalized by CityLearn's no-active-control baseline (lower = better), mean +/-
nanstd across the 3 eval seeds (55555, 1020, 1025).
Pure plotting from measured results -- no CityLearn/CHESCA computation, no GPU needed.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'figures')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'gate6b_group1_results.json')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',
})

KPI_LABELS = [
    'Carbon\nemissions', 'Discomfort', 'Ramping', 'Daily\n1-load factor',
    'Daily\npeak', 'Annual\npeak', 'M\n(1-thermal\nresilience)', 'S\n(unserved\nenergy)',
    'Average\nscore',
]

# Maps each plotted KPI to its key in the results JSON. 'M' and 'S' are the DPT-HVAC paper's short
# names for the two outage-conditional KPIs.
KEYS = ['carbon_emissions_total', 'discomfort_proportion', 'ramping_average',
        'daily_one_minus_load_factor_average', 'daily_peak_average', 'annual_peak_average',
        'one_minus_thermal_resilience_proportion', 'power_outage_normalized_unserved_energy_total',
        'average_score']
EVAL_SEEDS = [55555, 1020, 1025]
# The chosen model (r3) is evaluated at its own best-known h (see deploy/evaluate_and_report.py's
# BEST_KNOWN_H); Phase 3 of that script recorded it under the 'phase3_model_h24' key prefix.
POLICY_PREFIXES = {'model': 'phase3_model_h24', 'chesca': 'phase3_chesca', 'rbc': 'phase3_rbc'}

with open(RESULTS_PATH) as f:
    _results = json.load(f)


def _mean_std(policy, kpi_key):
    vals = [_results[f'{POLICY_PREFIXES[policy]}__seed{s}'][kpi_key] for s in EVAL_SEEDS]
    return float(np.nanmean(vals)), float(np.nanstd(vals))


DATA = {k: (*_mean_std('model', k), *_mean_std('chesca', k), *_mean_std('rbc', k)) for k in KEYS}

COLOR_MODEL, COLOR_CHESCA, COLOR_RBC = '#36A3EB', '#F1EA65', '#F35F5F'

model_means = np.array([DATA[k][0] for k in KEYS])
model_stds = np.array([DATA[k][1] for k in KEYS])
chesca_means = np.array([DATA[k][2] for k in KEYS])
chesca_stds = np.array([DATA[k][3] for k in KEYS])
rbc_means = np.array([DATA[k][4] for k in KEYS])
rbc_stds = np.array([DATA[k][5] for k in KEYS])

x = np.arange(len(KEYS))
width = 0.26

fig, ax = plt.subplots(figsize=(15, 7.5))

bars_model = ax.bar(x - width, model_means, width, yerr=model_stds, capsize=3,
                     color=COLOR_MODEL, label='DPT-CITYLEARN', zorder=3,
                     error_kw=dict(elinewidth=1.2, ecolor='black'))
bars_chesca = ax.bar(x, chesca_means, width, yerr=chesca_stds, capsize=3,
                      color=COLOR_CHESCA, label='CHESCA', zorder=3,
                      error_kw=dict(elinewidth=1.2, ecolor='black'))
bars_rbc = ax.bar(x + width, rbc_means, width, yerr=rbc_stds, capsize=3,
                   color=COLOR_RBC, label='RBC', zorder=3,
                   error_kw=dict(elinewidth=1.2, ecolor='black'))

ax.axhline(1.0, color='#444444', linestyle='--', linewidth=2, zorder=2)
ax.annotate('no-active-control baseline (=1.0)', xy=(2.5, 1.0), xytext=(0, 6),
            textcoords='offset points', fontsize=18, ha='center', color='#444444')

ax.set_xticks(x)
ax.set_xticklabels(KPI_LABELS, fontsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylabel('KPI value (norm. by no control baseline)', fontsize=16)
# ax.set_title('8-KPI breakdown + average score: DPT-CITYLEARN vs CHESCA vs RBC\n'
#              '(held-out task: Family C, capacity=1.075x; mean +/- std across 3 eval seeds)')
ax.legend(loc='upper right', frameon=True, fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', color='#DDDDDD', alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, 2.55)

# caption = (
#     "Values are normalized by CityLearn's official baseline: each building's own consumption WITHOUT any\n"
#     "active control at all (env.evaluate_citylearn_challenge()'s fixed baseline_condition) -- NOT a\n"
#     "re-run of the RBC bar shown here. 1.0 = same as doing nothing; below 1.0 = better than doing nothing.\n"
#     "Lower is better. The RBC bar is BasicRBC's own measured performance against that same no-active-\n"
#     "control baseline, so RBC CAN exceed 1.0 (e.g. carbon here) if its fixed rules do worse than no active\n"
#     "control at all on this capacity-scaled held-out task.\n"
#     "Error bars = std across 3 eval seeds (55555, 1020, 1025); M and S have LARGE error bars (+/-0.294, +/-0.026)\n"
#     "shown honestly, not hidden.\n"
#     "FOOTNOTE: M, S computed from 2 outage seeds (one seed had zero outages) -- resilience comparison limited."
# )
fig.subplots_adjust(top=0.86, bottom=0.30, left=0.07, right=0.98)
# fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9, family='monospace',
#           transform=fig.transFigure, wrap=True)

for name in ['kpi_comparison']:
    png_path = os.path.join(OUT_DIR, f'{name}.png')
    svg_path = os.path.join(OUT_DIR, f'{name}.svg')
    fig.savefig(png_path, dpi=400, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
plt.close(fig)
