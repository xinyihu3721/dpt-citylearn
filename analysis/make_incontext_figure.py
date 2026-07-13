"""
In-context learning curve (hero result): average_score vs context length h for the operating-point
model (r3), evaluated ZERO-SHOT on the fully held-out anchor family C (buildings 4/5/6, never seen
in training) across 4 unseen capacity multipliers (0.775, 0.925, 1.075, 1.225 -- 1.075 is the
original held-out point, the other 3 probe wider generalization). No gradient updates at deployment;
context is accumulated purely by conditioning. Values from gate6b_group1_results.json's Phase 2
sweep (2 eval seeds: 55555, 1020).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'gate6b_group1_results.json')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',
})

CAPS = [0.775, 0.925, 1.075, 1.225]
HS = [0, 24, 72]
PRIMARY_CAP = 1.075
COLORS = {0.775: '#9ECAE1', 0.925: '#4292C6', 1.075: '#08519C', 1.225: '#9ECAE1'}
MARKERS = {0.775: 's', 0.925: '^', 1.075: 'o', 1.225: 'D'}

with open(RESULTS_PATH) as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(8, 7.3))

for cap in CAPS:
    means, stds = [], []
    for h in HS:
        key = f'phase2_r3__cap{cap:.3f}__h{h}'
        scores = results[key]['scores']
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    means, stds = np.array(means), np.array(stds)
    is_primary = (cap == PRIMARY_CAP)
    ax.errorbar(HS, means, yerr=stds, marker=MARKERS[cap], markersize=9 if is_primary else 7,
                linewidth=2.6 if is_primary else 1.4, capsize=4,
                color='#08519C' if is_primary else '#9ECAE1',
                alpha=1.0 if is_primary else 0.75, zorder=5 if is_primary else 3,
                label=f'capacity={cap:.3f}x' + (' (original held-out point)' if is_primary else ' (unseen)'))

ax.axhline(1.0, color='#444444', linestyle='--', linewidth=1.2, zorder=1)
ax.annotate('No control normalized baseline (=1.0)', xy=(36, 1.0), xytext=(0, 6),
            textcoords='offset points', fontsize=9.5, ha='center', color='#444444')

ax.set_xticks(HS)
ax.set_ylim(0.45, 1.12)
ax.set_xlabel('Context length h (number of in-context transitions)')
ax.set_ylabel('average_score (lower = better)')
ax.set_title('In-context learning on the FULLY held-out Family C\n'
             'zero-shot: unseen buildings, unseen capacities, NO gradient updates at deployment',
             pad=18)
ax.legend(loc='center right', fontsize=9.5, frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', color='#DDDDDD', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

caption = (
    "Each point = mean +/- std across 2 eval seeds (55555, 1020) on Family C (buildings 4/5/6, never used\n"
    "in training). All 4 capacity multipliers are unseen; 1.075x is the original held-out point (bold). Score\n"
    "drops sharply from h=0 (no context, prior-only) to h=24, then holds -- pure in-context improvement,\n"
    "no parameter updates -- and the same qualitative pattern holds across all 4 unseen capacities."
)
fig.subplots_adjust(top=0.78, bottom=0.27, left=0.12, right=0.97)
fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=9, family='monospace',
          transform=fig.transFigure, wrap=True)

png_path = os.path.join(OUT_DIR, 'incontext_curve.png')
svg_path = os.path.join(OUT_DIR, 'incontext_curve.svg')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(svg_path, bbox_inches='tight')
print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
plt.close(fig)
