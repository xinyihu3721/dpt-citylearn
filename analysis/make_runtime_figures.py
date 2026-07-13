"""
Report-quality figures from the already-measured runtime-per-prediction data
(results_runtime_dpt.json, results_runtime_chesca_rbc.json). Pure plotting from saved JSON --
no CityLearn/CHESCA computation, no GPU needed, safe to run anywhere (including the login node).

Figure A: per-prediction latency comparison (DPT vs CHESCA vs RBC), log-scale y-axis, box+strip
  showing full per-step distributions (not just mean+-std) so CHESCA's periodic re-planning
  spikes are visible rather than hidden inside an error bar. Caption states the GPU/CPU hardware
  asymmetry explicitly.

Figure B: DPT latency vs context length (h=0/128/256 fixed microbenchmark) with the real-episode
  deployed mean shown as a separate reference line/marker, explicitly labeled as such (not
  conflated with the pure-forward-pass microbenchmark numbers).
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

DPT_PATH = os.path.join(PROJECT_ROOT, 'results', 'results_runtime_dpt.json')
CHESCA_RBC_PATH = os.path.join(PROJECT_ROOT, 'results', 'results_runtime_chesca_rbc.json')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',
})


def save_both(fig, name):
    png_path = os.path.join(OUT_DIR, f'{name}.png')
    svg_path = os.path.join(OUT_DIR, f'{name}.svg')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def figure_a(dpt, chesca_rbc):
    dpt_times = np.array(dpt['real_episode_times_ms_no_warmup'])
    chesca_times = np.array(chesca_rbc['chesca']['per_step_times_ms'][chesca_rbc['chesca']['warmup_steps_discarded']:])
    rbc_times = np.array(chesca_rbc['rbc']['per_step_times_ms'][chesca_rbc['rbc']['warmup_steps_discarded']:])

    data = [dpt_times, chesca_times, rbc_times]
    labels = ['DPT-CITYLEARN\n1x NVIDIA L40S GPU', 'CHESCA\n4 CPU cores\n(Xeon Gold 6242)',
              'RBC\n4 CPU cores\n(Xeon Gold 6242)']
    # Okabe-Ito colorblind-safe palette: DPT = vivid blue (hero), CHESCA = muted amber
    # (secondary, alpha-softened), RBC = neutral grey (trivial baseline).
    colors = ['#0072B2', '#E69F00', '#999999']
    positions = [1, 2, 3]
    means = [t.mean() for t in data]
    medians = [np.median(t) for t in data]

    BREAK = 28  # ms -- boundary between the two panels
    TOP_MAX = 235

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 7.5),
        gridspec_kw={'height_ratios': [1, 2.3], 'hspace': 0.08})

    for ax in (ax_top, ax_bot):
        bp = ax.boxplot(data, positions=positions, widths=0.35, showfliers=True, patch_artist=True,
                         whis=(5, 95), zorder=3,
                         flierprops=dict(marker='o', markersize=4, alpha=0.5, markeredgewidth=0))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        for flier, c in zip(bp['fliers'], colors):
            flier.set_markerfacecolor(c)
            flier.set_markeredgecolor(c)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
        for whisker in bp['whiskers']:
            whisker.set_color('#555555')
        for cap in bp['caps']:
            cap.set_color('#555555')

        rng = np.random.default_rng(0)
        scatter_alphas = {1: 0.18, 2: 0.55, 3: 0.18}  # CHESCA brighter to show its spike density
        for pos, times, c in zip(positions, data, colors):
            n_show = min(len(times), 500)
            idx = rng.choice(len(times), size=n_show, replace=False)
            jitter = rng.normal(0, 0.05, size=n_show)
            ax.scatter(pos + jitter, times[idx], s=7, alpha=scatter_alphas[pos], color=c,
                       zorder=1, linewidths=0)

        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', color='#DDDDDD', alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=13)

    # mean diamond markers: all values (<=19.85 ms) fall inside the bottom (0-28ms) panel, so draw
    # them ONLY on ax_bot -- drawing on ax_top too let the CHESCA marker's radius (data value 19.85,
    # just 8ms below the BREAK) poke across the clip boundary as a stray sliver.
    for pos, m in zip(positions, means):
        ax_bot.scatter([pos], [m], marker='D', s=50, color=colors[pos - 1],
                       edgecolor='black', linewidth=0.8, zorder=5)

    ax_top.set_ylim(BREAK, TOP_MAX)
    ax_bot.set_ylim(-1.5, BREAK)
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bot.tick_params(top=False)

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d * 2.5, +d * 2.5), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d * 2.5, +d * 2.5), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d * 1.1, 1 + d * 1.1), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d * 1.1, 1 + d * 1.1), **kwargs)

    # mean/median annotations -- all values (<=19.85 ms) fall inside the bottom (0-28ms) panel,
    # so no annotation ever needs to cross the axis break. RBC's median (0.010 ms) sits right at
    # y=0, near the bottom spine, so its label gets a smaller downward offset than the others so
    # it doesn't collide with the axis line.
    median_dy = {1: -14, 2: -14, 3: -8}
    for pos, m, med, c in zip(positions, means, medians, colors):
        ax_bot.annotate(f"mean={m:.3f} ms", (pos, m), textcoords="offset points", xytext=(16, 7),
                        fontsize=12, ha='left', color='black')
        ax_bot.annotate(f"median={med:.3f} ms", (pos, med), textcoords="offset points",
                        xytext=(16, median_dy[pos]), fontsize=12, ha='left', color='dimgray')

    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels(labels, fontsize=13)
    ax_bot.set_xlim(0.5, 3.7)
    fig.text(0.02, 0.5, 'Wall-clock time per prediction (ms)', va='center', ha='center',
              rotation='vertical', fontsize=15)

    legend_handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markeredgecolor='black',
                   markersize=9, label='mean'),
        plt.Line2D([0], [0], color='black', linewidth=1.8, label='median'),
    ]
    ax_top.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=11)

    ax_top.set_title('Per-prediction latency: DPT vs CHESCA vs RBC\n'
                     '(same held-out task: Family C, capacity=1.075x, seed=55555, 2207-step episode)',
                     pad=14)

    caption = (
        "Broken y-axis (// marks): lower panel 0-28 ms resolves DPT/RBC/CHESCA's typical range; upper panel\n"
        "28-235 ms shows CHESCA's periodic receding-horizon re-planning spikes (~4% of steps). Boxes = 5th-95th\n"
        "percentile + median (black line); dots = jittered per-step times; diamonds = mean.\n"
        "HARDWARE ASYMMETRY (not hidden): DPT ran on 1x NVIDIA L40S GPU; CHESCA and RBC ran on 4 CPU cores\n"
        "(Intel Xeon Gold 6242 @2.80GHz). This is NOT a hardware-normalized speed comparison -- do not read\n"
        "the gap as \"DPT is Nx faster\"; it reflects each controller's native deployment hardware."
    )
    fig.subplots_adjust(top=0.87, bottom=0.30, left=0.15, right=0.97)
    fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=8.5, family='monospace',
              transform=fig.transFigure, wrap=True)
    save_both(fig, 'runtime_latency_comparison')
    plt.close(fig)


def figure_b(dpt):
    fixed_h = dpt['fixed_h']
    hs = sorted(int(k) for k in fixed_h.keys())
    means = [fixed_h[str(h)]['mean_ms'] for h in hs]
    stds = [fixed_h[str(h)]['std_ms'] for h in hs]

    real_mean = dpt['real_episode']['mean_ms']
    real_std = dpt['real_episode']['std_ms']

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(hs, means, yerr=stds, marker='o', markersize=8, capsize=5, linewidth=2,
                color='#4C72B0', label='Fixed-h microbenchmark\n(pure forward pass, isolated repeats)')

    ax.axhline(real_mean, color='#C44E52', linestyle='--', linewidth=1.8,
               label=f'Real-episode deployed mean = {real_mean:.3f} ms\n(headline number; includes per-step\ncontext-assembly overhead)')
    ax.fill_between([min(hs) - 10, max(hs) + 10], real_mean - real_std, real_mean + real_std,
                     color='#C44E52', alpha=0.12)

    for h, m in zip(hs, means):
        ax.annotate(f"{m:.3f} ms", (h, m), textcoords="offset points", xytext=(0, -20),
                    fontsize=9, ha='center')

    ax.set_xlim(min(hs) - 15, max(hs) + 15)
    ax.set_ylim(1.0, real_mean + real_std + 0.09)
    ax.set_xticks(hs)
    ax.set_xlabel('Context length h (number of in-context transitions)')
    ax.set_ylabel('Wall-clock time per prediction (ms)')
    ax.set_title('DPT-CITYLEARN latency vs context length\n1x NVIDIA L40S GPU, Family C cap=1.075x held-out task')
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)

    caption = (
        "DPT per-prediction cost is low and near-flat: only ~8% growth from h=0 (1.061 ms) to h=256\n"
        "(1.149 ms), the trained H_max. The fixed-h points are a pure-forward-pass LOWER BOUND (isolated\n"
        "repeated calls on a frozen context snapshot); the real-episode mean sits slightly above them\n"
        "because it also includes per-step context-assembly overhead (fresh tensor construction each step).\n"
        "The real-episode mean, not the microbenchmark, is the number to quote for deployed cost."
    )
    fig.subplots_adjust(bottom=0.32)
    fig.text(0.02, 0.02, caption, ha='left', va='bottom', fontsize=8.5, family='monospace',
              transform=fig.transFigure, wrap=True)
    save_both(fig, 'runtime_dpt_context_length')
    plt.close(fig)


def main():
    with open(DPT_PATH) as f:
        dpt = json.load(f)
    with open(CHESCA_RBC_PATH) as f:
        chesca_rbc = json.load(f)

    warmup = dpt['real_episode']['warmup_steps_discarded']
    dpt['real_episode_times_ms_no_warmup'] = dpt['per_step_times_ms'][warmup:]

    figure_a(dpt, chesca_rbc)
    figure_b(dpt)


if __name__ == '__main__':
    main()
