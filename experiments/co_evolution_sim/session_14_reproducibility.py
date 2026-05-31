"""Session 14: 再現性確認

Question: Session 13（seed=42）の3つの主要発見は
         複数シードで再現されるか？

Seeds: 42-51 (10 seeds)
Criteria (reproducible if ≥8/10 seeds pass):
  1. new arch path length > old arch path length × 3
  2. new arch survivor ablation < new arch non-survivor ablation
  3. old arch clustering coefficient > new arch clustering coefficient
"""
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from session_13_anatomy import (
    collect_all,
    run_topology_anatomy,
    run_activity_patterns,
    run_information_flow,
    _GROUP_KEYS,
    _GROUP_LABELS,
    _GROUP_COLORS,
)

_SEEDS     = list(range(42, 52))
_CSV_PATH  = 'results_s14_raw.csv'
_REPRO_PNG = 'images/session_14/results_s14_reproducibility.png'
_SUMM_PNG  = 'images/session_14/results_s14_summary.png'


# ─── Per-seed metric extraction ───────────────────────────────────────────────

def _s14_extract_metrics(topo_data, act_data, flow_data):
    """Extract key scalar metrics from one seed's analysis results.

    Returns dict[group_key → dict of scalar metrics].
    """
    metrics = {}
    for gk in _GROUP_KEYS:
        t = topo_data[gk]
        f = flow_data[gk]
        metrics[gk] = {
            'n_edges':    float(np.mean([m['n_edges']    for m in t])),
            'clustering': float(np.mean([m['clustering'] for m in t])),
            'n_cycles':   float(np.mean([m['n_cycles']   for m in t])),
            'path_count': float(np.mean([m['path_count'] for m in t])),
            'cos_int':    act_data[gk]['cos_int'],
            'cos_out':    act_data[gk]['cos_out'],
            # mean across 4 input nodes, then across agents
            'depth_out':  float(np.mean([np.mean(m['depth_out']) for m in f])),
            'ablation':   float(np.mean([np.mean(m['ablation'])  for m in f])),
        }
    return metrics


def _s14_criterion_values(metrics):
    """Compute the 3 criterion scalars from extracted metrics.

    Returns (c1_ratio, c2_diff, c3_diff):
      c1_ratio = new_depth / old_depth   (pass if > 3.0)
      c2_diff  = new_surv_abl − new_non_abl   (pass if < 0)
      c3_diff  = old_clust − new_clust   (pass if > 0)
    """
    new_depth = (metrics['new_surv']['depth_out'] + metrics['new_non']['depth_out']) / 2
    old_depth = (metrics['old_surv']['depth_out'] + metrics['old_non']['depth_out']) / 2
    c1_ratio  = new_depth / old_depth if old_depth > 1e-12 else float('inf')

    c2_diff = metrics['new_surv']['ablation'] - metrics['new_non']['ablation']

    new_clust = (metrics['new_surv']['clustering'] + metrics['new_non']['clustering']) / 2
    old_clust = (metrics['old_surv']['clustering'] + metrics['old_non']['clustering']) / 2
    c3_diff   = old_clust - new_clust

    return c1_ratio, c2_diff, c3_diff


# ─── Multi-seed sweep ─────────────────────────────────────────────────────────

def run_s14_sweep(seeds=_SEEDS, n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    """Run all Session-13 analyses for each seed.

    Returns (all_metrics, all_criteria):
      all_metrics[seed]  = dict[gk → scalar metrics]
      all_criteria[seed] = {'c1_ratio', 'c2_diff', 'c3_diff',
                             'pass_c1', 'pass_c2', 'pass_c3'}
    """
    all_metrics  = {}
    all_criteria = {}

    for seed in seeds:
        print(f'\n=== seed={seed} ===')
        collected = collect_all(seed=seed, n_gen=n_gen, n_agents=n_agents,
                                n_ep=n_ep, n_surv=n_surv)

        print('  [Topology]')
        topo_data = run_topology_anatomy(collected)
        print('  [Activity]')
        act_data  = run_activity_patterns(collected, seed=seed)
        print('  [Info flow]')
        flow_data = run_information_flow(collected)

        metrics       = _s14_extract_metrics(topo_data, act_data, flow_data)
        c1, c2, c3    = _s14_criterion_values(metrics)
        all_metrics[seed]  = metrics
        all_criteria[seed] = {
            'c1_ratio': c1, 'c2_diff': c2, 'c3_diff': c3,
            'pass_c1':  c1 > 3.0,
            'pass_c2':  c2 < 0.0,
            'pass_c3':  c3 > 0.0,
        }
        print(f'  Criteria: '
              f'c1={c1:.2f}({"pass" if c1 > 3 else "fail"}), '
              f'c2={c2:.4f}({"pass" if c2 < 0 else "fail"}), '
              f'c3={c3:.4f}({"pass" if c3 > 0 else "fail"})')

    return all_metrics, all_criteria


# ─── CSV export ───────────────────────────────────────────────────────────────

_METRIC_KEYS = ['n_edges', 'clustering', 'n_cycles', 'path_count',
                'cos_int', 'cos_out', 'depth_out', 'ablation']


def save_s14_csv(all_metrics, all_criteria, fpath=_CSV_PATH):
    """Save per-seed metrics to CSV."""
    header = ['seed']
    for gk in _GROUP_KEYS:
        for mk in _METRIC_KEYS:
            header.append(f'{gk}__{mk}')
    header += ['c1_ratio', 'c2_diff', 'c3_diff', 'pass_c1', 'pass_c2', 'pass_c3']

    with open(fpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for seed in sorted(all_metrics.keys()):
            row = [seed]
            for gk in _GROUP_KEYS:
                for mk in _METRIC_KEYS:
                    row.append(f'{all_metrics[seed][gk][mk]:.6g}')
            cr = all_criteria[seed]
            row += [f'{cr["c1_ratio"]:.6g}', f'{cr["c2_diff"]:.6g}',
                    f'{cr["c3_diff"]:.6g}',
                    int(cr['pass_c1']), int(cr['pass_c2']), int(cr['pass_c3'])]
            writer.writerow(row)
    print(f'Saved {fpath}')


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_s14_reproducibility(all_criteria, seeds=_SEEDS, fname=_REPRO_PNG):
    """Box plots for the 3 criteria.

    Each panel: scatter of 10 seed values + box, seed=42 as red dot,
    threshold as dashed line, verdict annotation.
    """
    c1_vals = [all_criteria[s]['c1_ratio'] for s in seeds]
    c2_vals = [all_criteria[s]['c2_diff']  for s in seeds]
    c3_vals = [all_criteria[s]['c3_diff']  for s in seeds]

    n_pass = [
        sum(all_criteria[s]['pass_c1'] for s in seeds),
        sum(all_criteria[s]['pass_c2'] for s in seeds),
        sum(all_criteria[s]['pass_c3'] for s in seeds),
    ]

    specs = [
        (c1_vals, 3.0,
         'Criterion 1: New/Old Path Length Ratio\n(pass if > 3.0)',
         'Path Length Ratio (New / Old)'),
        (c2_vals, 0.0,
         'Criterion 2: New-Surv − New-Non Ablation\n(pass if < 0)',
         'Ablation Δ (Surv − Non)'),
        (c3_vals, 0.0,
         'Criterion 3: Old − New Clustering Coefficient\n(pass if > 0)',
         'Clustering Δ (Old − New)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Session 14: Reproducibility Check  (seeds 42–51, n=10)\n'
        'Criterion satisfied if ≥8/10 seeds pass  |  seed=42 shown as red dot',
        fontsize=10,
    )

    for ax, (vals, thr, title, ylabel), n in zip(axes, specs, n_pass):
        rng_j = np.random.default_rng(0)
        jitter = rng_j.uniform(-0.05, 0.05, len(vals))
        x_pos  = np.full(len(vals), 0.5) + jitter

        ax.boxplot(vals, positions=[0.5], widths=0.35,
                   patch_artist=True,
                   boxprops=dict(facecolor='lightsteelblue', alpha=0.6),
                   medianprops=dict(color='navy', linewidth=2),
                   whiskerprops=dict(color='gray'),
                   capprops=dict(color='gray'),
                   flierprops=dict(marker=''))

        ax.scatter(x_pos, vals, color='steelblue', s=30, zorder=5, alpha=0.85,
                   label='seeds 42–51')

        if 42 in all_criteria:
            idx42 = seeds.index(42)
            ax.scatter(0.5 + jitter[idx42], vals[idx42],
                       color='red', s=100, zorder=6, marker='o', label='seed=42')

        ax.axhline(thr, color='black', linestyle='--', linewidth=1.5,
                   label=f'threshold = {thr}')

        reproducible  = n >= 8
        verdict       = f'REPRODUCED ({n}/10)' if reproducible else f'NOT REPRODUCED ({n}/10)'
        verdict_color = 'green' if reproducible else 'red'

        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0.1, 0.9)
        ax.set_xticks([])
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.text(0.5, 0.03, verdict, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9.5,
                color=verdict_color, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_s14_summary(all_metrics, seeds=_SEEDS, fname=_SUMM_PNG):
    """Heatmap: conditions (rows) × seeds (columns), one panel per metric."""
    display = [
        ('n_edges',    'Edges'),
        ('clustering', 'Clustering'),
        ('depth_out',  'Depth\n(I→O)'),
        ('ablation',   'Ablation'),
        ('cos_int',    'cos_int'),
        ('cos_out',    'cos_out'),
    ]
    n_metrics = len(display)
    n_seeds   = len(seeds)
    n_groups  = len(_GROUP_KEYS)

    # data[mi, gi, si]
    data = np.zeros((n_metrics, n_groups, n_seeds))
    for si, seed in enumerate(seeds):
        for gi, gk in enumerate(_GROUP_KEYS):
            for mi, (mk, _) in enumerate(display):
                data[mi, gi, si] = all_metrics[seed][gk][mk]

    seed_labels = [str(s) for s in seeds]

    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3.5))
    axes = axes.flatten()

    fig.suptitle(
        'Session 14: Per-seed Summary  (10 seeds × 4 conditions × 6 metrics)\n'
        'Rows = conditions, Columns = seeds within each panel',
        fontsize=10,
    )

    cmaps = {
        'n_edges':    'Blues',
        'clustering': 'Greens',
        'depth_out':  'Purples',
        'ablation':   'Oranges',
        'cos_int':    'YlOrRd',
        'cos_out':    'YlOrRd',
    }

    for mi, (mk, mlabel) in enumerate(display):
        ax  = axes[mi]
        mat = data[mi]                       # (n_groups, n_seeds)
        vmin, vmax = float(mat.min()), float(mat.max())
        if vmax - vmin < 1e-9:
            vmin, vmax = vmin - 0.01, vmax + 0.01

        im = ax.imshow(mat, aspect='auto', cmap=cmaps.get(mk, 'viridis'),
                       vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n_seeds))
        ax.set_xticklabels(seed_labels, fontsize=7)
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(_GROUP_LABELS, fontsize=7.5)
        ax.set_title(mlabel, fontsize=9.5, pad=4)
        ax.set_xlabel('seed', fontsize=7.5)

        for gi in range(n_groups):
            for si in range(n_seeds):
                v = mat[gi, si]
                ax.text(si, gi, f'{v:.2f}', ha='center', va='center',
                        fontsize=5.5,
                        color='white' if (v - vmin) / (vmax - vmin) > 0.6 else 'black')

        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)

    # hide unused panels
    for mi in range(n_metrics, len(axes)):
        axes[mi].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 14: 再現性確認 (seeds 42–51) ===')
    print('Replicating Session 13 analyses across 10 seeds.\n')

    all_metrics, all_criteria = run_s14_sweep(
        seeds=_SEEDS, n_gen=50, n_agents=10, n_ep=5, n_surv=3,
    )

    print('\n[Summary] Criterion pass counts:')
    n_c1 = sum(all_criteria[s]['pass_c1'] for s in _SEEDS)
    n_c2 = sum(all_criteria[s]['pass_c2'] for s in _SEEDS)
    n_c3 = sum(all_criteria[s]['pass_c3'] for s in _SEEDS)
    for name, n in [('C1 (path ratio > 3.0)', n_c1),
                    ('C2 (surv abl < non)',    n_c2),
                    ('C3 (old clust > new)',   n_c3)]:
        verdict = 'REPRODUCED' if n >= 8 else 'NOT REPRODUCED'
        print(f'  {name}: {verdict} ({n}/10 seeds)')

    print(f'\n[Output] Saving CSV to {_CSV_PATH} ...')
    save_s14_csv(all_metrics, all_criteria)

    print('\n[Plot] Reproducibility box plots ...')
    plot_s14_reproducibility(all_criteria)

    print('\n[Plot] Summary heatmap ...')
    plot_s14_summary(all_metrics)

    print('\nDone.')
