"""Session 5: Association parameter sweep (pure Hebbian, no E/I)."""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from world_base import _make_graph

# ─── Session 5: Association parameter sweep (pure Hebbian, no E/I) ────────────

def _build_sweep_graph(N, rng, initial_density, initial_weight):
    """Build association graph for the parameter sweep.

    Stimulus: 0=food, 1=north, 2=south
    Loop Food:  3→4→5→3   (pre-wired entry: 0→3, weight 0.5)
    Loop North: 6→7→8→6   (pre-wired entry: 1→6, weight 0.5)
    Loop South: 9→10→11→9 (pre-wired entry: 2→9, weight 0.5)
    Background: 12..N-1
    Random background edges seeded at initial_weight (uniform).
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < initial_density:
                G.add_edge(i, j, weight=float(initial_weight))
    for s, d in [(0, 3), (1, 6), (2, 9)]:
        G.add_edge(s, d, weight=0.5)
    for s, d in [(3, 4), (4, 5), (5, 3)]:
        G.add_edge(s, d, weight=0.5)
    for s, d in [(6, 7), (7, 8), (8, 6)]:
        G.add_edge(s, d, weight=0.5)
    for s, d in [(9, 10), (10, 11), (11, 9)]:
        G.add_edge(s, d, weight=0.5)
    return G


def run_association_sweep(N=20, seed=42, K=5, T_phase=500, T_probe=200):
    """Sweep initial_density × initial_weight × hebbian_lr (27 conditions).

    Pure Hebbian + decay=0.01 (no E/I).  Probe: food-only (node0=0.8), measure
    node1 (north stimulus) and node2 (south stimulus) reactivation.
    """
    densities    = [0.2, 0.4, 0.6]
    init_weights = [0.05, 0.2, 0.5]
    lrs          = [0.05, 0.1, 0.2]

    loop_food  = [3, 4, 5]
    loop_north = [6, 7, 8]
    background = list(range(12, N))

    phases = [
        ({1: 0.8},         'North solo'),
        ({0: 0.8},         'Food solo'),
        ({0: 0.8, 1: 0.8}, 'North+Food'),
        ({0: 0.8, 2: 0.8}, 'South+Food'),
    ]

    results = []

    for d_idx, density in enumerate(densities):
        for w_idx, init_w in enumerate(init_weights):
            for lr_idx, lr in enumerate(lrs):
                cond_seed = seed + d_idx * 900 + w_idx * 30 + lr_idx
                rng = np.random.default_rng(cond_seed)
                G = _build_sweep_graph(N, rng, density, init_w)
                activity = np.zeros(N)

                def _count_cross_fn(a, b):
                    return sum(1 for i in a for j in b
                               if G.has_edge(i, j) or G.has_edge(j, i))

                def _train_hebbian(ext, steps, _G=G, _lr=lr, _iw=init_w, _rng=rng):
                    nonlocal activity
                    for t in range(steps):
                        new_act = np.zeros(N)
                        for i in range(N):
                            if i in ext:
                                new_act[i] = ext[i]
                            else:
                                s = sum(_G[j][i]['weight'] * activity[j]
                                        for j in _G.predecessors(i))
                                new_act[i] = np.tanh(s)
                        activity = new_act

                        if (t + 1) % K == 0:
                            to_rm = []
                            for i, j, d in list(_G.edges(data=True)):
                                w = d['weight']
                                if activity[i] > 0.5 and activity[j] > 0.5:
                                    w += _lr
                                w -= 0.01
                                if w < 0.01:
                                    to_rm.append((i, j))
                                else:
                                    _G[i][j]['weight'] = min(w, 1.0)
                            _G.remove_edges_from(to_rm)
                            existing = set(_G.edges())
                            for i in range(N):
                                for j in range(N):
                                    if i != j and (i, j) not in existing:
                                        if _rng.random() < 0.005:
                                            _G.add_edge(i, j, weight=float(_iw))

                phase_metrics = []
                for ext, pname in phases:
                    _train_hebbian(ext, T_phase)
                    pm = {
                        'name': pname,
                        'cross_fn': _count_cross_fn(loop_food, loop_north),
                        'active_n': int(np.sum(activity > 0.1)),
                        'mean_food':  float(np.mean(activity[loop_food])),
                        'mean_north': float(np.mean(activity[loop_north])),
                        'mean_bg':    float(np.mean(activity[background])) if background else 0.0,
                    }
                    phase_metrics.append(pm)

                # Probe: food only (node0=0.8), graph frozen
                act_p = activity.copy()
                probe_traj = []
                for _ in range(T_probe):
                    new_act = np.zeros(N)
                    new_act[0] = 0.8
                    for i in range(1, N):
                        s = sum(G[j][i]['weight'] * act_p[j]
                                for j in G.predecessors(i))
                        new_act[i] = np.tanh(s)
                    act_p = new_act
                    probe_traj.append(act_p.copy())

                probe_traj = np.array(probe_traj)  # (T_probe, N)
                mn = float(np.mean(probe_traj[:, 1]))
                ms = float(np.mean(probe_traj[:, 2]))
                assoc = bool(mn > 0.3 and mn > ms)

                print(f'd={density:.1f} w={init_w:.2f} lr={lr:.2f}  '
                      f'ce={phase_metrics[-1]["cross_fn"]:2d}  '
                      f'north={mn:.4f}  south={ms:.4f}  assoc={assoc}')

                results.append({
                    'density': density, 'd_idx': d_idx,
                    'init_weight': init_w, 'w_idx': w_idx,
                    'hebbian_lr': lr, 'lr_idx': lr_idx,
                    'phase_metrics': phase_metrics,
                    'probe_traj': probe_traj,
                    'mean_north_probe': mn,
                    'mean_south_probe': ms,
                    'association': assoc,
                })

    n_true = sum(r['association'] for r in results)
    print(f'\nAssociation=True: {n_true}/27 conditions')
    return results


def plot_association_sweep(results, fname='images/session_5/results_association_sweep.png'):
    densities    = [0.2, 0.4, 0.6]
    init_weights = [0.05, 0.2, 0.5]
    lrs          = [0.05, 0.1, 0.2]
    colors       = ['#1f77b4', '#ff7f0e', '#2ca02c']
    phase_labels = ['N\nsolo', 'F\nsolo', 'N+F', 'S+F']

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=False)
    fig.suptitle(
        'Association sweep: Food↔North cross-edges per phase\n'
        'rows=initial_density  cols=initial_weight  lines=hebbian_lr\n'
        '(green background = association=True, ◆ marker)',
        fontsize=11)

    for d_idx, density in enumerate(densities):
        for w_idx, init_w in enumerate(init_weights):
            ax = axes[d_idx][w_idx]
            cell = [r for r in results
                    if r['d_idx'] == d_idx and r['w_idx'] == w_idx]
            if any(r['association'] for r in cell):
                ax.set_facecolor('#e8f5e9')

            for r in cell:
                ys = [r['phase_metrics'][p]['cross_fn'] for p in range(4)]
                mk = 'D' if r['association'] else 'o'
                lbl = (f"lr={lrs[r['lr_idx']]:.2f}"
                       + (' ✓' if r['association'] else ''))
                ax.plot(range(1, 5), ys,
                        color=colors[r['lr_idx']], marker=mk, ms=5, label=lbl)

            ax.set_title(f'dens={density:.1f}  w₀={init_w:.2f}', fontsize=9)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(phase_labels, fontsize=7)
            ax.set_xlabel('Phase', fontsize=8)
            ax.set_ylabel('Cross-edges (F↔N)', fontsize=8)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_association_probe(results, fname='images/session_5/results_association_probe.png'):
    true_rs = [r for r in results if r['association']]
    if not true_rs:
        # Fall back: show top-6 by north reactivation
        true_rs = sorted(results, key=lambda r: r['mean_north_probe'], reverse=True)[:6]
        title_prefix = 'Top-6 by north reactivation (none reached assoc=True)'
    else:
        title_prefix = f'Probe trajectories — association=True ({len(true_rs)} conditions)'

    n = len(true_rs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title_prefix, fontsize=11)

    for idx, r in enumerate(true_rs):
        ax = axes[idx // cols][idx % cols]
        traj = r['probe_traj']   # (T_probe, N)
        ax.plot(traj[:, 1], color='steelblue', lw=1.5, label='node1 (north)')
        ax.plot(traj[:, 2], color='tomato', linestyle='--', lw=1.5,
                label='node2 (south)')
        ax.axhline(0.3, color='gray', linestyle=':', lw=0.8, label='thr=0.3')
        lbl = (f"d={r['density']:.1f}  w={r['init_weight']:.2f}"
               f"  lr={r['hebbian_lr']:.2f}\n"
               f"north={r['mean_north_probe']:.3f}  "
               f"south={r['mean_south_probe']:.3f}  assoc={r['association']}")
        ax.set_title(lbl, fontsize=8)
        ax.set_xlabel('Probe step', fontsize=8)
        ax.set_ylabel('Activity', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


