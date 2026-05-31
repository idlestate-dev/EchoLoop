"""Session 13: 500ステップ生存個体の解剖

Question: 新アーキテクチャ（ノイズ+睡眠）と旧アーキテクチャ（Readout+Co-evolution）の
         500ステップ生存個体は構造的に異なるか？

Measurements:
  1. Topology   : edge count, weight distribution, clustering, cycles, I/O paths
  2. Activity   : internal node patterns before food acquisition (NW vs SE)
  3. Info flow  : propagation depth (BFS), ablation sensitivity
"""
import os
from itertools import islice

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _s10_world_step, _s10_inp4, _s10_softmax_sample,
    _s10_run_ep_old,
    _N, _K, _GRID, _HP_MAX, _HP_DECAY,
    _FOOD_VAL, _RESPAWN, _FOOD_POS, _N_PROP, _MAX_STEPS, _ACTION_NAMES,
)
from session_11_noise_escape import _s11_hebb
from session_12_sleep_consolidation import _s12_run_ep, _BEST_EP, _BEST_AN

_T_CONSOL   = 200    # best T_consolidation from S12
_TOP_K      = 3      # agents to collect per group
_N_PROBE    = 10     # probe episodes for activity patterns
_CYCLE_CAP  = 500    # max directed cycles counted
_PATH_CAP   = 100    # max paths per (src,dst) pair
_PATH_CUT   = 5      # BFS cutoff for simple paths

_GROUP_KEYS    = ['new_surv', 'new_non', 'old_surv', 'old_non']
_GROUP_LABELS  = ['New-Surv', 'New-Non', 'Old-Surv', 'Old-Non']
_GROUP_COLORS  = ['steelblue', 'lightsteelblue', 'darkorange', 'moccasin']


# ─── Evolution + collection ───────────────────────────────────────────────────

def _s13_evolve_new_collect(seed, n_gen=50, n_agents=10, n_ep=5, n_surv=3,
                             T_consol=_T_CONSOL, top_k=_TOP_K):
    """Run new-arch evolution; return (surv_list, non_surv_list).
    Each element: {'G', 'W', 'mean_steps', 'max_steps'}
    Topology snapshot is taken before the final-generation episodes.
    """
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_agents):
        G = _s10_build_graph(rng)
        pop.append((G, _s10_get_W(G)))

    for gen in range(n_gen):
        fitnesses = []
        for G, W in pop:
            rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
            ep_steps = [
                _s12_run_ep(G, W,
                            np.random.default_rng(int(rng_ag.integers(0, 2**32))),
                            T_consol)[0]
                for _ in range(n_ep)
            ]
            fitnesses.append(float(np.mean(ep_steps)))

        if (gen + 1) % 10 == 0:
            print(f'    gen {gen+1:3d}: mean={np.mean(fitnesses):.1f}, '
                  f'best={max(fitnesses):.1f}')

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:n_surv]]
        new_pop    = list(survivors)
        while len(new_pop) < n_agents:
            G_p, _ = survivors[int(rng.integers(0, n_surv))]
            G_c    = _s10_mutate(G_p, rng)
            new_pop.append((G_c, _s10_get_W(G_c)))
        pop = new_pop

    # Final generation: snapshot before episodes, then evaluate
    final = []
    for G, W in pop:
        G_snap = G.copy()
        W_snap = W.copy()
        rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
        ep_steps = [
            _s12_run_ep(G, W,
                        np.random.default_rng(int(rng_ag.integers(0, 2**32))),
                        T_consol)[0]
            for _ in range(n_ep)
        ]
        final.append({
            'G':          G_snap,
            'W':          W_snap,
            'mean_steps': float(np.mean(ep_steps)),
            'max_steps':  float(max(ep_steps)),
        })

    final.sort(key=lambda x: -x['mean_steps'])
    surv500 = [f for f in final if f['max_steps'] >= 500]
    if len(surv500) < top_k:
        surv500 = final[:top_k]
    return surv500, final[-top_k:]


def _s13_evolve_old_collect(seed, n_gen=50, n_agents=10, n_ep=5, n_surv=3,
                             top_k=_TOP_K):
    """Run old-arch evolution; return (surv_list, non_surv_list).
    Each element: {'G', 'W', 'rw', 'mean_steps', 'max_steps'}
    """
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_agents):
        G  = _s10_build_graph(rng)
        rw = rng.standard_normal((16, 5)) * 0.5
        pop.append((G, _s10_get_W(G), rw))

    for gen in range(n_gen):
        fitnesses = []
        for G, W, rw in pop:
            rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
            ep_steps = [
                _s10_run_ep_old(G, W, rw,
                                np.random.default_rng(int(rng_ag.integers(0, 2**32))))
                for _ in range(n_ep)
            ]
            fitnesses.append(float(np.mean(ep_steps)))

        if (gen + 1) % 10 == 0:
            print(f'    gen {gen+1:3d}: mean={np.mean(fitnesses):.1f}, '
                  f'best={max(fitnesses):.1f}')

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:n_surv]]
        new_pop    = list(survivors)
        while len(new_pop) < n_agents:
            G_p, _, rw_p = survivors[int(rng.integers(0, n_surv))]
            G_c  = _s10_mutate(G_p, rng)
            rw_c = rw_p + rng.standard_normal(rw_p.shape) * 0.1
            new_pop.append((G_c, _s10_get_W(G_c), rw_c))
        pop = new_pop

    final = []
    for G, W, rw in pop:
        G_snap = G.copy()
        W_snap = W.copy()
        rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
        ep_steps = [
            _s10_run_ep_old(G, W, rw,
                            np.random.default_rng(int(rng_ag.integers(0, 2**32))))
            for _ in range(n_ep)
        ]
        final.append({
            'G':          G_snap,
            'W':          W_snap,
            'rw':         rw.copy(),
            'mean_steps': float(np.mean(ep_steps)),
            'max_steps':  float(max(ep_steps)),
        })

    final.sort(key=lambda x: -x['mean_steps'])
    surv500 = [f for f in final if f['max_steps'] >= 500]
    if len(surv500) < top_k:
        surv500 = final[:top_k]
    return surv500, final[-top_k:]


def collect_all(seed=42, n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    """Run both archs; return dict with new_surv, new_non, old_surv, old_non."""
    print('  [New arch: noise + sleep] ...')
    new_surv, new_non = _s13_evolve_new_collect(seed, n_gen, n_agents, n_ep, n_surv)
    n500_new = sum(1 for f in new_surv if f['max_steps'] >= 500)
    print(f'    actual 500-step survivors: {n500_new}  (collected top-{_TOP_K})')

    print('  [Old arch: readout + co-evo] ...')
    old_surv, old_non = _s13_evolve_old_collect(seed + 1, n_gen, n_agents, n_ep, n_surv)
    n500_old = sum(1 for f in old_surv if f['max_steps'] >= 500)
    print(f'    actual 500-step survivors: {n500_old}  (collected top-{_TOP_K})')

    return {'new_surv': new_surv, 'new_non': new_non,
            'old_surv': old_surv, 'old_non': old_non}


# ─── Topology analysis ────────────────────────────────────────────────────────

def _s13_topology_metrics(G):
    """Return topology metrics dict for directed graph G."""
    weights  = [d['weight'] for _, _, d in G.edges(data=True)]
    n_edges  = G.number_of_edges()
    mean_w   = float(np.mean(weights)) if weights else 0.0
    std_w    = float(np.std(weights))  if weights else 0.0

    G_un      = G.to_undirected()
    clustering = nx.average_clustering(G_un) if n_edges > 0 else 0.0

    cycles      = list(islice(nx.simple_cycles(G), _CYCLE_CAP))
    n_cycles    = len(cycles)
    cycle_sizes = [len(c) for c in cycles]

    path_count = 0
    for src in range(4):
        for dst in range(4, 9):
            try:
                paths = list(islice(
                    nx.all_simple_paths(G, src, dst, cutoff=_PATH_CUT),
                    _PATH_CAP,
                ))
                path_count += len(paths)
            except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
                pass

    return {
        'n_edges':    n_edges,
        'weights':    weights,
        'mean_w':     mean_w,
        'std_w':      std_w,
        'clustering': clustering,
        'n_cycles':   n_cycles,
        'cycle_sizes': cycle_sizes,
        'path_count': path_count,
    }


def run_topology_anatomy(collected):
    """Compute topology metrics for all 4 groups."""
    results = {}
    for gk in _GROUP_KEYS:
        metrics = [_s13_topology_metrics(a['G']) for a in collected[gk]]
        results[gk] = metrics
        print(f'  {gk}: edges={[m["n_edges"] for m in metrics]}, '
              f'clust={[round(m["clustering"], 3) for m in metrics]}, '
              f'cycles={[m["n_cycles"] for m in metrics]}, '
              f'paths={[m["path_count"] for m in metrics]}')
    return results


# ─── Activity pattern analysis ────────────────────────────────────────────────

def _cosine_dist(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float(1.0 - np.dot(a, b) / (na * nb))


_N_TRANSIENT = 2   # steps for transient (feedforward) probe — input reaches nodes in 1-2 hops
_N_STEADY    = 30  # steps for steady-state probe


def _s13_forced_activity(W, inp4, n_steps=_N_TRANSIENT):
    """Return (internal_act[11], output_act[5]) after n_steps propagation
    from zero initial state under constant inp4.  Frozen topology, no noise.

    n_steps=2  captures the first feedforward wave before recurrence saturates.
    n_steps=30 shows the fixed-point attractor (may be input-independent for dense W).
    """
    act = np.zeros(_N)
    for _ in range(n_steps):
        act = _s10_propagate(W, act, inp4)
    return act[9:].copy(), act[4:9].copy()


def run_activity_patterns(collected, seed=42):
    """Forced-location probe: measure steady-state activity when the network
    receives NW-food-location input vs SE-food-location input.

    NW scenario: agent at (row=0,col=0), only NW food available, hp=100
    SE scenario: agent at (row=4,col=4), only SE food available, hp=100

    Internal nodes (9-19): recurrent context state
    Output  nodes  (4-8) : action selection (North/South/West/East/Eat)
    """
    inp_nw = _s10_inp4(0, 0, 100, [True,  False])
    inp_se = _s10_inp4(4, 4, 100, [False, True])

    results = {}
    for gk in _GROUP_KEYS:
        int_nw, int_se = [], []
        out_nw, out_se = [], []
        for a in collected[gk]:
            i_nw, o_nw = _s13_forced_activity(a['W'], inp_nw)
            i_se, o_se = _s13_forced_activity(a['W'], inp_se)
            int_nw.append(i_nw);  int_se.append(i_se)
            out_nw.append(o_nw);  out_se.append(o_se)

        cos_int = [_cosine_dist(a, b) for a, b in zip(int_nw, int_se)]
        cos_out = [_cosine_dist(a, b) for a, b in zip(out_nw, out_se)]

        results[gk] = {
            'int_nw':   np.mean(int_nw, axis=0),
            'int_se':   np.mean(int_se, axis=0),
            'out_nw':   np.mean(out_nw, axis=0),
            'out_se':   np.mean(out_se, axis=0),
            'n_agents': len(int_nw),
            'cos_int':  float(np.nanmean(cos_int)),
            'cos_out':  float(np.nanmean(cos_out)),
        }
        print(f'  {gk}: n_agents={len(int_nw)}, '
              f'cos_dist_int={results[gk]["cos_int"]:.4f}, '
              f'cos_dist_out={results[gk]["cos_out"]:.4f}')
    return results


# ─── Information flow analysis ────────────────────────────────────────────────

def _s13_info_flow(G, W):
    """Return propagation depth (BFS) and ablation sensitivity for agent (G, W)."""
    depth_out, depth_int = [], []
    for src in range(4):
        lengths   = nx.single_source_shortest_path_length(G, src)
        d_out = [lengths.get(d, _N) for d in range(4, 9)]
        d_int = [lengths.get(d, _N) for d in range(9, 20)]
        depth_out.append(float(np.mean(d_out)))
        depth_int.append(float(np.mean(d_int)))

    # Ablation using _N_PROP steps from zero (one simulation-step equivalent).
    # Input propagates ~1 hop per _N_PROP steps; detects direct input→output paths.
    baseline_inp4 = np.array([0.5, 0.5, 0.7, 1.0])
    act_base = np.zeros(_N)
    for _ in range(_N_PROP):
        act_base = _s10_propagate(W, act_base, baseline_inp4)
    out_base = act_base[4:9]

    ablation = []
    for k in range(4):
        inp_k = baseline_inp4.copy()
        inp_k[k] = 0.0
        act_k = np.zeros(_N)
        for _ in range(_N_PROP):
            act_k = _s10_propagate(W, act_k, inp_k)
        ablation.append(float(np.linalg.norm(act_k[4:9] - out_base)))

    return {
        'depth_out': depth_out,
        'depth_int': depth_int,
        'ablation':  ablation,
    }


def run_information_flow(collected):
    """Compute info-flow metrics for all 4 groups."""
    results = {}
    for gk in _GROUP_KEYS:
        metrics    = [_s13_info_flow(a['G'], a['W']) for a in collected[gk]]
        results[gk] = metrics
        m_abl  = np.mean([m['ablation']  for m in metrics], axis=0).round(4)
        m_dep  = np.mean([m['depth_out'] for m in metrics], axis=0).round(2)
        print(f'  {gk}: depth_to_output={m_dep}, ablation={m_abl}')
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_topology_anatomy(
        topo_data,
        fname='images/session_13/results_s13_topology_anatomy.png'):

    def _vals(gk, key):
        return [m[key] for m in topo_data[gk]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        'Session 13: Topology Anatomy — 500-step survivors vs non-survivors\n'
        'New arch (noise+sleep) vs Old arch (readout+co-evo)  |  seed=42',
        fontsize=10,
    )

    specs = [
        ('n_edges',    'Edge Count',                            None),
        ('clustering', 'Clustering Coefficient (undirected)',   (0, 1)),
        ('n_cycles',   f'Directed Cycles (cap={_CYCLE_CAP})',  None),
        ('path_count', f'Input→Output Paths\n(cutoff={_PATH_CUT}, cap={_PATH_CAP})', None),
    ]

    for ax, (key, title, ylim) in zip(axes, specs):
        for xi, (gk, lb, col) in enumerate(
                zip(_GROUP_KEYS, _GROUP_LABELS, _GROUP_COLORS)):
            vals = _vals(gk, key)
            ax.bar(xi, np.mean(vals), color=col, edgecolor='gray', linewidth=0.8,
                   yerr=np.std(vals) if len(vals) > 1 else 0,
                   capsize=5, error_kw={'elinewidth': 1.5})
            for v in vals:
                ax.scatter(xi, v, color='black', s=18, zorder=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(_GROUP_LABELS, fontsize=8)
        ax.set_title(title, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_activity_pattern(
        act_data,
        fname='images/session_13/results_s13_activity_pattern.png'):
    """4 rows × 4 cols heatmap.
    Rows: (int-NW, int-SE, out-NW, out-SE)   Cols: (New-Surv, Old-Surv, New-Non, Old-Non)
    """
    col_order   = ['new_surv', 'old_surv', 'new_non', 'old_non']
    col_titles  = ['New-Surv', 'Old-Surv', 'New-Non', 'Old-Non']
    int_labels  = [f'int{i}' for i in range(11)]
    out_labels  = [f'out{i}\n({_ACTION_NAMES[i][:3]})' for i in range(5)]

    row_specs = [
        ('int_nw', int_labels, 'Internal (9-19)  NW food'),
        ('int_se', int_labels, 'Internal (9-19)  SE food'),
        ('out_nw', out_labels, 'Output  (4-8)    NW food'),
        ('out_se', out_labels, 'Output  (4-8)    SE food'),
    ]

    # Determine colour range per block type (int vs out)
    int_vals = np.concatenate([[act_data[gk]['int_nw'], act_data[gk]['int_se']]
                               for gk in col_order])
    out_vals = np.concatenate([[act_data[gk]['out_nw'], act_data[gk]['out_se']]
                               for gk in col_order])
    vmax_int = max(float(np.max(np.abs(int_vals))), 0.05)
    vmax_out = max(float(np.max(np.abs(out_vals))), 0.05)

    fig, axes = plt.subplots(4, 4, figsize=(18, 10))
    fig.suptitle(
        f'Session 13: Transient Activity — Forced Location Probe  ({_N_TRANSIENT}-step from zero)\n'
        'NW=(row=0,col=0, food0 only)  SE=(row=4,col=4, food1 only)\n'
        'Transient captures feedforward signal before recurrent dynamics saturate\n'
        'cos_dist_int / cos_dist_out shown in column title',
        fontsize=9,
    )

    for ci, (gk, ctitle) in enumerate(zip(col_order, col_titles)):
        d      = act_data[gk]
        n_ag   = d['n_agents']
        cd_int = d['cos_int']
        cd_out = d['cos_out']
        axes[0][ci].set_title(
            f'{ctitle}  (n={n_ag})\n'
            f'cos_int={cd_int:.4f}  cos_out={cd_out:.4f}',
            fontsize=7.5,
        )

        for ri, (dkey, labels, row_title) in enumerate(row_specs):
            ax   = axes[ri][ci]
            vmax = vmax_int if ri < 2 else vmax_out
            mat  = d[dkey].reshape(1, -1)
            im   = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=5.5, rotation=45)
            ax.set_yticks([])
            for ci2, val in enumerate(d[dkey]):
                ax.text(ci2, 0, f'{val:.2f}', ha='center', va='center', fontsize=5.5)
            if ci == 0:
                ax.set_ylabel(row_title, fontsize=7, rotation=0,
                              labelpad=80, va='center')
            if ci == 3:
                plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_information_flow(
        flow_data,
        fname='images/session_13/results_s13_information_flow.png'):

    input_labels = ['node0\n(col/x)', 'node1\n(row/y)', 'node2\n(hp)', 'node3\n(food)']
    x     = np.arange(4)
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 13: Information Flow — Propagation Depth & Ablation Sensitivity\n'
        'New arch (noise+sleep) vs Old arch (readout+co-evo)  |  seed=42',
        fontsize=10,
    )

    for ax, (metric_key, ylabel, title) in zip(axes, [
        ('depth_out',
         'Mean BFS Distance (input → output nodes 4-8)',
         'Propagation Depth: Input → Output'),
        ('ablation',
         'Output Activity Change (L2) when Input = 0',
         'Ablation Sensitivity: Each Input Node\'s Influence'),
    ]):
        for gi, (gk, lb, col) in enumerate(
                zip(_GROUP_KEYS, _GROUP_LABELS, _GROUP_COLORS)):
            metrics = flow_data[gk]
            means   = np.mean([m[metric_key] for m in metrics], axis=0)
            stds    = np.std( [m[metric_key] for m in metrics], axis=0)
            offset  = (gi - 1.5) * width
            ax.bar(x + offset, means, width=width, label=lb, color=col,
                   edgecolor='gray', linewidth=0.7,
                   yerr=stds, capsize=3, error_kw={'elinewidth': 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(input_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 13: 500ステップ生存個体の解剖 ===')

    print('\n[Collection] Evolution: 50 gen × 10 agents × 5 ep, seed=42')
    collected = collect_all(seed=42, n_gen=50, n_agents=10, n_ep=5, n_surv=3)

    print('\n  Summary:')
    for gk, lb in zip(_GROUP_KEYS, _GROUP_LABELS):
        agents = collected[gk]
        print(f'  {lb}: mean_steps={[round(a["mean_steps"],1) for a in agents]}, '
              f'max_steps={[int(a["max_steps"]) for a in agents]}')

    print('\n[Analysis 1] Topology anatomy ...')
    topo_data = run_topology_anatomy(collected)
    plot_topology_anatomy(topo_data)

    print('\n[Analysis 2] Activity patterns (frozen topology probe) ...')
    act_data = run_activity_patterns(collected, seed=42)
    plot_activity_pattern(act_data)

    print('\n[Analysis 3] Information flow ...')
    flow_data = run_information_flow(collected)
    plot_information_flow(flow_data)

    print('\nDone.')
