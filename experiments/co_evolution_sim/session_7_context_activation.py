"""Session 7: Context-dependent activation patterns."""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ─── Session 7: Context-dependent activation patterns ────────────────────────

_S7_COND     = {'density': 0.2, 'init_weight': 0.05, 'lr': 0.05}
_S7_INTERNAL = list(range(3, 20))   # 17 internal nodes
_S7_K        = 5


def _s7_build_graph(N, rng, density=0.2, init_weight=0.05):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < density:
                G.add_edge(i, j, weight=init_weight)
    return G


def _s7_step(G, activity, ext_vals, N):
    new_act = np.zeros(N)
    for i in range(N):
        if i in ext_vals:
            new_act[i] = ext_vals[i]
        else:
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_act[i] = np.tanh(s)
    return new_act


def _s7_hebb(G, activity, N, rng, lr=0.05, init_weight=0.05):
    edges_to_remove = []
    for i, j, data in list(G.edges(data=True)):
        w = data['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += lr
        w -= 0.01
        if w < 0.01:
            edges_to_remove.append((i, j))
        else:
            G[i][j]['weight'] = min(w, 1.0)
    G.remove_edges_from(edges_to_remove)
    existing = set(G.edges())
    for i in range(N):
        for j in range(N):
            if i != j and (i, j) not in existing and rng.random() < 0.01:
                G.add_edge(i, j, weight=init_weight)


def _s7_train_phase(G, activity, ext_vals, T, N, rng, K=5, lr=0.05,
                    init_weight=0.05, track_topology=False):
    n_edges_list, clustering_list, act_var_list = [], [], []
    for t in range(T):
        activity[:] = _s7_step(G, activity, ext_vals, N)
        if (t + 1) % K == 0:
            _s7_hebb(G, activity, N, rng, lr, init_weight)
            if track_topology:
                n_edges_list.append(G.number_of_edges())
                clustering_list.append(nx.average_clustering(G))
                act_var_list.append(float(np.var(activity)))
    if track_topology:
        return {'n_edges': n_edges_list, 'clustering': clustering_list,
                'act_var': act_var_list}
    return None


def _s7_probe(G, N, T_probe=200):
    """Run ProbeA (Food+North) and ProbeB (Food+South). Returns metrics dict."""
    internal = _S7_INTERNAL

    ext_A = {0: 0.8, 1: 0.3, 2: 0.0}
    act = np.zeros(N)
    acc_A = np.zeros(N)
    for _ in range(T_probe):
        act = _s7_step(G, act, ext_A, N)
        acc_A += act
    mean_A = acc_A / T_probe

    ext_B = {0: 0.8, 1: 0.0, 2: 0.3}
    act = np.zeros(N)
    acc_B = np.zeros(N)
    for _ in range(T_probe):
        act = _s7_step(G, act, ext_B, N)
        acc_B += act
    mean_B = acc_B / T_probe

    int_A  = mean_A[internal]
    int_B  = mean_B[internal]
    norm_A = float(np.linalg.norm(int_A))
    norm_B = float(np.linalg.norm(int_B))
    norm   = norm_A * norm_B
    # Guard degenerate cases: if either pattern is ~zero, cos_dist is undefined.
    cos_dist = float(1.0 - np.dot(int_A, int_B) / (norm + 1e-10)) if norm > 1e-6 else float('nan')

    # Bias: mean activity of internal nodes that prefer each context.
    diff = int_A - int_B
    north_bias = float(np.mean(diff[diff > 0])) if np.any(diff > 0) else 0.0
    south_bias = float(np.mean(-diff[diff < 0])) if np.any(diff < 0) else 0.0

    return {
        'mean_A': mean_A, 'mean_B': mean_B,
        'int_A': int_A, 'int_B': int_B,
        'norm_A': norm_A, 'norm_B': norm_B,
        'cos_dist': cos_dist, 'north_bias': north_bias, 'south_bias': south_bias,
    }


def run_context_activation_experiment(N=20, seed=42,
                                       switch_intervals=(50, 100, 200),
                                       phase_lengths=(500, 1000, 2000),
                                       T_probe=200):
    density = _S7_COND['density']
    init_w  = _S7_COND['init_weight']
    lr      = _S7_COND['lr']
    K       = _S7_K

    sweep_results = {}
    topo_results  = {}

    print('=== Session 7: Context-dependent activation parameter sweep ===')
    for T_phase in phase_lengths:
        for sw in switch_intervals:
            print(f'  T_phase={T_phase}  switch_interval={sw}')
            rng = np.random.default_rng(seed)
            G   = _s7_build_graph(N, rng, density, init_w)
            act = np.zeros(N)
            all_n_edges, all_clustering, all_act_var = [], [], []
            phase_bdry = []

            m1 = _s7_train_phase(G, act, {0: 0.8, 1: 0.8, 2: 0.0},
                                  T_phase, N, rng, K, lr, init_w, track_topology=True)
            all_n_edges.extend(m1['n_edges'])
            all_clustering.extend(m1['clustering'])
            all_act_var.extend(m1['act_var'])
            phase_bdry.append(len(all_n_edges))

            m2 = _s7_train_phase(G, act, {0: 0.8, 1: 0.0, 2: 0.8},
                                  T_phase, N, rng, K, lr, init_w, track_topology=True)
            all_n_edges.extend(m2['n_edges'])
            all_clustering.extend(m2['clustering'])
            all_act_var.extend(m2['act_var'])
            phase_bdry.append(len(all_n_edges))

            T3 = 2000
            ext_north = {0: 0.8, 1: 0.8, 2: 0.0}
            ext_south  = {0: 0.8, 1: 0.0, 2: 0.8}
            for t in range(T3):
                ext = ext_north if (t // sw) % 2 == 0 else ext_south
                act[:] = _s7_step(G, act, ext, N)
                if (t + 1) % K == 0:
                    _s7_hebb(G, act, N, rng, lr, init_w)
                    all_n_edges.append(G.number_of_edges())
                    all_clustering.append(nx.average_clustering(G))
                    all_act_var.append(float(np.var(act)))
            phase_bdry.append(len(all_n_edges))

            probe = _s7_probe(G, N, T_probe)
            sweep_results[(T_phase, sw)] = probe
            topo_results[(T_phase, sw)]  = {
                'n_edges': all_n_edges, 'clustering': all_clustering,
                'act_var': all_act_var, 'phase_boundaries': phase_bdry,
            }
            print(f'    cos_dist={probe["cos_dist"]:.4f}  '
                  f'north_bias={probe["north_bias"]:.4f}  '
                  f'south_bias={probe["south_bias"]:.4f}')

    return {'sweep': sweep_results, 'topology': topo_results}


def run_context_control_experiment(best_key, N=20, seed=42, T_probe=200):
    density = _S7_COND['density']
    init_w  = _S7_COND['init_weight']
    lr      = _S7_COND['lr']
    K       = _S7_K
    T_phase = best_key[0]

    print('=== Session 7: Control experiments ===')

    print('  Control A (no Phase3)...')
    rng = np.random.default_rng(seed)
    G   = _s7_build_graph(N, rng, density, init_w)
    act = np.zeros(N)
    _s7_train_phase(G, act, {0: 0.8, 1: 0.8, 2: 0.0}, T_phase, N, rng, K, lr, init_w)
    _s7_train_phase(G, act, {0: 0.8, 1: 0.0, 2: 0.8}, T_phase, N, rng, K, lr, init_w)
    ctrl_A = _s7_probe(G, N, T_probe)
    print(f'    cos_dist={ctrl_A["cos_dist"]:.4f}  '
          f'north_bias={ctrl_A["north_bias"]:.4f}  south_bias={ctrl_A["south_bias"]:.4f}')

    print('  Control B (mixed training)...')
    rng = np.random.default_rng(seed)
    G   = _s7_build_graph(N, rng, density, init_w)
    act = np.zeros(N)
    _s7_train_phase(G, act, {0: 0.8, 1: 0.5, 2: 0.5}, 2000, N, rng, K, lr, init_w)
    ctrl_B = _s7_probe(G, N, T_probe)
    print(f'    cos_dist={ctrl_B["cos_dist"]:.4f}  '
          f'north_bias={ctrl_B["north_bias"]:.4f}  south_bias={ctrl_B["south_bias"]:.4f}')

    return {'ctrl_A': ctrl_A, 'ctrl_B': ctrl_B}


def plot_context_activation(data, fname='images/results_context_activation.png'):
    sweep         = data['sweep']
    phase_lengths = [500, 1000, 2000]
    sw_intervals  = [50, 100, 200]
    n_int         = len(_S7_INTERNAL)

    fig, axes = plt.subplots(
        len(phase_lengths), len(sw_intervals),
        figsize=(5.0 * len(sw_intervals), 3.2 * len(phase_lengths)),
        squeeze=False,
    )
    fig.suptitle(
        'Session 7: Context-dependent activation\n'
        'ProbeA = Food+North hint  |  ProbeB = Food+South hint',
        fontsize=11, y=1.01,
    )

    for ri, T_phase in enumerate(phase_lengths):
        for ci, sw in enumerate(sw_intervals):
            ax    = axes[ri][ci]
            probe = sweep[(T_phase, sw)]
            mat   = np.stack([probe['int_A'], probe['int_B']], axis=0)  # (2, 17)

            im = ax.imshow(mat, aspect='auto', cmap='hot', vmin=0.0, vmax=1.0,
                           interpolation='nearest')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['ProbeA\n(N)', 'ProbeB\n(S)'], fontsize=7)
            ax.set_xticks(range(n_int))
            ax.set_xticklabels([str(i) for i in _S7_INTERNAL], fontsize=5, rotation=45)

            cos  = probe['cos_dist']
            nb   = probe['north_bias']
            sb   = probe['south_bias']
            nA   = probe['norm_A']
            nB   = probe['norm_B']
            cos_str = f'{cos:.3f}' if not (cos != cos) else 'nan'
            ax.set_title(
                f'T_phase={T_phase}, sw={sw}\n'
                f'cos={cos_str}  nb={nb:.3f}  sb={sb:.3f}\n'
                f'‖A‖={nA:.3f}  ‖B‖={nB:.3f}',
                fontsize=7,
            )

            if ri == 0 and ci == len(sw_intervals) - 1:
                plt.colorbar(im, ax=ax, fraction=0.07, pad=0.02, label='Mean act.')

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context_topology(data, fname='images/results_context_topology.png'):
    topo          = data['topology']
    phase_lengths = [500, 1000, 2000]
    sw_intervals  = [50, 100, 200]

    colors = {500: '#1f77b4', 1000: '#ff7f0e', 2000: '#2ca02c'}
    styles = {50: '-', 100: '--', 200: ':'}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    fig.suptitle(
        'Session 7: Topology evolution — all 9 conditions overlaid\n'
        '(x-axis normalised to [0,1]; dashed verticals = phase transitions)',
        fontsize=11,
    )

    for T_phase in phase_lengths:
        for sw in sw_intervals:
            res  = topo[(T_phase, sw)]
            n    = len(res['n_edges'])
            xs   = np.linspace(0.0, 1.0, n)
            bdry = [b / n for b in res['phase_boundaries'][:-1]]
            label = f'T={T_phase}, sw={sw}'
            kw    = dict(color=colors[T_phase], ls=styles[sw], lw=1.0, alpha=0.8)

            axes[0].plot(xs, res['n_edges'],    label=label, **kw)
            axes[1].plot(xs, res['clustering'], **kw)
            axes[2].plot(xs, res['act_var'],    **kw)

            for b in bdry:
                for ax in axes:
                    ax.axvline(b, color=colors[T_phase], lw=0.4, ls=':', alpha=0.35)

    row_labels = ['Number of edges', 'Mean clustering coeff.', 'Activity variance']
    for ax, lbl in zip(axes, row_labels):
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_xlabel('Normalised training step', fontsize=9)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=7, ncol=3,
               bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context_control(sweep_data, ctrl_data, best_key,
                          fname='images/results_context_control.png'):
    best_probe = sweep_data['sweep'][best_key]
    ctrl_A     = ctrl_data['ctrl_A']
    ctrl_B     = ctrl_data['ctrl_B']

    conditions = [
        (f'Best\n(T={best_key[0]}, sw={best_key[1]})', best_probe, 'steelblue'),
        ('Control A\n(no Phase3)',                        ctrl_A,    'tomato'),
        ('Control B\n(mixed)',                             ctrl_B,    'seagreen'),
    ]
    metrics = [
        ('cos_dist',    'Cosine distance\n(ProbeA vs ProbeB)'),
        ('north_bias',  'North bias\n(ProbeA node1 − ProbeB node1)'),
        ('south_bias',  'South bias\n(ProbeB node2 − ProbeA node2)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        'Session 7: Control comparison\n'
        'Best sweep vs no-Phase3 (A) vs mixed training (B)',
        fontsize=11,
    )

    xs = np.arange(len(conditions))
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals   = [c[1][metric] for c in conditions]
        colrs  = [c[2] for c in conditions]
        bars   = ax.bar(xs, vals, color=colrs, alpha=0.85, width=0.55)
        for bar, val in zip(bars, vals):
            yoff = 0.005 if val >= 0 else -0.025
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([c[0] for c in conditions], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


