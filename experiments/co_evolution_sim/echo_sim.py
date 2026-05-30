import itertools
from collections import deque

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


METRIC_LABELS = [
    'Mean edge weight',
    'Number of edges',
    'Mean clustering coeff.',
    'Mean shortest path\n(largest WCC)',
    'Activity variance',
    'Degree std (out)',
]

INPUT_PATTERNS = ['constant', 'alternating', 'random']


def _node0_input(pattern, t, rng_input):
    if pattern == 'constant':
        return 0.5
    elif pattern == 'alternating':
        return 0.8 if (t // 50) % 2 == 0 else 0.2
    elif pattern == 'random':
        return rng_input.uniform(0.1, 0.9)
    raise ValueError(f'Unknown input_pattern: {pattern}')


def run_simulation(K, T=2000, N=20, seed=42, input_pattern='constant') -> dict:
    rng = np.random.default_rng(seed)
    # Separate RNG stream for random input so it doesn't perturb topology RNG
    rng_input = np.random.default_rng(seed + 1000)

    # Initialize directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    # Initialize activity
    activity = rng.random(N)
    activity[0] = _node0_input(input_pattern, 0, rng_input)

    # Tracking lists (one entry per topology update)
    mean_weight = []
    num_edges = []
    clustering = []
    shortest_path = []
    activity_variance = []
    activity_mean = []
    degree_std = []
    topology_delta = []

    for t in range(T):
        # --- Activity update ---
        new_activity = np.zeros(N)
        for i in range(N):
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_activity[i] = np.tanh(s)
        new_activity[0] = _node0_input(input_pattern, t, rng_input)
        activity = new_activity

        # --- Topology update every K steps ---
        if (t + 1) % K == 0:
            weights_before = {(i, j): data['weight'] for i, j, data in G.edges(data=True)}

            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                w = data['weight']
                # Strengthening
                if activity[i] > 0.5 and activity[j] > 0.5:
                    w += 0.05
                # Decay
                w -= 0.01
                if w < 0.01:
                    edges_to_remove.append((i, j))
                else:
                    G[i][j]['weight'] = min(w, 1.0)
            G.remove_edges_from(edges_to_remove)

            # Addition
            existing = set(G.edges())
            for i in range(N):
                for j in range(N):
                    if i != j and (i, j) not in existing and rng.random() < 0.01:
                        G.add_edge(i, j, weight=0.05)

            # Topology change magnitude
            edges_before_set = set(weights_before.keys())
            edges_after_set = set(G.edges())
            delta = 0.0
            for e in edges_before_set & edges_after_set:
                delta += abs(G[e[0]][e[1]]['weight'] - weights_before[e])
            for e in edges_before_set - edges_after_set:
                delta += weights_before[e]
            for e in edges_after_set - edges_before_set:
                delta += G[e[0]][e[1]]['weight']
            topology_delta.append(delta)

            # Record metrics
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            mean_weight.append(np.mean(weights) if weights else 0.0)
            num_edges.append(G.number_of_edges())
            clustering.append(nx.average_clustering(G))

            # Mean shortest path on largest weakly connected component
            wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            if wccs:
                lcc = G.subgraph(wccs[0]).copy()
                if lcc.number_of_nodes() > 1:
                    try:
                        shortest_path.append(nx.average_shortest_path_length(lcc))
                    except nx.NetworkXError:
                        shortest_path.append(float('nan'))
                else:
                    shortest_path.append(float('nan'))
            else:
                shortest_path.append(float('nan'))

            activity_variance.append(float(np.var(activity)))
            activity_mean.append(float(np.mean(activity)))
            out_degrees = [d for _, d in G.out_degree()]
            degree_std.append(float(np.std(out_degrees)))

    n_topo_steps = T // K
    timesteps = list(np.arange(1, n_topo_steps + 1) * K)

    # Topology fingerprint (final state)
    final_out_degrees = [d for _, d in G.out_degree()]
    final_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    clustering_per_node = list(nx.clustering(G).values())

    return {
        'mean_weight': mean_weight,
        'num_edges': num_edges,
        'clustering': clustering,
        'shortest_path': shortest_path,
        'activity_variance': activity_variance,
        'activity_mean': activity_mean,
        'degree_std': degree_std,
        'topology_delta': topology_delta,
        'timesteps': timesteps,
        'fingerprint': {
            'out_degrees': final_out_degrees,
            'edge_weights': final_weights,
            'clustering_per_node': clustering_per_node,
        },
    }


def run_comparison(K_values, T=2000, seed=42) -> dict:
    results = {}
    for K in K_values:
        print(f'Running K={K}...')
        results[K] = run_simulation(K=K, T=T, seed=seed)
    return results


def run_input_comparison(K=20, T=2000, seed=42) -> dict:
    results = {}
    for pattern in INPUT_PATTERNS:
        print(f'Running input_pattern={pattern!r}...')
        results[pattern] = run_simulation(K=K, T=T, seed=seed, input_pattern=pattern)
    return results


def plot_comparison(results, K_values):
    metric_keys = ['mean_weight', 'num_edges', 'clustering', 'shortest_path', 'activity_variance', 'degree_std']
    n_rows = len(metric_keys)
    n_cols = len(K_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle('Co-evolution simulation: multi-K comparison', fontsize=13, y=1.01)

    # Compute per-row y-limits (ignoring nan)
    row_ylims = []
    for key in metric_keys:
        all_vals = []
        for K in K_values:
            all_vals.extend(v for v in results[K][key] if not np.isnan(v))
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05 if hi != lo else 0.1
            row_ylims.append((lo - margin, hi + margin))
        else:
            row_ylims.append((0, 1))

    for col, K in enumerate(K_values):
        data = results[K]
        ts = data['timesteps']
        axes[0][col].set_title(f'K = {K}', fontsize=11)

        for row, (key, label, ylim) in enumerate(zip(metric_keys, METRIC_LABELS, row_ylims)):
            ax = axes[row][col]
            ax.plot(ts, data[key])
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Activity timestep', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_comparison_fine.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_input_comparison(results):
    metric_keys = ['mean_weight', 'num_edges', 'clustering', 'shortest_path', 'activity_variance', 'degree_std']
    patterns = INPUT_PATTERNS
    n_rows = len(metric_keys)
    n_cols = len(patterns)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle('Co-evolution simulation: input pattern comparison (K=20)', fontsize=13, y=1.01)

    # Shared y-limits per row across all patterns
    row_ylims = []
    for key in metric_keys:
        all_vals = []
        for p in patterns:
            all_vals.extend(v for v in results[p][key] if not np.isnan(v))
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05 if hi != lo else 0.1
            row_ylims.append((lo - margin, hi + margin))
        else:
            row_ylims.append((0, 1))

    for col, pattern in enumerate(patterns):
        data = results[pattern]
        ts = data['timesteps']
        axes[0][col].set_title(pattern, fontsize=11)

        for row, (key, label, ylim) in enumerate(zip(metric_keys, METRIC_LABELS, row_ylims)):
            ax = axes[row][col]
            ax.plot(ts, data[key])
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Activity timestep', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_input_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_topology_fingerprint(results):
    patterns = INPUT_PATTERNS
    n_cols = len(patterns)

    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 9), squeeze=False)
    fig.suptitle('Topology fingerprint at T=2000 (K=20)', fontsize=13, y=1.01)

    # Collect global ranges for shared axes
    all_degrees = [d for p in patterns for d in results[p]['fingerprint']['out_degrees']]
    all_weights = [w for p in patterns for w in results[p]['fingerprint']['edge_weights']]
    deg_range = (min(all_degrees), max(all_degrees))
    wgt_range = (min(all_weights) if all_weights else 0, max(all_weights) if all_weights else 1)

    degree_bins = range(deg_range[0], deg_range[1] + 2)

    for col, pattern in enumerate(patterns):
        fp = results[pattern]['fingerprint']
        axes[0][col].set_title(pattern, fontsize=11)

        # Row 0: out-degree histogram
        ax = axes[0][col]
        ax.hist(fp['out_degrees'], bins=degree_bins, align='left', rwidth=0.8)
        ax.set_xlim(deg_range[0] - 0.5, deg_range[1] + 1.5)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Out-degree distribution', fontsize=9)
        ax.set_xlabel('Out-degree', fontsize=8)

        # Row 1: edge weight histogram
        ax = axes[1][col]
        ax.hist(fp['edge_weights'], bins=20, range=wgt_range)
        ax.set_xlim(wgt_range)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Edge weight distribution', fontsize=9)
        ax.set_xlabel('Weight', fontsize=8)

        # Row 2: out-degree vs clustering scatter
        ax = axes[2][col]
        ax.scatter(fp['out_degrees'], fp['clustering_per_node'], alpha=0.7, s=40)
        ax.set_xlim(deg_range[0] - 0.5, deg_range[1] + 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Out-degree vs clustering', fontsize=9)
        ax.set_xlabel('Out-degree', fontsize=8)
        ax.set_ylabel('Clustering coeff.', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_topology_fingerprint.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_encoding_experiment(K=20, T_train=2000, T_silence=500, T_probe=500, N=20, seed=42):
    # If final Euclidean distance > 1.0, topology encodes input history.
    # If final Euclidean distance < 0.1, topology does not encode input history.
    # Values in between are ambiguous.

    def _phase1_train(pattern):
        rng = np.random.default_rng(seed)
        rng_input = np.random.default_rng(seed + 1000)
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(N):
                if i != j and rng.random() < 0.2:
                    G.add_edge(i, j, weight=rng.random())
        activity = rng.random(N)
        activity[0] = _node0_input(pattern, 0, rng_input)

        for t in range(T_train):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            new_activity[0] = _node0_input(pattern, t, rng_input)
            activity = new_activity

            if (t + 1) % K == 0:
                edges_to_remove = []
                for i, j, data in list(G.edges(data=True)):
                    w = data['weight']
                    if activity[i] > 0.5 and activity[j] > 0.5:
                        w += 0.05
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
                            G.add_edge(i, j, weight=0.05)
        return G

    def _phase2_silence(G):
        G = G.copy()
        for t in range(T_silence):
            if (t + 1) % K == 0:
                edges_to_remove = []
                for i, j, data in list(G.edges(data=True)):
                    w = data['weight'] - 0.01
                    if w < 0.01:
                        edges_to_remove.append((i, j))
                    else:
                        G[i][j]['weight'] = w
                G.remove_edges_from(edges_to_remove)
        return G

    def _phase3_probe(G):
        activity = np.zeros(N)
        trajectory = []
        for _ in range(T_probe):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            new_activity[0] = 0.3
            activity = new_activity
            trajectory.append(activity.copy())
        return np.array(trajectory)  # shape (T_probe, N)

    print('Encoding experiment — Phase 1: Training...')
    G_A = _phase1_train('constant')
    G_B = _phase1_train('alternating')

    print('Encoding experiment — Phase 2: Silence...')
    G_A_silent = _phase2_silence(G_A)
    G_B_silent = _phase2_silence(G_B)

    def _topology_stats(G, label):
        n_edges = G.number_of_edges()
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        mean_w = np.mean(weights) if weights else 0.0
        max_edges = N * (N - 1)
        density = n_edges / max_edges
        print(f'{label}: {n_edges} edges, mean weight {mean_w:.4f}, density {density:.4f}')

    _topology_stats(G_A_silent, 'Topology A (after silence)')
    _topology_stats(G_B_silent, 'Topology B (after silence)')

    print('Encoding experiment — Phase 3: Probe...')
    traj_A = _phase3_probe(G_A_silent)  # (T_probe, N)
    traj_B = _phase3_probe(G_B_silent)

    mean_act_A = traj_A.mean(axis=0)       # (N,)
    mean_act_B = traj_B.mean(axis=0)
    per_node_diff = mean_act_A - mean_act_B

    inst_dist = np.linalg.norm(traj_A - traj_B, axis=1)  # (T_probe,)
    cumulative_dist = np.cumsum(inst_dist)
    final_dist = float(cumulative_dist[-1])

    print(f'Final Euclidean distance between A and B probe trajectories: {final_dist:.4f}')

    return {
        'mean_act_A': mean_act_A,
        'mean_act_B': mean_act_B,
        'per_node_diff': per_node_diff,
        'cumulative_dist': cumulative_dist,
        'final_dist': final_dist,
    }


def plot_encoding(enc):
    N = len(enc['mean_act_A'])
    nodes = np.arange(N)
    T_probe = len(enc['cumulative_dist'])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    fig.suptitle('Topology encoding of input history (K=20)', fontsize=13)

    ax = axes[0]
    ax.plot(nodes, enc['mean_act_A'], label='A (constant)', marker='o', ms=4)
    ax.plot(nodes, enc['mean_act_B'], label='B (alternating)', marker='s', ms=4)
    ax.set_ylabel('Mean activity per node')
    ax.set_xlabel('Node index')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(nodes, enc['per_node_diff'])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel('Mean activity difference\n(A minus B)')
    ax.set_xlabel('Node index')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(np.arange(T_probe), enc['cumulative_dist'])
    ax.set_ylabel('Cumulative Euclidean distance\n(A vs B)')
    ax.set_xlabel('Probe timestep')
    ax.set_title(f'Final distance: {enc["final_dist"]:.4f}', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = 'images/results_encoding.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_stdp_comparison(results, patterns):
    metric_keys = ['mean_weight', 'num_edges', 'clustering', 'shortest_path', 'activity_variance', 'degree_std']
    n_rows = len(metric_keys)
    n_cols = len(patterns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.suptitle('STDP simulation: input pattern comparison (K=20)', fontsize=13, y=1.01)

    row_ylims = []
    for key in metric_keys:
        all_vals = []
        for p in patterns:
            all_vals.extend(v for v in results[p][key] if not np.isnan(v))
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05 if hi != lo else 0.1
            row_ylims.append((lo - margin, hi + margin))
        else:
            row_ylims.append((0, 1))

    for col, pattern in enumerate(patterns):
        data = results[pattern]
        ts = data['timesteps']
        axes[0][col].set_title(pattern, fontsize=11)
        for row, (key, label, ylim) in enumerate(zip(metric_keys, METRIC_LABELS, row_ylims)):
            ax = axes[row][col]
            ax.plot(ts, data[key])
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Activity timestep', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_stdp_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_stdp_fingerprint(results, patterns):
    n_cols = len(patterns)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 9), squeeze=False)
    fig.suptitle('STDP topology fingerprint at T=2000 (K=20)', fontsize=13, y=1.01)

    all_degrees = [d for p in patterns for d in results[p]['fingerprint']['out_degrees']]
    all_weights = [w for p in patterns for w in results[p]['fingerprint']['edge_weights']]
    deg_range = (min(all_degrees), max(all_degrees))
    wgt_range = (min(all_weights) if all_weights else 0, max(all_weights) if all_weights else 1)
    degree_bins = range(deg_range[0], deg_range[1] + 2)

    for col, pattern in enumerate(patterns):
        fp = results[pattern]['fingerprint']
        axes[0][col].set_title(pattern, fontsize=11)

        ax = axes[0][col]
        ax.hist(fp['out_degrees'], bins=degree_bins, align='left', rwidth=0.8)
        ax.set_xlim(deg_range[0] - 0.5, deg_range[1] + 1.5)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Out-degree distribution', fontsize=9)
        ax.set_xlabel('Out-degree', fontsize=8)

        ax = axes[1][col]
        ax.hist(fp['edge_weights'], bins=20, range=wgt_range)
        ax.set_xlim(wgt_range)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Edge weight distribution', fontsize=9)
        ax.set_xlabel('Weight', fontsize=8)

        ax = axes[2][col]
        ax.scatter(fp['out_degrees'], fp['clustering_per_node'], alpha=0.7, s=40)
        ax.set_xlim(deg_range[0] - 0.5, deg_range[1] + 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Out-degree vs clustering', fontsize=9)
        ax.set_xlabel('Out-degree', fontsize=8)
        ax.set_ylabel('Clustering coeff.', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_stdp_fingerprint.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_stdp_encoding(enc):
    N = len(enc['mean_act_A'])
    nodes = np.arange(N)
    T_probe = len(enc['cumulative_dist'])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    fig.suptitle('STDP topology encoding of input history (K=20)', fontsize=13)

    ax = axes[0]
    ax.plot(nodes, enc['mean_act_A'], label='A (constant)', marker='o', ms=4)
    ax.plot(nodes, enc['mean_act_B'], label='B (alternating)', marker='s', ms=4)
    ax.set_ylabel('Mean activity per node')
    ax.set_xlabel('Node index')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(nodes, enc['per_node_diff'])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel('Mean activity difference\n(A minus B)')
    ax.set_xlabel('Node index')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(np.arange(T_probe), enc['cumulative_dist'])
    ax.set_ylabel('Cumulative Euclidean distance\n(A vs B)')
    ax.set_xlabel('Probe timestep')
    ax.set_title(f'Final distance: {enc["final_dist"]:.4f}', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = 'images/results_stdp_encoding.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_stdp_encoding_experiment(K=20, T_train=2000, T_silence=500, T_probe=500, N=20, seed=42):
    # If final Euclidean distance > 1.0, topology encodes input history.
    # If final Euclidean distance < 0.1, topology does not encode input history.
    # Values in between are ambiguous.

    def _phase1_train(pattern):
        rng = np.random.default_rng(seed)
        rng_input = np.random.default_rng(seed + 1000)
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(N):
                if i != j and rng.random() < 0.2:
                    G.add_edge(i, j, weight=rng.random())
        activity = rng.random(N)
        activity[0] = _node0_input(pattern, 0, rng_input)
        prev_activity = np.zeros(N)

        mean_weight, num_edges, clustering_list = [], [], []
        shortest_path, activity_variance, degree_std = [], [], []

        for t in range(T_train):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            new_activity[0] = _node0_input(pattern, t, rng_input)
            prev_activity = activity
            activity = new_activity

            if (t + 1) % K == 0:
                edges_to_remove = []
                for i, j, data in list(G.edges(data=True)):
                    w = data['weight']
                    # STDP potentiation: i fired before j
                    if prev_activity[i] > 0.5 and activity[j] > 0.5 and prev_activity[j] <= 0.5:
                        w += 0.05
                    # STDP depression: j fired before i
                    elif prev_activity[j] > 0.5 and activity[i] > 0.5 and prev_activity[i] <= 0.5:
                        w -= 0.03
                    # Decay
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
                            G.add_edge(i, j, weight=0.05)

                weights = [d['weight'] for _, _, d in G.edges(data=True)]
                mean_weight.append(np.mean(weights) if weights else 0.0)
                num_edges.append(G.number_of_edges())
                clustering_list.append(nx.average_clustering(G))

                wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
                if wccs:
                    lcc = G.subgraph(wccs[0]).copy()
                    if lcc.number_of_nodes() > 1:
                        try:
                            shortest_path.append(nx.average_shortest_path_length(lcc))
                        except nx.NetworkXError:
                            shortest_path.append(float('nan'))
                    else:
                        shortest_path.append(float('nan'))
                else:
                    shortest_path.append(float('nan'))

                activity_variance.append(float(np.var(activity)))
                out_degrees = [d for _, d in G.out_degree()]
                degree_std.append(float(np.std(out_degrees)))

        n_topo_steps = T_train // K
        timesteps = list(np.arange(1, n_topo_steps + 1) * K)
        metrics = {
            'mean_weight': mean_weight,
            'num_edges': num_edges,
            'clustering': clustering_list,
            'shortest_path': shortest_path,
            'activity_variance': activity_variance,
            'degree_std': degree_std,
            'timesteps': timesteps,
            'fingerprint': {
                'out_degrees': [d for _, d in G.out_degree()],
                'edge_weights': [d['weight'] for _, _, d in G.edges(data=True)],
                'clustering_per_node': list(nx.clustering(G).values()),
            },
        }
        return G, metrics

    def _phase2_silence(G):
        G = G.copy()
        for t in range(T_silence):
            if (t + 1) % K == 0:
                edges_to_remove = []
                for i, j, data in list(G.edges(data=True)):
                    w = data['weight'] - 0.01
                    if w < 0.01:
                        edges_to_remove.append((i, j))
                    else:
                        G[i][j]['weight'] = w
                G.remove_edges_from(edges_to_remove)
        return G

    def _phase3_probe(G):
        activity = np.zeros(N)
        trajectory = []
        for _ in range(T_probe):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            new_activity[0] = 0.3
            activity = new_activity
            trajectory.append(activity.copy())
        return np.array(trajectory)

    def _topology_stats(G, label):
        n_edges = G.number_of_edges()
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        mean_w = np.mean(weights) if weights else 0.0
        density = n_edges / (N * (N - 1))
        print(f'{label}: {n_edges} edges, mean weight {mean_w:.4f}, density {density:.4f}')

    print('STDP Encoding experiment — Phase 1: Training...')
    G_A, metrics_A = _phase1_train('constant')
    G_B, metrics_B = _phase1_train('alternating')

    patterns = ['constant', 'alternating']
    plot_stdp_comparison({'constant': metrics_A, 'alternating': metrics_B}, patterns)
    plot_stdp_fingerprint({'constant': metrics_A, 'alternating': metrics_B}, patterns)

    print('STDP Encoding experiment — Phase 2: Silence...')
    G_A_silent = _phase2_silence(G_A)
    G_B_silent = _phase2_silence(G_B)

    _topology_stats(G_A_silent, 'STDP Topology A (after silence)')
    _topology_stats(G_B_silent, 'STDP Topology B (after silence)')

    print('STDP Encoding experiment — Phase 3: Probe...')
    traj_A = _phase3_probe(G_A_silent)
    traj_B = _phase3_probe(G_B_silent)

    mean_act_A = traj_A.mean(axis=0)
    mean_act_B = traj_B.mean(axis=0)
    per_node_diff = mean_act_A - mean_act_B

    inst_dist = np.linalg.norm(traj_A - traj_B, axis=1)
    cumulative_dist = np.cumsum(inst_dist)
    final_dist = float(cumulative_dist[-1])

    print(f'STDP Final Euclidean distance: {final_dist:.4f}')

    enc = {
        'mean_act_A': mean_act_A,
        'mean_act_B': mean_act_B,
        'per_node_diff': per_node_diff,
        'cumulative_dist': cumulative_dist,
        'final_dist': final_dist,
    }
    plot_stdp_encoding(enc)
    return enc


def run_delta_comparison(K_values, T=2000, seed=42) -> dict:
    results = {}
    for K in K_values:
        print(f'Running delta comparison K={K}...')
        results[K] = run_simulation(K=K, T=T, seed=seed)
    return results


def plot_delta_comparison(results, K_values):
    n_cols = len(K_values)

    # Shared y-limits per row
    def _ylim(vals_by_K):
        all_vals = [v for K in K_values for v in vals_by_K(K) if not np.isnan(v)]
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - margin, hi + margin)

    ylim_delta  = _ylim(lambda K: results[K]['topology_delta'])
    ylim_cumsum = _ylim(lambda K: np.cumsum(results[K]['topology_delta']).tolist())
    ylim_var    = _ylim(lambda K: results[K]['activity_variance'])
    row_ylims   = [ylim_delta, ylim_cumsum, ylim_var]
    row_labels  = ['Topology delta\n(per update)', 'Cumulative\ntopology delta', 'Activity variance']

    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 8), squeeze=False)
    fig.suptitle('Topology change magnitude vs K', fontsize=13, y=1.01)

    for col, K in enumerate(K_values):
        data = results[K]
        ts   = data['timesteps']
        axes[0][col].set_title(f'K = {K}', fontsize=11)

        axes[0][col].plot(ts, data['topology_delta'])
        axes[1][col].plot(ts, np.cumsum(data['topology_delta']))
        axes[2][col].plot(ts, data['activity_variance'])

        for row in range(3):
            ax = axes[row][col]
            ax.set_ylim(row_ylims[row])
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 2:
                ax.set_xlabel('Activity timestep', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_delta_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_convergence_test(T=10000, seed=42):
    K_values = [1, 50]
    results = {}
    for K in K_values:
        print(f'Running convergence test K={K}...')
        results[K] = run_simulation(K=K, T=T, seed=seed)

    for K in K_values:
        data = results[K]
        last_100_delta = data['topology_delta'][-100:]
        mean_last_delta = float(np.mean(last_100_delta)) if last_100_delta else 0.0
        weights = data['mean_weight']
        n_edges_final = data['num_edges'][-1]
        mean_w_final = weights[-1]
        density_final = n_edges_final / (20 * 19)
        clustering_final = data['clustering'][-1]
        print(
            f'K={K} final state:\n'
            f'  edges: {n_edges_final}\n'
            f'  mean weight: {mean_w_final:.4f}\n'
            f'  density: {density_final:.4f}\n'
            f'  clustering coefficient: {clustering_final:.4f}\n'
            f'  topology delta (last 100 updates, mean): {mean_last_delta:.4f}'
        )

    return results


def plot_convergence_test(results):
    K_values = [1, 50]
    row_keys   = ['topology_delta', None, 'num_edges', 'clustering']
    row_labels = ['Topology delta\n(per update)', 'Cumulative\ntopology delta',
                  'Number of edges', 'Mean clustering coeff.']

    def _ylim(series_list):
        all_vals = [v for s in series_list for v in s if not np.isnan(v)]
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - margin, hi + margin)

    row_ylims = [
        _ylim([results[K]['topology_delta'] for K in K_values]),
        _ylim([np.cumsum(results[K]['topology_delta']).tolist() for K in K_values]),
        _ylim([results[K]['num_edges'] for K in K_values]),
        _ylim([results[K]['clustering'] for K in K_values]),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), squeeze=False)
    fig.suptitle('Convergence test: K=1 vs K=50 (T=10000)', fontsize=13)

    for col, K in enumerate(K_values):
        data = results[K]
        ts = data['timesteps']
        axes[0][col].set_title(f'K = {K}', fontsize=11)

        axes[0][col].plot(ts, data['topology_delta'])
        axes[1][col].plot(ts, np.cumsum(data['topology_delta']))
        axes[2][col].plot(ts, data['num_edges'])
        axes[3][col].plot(ts, data['clustering'])

        for row in range(4):
            ax = axes[row][col]
            ax.set_ylim(row_ylims[row])
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 3:
                ax.set_xlabel('Activity timestep', fontsize=8)

    plt.tight_layout()
    fname = 'images/results_convergence.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_cycle_analysis(K=1, T=10000, N=20, seed=42):
    MAX_CYCLES = 10000
    SNAPSHOT_TIMES = {100, 500, 1000, 2000, 5000, 10000}

    rng = np.random.default_rng(seed)
    rng_input = np.random.default_rng(seed + 1000)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = rng.random(N)
    activity[0] = _node0_input('constant', 0, rng_input)

    snapshots = {}

    for t in range(T):
        # Activity update
        new_activity = np.zeros(N)
        for i in range(N):
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_activity[i] = np.tanh(s)
        new_activity[0] = _node0_input('constant', t, rng_input)
        activity = new_activity

        # Topology update every K steps
        if (t + 1) % K == 0:
            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                w = data['weight']
                if activity[i] > 0.5 and activity[j] > 0.5:
                    w += 0.05
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
                        G.add_edge(i, j, weight=0.05)

        # Snapshot at specified timepoints (after all updates at step t+1)
        step = t + 1
        if step in SNAPSHOT_TIMES:
            cycles_sample = list(itertools.islice(nx.simple_cycles(G), MAX_CYCLES + 1))
            if len(cycles_sample) > MAX_CYCLES:
                total_cycles = '>10000'
                cycle_dist = None
            else:
                total_cycles = len(cycles_sample)
                cycle_dist = {}
                for c in cycles_sample:
                    cycle_dist[len(c)] = cycle_dist.get(len(c), 0) + 1

            sccs = list(nx.strongly_connected_components(G))
            large_sccs = [s for s in sccs if len(s) > 1]
            largest_scc = max((len(s) for s in sccs), default=0)

            snapshots[step] = {
                'total_cycles': total_cycles,
                'cycle_distribution': cycle_dist,
                'scc_count': len(large_sccs),
                'largest_scc': largest_scc,
            }

            dist_str = str(dict(sorted(cycle_dist.items()))) if cycle_dist is not None else 'N/A (>10000)'
            print(
                f'T={step}:\n'
                f'  total cycles: {total_cycles}\n'
                f'  cycle length distribution: {dist_str}\n'
                f'  strongly connected components (size > 1): {len(large_sccs)}\n'
                f'  largest SCC size: {largest_scc}'
            )

    return snapshots


def plot_cycle_analysis(snapshots):
    MAX_CYCLES = 10000
    timepoints = sorted(snapshots.keys())
    x = np.arange(len(timepoints))
    x_labels = [f'T={t}' for t in timepoints]

    # Numeric total_cycles for plotting; use MAX_CYCLES+1 as sentinel for overflow
    total_plot = []
    overflow = []
    for t in timepoints:
        tc = snapshots[t]['total_cycles']
        if isinstance(tc, str):
            total_plot.append(MAX_CYCLES + 1)
            overflow.append(True)
        else:
            total_plot.append(tc)
            overflow.append(False)

    largest_scc = [snapshots[t]['largest_scc'] for t in timepoints]

    # Collect all cycle lengths present across non-overflow snapshots
    all_lengths = sorted({
        length
        for t in timepoints
        if snapshots[t]['cycle_distribution'] is not None
        for length in snapshots[t]['cycle_distribution']
    })

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    fig.suptitle('Directed cycle analysis: K=1, T=10000', fontsize=13)

    # Row 1: total cycle count
    ax = axes[0]
    bar_colors = ['tomato' if ov else 'steelblue' for ov in overflow]
    ax.bar(x, total_plot, color=bar_colors)
    for i, ov in enumerate(overflow):
        if ov:
            ax.text(x[i], total_plot[i] * 0.98, '>10000', ha='center', va='top',
                    fontsize=8, color='white', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('Total cycles')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 2: stacked bar by cycle length (overflow timepoints show empty bar)
    ax = axes[1]
    colors = plt.cm.tab10.colors
    bottom = np.zeros(len(timepoints))
    for li, length in enumerate(all_lengths):
        counts = np.array([
            (snapshots[t]['cycle_distribution'].get(length, 0)
             if snapshots[t]['cycle_distribution'] is not None else 0)
            for t in timepoints
        ], dtype=float)
        ax.bar(x, counts, bottom=bottom, label=f'len {length}',
               color=colors[li % len(colors)])
        bottom += counts
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('Cycle count by length')
    if all_lengths:
        ax.legend(fontsize=8, loc='upper left', ncol=2)
    else:
        ax.text(0.5, 0.5, 'All timepoints exceeded\ncycle count limit (>10000)',
                transform=ax.transAxes, ha='center', va='center', fontsize=9,
                color='gray')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3: largest SCC size
    ax = axes[2]
    ax.plot(timepoints, largest_scc, marker='o')
    ax.set_ylabel('Largest SCC size')
    ax.set_xlabel('Timestep')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = 'images/results_cycle_analysis.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def _scc_snapshots(K, T, N, seed, snapshot_times):
    """Run rate-based simulation quietly; return SCC and cycle data at each snapshot time."""
    MAX_CYCLES = 10000
    rng = np.random.default_rng(seed)
    rng_input = np.random.default_rng(seed + 1000)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())
    activity = rng.random(N)
    activity[0] = 0.5
    snapshots = {}
    for t in range(T):
        new_activity = np.zeros(N)
        for i in range(N):
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_activity[i] = np.tanh(s)
        new_activity[0] = 0.5
        activity = new_activity
        if (t + 1) % K == 0:
            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                w = data['weight']
                if activity[i] > 0.5 and activity[j] > 0.5:
                    w += 0.05
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
                        G.add_edge(i, j, weight=0.05)
        step = t + 1
        if step in snapshot_times:
            cycles_sample = list(itertools.islice(nx.simple_cycles(G), MAX_CYCLES + 1))
            total_cycles = '>10000' if len(cycles_sample) > MAX_CYCLES else len(cycles_sample)
            sccs = list(nx.strongly_connected_components(G))
            largest_scc = max((len(s) for s in sccs), default=0)
            snapshots[step] = {'largest_scc': largest_scc, 'total_cycles': total_cycles}
    return snapshots


def run_ei_simulation(K=1, T=5000, N=20, seed=42, ei_threshold=0.7, ei_window=100,
                      inhibition_strength=0.3):
    MAX_CYCLES = 10000
    SNAPSHOT_TIMES = {500, 1000, 2000, 5000}

    rng = np.random.default_rng(seed)
    rng_input = np.random.default_rng(seed + 1000)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = rng.random(N)
    activity[0] = 0.5
    node_type = np.ones(N, dtype=float)       # +1 excitatory, -1 inhibitory
    type_strength = np.ones(N, dtype=float)   # +1.0 excitatory, -inhibition_strength inhibitory
    activity_histories = [deque(maxlen=ei_window) for _ in range(N)]

    mean_weight, num_edges, clustering_list = [], [], []
    shortest_path, activity_variance, activity_mean_list = [], [], []
    degree_std, topology_delta = [], []
    num_excitatory, num_inhibitory = [], []
    mean_act_excitatory, mean_act_inhibitory = [], []
    timesteps = []
    cycle_snapshots = {}

    for t in range(T):
        # Activity update with signed type influence
        new_activity = np.zeros(N)
        for i in range(N):
            influence = sum(
                G[j][i]['weight'] * activity[j] * type_strength[j]
                for j in G.predecessors(i)
            )
            new_activity[i] = np.tanh(max(0.0, influence))
        new_activity[0] = 0.5
        activity = new_activity

        if (t + 1) % K == 0:
            weights_before = {(i, j): d['weight'] for i, j, d in G.edges(data=True)}

            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                w = data['weight']
                if activity[i] > 0.5 and activity[j] > 0.5:
                    w += 0.05
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
                        G.add_edge(i, j, weight=0.05)

            # Topology delta
            eb = set(weights_before.keys())
            ea = set(G.edges())
            delta = (
                sum(abs(G[e[0]][e[1]]['weight'] - weights_before[e]) for e in eb & ea)
                + sum(weights_before[e] for e in eb - ea)
                + sum(G[e[0]][e[1]]['weight'] for e in ea - eb)
            )
            topology_delta.append(delta)

            # Record activity history
            for i in range(N):
                activity_histories[i].append(activity[i])

            # E/I switching: at most 1 node per step in each direction
            recent_means = [float(np.mean(activity_histories[i])) for i in range(N)]
            exc_candidates = [i for i in range(N)
                              if node_type[i] == 1 and recent_means[i] > ei_threshold]
            inh_candidates = [i for i in range(N)
                              if node_type[i] == -1 and recent_means[i] < ei_threshold * 0.5]
            if exc_candidates:
                switch_i = max(exc_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = -1
                type_strength[switch_i] = -inhibition_strength
            if inh_candidates:
                switch_i = min(inh_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = 1
                type_strength[switch_i] = 1.0

            # Standard metrics
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            mean_weight.append(np.mean(weights) if weights else 0.0)
            num_edges.append(G.number_of_edges())
            clustering_list.append(nx.average_clustering(G))

            wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            if wccs:
                lcc = G.subgraph(wccs[0]).copy()
                if lcc.number_of_nodes() > 1:
                    try:
                        shortest_path.append(nx.average_shortest_path_length(lcc))
                    except nx.NetworkXError:
                        shortest_path.append(float('nan'))
                else:
                    shortest_path.append(float('nan'))
            else:
                shortest_path.append(float('nan'))

            activity_variance.append(float(np.var(activity)))
            activity_mean_list.append(float(np.mean(activity)))
            out_degrees = [d for _, d in G.out_degree()]
            degree_std.append(float(np.std(out_degrees)))

            # E/I specific metrics
            exc_mask = node_type == 1
            inh_mask = node_type == -1
            num_excitatory.append(int(np.sum(exc_mask)))
            num_inhibitory.append(int(np.sum(inh_mask)))
            mean_act_excitatory.append(float(np.mean(activity[exc_mask])) if exc_mask.any() else 0.0)
            mean_act_inhibitory.append(float(np.mean(activity[inh_mask])) if inh_mask.any() else 0.0)
            timesteps.append(t + 1)

        # Cycle snapshot (taken after all updates at this step)
        step = t + 1
        if step in SNAPSHOT_TIMES:
            cycles_sample = list(itertools.islice(nx.simple_cycles(G), MAX_CYCLES + 1))
            if len(cycles_sample) > MAX_CYCLES:
                total_cycles = '>10000'
                cycle_dist = None
            else:
                total_cycles = len(cycles_sample)
                cycle_dist = {}
                for c in cycles_sample:
                    cycle_dist[len(c)] = cycle_dist.get(len(c), 0) + 1

            sccs = list(nx.strongly_connected_components(G))
            large_sccs = [s for s in sccs if len(s) > 1]
            largest_scc = max((len(s) for s in sccs), default=0)
            cycle_snapshots[step] = {
                'total_cycles': total_cycles,
                'cycle_distribution': cycle_dist,
                'scc_count': len(large_sccs),
                'largest_scc': largest_scc,
            }
            dist_str = str(dict(sorted(cycle_dist.items()))) if cycle_dist is not None else 'N/A (>10000)'
            print(
                f'T={step}:\n'
                f'  total cycles: {total_cycles}\n'
                f'  cycle length distribution: {dist_str}\n'
                f'  strongly connected components (size > 1): {len(large_sccs)}\n'
                f'  largest SCC size: {largest_scc}'
            )

    return {
        'mean_weight': mean_weight,
        'num_edges': num_edges,
        'clustering': clustering_list,
        'shortest_path': shortest_path,
        'activity_variance': activity_variance,
        'activity_mean': activity_mean_list,
        'degree_std': degree_std,
        'topology_delta': topology_delta,
        'num_excitatory': num_excitatory,
        'num_inhibitory': num_inhibitory,
        'mean_act_excitatory': mean_act_excitatory,
        'mean_act_inhibitory': mean_act_inhibitory,
        'timesteps': timesteps,
        'cycle_snapshots': cycle_snapshots,
    }


def plot_ei_comparison(orig, orig_snapshots, ei):
    N = 20
    orig_ts = orig['timesteps']
    ei_ts   = ei['timesteps']

    def _ylim(a, b):
        vals = [v for v in list(a) + list(b) if not np.isnan(float(v))]
        if not vals:
            return (0, 1)
        lo, hi = min(vals), max(vals)
        m = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - m, hi + m)

    snapshot_times = sorted(orig_snapshots.keys())
    orig_scc = [orig_snapshots[t]['largest_scc'] for t in snapshot_times]
    ei_scc   = [ei['cycle_snapshots'][t]['largest_scc'] for t in snapshot_times]

    fig, axes = plt.subplots(4, 2, figsize=(11, 14), squeeze=False)
    fig.suptitle('E/I dynamics comparison: original vs E/I switching (K=1, T=5000)', fontsize=13)
    axes[0][0].set_title('Original (rate-based)', fontsize=11)
    axes[0][1].set_title('E/I switching', fontsize=11)

    # Row 1: number of edges
    yl = _ylim(orig['num_edges'], ei['num_edges'])
    for col, (ts, data) in enumerate([(orig_ts, orig), (ei_ts, ei)]):
        axes[0][col].plot(ts, data['num_edges'])
        axes[0][col].set_ylim(yl)
        axes[0][col].set_ylabel('Number of edges', fontsize=9)
        axes[0][col].grid(True, alpha=0.3)

    # Row 2: activity variance
    yl = _ylim(orig['activity_variance'], ei['activity_variance'])
    for col, (ts, data) in enumerate([(orig_ts, orig), (ei_ts, ei)]):
        axes[1][col].plot(ts, data['activity_variance'])
        axes[1][col].set_ylim(yl)
        axes[1][col].set_ylabel('Activity variance', fontsize=9)
        axes[1][col].grid(True, alpha=0.3)

    # Row 3: E/I node counts (stacked area)
    orig_exc = [N] * len(orig_ts)
    orig_inh = [0] * len(orig_ts)
    axes[2][0].stackplot(orig_ts, orig_exc, orig_inh,
                         labels=['Excitatory', 'Inhibitory'],
                         colors=['steelblue', 'tomato'], alpha=0.8)
    axes[2][1].stackplot(ei_ts, ei['num_excitatory'], ei['num_inhibitory'],
                         labels=['Excitatory', 'Inhibitory'],
                         colors=['steelblue', 'tomato'], alpha=0.8)
    for col in range(2):
        axes[2][col].set_ylim(0, N + 1)
        axes[2][col].set_ylabel('Node count', fontsize=9)
        axes[2][col].legend(fontsize=8, loc='upper right')
        axes[2][col].grid(True, alpha=0.3)

    # Row 4: largest SCC at snapshot times (scatter)
    yl = _ylim(orig_scc, ei_scc)
    for col, scc_vals in enumerate([orig_scc, ei_scc]):
        ax = axes[3][col]
        ax.scatter(snapshot_times, scc_vals, marker='o', zorder=3)
        ax.plot(snapshot_times, scc_vals, linestyle='--', alpha=0.5)
        ax.set_ylim(yl)
        ax.set_ylabel('Largest SCC size', fontsize=9)
        ax.set_xlabel('Activity timestep', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = 'images/results_ei_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_ei_comparison():
    N = 20
    SNAPSHOT_TIMES = {500, 1000, 2000, 5000}

    print('Running original simulation (K=1, T=5000)...')
    orig = run_simulation(K=1, T=5000, seed=42)

    print('Computing original SCC/cycle snapshots...')
    orig_snapshots = _scc_snapshots(K=1, T=5000, N=N, seed=42, snapshot_times=SNAPSHOT_TIMES)

    print('Running E/I simulation (K=1, T=5000)...')
    ei = run_ei_simulation(K=1, T=5000, N=N, seed=42, ei_threshold=0.7, ei_window=10)

    orig_cycles_final = orig_snapshots[5000]['total_cycles']
    ei_cycles_final   = ei['cycle_snapshots'][5000]['total_cycles']

    print(
        f"Original final state: edges={orig['num_edges'][-1]}, "
        f"clustering={orig['clustering'][-1]:.4f}, "
        f"activity_variance={orig['activity_variance'][-1]:.4f}, "
        f"cycles={orig_cycles_final}"
    )
    print(
        f"E/I final state: edges={ei['num_edges'][-1]}, "
        f"clustering={ei['clustering'][-1]:.4f}, "
        f"activity_variance={ei['activity_variance'][-1]:.4f}, "
        f"cycles={ei_cycles_final}"
    )
    print(
        f"E/I node type history: "
        f"min_excitatory={min(ei['num_excitatory'])}, "
        f"max_inhibitory={max(ei['num_inhibitory'])}"
    )

    plot_ei_comparison(orig, orig_snapshots, ei)
    return orig, orig_snapshots, ei


def run_inhibition_sweep():
    conditions = [
        {'inhibition_strength': 0.1, 'ei_threshold': 0.7},
        {'inhibition_strength': 0.3, 'ei_threshold': 0.7},
        {'inhibition_strength': 0.5, 'ei_threshold': 0.7},
        {'inhibition_strength': 1.0, 'ei_threshold': 0.7},
    ]
    results = []
    for cond in conditions:
        inh = cond['inhibition_strength']
        print(f"Running E/I simulation: inhibition_strength={inh}...")
        data = run_ei_simulation(K=1, T=5000, N=20, seed=42,
                                 ei_threshold=cond['ei_threshold'],
                                 ei_window=10,
                                 inhibition_strength=inh)
        results.append((cond, data))

        final_cycles = data['cycle_snapshots'][5000]['total_cycles']
        print(
            f"inhibition_strength={inh}:\n"
            f"  final edges: {data['num_edges'][-1]}\n"
            f"  final activity_variance: {data['activity_variance'][-1]:.4f}\n"
            f"  final clustering: {data['clustering'][-1]:.4f}\n"
            f"  min_excitatory: {min(data['num_excitatory'])}, "
            f"max_inhibitory: {max(data['num_inhibitory'])}\n"
            f"  final cycles: {final_cycles}"
        )
    return results


def plot_inhibition_sweep(results):
    N = 20
    SNAPSHOT_TIMES = [500, 1000, 2000, 5000]
    n_cols = len(results)

    def _row_ylim(key, transform=None):
        all_vals = []
        for _, data in results:
            vals = transform(data[key]) if transform else data[key]
            all_vals.extend(v for v in vals if not np.isnan(float(v)))
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        m = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - m, hi + m)

    yl_edges   = _row_ylim('num_edges')
    yl_var     = _row_ylim('activity_variance')
    yl_scc     = (0, N + 1)

    # SCC y-limit from cycle snapshots
    scc_all = [
        data['cycle_snapshots'][t]['largest_scc']
        for _, data in results
        for t in SNAPSHOT_TIMES
    ]
    yl_scc_scatter = (0, max(scc_all) + 1) if scc_all else (0, N + 1)

    fig, axes = plt.subplots(4, n_cols, figsize=(3.5 * n_cols, 14), squeeze=False)
    fig.suptitle('E/I inhibition strength sweep (K=1, T=5000)', fontsize=13, y=1.01)

    row_labels = [
        'Number of edges',
        'Activity variance',
        'Node count (E/I)',
        'Largest SCC size',
    ]

    for col, (cond, data) in enumerate(results):
        inh = cond['inhibition_strength']
        ts  = data['timesteps']
        axes[0][col].set_title(f'inh={inh}', fontsize=11)

        # Row 1: edges
        axes[0][col].plot(ts, data['num_edges'])
        axes[0][col].set_ylim(yl_edges)
        axes[0][col].grid(True, alpha=0.3)

        # Row 2: activity variance
        axes[1][col].plot(ts, data['activity_variance'])
        axes[1][col].set_ylim(yl_var)
        axes[1][col].grid(True, alpha=0.3)

        # Row 3: E/I stacked area
        axes[2][col].stackplot(ts, data['num_excitatory'], data['num_inhibitory'],
                               labels=['Excitatory', 'Inhibitory'],
                               colors=['steelblue', 'tomato'], alpha=0.8)
        axes[2][col].set_ylim(0, N + 1)
        axes[2][col].grid(True, alpha=0.3)
        if col == n_cols - 1:
            axes[2][col].legend(fontsize=7, loc='upper right')

        # Row 4: largest SCC scatter
        scc_vals = [data['cycle_snapshots'][t]['largest_scc'] for t in SNAPSHOT_TIMES]
        axes[3][col].scatter(SNAPSHOT_TIMES, scc_vals, marker='o', zorder=3)
        axes[3][col].plot(SNAPSHOT_TIMES, scc_vals, linestyle='--', alpha=0.5)
        axes[3][col].set_ylim(yl_scc_scatter)
        axes[3][col].set_xlabel('Activity timestep', fontsize=8)
        axes[3][col].grid(True, alpha=0.3)

        for row in range(4):
            if col == 0:
                axes[row][col].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = 'images/results_ei_inhibition.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_threshold_sweep():
    conditions = [
        {'ei_threshold': 0.6, 'ei_window': 100},
        {'ei_threshold': 0.7, 'ei_window': 100},
        {'ei_threshold': 0.8, 'ei_window': 100},
        {'ei_threshold': 0.9, 'ei_window': 100},
    ]
    results = []
    for cond in conditions:
        thr = cond['ei_threshold']
        print(f"Running E/I simulation: ei_threshold={thr}...")
        data = run_ei_simulation(K=1, T=5000, N=20, seed=42,
                                 ei_threshold=thr,
                                 ei_window=cond['ei_window'],
                                 inhibition_strength=0.3)
        results.append((cond, data))

        final_cycles = data['cycle_snapshots'][5000]['total_cycles']
        print(
            f"inhibition_strength=0.3, ei_threshold={thr}:\n"
            f"  final edges: {data['num_edges'][-1]}\n"
            f"  final activity_variance: {data['activity_variance'][-1]:.4f}\n"
            f"  final clustering: {data['clustering'][-1]:.4f}\n"
            f"  min_excitatory: {min(data['num_excitatory'])}, "
            f"max_inhibitory: {max(data['num_inhibitory'])}\n"
            f"  final cycles: {final_cycles}"
        )
    return results


def plot_threshold_sweep(results):
    N = 20
    SNAPSHOT_TIMES = [500, 1000, 2000, 5000]
    n_cols = len(results)

    def _row_ylim(key):
        all_vals = [v for _, data in results for v in data[key] if not np.isnan(float(v))]
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        m = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - m, hi + m)

    yl_edges = _row_ylim('num_edges')
    yl_var   = _row_ylim('activity_variance')

    scc_all = [data['cycle_snapshots'][t]['largest_scc']
               for _, data in results for t in SNAPSHOT_TIMES]
    yl_scc = (0, max(scc_all) + 1) if scc_all else (0, N + 1)

    row_labels = ['Number of edges', 'Activity variance', 'Node count (E/I)', 'Largest SCC size']

    fig, axes = plt.subplots(4, n_cols, figsize=(3.5 * n_cols, 14), squeeze=False)
    fig.suptitle('E/I threshold sweep (inh=0.3, ei_window=100, K=1, T=5000)', fontsize=13, y=1.01)

    for col, (cond, data) in enumerate(results):
        thr = cond['ei_threshold']
        ts  = data['timesteps']
        axes[0][col].set_title(f'thr={thr}', fontsize=11)

        axes[0][col].plot(ts, data['num_edges'])
        axes[0][col].set_ylim(yl_edges)
        axes[0][col].grid(True, alpha=0.3)

        axes[1][col].plot(ts, data['activity_variance'])
        axes[1][col].set_ylim(yl_var)
        axes[1][col].grid(True, alpha=0.3)

        axes[2][col].stackplot(ts, data['num_excitatory'], data['num_inhibitory'],
                               labels=['Excitatory', 'Inhibitory'],
                               colors=['steelblue', 'tomato'], alpha=0.8)
        axes[2][col].set_ylim(0, N + 1)
        axes[2][col].grid(True, alpha=0.3)
        if col == n_cols - 1:
            axes[2][col].legend(fontsize=7, loc='upper right')

        scc_vals = [data['cycle_snapshots'][t]['largest_scc'] for t in SNAPSHOT_TIMES]
        axes[3][col].scatter(SNAPSHOT_TIMES, scc_vals, marker='o', zorder=3)
        axes[3][col].plot(SNAPSHOT_TIMES, scc_vals, linestyle='--', alpha=0.5)
        axes[3][col].set_ylim(yl_scc)
        axes[3][col].set_xlabel('Activity timestep', fontsize=8)
        axes[3][col].grid(True, alpha=0.3)

        for row in range(4):
            if col == 0:
                axes[row][col].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = 'images/results_ei_threshold.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_ei_silencing_simulation(K=1, T=5000, N=20, seed=42, ei_threshold=0.7, ei_window=100):
    MAX_CYCLES = 10000
    SNAPSHOT_TIMES = {500, 1000, 2000, 5000}

    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = rng.random(N)
    activity[0] = 0.5
    node_type = np.ones(N, dtype=float)  # +1 excitatory, -1 inhibitory
    activity_histories = [deque(maxlen=ei_window) for _ in range(N)]

    mean_weight, num_edges_active, num_edges_silenced = [], [], []
    clustering_list, shortest_path = [], []
    activity_variance, activity_mean_list = [], []
    degree_std, topology_delta = [], []
    num_excitatory, num_inhibitory = [], []
    mean_act_excitatory, mean_act_inhibitory = [], []
    timesteps = []
    cycle_snapshots = {}

    for t in range(T):
        # Activity propagation: plain tanh, no type_strength
        new_activity = np.zeros(N)
        for i in range(N):
            influence = sum(
                G[j][i]['weight'] * activity[j]
                for j in G.predecessors(i)
            )
            new_activity[i] = np.tanh(influence)
        new_activity[0] = 0.5
        activity = new_activity

        if (t + 1) % K == 0:
            weights_before = {(i, j): d['weight'] for i, j, d in G.edges(data=True)}

            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                if node_type[i] == 1 and node_type[j] == 1:
                    # Active edge: apply strengthening, decay, deletion
                    w = data['weight']
                    if activity[i] > 0.5 and activity[j] > 0.5:
                        w += 0.05
                    w -= 0.01
                    if w < 0.01:
                        edges_to_remove.append((i, j))
                    else:
                        G[i][j]['weight'] = min(w, 1.0)
                # Silenced edge: weight unchanged, no deletion
            G.remove_edges_from(edges_to_remove)

            # Addition only between two excitatory nodes
            existing = set(G.edges())
            for i in range(N):
                for j in range(N):
                    if i != j and node_type[i] == 1 and node_type[j] == 1:
                        if (i, j) not in existing and rng.random() < 0.01:
                            G.add_edge(i, j, weight=0.05)

            # Topology delta
            eb = set(weights_before.keys())
            ea = set(G.edges())
            delta = (
                sum(abs(G[e[0]][e[1]]['weight'] - weights_before[e]) for e in eb & ea)
                + sum(weights_before[e] for e in eb - ea)
                + sum(G[e[0]][e[1]]['weight'] for e in ea - eb)
            )
            topology_delta.append(delta)

            # Record activity history
            for i in range(N):
                activity_histories[i].append(activity[i])

            # E/I switching: at most 1 node per step in each direction
            recent_means = [float(np.mean(activity_histories[i])) for i in range(N)]
            exc_candidates = [i for i in range(N)
                              if node_type[i] == 1 and recent_means[i] > ei_threshold]
            inh_candidates = [i for i in range(N)
                              if node_type[i] == -1 and recent_means[i] < ei_threshold * 0.5]
            if exc_candidates:
                switch_i = max(exc_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = -1
            if inh_candidates:
                switch_i = min(inh_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = 1

            # Count active (exc-exc) vs silenced edges
            n_active = sum(
                1 for i, j in G.edges()
                if node_type[i] == 1 and node_type[j] == 1
            )
            n_silenced = G.number_of_edges() - n_active

            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            mean_weight.append(np.mean(weights) if weights else 0.0)
            num_edges_active.append(n_active)
            num_edges_silenced.append(n_silenced)
            clustering_list.append(nx.average_clustering(G))

            wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            if wccs:
                lcc = G.subgraph(wccs[0]).copy()
                if lcc.number_of_nodes() > 1:
                    try:
                        shortest_path.append(nx.average_shortest_path_length(lcc))
                    except nx.NetworkXError:
                        shortest_path.append(float('nan'))
                else:
                    shortest_path.append(float('nan'))
            else:
                shortest_path.append(float('nan'))

            activity_variance.append(float(np.var(activity)))
            activity_mean_list.append(float(np.mean(activity)))
            out_degrees = [d for _, d in G.out_degree()]
            degree_std.append(float(np.std(out_degrees)))

            exc_mask = node_type == 1
            inh_mask = node_type == -1
            num_excitatory.append(int(np.sum(exc_mask)))
            num_inhibitory.append(int(np.sum(inh_mask)))
            mean_act_excitatory.append(float(np.mean(activity[exc_mask])) if exc_mask.any() else 0.0)
            mean_act_inhibitory.append(float(np.mean(activity[inh_mask])) if inh_mask.any() else 0.0)
            timesteps.append(t + 1)

        step = t + 1
        if step in SNAPSHOT_TIMES:
            cycles_sample = list(itertools.islice(nx.simple_cycles(G), MAX_CYCLES + 1))
            if len(cycles_sample) > MAX_CYCLES:
                total_cycles = '>10000'
                cycle_dist = None
            else:
                total_cycles = len(cycles_sample)
                cycle_dist = {}
                for c in cycles_sample:
                    cycle_dist[len(c)] = cycle_dist.get(len(c), 0) + 1

            sccs = list(nx.strongly_connected_components(G))
            large_sccs = [s for s in sccs if len(s) > 1]
            largest_scc = max((len(s) for s in sccs), default=0)
            cycle_snapshots[step] = {
                'total_cycles': total_cycles,
                'cycle_distribution': cycle_dist,
                'scc_count': len(large_sccs),
                'largest_scc': largest_scc,
            }
            dist_str = str(dict(sorted(cycle_dist.items()))) if cycle_dist is not None else 'N/A (>10000)'
            print(
                f'T={step}:\n'
                f'  total cycles: {total_cycles}\n'
                f'  cycle length distribution: {dist_str}\n'
                f'  strongly connected components (size > 1): {len(large_sccs)}\n'
                f'  largest SCC size: {largest_scc}'
            )
            if step == 5000:
                exc_mask = node_type == 1
                inh_mask = node_type == -1
                mean_exc = float(np.mean(activity[exc_mask])) if exc_mask.any() else float('nan')
                mean_inh = float(np.mean(activity[inh_mask])) if inh_mask.any() else float('nan')
                print(
                    f'  mean activity of excitatory nodes: {mean_exc:.4f}\n'
                    f'  mean activity of inhibitory nodes: {mean_inh:.4f}'
                )

    return {
        'mean_weight': mean_weight,
        'num_edges_active': num_edges_active,
        'num_edges_silenced': num_edges_silenced,
        'clustering': clustering_list,
        'shortest_path': shortest_path,
        'activity_variance': activity_variance,
        'activity_mean': activity_mean_list,
        'degree_std': degree_std,
        'topology_delta': topology_delta,
        'num_excitatory': num_excitatory,
        'num_inhibitory': num_inhibitory,
        'mean_act_excitatory': mean_act_excitatory,
        'mean_act_inhibitory': mean_act_inhibitory,
        'timesteps': timesteps,
        'cycle_snapshots': cycle_snapshots,
    }


def run_silencing_sweep():
    conditions = [
        {'ei_threshold': 0.6},
        {'ei_threshold': 0.7},
        {'ei_threshold': 0.8},
        {'ei_threshold': 0.9},
    ]
    results = []
    for cond in conditions:
        thr = cond['ei_threshold']
        print(f'Running E/I silencing simulation: ei_threshold={thr}...')
        data = run_ei_silencing_simulation(K=1, T=5000, N=20, seed=42,
                                           ei_threshold=thr,
                                           ei_window=100)
        results.append((cond, data))

        final_cycles = data['cycle_snapshots'][5000]['total_cycles']
        print(
            f'ei_threshold={thr}:\n'
            f'  final edges (active): {data["num_edges_active"][-1]}\n'
            f'  final edges (silenced): {data["num_edges_silenced"][-1]}\n'
            f'  final activity_variance: {data["activity_variance"][-1]:.4f}\n'
            f'  final clustering: {data["clustering"][-1]:.4f}\n'
            f'  min_excitatory: {min(data["num_excitatory"])}, '
            f'max_inhibitory: {max(data["num_inhibitory"])}\n'
            f'  final cycles: {final_cycles}'
        )
    return results


def plot_silencing_sweep(results):
    N = 20
    SNAPSHOT_TIMES = [500, 1000, 2000, 5000]
    n_cols = len(results)

    def _row_ylim(key):
        all_vals = [v for _, data in results for v in data[key] if not np.isnan(float(v))]
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        m = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - m, hi + m)

    # Edge row: y-limit from total edges (active + silenced)
    edge_totals = [
        a + s
        for _, data in results
        for a, s in zip(data['num_edges_active'], data['num_edges_silenced'])
    ]
    yl_edges = (0, (max(edge_totals) * 1.05) if edge_totals else N)

    yl_var = _row_ylim('activity_variance')
    scc_all = [data['cycle_snapshots'][t]['largest_scc']
               for _, data in results for t in SNAPSHOT_TIMES]
    yl_scc = (0, max(scc_all) + 1) if scc_all else (0, N + 1)

    row_labels = [
        'Number of edges\n(active / silenced)',
        'Activity variance',
        'Node count (E/I)',
        'Largest SCC size',
    ]

    fig, axes = plt.subplots(4, n_cols, figsize=(3.5 * n_cols, 14), squeeze=False)
    fig.suptitle('E/I silencing sweep (ei_window=100, K=1, T=5000)', fontsize=13, y=1.01)

    for col, (cond, data) in enumerate(results):
        thr = cond['ei_threshold']
        ts = data['timesteps']
        axes[0][col].set_title(f'thr={thr}', fontsize=11)

        # Row 1: stacked active vs silenced edges
        axes[0][col].stackplot(
            ts, data['num_edges_active'], data['num_edges_silenced'],
            labels=['Active (exc-exc)', 'Silenced'],
            colors=['steelblue', 'lightcoral'], alpha=0.8,
        )
        axes[0][col].set_ylim(yl_edges)
        axes[0][col].grid(True, alpha=0.3)
        if col == n_cols - 1:
            axes[0][col].legend(fontsize=7, loc='upper right')

        # Row 2: activity variance
        axes[1][col].plot(ts, data['activity_variance'])
        axes[1][col].set_ylim(yl_var)
        axes[1][col].grid(True, alpha=0.3)

        # Row 3: E/I node count stacked area
        axes[2][col].stackplot(
            ts, data['num_excitatory'], data['num_inhibitory'],
            labels=['Excitatory', 'Inhibitory'],
            colors=['steelblue', 'tomato'], alpha=0.8,
        )
        axes[2][col].set_ylim(0, N + 1)
        axes[2][col].grid(True, alpha=0.3)
        if col == n_cols - 1:
            axes[2][col].legend(fontsize=7, loc='upper right')

        # Row 4: largest SCC at snapshot times
        scc_vals = [data['cycle_snapshots'][t]['largest_scc'] for t in SNAPSHOT_TIMES]
        axes[3][col].scatter(SNAPSHOT_TIMES, scc_vals, marker='o', zorder=3)
        axes[3][col].plot(SNAPSHOT_TIMES, scc_vals, linestyle='--', alpha=0.5)
        axes[3][col].set_ylim(yl_scc)
        axes[3][col].set_xlabel('Activity timestep', fontsize=8)
        axes[3][col].grid(True, alpha=0.3)

        for row in range(4):
            if col == 0:
                axes[row][col].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = 'images/results_ei_silencing.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_ei_isolation_simulation(K=1, T=5000, N=20, seed=42, ei_threshold=0.7, ei_window=100):
    MAX_CYCLES = 10000
    SNAPSHOT_TIMES = {500, 1000, 2000, 5000}

    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = rng.random(N)
    activity[0] = 0.5
    node_type = np.ones(N, dtype=float)  # +1 excitatory, -1 inhibitory
    activity_histories = [deque(maxlen=ei_window) for _ in range(N)]

    mean_weight, num_edges_active, num_edges_silenced = [], [], []
    clustering_list, shortest_path = [], []
    activity_variance, activity_mean_list = [], []
    degree_std, topology_delta = [], []
    num_excitatory, num_inhibitory = [], []
    mean_act_excitatory, mean_act_inhibitory = [], []
    timesteps = []
    cycle_snapshots = {}
    recovered_count = 0  # cumulative inh->exc switches

    for t in range(T):
        # Activity propagation: only through exc-exc edges; inhibitory nodes decay
        new_activity = np.zeros(N)
        for i in range(N):
            if node_type[i] == 1:
                influence = sum(
                    G[j][i]['weight'] * activity[j]
                    for j in G.predecessors(i)
                    if node_type[j] == 1
                )
                new_activity[i] = np.tanh(influence)
            else:
                new_activity[i] = activity[i] * 0.9
        new_activity[0] = 0.5
        activity = new_activity

        if (t + 1) % K == 0:
            weights_before = {(i, j): d['weight'] for i, j, d in G.edges(data=True)}

            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                if node_type[i] == 1 and node_type[j] == 1:
                    w = data['weight']
                    if activity[i] > 0.5 and activity[j] > 0.5:
                        w += 0.05
                    w -= 0.01
                    if w < 0.01:
                        edges_to_remove.append((i, j))
                    else:
                        G[i][j]['weight'] = min(w, 1.0)
            G.remove_edges_from(edges_to_remove)

            existing = set(G.edges())
            for i in range(N):
                for j in range(N):
                    if i != j and node_type[i] == 1 and node_type[j] == 1:
                        if (i, j) not in existing and rng.random() < 0.01:
                            G.add_edge(i, j, weight=0.05)

            eb = set(weights_before.keys())
            ea = set(G.edges())
            delta = (
                sum(abs(G[e[0]][e[1]]['weight'] - weights_before[e]) for e in eb & ea)
                + sum(weights_before[e] for e in eb - ea)
                + sum(G[e[0]][e[1]]['weight'] for e in ea - eb)
            )
            topology_delta.append(delta)

            for i in range(N):
                activity_histories[i].append(activity[i])

            recent_means = [float(np.mean(activity_histories[i])) for i in range(N)]
            exc_candidates = [i for i in range(N)
                              if node_type[i] == 1 and recent_means[i] > ei_threshold]
            inh_candidates = [i for i in range(N)
                              if node_type[i] == -1 and recent_means[i] < ei_threshold * 0.5]
            if exc_candidates:
                switch_i = max(exc_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = -1
            if inh_candidates:
                switch_i = min(inh_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = 1
                recovered_count += 1

            n_active = sum(
                1 for i, j in G.edges()
                if node_type[i] == 1 and node_type[j] == 1
            )
            n_silenced = G.number_of_edges() - n_active

            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            mean_weight.append(np.mean(weights) if weights else 0.0)
            num_edges_active.append(n_active)
            num_edges_silenced.append(n_silenced)
            clustering_list.append(nx.average_clustering(G))

            wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            if wccs:
                lcc = G.subgraph(wccs[0]).copy()
                if lcc.number_of_nodes() > 1:
                    try:
                        shortest_path.append(nx.average_shortest_path_length(lcc))
                    except nx.NetworkXError:
                        shortest_path.append(float('nan'))
                else:
                    shortest_path.append(float('nan'))
            else:
                shortest_path.append(float('nan'))

            activity_variance.append(float(np.var(activity)))
            activity_mean_list.append(float(np.mean(activity)))
            out_degrees = [d for _, d in G.out_degree()]
            degree_std.append(float(np.std(out_degrees)))

            exc_mask = node_type == 1
            inh_mask = node_type == -1
            num_excitatory.append(int(np.sum(exc_mask)))
            num_inhibitory.append(int(np.sum(inh_mask)))
            mean_act_excitatory.append(float(np.mean(activity[exc_mask])) if exc_mask.any() else 0.0)
            mean_act_inhibitory.append(float(np.mean(activity[inh_mask])) if inh_mask.any() else 0.0)
            timesteps.append(t + 1)

        step = t + 1
        if step in SNAPSHOT_TIMES:
            cycles_sample = list(itertools.islice(nx.simple_cycles(G), MAX_CYCLES + 1))
            if len(cycles_sample) > MAX_CYCLES:
                total_cycles = '>10000'
                cycle_dist = None
            else:
                total_cycles = len(cycles_sample)
                cycle_dist = {}
                for c in cycles_sample:
                    cycle_dist[len(c)] = cycle_dist.get(len(c), 0) + 1

            sccs = list(nx.strongly_connected_components(G))
            large_sccs = [s for s in sccs if len(s) > 1]
            largest_scc = max((len(s) for s in sccs), default=0)
            cycle_snapshots[step] = {
                'total_cycles': total_cycles,
                'cycle_distribution': cycle_dist,
                'scc_count': len(large_sccs),
                'largest_scc': largest_scc,
            }
            dist_str = str(dict(sorted(cycle_dist.items()))) if cycle_dist is not None else 'N/A (>10000)'
            print(
                f'T={step}:\n'
                f'  total cycles: {total_cycles}\n'
                f'  cycle length distribution: {dist_str}\n'
                f'  strongly connected components (size > 1): {len(large_sccs)}\n'
                f'  largest SCC size: {largest_scc}'
            )
            if step == 5000:
                exc_mask = node_type == 1
                inh_mask = node_type == -1
                mean_exc = float(np.mean(activity[exc_mask])) if exc_mask.any() else float('nan')
                mean_inh = float(np.mean(activity[inh_mask])) if inh_mask.any() else float('nan')
                print(
                    f'  mean activity of excitatory nodes: {mean_exc:.4f}\n'
                    f'  mean activity of inhibitory nodes: {mean_inh:.4f}'
                )

    return {
        'mean_weight': mean_weight,
        'num_edges_active': num_edges_active,
        'num_edges_silenced': num_edges_silenced,
        'clustering': clustering_list,
        'shortest_path': shortest_path,
        'activity_variance': activity_variance,
        'activity_mean': activity_mean_list,
        'degree_std': degree_std,
        'topology_delta': topology_delta,
        'num_excitatory': num_excitatory,
        'num_inhibitory': num_inhibitory,
        'mean_act_excitatory': mean_act_excitatory,
        'mean_act_inhibitory': mean_act_inhibitory,
        'timesteps': timesteps,
        'cycle_snapshots': cycle_snapshots,
        'recovered_count': recovered_count,
    }


def run_isolation_sweep():
    conditions = [
        {'ei_threshold': 0.6},
        {'ei_threshold': 0.7},
        {'ei_threshold': 0.8},
        {'ei_threshold': 0.9},
    ]
    results = []
    for cond in conditions:
        thr = cond['ei_threshold']
        print(f'Running E/I isolation simulation: ei_threshold={thr}...')
        data = run_ei_isolation_simulation(K=1, T=5000, N=20, seed=42,
                                           ei_threshold=thr,
                                           ei_window=100)
        results.append((cond, data))

        final_cycles = data['cycle_snapshots'][5000]['total_cycles']
        print(
            f'ei_threshold={thr}:\n'
            f'  final edges (active): {data["num_edges_active"][-1]}\n'
            f'  final edges (silenced): {data["num_edges_silenced"][-1]}\n'
            f'  final activity_variance: {data["activity_variance"][-1]:.4f}\n'
            f'  final clustering: {data["clustering"][-1]:.4f}\n'
            f'  min_excitatory: {min(data["num_excitatory"])}, '
            f'max_inhibitory: {max(data["num_inhibitory"])}\n'
            f'  mean activity of excitatory nodes: {data["mean_act_excitatory"][-1]:.4f}\n'
            f'  mean activity of inhibitory nodes: {data["mean_act_inhibitory"][-1]:.4f}\n'
            f'  inhibitory nodes that recovered to excitatory: {data["recovered_count"]}\n'
            f'  final cycles: {final_cycles}'
        )
    return results


def plot_isolation_sweep(results):
    N = 20
    SNAPSHOT_TIMES = [500, 1000, 2000, 5000]
    n_cols = len(results)

    def _row_ylim(key):
        all_vals = [v for _, data in results for v in data[key] if not np.isnan(float(v))]
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        m = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - m, hi + m)

    edge_totals = [
        a + s
        for _, data in results
        for a, s in zip(data['num_edges_active'], data['num_edges_silenced'])
    ]
    yl_edges = (0, (max(edge_totals) * 1.05) if edge_totals else N)
    yl_var = _row_ylim('activity_variance')
    scc_all = [data['cycle_snapshots'][t]['largest_scc']
               for _, data in results for t in SNAPSHOT_TIMES]
    yl_scc = (0, max(scc_all) + 1) if scc_all else (0, N + 1)

    row_labels = [
        'Number of edges\n(active / silenced)',
        'Activity variance',
        'Node count (E/I)',
        'Largest SCC size',
    ]

    fig, axes = plt.subplots(4, n_cols, figsize=(3.5 * n_cols, 14), squeeze=False)
    fig.suptitle('E/I isolation sweep (ei_window=100, K=1, T=5000)', fontsize=13, y=1.01)

    for col, (cond, data) in enumerate(results):
        thr = cond['ei_threshold']
        ts = data['timesteps']
        axes[0][col].set_title(f'thr={thr}', fontsize=11)

        axes[0][col].stackplot(
            ts, data['num_edges_active'], data['num_edges_silenced'],
            labels=['Active (exc-exc)', 'Silenced'],
            colors=['steelblue', 'lightcoral'], alpha=0.8,
        )
        axes[0][col].set_ylim(yl_edges)
        axes[0][col].grid(True, alpha=0.3)
        if col == n_cols - 1:
            axes[0][col].legend(fontsize=7, loc='upper right')

        axes[1][col].plot(ts, data['activity_variance'])
        axes[1][col].set_ylim(yl_var)
        axes[1][col].grid(True, alpha=0.3)

        axes[2][col].stackplot(
            ts, data['num_excitatory'], data['num_inhibitory'],
            labels=['Excitatory', 'Inhibitory'],
            colors=['steelblue', 'tomato'], alpha=0.8,
        )
        axes[2][col].set_ylim(0, N + 1)
        axes[2][col].grid(True, alpha=0.3)
        if col == n_cols - 1:
            axes[2][col].legend(fontsize=7, loc='upper right')

        scc_vals = [data['cycle_snapshots'][t]['largest_scc'] for t in SNAPSHOT_TIMES]
        axes[3][col].scatter(SNAPSHOT_TIMES, scc_vals, marker='o', zorder=3)
        axes[3][col].plot(SNAPSHOT_TIMES, scc_vals, linestyle='--', alpha=0.5)
        axes[3][col].set_ylim(yl_scc)
        axes[3][col].set_xlabel('Activity timestep', fontsize=8)
        axes[3][col].grid(True, alpha=0.3)

        for row in range(4):
            if col == 0:
                axes[row][col].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = 'images/results_ei_isolation.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_input_removal_test(ei_threshold=0.9, K=1, N=20, seed=42, ei_window=100):
    T_train = 3000
    T_removal = 3000
    T_restore = 2000
    T_total = T_train + T_removal + T_restore

    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = rng.random(N)
    activity[0] = 0.5
    node_type = np.ones(N, dtype=float)
    activity_histories = [deque(maxlen=ei_window) for _ in range(N)]

    activity_variance = []
    num_edges_active = []
    num_excitatory_list = []
    num_inhibitory_list = []
    mean_act_excitatory = []
    mean_act_inhibitory = []
    timesteps = []

    for t in range(T_total):
        if t < T_train:
            node0_input = 0.5
        elif t < T_train + T_removal:
            node0_input = 0.0
        else:
            node0_input = 0.5

        new_activity = np.zeros(N)
        for i in range(N):
            if node_type[i] == 1:
                influence = sum(
                    G[j][i]['weight'] * activity[j]
                    for j in G.predecessors(i)
                    if node_type[j] == 1
                )
                new_activity[i] = np.tanh(influence)
            else:
                new_activity[i] = activity[i] * 0.9
        new_activity[0] = node0_input
        activity = new_activity

        if (t + 1) % K == 0:
            edges_to_remove = []
            for i, j, data in list(G.edges(data=True)):
                if node_type[i] == 1 and node_type[j] == 1:
                    w = data['weight']
                    if activity[i] > 0.5 and activity[j] > 0.5:
                        w += 0.05
                    w -= 0.01
                    if w < 0.01:
                        edges_to_remove.append((i, j))
                    else:
                        G[i][j]['weight'] = min(w, 1.0)
            G.remove_edges_from(edges_to_remove)

            existing = set(G.edges())
            for i in range(N):
                for j in range(N):
                    if i != j and node_type[i] == 1 and node_type[j] == 1:
                        if (i, j) not in existing and rng.random() < 0.01:
                            G.add_edge(i, j, weight=0.05)

            for i in range(N):
                activity_histories[i].append(activity[i])

            recent_means = [float(np.mean(activity_histories[i])) for i in range(N)]
            exc_candidates = [i for i in range(N)
                              if node_type[i] == 1 and recent_means[i] > ei_threshold]
            inh_candidates = [i for i in range(N)
                              if node_type[i] == -1 and recent_means[i] < ei_threshold * 0.5]
            if exc_candidates:
                switch_i = max(exc_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = -1
            if inh_candidates:
                switch_i = min(inh_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = 1

            n_active = sum(
                1 for i, j in G.edges()
                if node_type[i] == 1 and node_type[j] == 1
            )
            exc_mask = node_type == 1
            inh_mask = node_type == -1

            activity_variance.append(float(np.var(activity)))
            num_edges_active.append(n_active)
            num_excitatory_list.append(int(np.sum(exc_mask)))
            num_inhibitory_list.append(int(np.sum(inh_mask)))
            mean_act_excitatory.append(float(np.mean(activity[exc_mask])) if exc_mask.any() else 0.0)
            mean_act_inhibitory.append(float(np.mean(activity[inh_mask])) if inh_mask.any() else 0.0)
            timesteps.append(t + 1)

    # Phase boundary indices (K=1 so timesteps[i] == i+1)
    train_idx = T_train - 1
    removal_idx = T_train + T_removal - 1

    print(
        f'End of training:\n'
        f'  activity_variance: {activity_variance[train_idx]:.4f}\n'
        f'  active_edges: {num_edges_active[train_idx]}\n'
        f'  excitatory_nodes: {num_excitatory_list[train_idx]}'
    )
    maintained = activity_variance[removal_idx] > 0.01
    print(
        f'\nEnd of no-input phase:\n'
        f'  activity_variance: {activity_variance[removal_idx]:.4f}\n'
        f'  active_edges: {num_edges_active[removal_idx]}\n'
        f'  excitatory_nodes: {num_excitatory_list[removal_idx]}\n'
        f'  activity_maintained: {maintained}'
    )
    print(
        f'\nEnd of restored phase:\n'
        f'  activity_variance: {activity_variance[-1]:.4f}\n'
        f'  active_edges: {num_edges_active[-1]}\n'
        f'  excitatory_nodes: {num_excitatory_list[-1]}'
    )

    return {
        'activity_variance': activity_variance,
        'num_edges_active': num_edges_active,
        'num_excitatory': num_excitatory_list,
        'num_inhibitory': num_inhibitory_list,
        'mean_act_excitatory': mean_act_excitatory,
        'mean_act_inhibitory': mean_act_inhibitory,
        'timesteps': timesteps,
        'T_train': T_train,
        'T_removal': T_removal,
        'T_restore': T_restore,
        'N': N,
    }


def plot_input_removal(data):
    N = data['N']
    ts = data['timesteps']
    T_train = data['T_train']
    T_removal = data['T_removal']
    T_restore = data['T_restore']

    vlines = [T_train, T_train + T_removal]
    phase_mids = [
        T_train / 2,
        T_train + T_removal / 2,
        T_train + T_removal + T_restore / 2,
    ]
    phase_labels = ['Training', 'No input', 'Restored']

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), squeeze=False)
    fig.suptitle('Input removal test (ei_threshold=0.9, K=1, N=20)', fontsize=13)

    # Row 1: activity variance
    axes[0][0].plot(ts, data['activity_variance'])
    axes[0][0].axhline(0.01, color='tomato', linestyle=':', alpha=0.8, linewidth=1,
                       label='maintained threshold (0.01)')
    axes[0][0].set_ylabel('Activity variance', fontsize=9)
    axes[0][0].legend(fontsize=8, loc='upper right')
    axes[0][0].grid(True, alpha=0.3)

    # Row 2: active edges
    axes[1][0].plot(ts, data['num_edges_active'])
    axes[1][0].set_ylabel('Active edges (exc-exc)', fontsize=9)
    axes[1][0].grid(True, alpha=0.3)

    # Row 3: E/I node count stacked area
    axes[2][0].stackplot(
        ts, data['num_excitatory'], data['num_inhibitory'],
        labels=['Excitatory', 'Inhibitory'],
        colors=['steelblue', 'tomato'], alpha=0.8,
    )
    axes[2][0].set_ylim(0, N + 1)
    axes[2][0].set_ylabel('Node count (E/I)', fontsize=9)
    axes[2][0].legend(fontsize=8, loc='upper right')
    axes[2][0].grid(True, alpha=0.3)

    # Row 4: mean activity
    axes[3][0].plot(ts, data['mean_act_excitatory'], label='Excitatory', color='steelblue')
    axes[3][0].plot(ts, data['mean_act_inhibitory'], label='Inhibitory',
                    color='tomato', alpha=0.7)
    axes[3][0].set_ylabel('Mean activity', fontsize=9)
    axes[3][0].set_xlabel('Activity timestep', fontsize=8)
    axes[3][0].legend(fontsize=8, loc='upper right')
    axes[3][0].grid(True, alpha=0.3)

    # Vertical phase lines and labels on all axes
    for ax in axes[:, 0]:
        for vl in vlines:
            ax.axvline(vl, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        for mid, label in zip(phase_mids, phase_labels):
            ax.text(mid, 1.0, label, ha='center', va='bottom',
                    transform=ax.get_xaxis_transform(),
                    fontsize=8, color='dimgray', style='italic')

    plt.tight_layout()
    fname = 'images/results_input_removal.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def _run_ei_isolation_phase(G, activity, node_type, activity_histories, rng,
                             ei_threshold, ei_window, K, T, node0_input,
                             topology_dynamic=True):
    """Run one phase of the isolation model. Mutates G, activity, node_type in-place.
    Returns dict of per-step metrics."""
    N = len(activity)
    result = {
        'activity_variance': [],
        'num_excitatory': [],
        'mean_act_excitatory': [],
        'timesteps': [],
    }

    for t in range(T):
        new_activity = np.zeros(N)
        for i in range(N):
            if node_type[i] == 1:
                influence = sum(
                    G[j][i]['weight'] * activity[j]
                    for j in G.predecessors(i)
                    if node_type[j] == 1
                )
                new_activity[i] = np.tanh(influence)
            else:
                new_activity[i] = activity[i] * 0.9
        new_activity[0] = node0_input
        activity[:] = new_activity

        if (t + 1) % K == 0:
            if topology_dynamic:
                edges_to_remove = []
                for i, j, data in list(G.edges(data=True)):
                    if node_type[i] == 1 and node_type[j] == 1:
                        w = data['weight']
                        if activity[i] > 0.5 and activity[j] > 0.5:
                            w += 0.05
                        w -= 0.01
                        if w < 0.01:
                            edges_to_remove.append((i, j))
                        else:
                            G[i][j]['weight'] = min(w, 1.0)
                G.remove_edges_from(edges_to_remove)

                existing = set(G.edges())
                for i in range(N):
                    for j in range(N):
                        if i != j and node_type[i] == 1 and node_type[j] == 1:
                            if (i, j) not in existing and rng.random() < 0.01:
                                G.add_edge(i, j, weight=0.05)

            for i in range(N):
                activity_histories[i].append(activity[i])

            recent_means = [float(np.mean(activity_histories[i])) for i in range(N)]
            exc_candidates = [i for i in range(N)
                              if node_type[i] == 1 and recent_means[i] > ei_threshold]
            inh_candidates = [i for i in range(N)
                              if node_type[i] == -1 and recent_means[i] < ei_threshold * 0.5]
            if exc_candidates:
                switch_i = max(exc_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = -1
            if inh_candidates:
                switch_i = min(inh_candidates, key=lambda i: recent_means[i])
                node_type[switch_i] = 1

            exc_mask = node_type == 1
            result['activity_variance'].append(float(np.var(activity)))
            result['num_excitatory'].append(int(np.sum(exc_mask)))
            result['mean_act_excitatory'].append(
                float(np.mean(activity[exc_mask])) if exc_mask.any() else 0.0
            )
            result['timesteps'].append(t + 1)

    return result


def run_fixed_topology_ei_test(ei_threshold=0.9, K=1, N=20, seed=42, ei_window=100,
                                T_train=3000, T_test=5000):
    rng = np.random.default_rng(seed)

    # Initialize
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = rng.random(N)
    activity[0] = 0.5
    node_type = np.ones(N, dtype=float)
    activity_histories = [deque(maxlen=ei_window) for _ in range(N)]

    # Training phase
    print(f'Training (T={T_train}, ei_threshold={ei_threshold})...')
    _run_ei_isolation_phase(G, activity, node_type, activity_histories, rng,
                             ei_threshold, ei_window, K, T_train,
                             node0_input=0.5, topology_dynamic=True)

    # Snapshot trained state for both conditions
    trained_G = G.copy()
    trained_activity = activity.copy()
    trained_node_type = node_type.copy()
    trained_histories = [deque(ah, maxlen=ei_window) for ah in activity_histories]

    # Condition A: Dynamic topology, no input
    print('Running condition A: dynamic topology, no input...')
    G_A = trained_G.copy()
    act_A = trained_activity.copy()
    nt_A = trained_node_type.copy()
    hist_A = [deque(ah, maxlen=ei_window) for ah in trained_histories]
    rng_A = np.random.default_rng(seed + 1)
    res_A = _run_ei_isolation_phase(G_A, act_A, nt_A, hist_A, rng_A,
                                     ei_threshold, ei_window, K, T_test,
                                     node0_input=0.0, topology_dynamic=True)

    # Condition B: Fixed topology, no input
    print('Running condition B: fixed topology, no input...')
    G_B = trained_G.copy()
    act_B = trained_activity.copy()
    nt_B = trained_node_type.copy()
    hist_B = [deque(ah, maxlen=ei_window) for ah in trained_histories]
    rng_B = np.random.default_rng(seed + 2)  # unused for fixed topology
    res_B = _run_ei_isolation_phase(G_B, act_B, nt_B, hist_B, rng_B,
                                     ei_threshold, ei_window, K, T_test,
                                     node0_input=0.0, topology_dynamic=False)

    # Console output
    for label, res in [('Dynamic topology', res_A), ('Fixed topology', res_B)]:
        var_last = res['activity_variance'][-1000:]
        osc = float(np.std(var_last)) > 0.01
        print(
            f'\n{label}:\n'
            f'  activity_variance (mean over last 1000 steps): {np.mean(var_last):.4f}\n'
            f'  oscillation detected: {osc}'
        )

    return res_A, res_B


def plot_fixed_topology_control(res_A, res_B):
    N = 20

    def _row_ylim(key):
        all_vals = [v for data in [res_A, res_B] for v in data[key]]
        if not all_vals:
            return (0, 1)
        lo, hi = min(all_vals), max(all_vals)
        m = (hi - lo) * 0.05 if hi != lo else 0.1
        return (lo - m, hi + m)

    yl_var = _row_ylim('activity_variance')
    yl_exc = (0, N + 1)
    yl_mean = _row_ylim('mean_act_excitatory')

    row_labels = [
        'Activity variance',
        'Excitatory node count',
        'Mean activity (excitatory)',
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), squeeze=False)
    fig.suptitle(
        'Fixed topology control (ei_threshold=0.9, K=1, no input after training)',
        fontsize=13,
    )
    axes[0][0].set_title('Dynamic topology', fontsize=11)
    axes[0][1].set_title('Fixed topology', fontsize=11)

    for col, data in enumerate([res_A, res_B]):
        ts = data['timesteps']

        axes[0][col].plot(ts, data['activity_variance'])
        axes[0][col].set_ylim(yl_var)
        axes[0][col].grid(True, alpha=0.3)

        axes[1][col].plot(ts, data['num_excitatory'])
        axes[1][col].set_ylim(yl_exc)
        axes[1][col].grid(True, alpha=0.3)

        axes[2][col].plot(ts, data['mean_act_excitatory'])
        axes[2][col].set_ylim(yl_mean)
        axes[2][col].set_xlabel('Activity timestep', fontsize=8)
        axes[2][col].grid(True, alpha=0.3)

    for row in range(3):
        axes[row][0].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = 'images/results_fixed_topology_control.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_pattern_encoding_experiment(K=10, T_pretrain=2000, T_encode=1000, T_test=500,
                                     N=20, seed=42):
    PATTERN_SWITCH = 100

    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = np.zeros(N)
    activity[0] = 0.5

    def _hebbian_step(G, activity, rng):
        edges_to_remove = []
        for i, j, data in list(G.edges(data=True)):
            w = data['weight']
            if activity[i] > 0.5 and activity[j] > 0.5:
                w += 0.05
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
                    G.add_edge(i, j, weight=0.05)

    # Phase 1: Pretrain with alternating X/Y
    train_timesteps = []
    train_act_node0, train_act_node1, train_act_node2 = [], [], []
    train_num_edges, train_mean_weight, train_act_variance = [], [], []

    for t in range(T_pretrain):
        block = t // PATTERN_SWITCH
        if block % 2 == 0:
            ext = {0: 0.8, 1: 0.8, 2: 0.0}
        else:
            ext = {0: 0.8, 1: 0.0, 2: 0.8}
        new_activity = np.zeros(N)
        for i in range(N):
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_activity[i] = np.tanh(s)
        for node, val in ext.items():
            new_activity[node] = val
        activity = new_activity
        if (t + 1) % K == 0:
            _hebbian_step(G, activity, rng)
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            train_timesteps.append(t + 1)
            train_act_node0.append(float(activity[0]))
            train_act_node1.append(float(activity[1]))
            train_act_node2.append(float(activity[2]))
            train_num_edges.append(G.number_of_edges())
            train_mean_weight.append(float(np.mean(weights)) if weights else 0.0)
            train_act_variance.append(float(np.var(activity)))

    G_pretrain = G.copy()
    act_pretrain = activity.copy()
    print(f'Pretrain complete (T={T_pretrain}, alternating X/Y, '
          f'{G_pretrain.number_of_edges()} edges).')

    # Phase 2a: Encode X-only from pretrained topology.
    # Edges into node 2 (inactive in X) will decay and be deleted.
    G_X = G_pretrain.copy()
    act = act_pretrain.copy()
    rng_X = np.random.default_rng(seed + 1)
    for t in range(T_encode):
        new_act = np.zeros(N)
        for i in range(N):
            s = sum(G_X[j][i]['weight'] * act[j] for j in G_X.predecessors(i))
            new_act[i] = np.tanh(s)
        new_act[0], new_act[1], new_act[2] = 0.8, 0.8, 0.0
        act = new_act
        if (t + 1) % K == 0:
            _hebbian_step(G_X, act, rng_X)
    print(f'X-only encoding complete ({G_X.number_of_edges()} edges remaining).')

    # Phase 2b: Encode Y-only from the same pretrained topology (separate copy).
    # Edges into node 1 (inactive in Y) will decay and be deleted.
    G_Y = G_pretrain.copy()
    act = act_pretrain.copy()
    rng_Y = np.random.default_rng(seed + 2)
    for t in range(T_encode):
        new_act = np.zeros(N)
        for i in range(N):
            s = sum(G_Y[j][i]['weight'] * act[j] for j in G_Y.predecessors(i))
            new_act[i] = np.tanh(s)
        new_act[0], new_act[1], new_act[2] = 0.8, 0.0, 0.8
        act = new_act
        if (t + 1) % K == 0:
            _hebbian_step(G_Y, act, rng_Y)
    print(f'Y-only encoding complete ({G_Y.number_of_edges()} edges remaining).')

    # Test: zero initial activity, food probe only, topology frozen
    def _run_test(test_G):
        act = np.zeros(N)
        node1_trace, node2_trace = [], []
        internal_traces = [[] for _ in range(3, N)]
        for _ in range(T_test):
            new_act = np.zeros(N)
            for i in range(N):
                s = sum(test_G[j][i]['weight'] * act[j] for j in test_G.predecessors(i))
                new_act[i] = np.tanh(s)
            new_act[0] = 0.5   # food probe — nodes 1 and 2 evolve freely
            act = new_act
            node1_trace.append(float(act[1]))
            node2_trace.append(float(act[2]))
            for idx, node in enumerate(range(3, N)):
                internal_traces[idx].append(float(act[node]))
        return node1_trace, node2_trace, internal_traces

    node1_X, node2_X, internal_X = _run_test(G_X)
    mean_n1_X = float(np.mean(node1_X))
    mean_n2_X = float(np.mean(node2_X))
    bias_X = mean_n1_X - mean_n2_X
    print(
        f'\nTest after X:\n'
        f'  mean activity node 1 (north): {mean_n1_X:.4f}\n'
        f'  mean activity node 2 (south): {mean_n2_X:.4f}\n'
        f'  bias toward north: {bias_X:.4f} (positive = north, negative = south)'
    )

    node1_Y, node2_Y, internal_Y = _run_test(G_Y)
    mean_n1_Y = float(np.mean(node1_Y))
    mean_n2_Y = float(np.mean(node2_Y))
    bias_Y = mean_n2_Y - mean_n1_Y
    print(
        f'\nTest after Y:\n'
        f'  mean activity node 1 (north): {mean_n1_Y:.4f}\n'
        f'  mean activity node 2 (south): {mean_n2_Y:.4f}\n'
        f'  bias toward south: {bias_Y:.4f} (positive = south, negative = north)'
    )

    encoding_detected = bias_X > 0 and bias_Y > 0
    print(f'\nPattern encoding detected: {encoding_detected}')

    return {
        'node1_X': node1_X, 'node2_X': node2_X, 'internal_X': internal_X,
        'node1_Y': node1_Y, 'node2_Y': node2_Y, 'internal_Y': internal_Y,
        'mean_n1_X': mean_n1_X, 'mean_n2_X': mean_n2_X,
        'mean_n1_Y': mean_n1_Y, 'mean_n2_Y': mean_n2_Y,
        'train_timesteps': train_timesteps,
        'train_act_node0': train_act_node0,
        'train_act_node1': train_act_node1,
        'train_act_node2': train_act_node2,
        'train_num_edges': train_num_edges,
        'train_mean_weight': train_mean_weight,
        'train_act_variance': train_act_variance,
        'T_pretrain': T_pretrain,
        'T_encode': T_encode,
        'T_test': T_test,
        'N': N,
    }


def plot_pattern_encoding(data):
    N = data['N']
    T_test = data['T_test']
    T_encode = data['T_encode']
    test_ts = list(range(1, T_test + 1))

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), squeeze=False)
    fig.suptitle(
        f'Pattern encoding test (K=10, T_pretrain={data["T_pretrain"]}, '
        f'T_encode={T_encode}, T_test={T_test})',
        fontsize=13,
    )
    axes[0][0].set_title(f'Test after X  (X-only encoding, T={T_encode})', fontsize=10)
    axes[0][1].set_title(f'Test after Y  (Y-only encoding, T={T_encode})', fontsize=10)

    # Row 1: node 1 and node 2 activity traces
    for col, (n1, n2) in enumerate([
        (data['node1_X'], data['node2_X']),
        (data['node1_Y'], data['node2_Y']),
    ]):
        axes[0][col].plot(test_ts, n1, label='Node 1 (north)', color='steelblue')
        axes[0][col].plot(test_ts, n2, label='Node 2 (south)', color='tomato')
        axes[0][col].set_ylabel('Activity', fontsize=9)
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3)

    # Row 2: internal node (3–19) activity traces
    for col, internal in enumerate([data['internal_X'], data['internal_Y']]):
        for trace in internal:
            axes[1][col].plot(test_ts, trace, alpha=0.4, linewidth=0.7)
        axes[1][col].set_ylabel('Activity (nodes 3–19)', fontsize=9)
        axes[1][col].grid(True, alpha=0.3)

    # Row 3: bar chart of mean node 1 vs node 2
    bar_labels = ['North\n(node 1)', 'South\n(node 2)']
    for col, (mn1, mn2) in enumerate([
        (data['mean_n1_X'], data['mean_n2_X']),
        (data['mean_n1_Y'], data['mean_n2_Y']),
    ]):
        axes[2][col].bar(bar_labels, [mn1, mn2],
                         color=['steelblue', 'tomato'], alpha=0.8, width=0.5)
        axes[2][col].set_ylabel('Mean activity', fontsize=9)
        axes[2][col].set_ylim(0, max(mn1, mn2) * 1.25 + 1e-6)
        axes[2][col].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = 'images/results_pattern_encoding.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_pattern_encoding_training(data):
    T_pretrain = data['T_pretrain']
    ts = data['train_timesteps']
    switches = list(range(100, T_pretrain, 100))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), squeeze=False)
    fig.suptitle(
        f'Pattern encoding — pretrain dynamics (K=10, T={T_pretrain}, alternating X/Y)',
        fontsize=13,
    )

    # Row 1: activity of nodes 0, 1, 2
    axes[0][0].plot(ts, data['train_act_node0'], label='Node 0 (food)', color='seagreen')
    axes[0][0].plot(ts, data['train_act_node1'], label='Node 1 (north)', color='steelblue')
    axes[0][0].plot(ts, data['train_act_node2'], label='Node 2 (south)', color='tomato')
    axes[0][0].set_ylabel('Activity', fontsize=9)
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(True, alpha=0.3)

    # Row 2: number of edges
    axes[1][0].plot(ts, data['train_num_edges'], color='steelblue')
    axes[1][0].set_ylabel('Number of edges', fontsize=9)
    axes[1][0].grid(True, alpha=0.3)

    # Row 3: mean edge weight
    axes[2][0].plot(ts, data['train_mean_weight'], color='seagreen')
    axes[2][0].set_ylabel('Mean edge weight', fontsize=9)
    axes[2][0].grid(True, alpha=0.3)

    # Row 4: activity variance
    axes[3][0].plot(ts, data['train_act_variance'], color='dimgray')
    axes[3][0].set_ylabel('Activity variance', fontsize=9)
    axes[3][0].set_xlabel('Training timestep', fontsize=8)
    axes[3][0].grid(True, alpha=0.3)

    # Pattern switch vertical lines and X/Y shading on all axes
    for ax in axes[:, 0]:
        for sw in switches:
            ax.axvline(sw, color='gray', linestyle='--', alpha=0.25, linewidth=0.8)

    # X/Y block labels on the top axis only (every other block to reduce clutter)
    ax_top = axes[0][0]
    block_edges = [0] + switches + [T_pretrain]
    for b in range(0, len(block_edges) - 1, 2):  # every 2 blocks
        left, right = block_edges[b], block_edges[b + 1]
        mid = (left + right) / 2
        label = 'X' if (b // 1) % 2 == 0 else 'Y'
        ax_top.text(mid, 1.01, label, ha='center', va='bottom',
                    transform=ax_top.get_xaxis_transform(),
                    fontsize=6, color='dimgray')

    plt.tight_layout()
    fname = 'images/results_pattern_encoding_training.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_pattern_encoding_experiment_v2(K=10, T_pretrain=2000, T_encode=1000, T_test=500,
                                        N=20, seed=42):
    """Same as v1 but adds a no-input control test and confirms node 0 is active in all phases."""
    PATTERN_SWITCH = 100

    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=rng.random())

    activity = np.zeros(N)
    activity[0] = 0.5

    def _hebbian_step(G, activity, rng):
        edges_to_remove = []
        for i, j, data in list(G.edges(data=True)):
            w = data['weight']
            if activity[i] > 0.5 and activity[j] > 0.5:
                w += 0.05
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
                    G.add_edge(i, j, weight=0.05)

    # Phase 1: Pretrain with alternating X/Y (node 0 = 0.8 in both)
    train_timesteps = []
    train_act_node0, train_act_node1, train_act_node2 = [], [], []
    train_num_edges, train_mean_weight, train_act_variance = [], [], []
    node0_pretrain_first100 = []

    for t in range(T_pretrain):
        block = t // PATTERN_SWITCH
        if block % 2 == 0:
            ext = {0: 0.8, 1: 0.8, 2: 0.0}   # Pattern X
        else:
            ext = {0: 0.8, 1: 0.0, 2: 0.8}   # Pattern Y
        new_activity = np.zeros(N)
        for i in range(N):
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_activity[i] = np.tanh(s)
        for node, val in ext.items():
            new_activity[node] = val
        activity = new_activity
        if t < 100:
            node0_pretrain_first100.append(float(activity[0]))
        if (t + 1) % K == 0:
            _hebbian_step(G, activity, rng)
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            train_timesteps.append(t + 1)
            train_act_node0.append(float(activity[0]))
            train_act_node1.append(float(activity[1]))
            train_act_node2.append(float(activity[2]))
            train_num_edges.append(G.number_of_edges())
            train_mean_weight.append(float(np.mean(weights)) if weights else 0.0)
            train_act_variance.append(float(np.var(activity)))

    print(f'Node 0 mean activity (pretrain first 100 steps): '
          f'{float(np.mean(node0_pretrain_first100)):.4f}')

    G_pretrain = G.copy()
    act_pretrain = activity.copy()
    print(f'Pretrain complete (T={T_pretrain}, alternating X/Y, '
          f'{G_pretrain.number_of_edges()} edges).')

    # Phase 2: X-only encoding (node 0 = 0.8, node 1 = 0.8, node 2 = 0.0)
    G_X = G_pretrain.copy()
    act = act_pretrain.copy()
    rng_X = np.random.default_rng(seed + 1)
    node0_x_enc = []
    for t in range(T_encode):
        new_act = np.zeros(N)
        for i in range(N):
            s = sum(G_X[j][i]['weight'] * act[j] for j in G_X.predecessors(i))
            new_act[i] = np.tanh(s)
        new_act[0], new_act[1], new_act[2] = 0.8, 0.8, 0.0
        act = new_act
        node0_x_enc.append(float(act[0]))
        if (t + 1) % K == 0:
            _hebbian_step(G_X, act, rng_X)
    print(f'Node 0 mean activity (X-only encoding): '
          f'{float(np.mean(node0_x_enc)):.4f}')
    print(f'X-only encoding complete ({G_X.number_of_edges()} edges remaining).')

    # Phase 3: Y-only encoding (node 0 = 0.8, node 1 = 0.0, node 2 = 0.8)
    G_Y = G_pretrain.copy()
    act = act_pretrain.copy()
    rng_Y = np.random.default_rng(seed + 2)
    for t in range(T_encode):
        new_act = np.zeros(N)
        for i in range(N):
            s = sum(G_Y[j][i]['weight'] * act[j] for j in G_Y.predecessors(i))
            new_act[i] = np.tanh(s)
        new_act[0], new_act[1], new_act[2] = 0.8, 0.0, 0.8
        act = new_act
        if (t + 1) % K == 0:
            _hebbian_step(G_Y, act, rng_Y)
    print(f'Y-only encoding complete ({G_Y.number_of_edges()} edges remaining).')

    # Test: zero initial activity, topology frozen, only node 0 varies
    def _run_test(test_G, node0_val):
        act = np.zeros(N)
        node1_trace, node2_trace = [], []
        internal_traces = [[] for _ in range(3, N)]
        for _ in range(T_test):
            new_act = np.zeros(N)
            for i in range(N):
                s = sum(test_G[j][i]['weight'] * act[j] for j in test_G.predecessors(i))
                new_act[i] = np.tanh(s)
            new_act[0] = node0_val
            act = new_act
            node1_trace.append(float(act[1]))
            node2_trace.append(float(act[2]))
            for idx, node in enumerate(range(3, N)):
                internal_traces[idx].append(float(act[node]))
        return node1_trace, node2_trace, internal_traces

    # Probe: initial hint for nodes 1 and 2, node 0 fixed, topology frozen
    def _run_probe(test_G, node0_val, node1_init=0.0, node2_init=0.0):
        act = np.zeros(N)
        act[0] = node0_val
        act[1] = node1_init
        act[2] = node2_init
        node1_trace, node2_trace = [], []
        internal_traces = [[] for _ in range(3, N)]
        for _ in range(T_test):
            new_act = np.zeros(N)
            for i in range(N):
                s = sum(test_G[j][i]['weight'] * act[j] for j in test_G.predecessors(i))
                new_act[i] = np.tanh(s)
            new_act[0] = node0_val
            act = new_act
            node1_trace.append(float(act[1]))
            node2_trace.append(float(act[2]))
            for idx, node in enumerate(range(3, N)):
                internal_traces[idx].append(float(act[node]))
        return node1_trace, node2_trace, internal_traces

    node1_X, node2_X, internal_X = _run_test(G_X, node0_val=0.5)
    mean_n1_X = float(np.mean(node1_X))
    mean_n2_X = float(np.mean(node2_X))
    bias_X = mean_n1_X - mean_n2_X
    print(
        f'\nTest after X:\n'
        f'  mean activity node 1 (north): {mean_n1_X:.4f}\n'
        f'  mean activity node 2 (south): {mean_n2_X:.4f}\n'
        f'  bias toward north: {bias_X:.4f} (positive = north, negative = south)'
    )

    node1_Y, node2_Y, internal_Y = _run_test(G_Y, node0_val=0.5)
    mean_n1_Y = float(np.mean(node1_Y))
    mean_n2_Y = float(np.mean(node2_Y))
    bias_Y = mean_n2_Y - mean_n1_Y
    print(
        f'\nTest after Y:\n'
        f'  mean activity node 1 (north): {mean_n1_Y:.4f}\n'
        f'  mean activity node 2 (south): {mean_n2_Y:.4f}\n'
        f'  bias toward south: {bias_Y:.4f} (positive = south, negative = north)'
    )

    # Control: no input (node 0 = 0.0) — confirms responses are input-driven
    node1_ctrl, node2_ctrl, internal_ctrl = _run_test(G_X, node0_val=0.0)
    mean_n1_ctrl = float(np.mean(node1_ctrl))
    mean_n2_ctrl = float(np.mean(node2_ctrl))
    print(
        f'\nControl (no input):\n'
        f'  mean activity node 1: {mean_n1_ctrl:.4f}\n'
        f'  mean activity node 2: {mean_n2_ctrl:.4f}'
    )

    encoding_detected = bias_X > 0 and bias_Y > 0
    input_driven = mean_n1_ctrl < 0.05 and mean_n2_ctrl < 0.05
    print(f'\nPattern encoding detected: {encoding_detected}')
    print(f'Encoding is input-driven: {input_driven}')

    # Coexistence test: probe G_pretrain with weak directional hints
    node1_north, node2_north, internal_north = _run_probe(G_pretrain, 0.5, 0.3, 0.0)
    node1_south, node2_south, internal_south = _run_probe(G_pretrain, 0.5, 0.0, 0.3)
    node1_coex_ctrl, node2_coex_ctrl, internal_coex_ctrl = _run_probe(G_pretrain, 0.0, 0.0, 0.0)

    mean_n1_north = float(np.mean(node1_north))
    mean_n2_north = float(np.mean(node2_north))
    mean_n1_south = float(np.mean(node1_south))
    mean_n2_south = float(np.mean(node2_south))
    patterns_coexist = mean_n1_north > mean_n2_north and mean_n2_south > mean_n1_south

    print(
        f'\nCoexistence test (after pretrain only):\n'
        f'  Probe with north hint:\n'
        f'    mean activity node 1 (north): {mean_n1_north:.4f}\n'
        f'    mean activity node 2 (south): {mean_n2_north:.4f}\n'
        f'  Probe with south hint:\n'
        f'    mean activity node 1 (north): {mean_n1_south:.4f}\n'
        f'    mean activity node 2 (south): {mean_n2_south:.4f}\n'
        f'  Patterns coexist: {patterns_coexist}'
    )

    return {
        'node1_X': node1_X, 'node2_X': node2_X, 'internal_X': internal_X,
        'node1_Y': node1_Y, 'node2_Y': node2_Y, 'internal_Y': internal_Y,
        'node1_ctrl': node1_ctrl, 'node2_ctrl': node2_ctrl, 'internal_ctrl': internal_ctrl,
        'mean_n1_X': mean_n1_X, 'mean_n2_X': mean_n2_X,
        'mean_n1_Y': mean_n1_Y, 'mean_n2_Y': mean_n2_Y,
        'mean_n1_ctrl': mean_n1_ctrl, 'mean_n2_ctrl': mean_n2_ctrl,
        'node1_north': node1_north, 'node2_north': node2_north, 'internal_north': internal_north,
        'node1_south': node1_south, 'node2_south': node2_south, 'internal_south': internal_south,
        'node1_coex_ctrl': node1_coex_ctrl, 'node2_coex_ctrl': node2_coex_ctrl,
        'internal_coex_ctrl': internal_coex_ctrl,
        'mean_n1_north': mean_n1_north, 'mean_n2_north': mean_n2_north,
        'mean_n1_south': mean_n1_south, 'mean_n2_south': mean_n2_south,
        'patterns_coexist': patterns_coexist,
        'train_timesteps': train_timesteps,
        'train_act_node0': train_act_node0,
        'train_act_node1': train_act_node1,
        'train_act_node2': train_act_node2,
        'train_num_edges': train_num_edges,
        'train_mean_weight': train_mean_weight,
        'train_act_variance': train_act_variance,
        'T_pretrain': T_pretrain,
        'T_encode': T_encode,
        'T_test': T_test,
        'N': N,
    }


def plot_pattern_encoding_v2(data):
    N = data['N']
    T_test = data['T_test']
    T_encode = data['T_encode']
    test_ts = list(range(1, T_test + 1))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    fig.suptitle(
        f'Pattern encoding v2 (K=10, T_pretrain={data["T_pretrain"]}, '
        f'T_encode={T_encode}, T_test={T_test})',
        fontsize=13,
    )
    axes[0][0].set_title(f'Test after X  (X-only, T={T_encode})', fontsize=10)
    axes[0][1].set_title(f'Test after Y  (Y-only, T={T_encode})', fontsize=10)
    axes[0][2].set_title('Control  (no input, node 0 = 0.0)', fontsize=10)

    cols = [
        (data['node1_X'], data['node2_X'], data['internal_X'],
         data['mean_n1_X'], data['mean_n2_X']),
        (data['node1_Y'], data['node2_Y'], data['internal_Y'],
         data['mean_n1_Y'], data['mean_n2_Y']),
        (data['node1_ctrl'], data['node2_ctrl'], data['internal_ctrl'],
         data['mean_n1_ctrl'], data['mean_n2_ctrl']),
    ]

    for col, (n1, n2, internal, mn1, mn2) in enumerate(cols):
        # Row 1: node 1 and node 2 traces
        axes[0][col].plot(test_ts, n1, label='Node 1 (north)', color='steelblue')
        axes[0][col].plot(test_ts, n2, label='Node 2 (south)', color='tomato')
        axes[0][col].set_ylim(-0.05, 1.1)
        axes[0][col].set_ylabel('Activity', fontsize=9)
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3)

        # Row 2: internal node traces
        for trace in internal:
            axes[1][col].plot(test_ts, trace, alpha=0.4, linewidth=0.7)
        axes[1][col].set_ylim(-0.05, 1.1)
        axes[1][col].set_ylabel('Activity (nodes 3–19)', fontsize=9)
        axes[1][col].grid(True, alpha=0.3)

        # Row 3: bar chart
        axes[2][col].bar(
            ['North\n(node 1)', 'South\n(node 2)'],
            [mn1, mn2],
            color=['steelblue', 'tomato'], alpha=0.8, width=0.5,
        )
        axes[2][col].set_ylim(0, 1.1)
        axes[2][col].set_ylabel('Mean activity', fontsize=9)
        axes[2][col].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = 'images/results_pattern_encoding_v2.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_pattern_encoding_training_v2(data):
    T_pretrain = data['T_pretrain']
    ts = data['train_timesteps']
    switches = list(range(100, T_pretrain, 100))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), squeeze=False)
    fig.suptitle(
        f'Pattern encoding v2 — pretrain dynamics (K=10, T={T_pretrain}, alternating X/Y)',
        fontsize=13,
    )

    axes[0][0].plot(ts, data['train_act_node0'], label='Node 0 (food)', color='seagreen')
    axes[0][0].plot(ts, data['train_act_node1'], label='Node 1 (north)', color='steelblue')
    axes[0][0].plot(ts, data['train_act_node2'], label='Node 2 (south)', color='tomato')
    axes[0][0].set_ylabel('Activity', fontsize=9)
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(True, alpha=0.3)

    axes[1][0].plot(ts, data['train_num_edges'], color='steelblue')
    axes[1][0].set_ylabel('Number of edges', fontsize=9)
    axes[1][0].grid(True, alpha=0.3)

    axes[2][0].plot(ts, data['train_mean_weight'], color='seagreen')
    axes[2][0].set_ylabel('Mean edge weight', fontsize=9)
    axes[2][0].grid(True, alpha=0.3)

    axes[3][0].plot(ts, data['train_act_variance'], color='dimgray')
    axes[3][0].set_ylabel('Activity variance', fontsize=9)
    axes[3][0].set_xlabel('Training timestep', fontsize=8)
    axes[3][0].grid(True, alpha=0.3)

    for ax in axes[:, 0]:
        for sw in switches:
            ax.axvline(sw, color='gray', linestyle='--', alpha=0.25, linewidth=0.8)

    ax_top = axes[0][0]
    block_edges = [0] + switches + [T_pretrain]
    for b in range(0, len(block_edges) - 1, 2):
        left, right = block_edges[b], block_edges[b + 1]
        mid = (left + right) / 2
        label = 'X' if (b // 1) % 2 == 0 else 'Y'
        ax_top.text(mid, 1.01, label, ha='center', va='bottom',
                    transform=ax_top.get_xaxis_transform(),
                    fontsize=6, color='dimgray')

    plt.tight_layout()
    fname = 'images/results_pattern_encoding_training_v2.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def plot_pattern_coexistence(data):
    T_test = data['T_test']
    test_ts = list(range(1, T_test + 1))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    fig.suptitle(
        f'Pattern coexistence test (K=10, T_pretrain={data["T_pretrain"]}, T_test={T_test})',
        fontsize=13,
    )
    axes[0][0].set_title('Probe with north hint\n(node0=0.5, node1=0.3, node2=0.0)', fontsize=10)
    axes[0][1].set_title('Probe with south hint\n(node0=0.5, node1=0.0, node2=0.3)', fontsize=10)
    axes[0][2].set_title('Control  (no input)', fontsize=10)

    mn1_coex_ctrl = float(np.mean(data['node1_coex_ctrl']))
    mn2_coex_ctrl = float(np.mean(data['node2_coex_ctrl']))

    cols = [
        (data['node1_north'], data['node2_north'], data['internal_north'],
         data['mean_n1_north'], data['mean_n2_north']),
        (data['node1_south'], data['node2_south'], data['internal_south'],
         data['mean_n1_south'], data['mean_n2_south']),
        (data['node1_coex_ctrl'], data['node2_coex_ctrl'], data['internal_coex_ctrl'],
         mn1_coex_ctrl, mn2_coex_ctrl),
    ]

    for col, (n1, n2, internal, mn1, mn2) in enumerate(cols):
        axes[0][col].plot(test_ts, n1, label='Node 1 (north)', color='steelblue')
        axes[0][col].plot(test_ts, n2, label='Node 2 (south)', color='tomato')
        axes[0][col].set_ylim(-0.05, 1.1)
        axes[0][col].set_ylabel('Activity', fontsize=9)
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3)

        for trace in internal:
            axes[1][col].plot(test_ts, trace, alpha=0.4, linewidth=0.7)
        axes[1][col].set_ylim(-0.05, 1.1)
        axes[1][col].set_ylabel('Activity (nodes 3–19)', fontsize=9)
        axes[1][col].grid(True, alpha=0.3)

        axes[2][col].bar(
            ['North\n(node 1)', 'South\n(node 2)'],
            [mn1, mn2],
            color=['steelblue', 'tomato'], alpha=0.8, width=0.5,
        )
        axes[2][col].set_ylim(0, 1.1)
        axes[2][col].set_ylabel('Mean activity', fontsize=9)
        axes[2][col].grid(True, alpha=0.3, axis='y')

    coexist_str = 'True' if data['patterns_coexist'] else 'False'
    fig.text(0.5, 0.01,
             f'Patterns coexist: {coexist_str}  '
             f'(north hint → node1 > node2 AND south hint → node2 > node1)',
             ha='center', fontsize=10, color='darkgreen' if data['patterns_coexist'] else 'firebrick')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = 'images/results_pattern_coexistence.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


if __name__ == "__main__":
    # results = run_input_comparison(K=20)
    # plot_input_comparison(results)
    # plot_topology_fingerprint(results)

    # enc = run_encoding_experiment()
    # plot_encoding(enc)

    # run_stdp_encoding_experiment()

    # delta_results = run_delta_comparison([1, 5, 10, 20, 30, 50])
    # plot_delta_comparison(delta_results, [1, 5, 10, 20, 30, 50])

    # conv_results = run_convergence_test()
    # plot_convergence_test(conv_results)

    # cycle_snapshots = run_cycle_analysis()
    # plot_cycle_analysis(cycle_snapshots)

    # run_ei_comparison()

    # sweep = run_inhibition_sweep()
    # plot_inhibition_sweep(sweep)

    # threshold_sweep = run_threshold_sweep()
    # plot_threshold_sweep(threshold_sweep)

    # silencing_sweep = run_silencing_sweep()
    # plot_silencing_sweep(silencing_sweep)

    # isolation_sweep = run_isolation_sweep()
    # plot_isolation_sweep(isolation_sweep)

    # removal_data = run_input_removal_test()
    # plot_input_removal(removal_data)

    # res_A, res_B = run_fixed_topology_ei_test()
    # plot_fixed_topology_control(res_A, res_B)

    # enc_data = run_pattern_encoding_experiment()
    # plot_pattern_encoding(enc_data)
    # plot_pattern_encoding_training(enc_data)

    enc_data_v2 = run_pattern_encoding_experiment_v2()
    plot_pattern_encoding_v2(enc_data_v2)
    plot_pattern_encoding_training_v2(enc_data_v2)
    plot_pattern_coexistence(enc_data_v2)
