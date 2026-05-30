import itertools
import os
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


def run_loop_resonance_experiment(N=15, seed=42, K=5, T_train=2000, T_probe=200):
    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    # Sparse background edges (all node pairs), weight 0.1-0.3
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.15:
                G.add_edge(i, j, weight=float(rng.uniform(0.1, 0.3)))

    # Loop A: 0 → 1 → 2 → 3 → 0, weight 0.8
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        G.add_edge(i, j, weight=0.8)

    # Loop B: 0 → 4 → 5 → 6 → 0, weight 0.8
    for i, j in [(0, 4), (4, 5), (5, 6), (6, 0)]:
        G.add_edge(i, j, weight=0.8)

    loop_a_nodes = [1, 2, 3]
    loop_b_nodes = [4, 5, 6]
    background_nodes = list(range(7, N))

    activity = np.zeros(N)

    def _train_phase(ext, steps):
        nonlocal activity
        for t in range(steps):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            for node, val in ext.items():
                new_activity[node] = val
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
                        if i != j and (i, j) not in existing and rng.random() < 0.005:
                            G.add_edge(i, j, weight=0.05)

    # Phase 2a: Pattern A only (T=1500)
    _train_phase({0: 0.8, 1: 0.8}, 1500)

    print('After A training only (before B):')
    print('Loop A edges:')
    for src, dst in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        if G.has_edge(src, dst):
            print(f'  {src}→{dst}: {G[src][dst]["weight"]:.3f}')
        else:
            print(f'  {src}→{dst}: DELETED')
    print('Loop B edges:')
    for src, dst in [(0, 4), (4, 5), (5, 6), (6, 0)]:
        if G.has_edge(src, dst):
            print(f'  {src}→{dst}: {G[src][dst]["weight"]:.3f}')
        else:
            print(f'  {src}→{dst}: DELETED')

    # Phase 2b: Pattern B only (T=500)
    _train_phase({0: 0.8, 4: 0.8}, 500)

    print('Edge weights after training:')
    print('Loop A edges (0→1, 1→2, 2→3, 3→0):')
    for src, dst in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        if G.has_edge(src, dst):
            print(f'  {src}→{dst}: {G[src][dst]["weight"]:.3f}')
    print('Loop B edges (0→4, 4→5, 5→6, 6→0):')
    for src, dst in [(0, 4), (4, 5), (5, 6), (6, 0)]:
        if G.has_edge(src, dst):
            print(f'  {src}→{dst}: {G[src][dst]["weight"]:.3f}')

    def _run_probe(init_vals):
        """Run T_probe steps with init_vals clamped throughout; return trajectory array."""
        act = np.zeros(N)
        for node, val in init_vals.items():
            act[node] = val
        trajectories = []
        for _ in range(T_probe):
            new_act = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * act[j] for j in G.predecessors(i))
                new_act[i] = np.tanh(s)
            for node, val in init_vals.items():
                new_act[node] = val
            act = new_act
            trajectories.append(act.copy())
        return np.array(trajectories)  # (T_probe, N)

    traj1 = _run_probe({0: 0.5})
    traj2 = _run_probe({0: 0.5, 1: 0.3})
    traj3 = _run_probe({0: 0.5, 4: 0.3})

    def _group_means(traj):
        ma = float(np.mean(traj[:, loop_a_nodes]))
        mb = float(np.mean(traj[:, loop_b_nodes]))
        mbg = float(np.mean(traj[:, background_nodes])) if background_nodes else 0.0
        return ma, mb, mbg

    ma1, mb1, mbg1 = _group_means(traj1)
    ma2, mb2, mbg2 = _group_means(traj2)
    ma3, mb3, mbg3 = _group_means(traj3)

    both_activated = ma1 > 0.1 and mb1 > 0.1
    a_selective = ma2 > mb2 + 0.1
    b_selective = mb3 > ma3 + 0.1
    resonance_supported = both_activated and a_selective and b_selective

    print('=== Loop Resonance Experiment ===')
    print()
    print('Probe 1 (shared node only):')
    print(f'  Loop A mean activity: {ma1:.4f}')
    print(f'  Loop B mean activity: {mb1:.4f}')
    print(f'  Background mean activity: {mbg1:.4f}')
    print(f'  Both loops activated: {both_activated} (True if both > 0.1)')
    print()
    print('Probe 2 (shared + loop A hint):')
    print(f'  Loop A mean activity: {ma2:.4f}')
    print(f'  Loop B mean activity: {mb2:.4f}')
    print(f'  Loop A selective: {a_selective} (True if loop A > loop B + 0.1)')
    print()
    print('Probe 3 (shared + loop B hint):')
    print(f'  Loop A mean activity: {ma3:.4f}')
    print(f'  Loop B mean activity: {mb3:.4f}')
    print(f'  Loop B selective: {b_selective} (True if loop B > loop A + 0.1)')
    print()
    print(f'Resonance hypothesis supported: {resonance_supported}')

    return {
        'traj1': traj1, 'traj2': traj2, 'traj3': traj3,
        'ma1': ma1, 'mb1': mb1, 'mbg1': mbg1,
        'ma2': ma2, 'mb2': mb2, 'mbg2': mbg2,
        'ma3': ma3, 'mb3': mb3, 'mbg3': mbg3,
        'both_activated': both_activated,
        'a_selective': a_selective,
        'b_selective': b_selective,
        'resonance_supported': resonance_supported,
        'loop_a_nodes': loop_a_nodes,
        'loop_b_nodes': loop_b_nodes,
        'background_nodes': background_nodes,
        'T_probe': T_probe,
    }


def plot_loop_resonance(data):
    T_probe = data['T_probe']
    ts = np.arange(1, T_probe + 1)
    loop_a_nodes = data['loop_a_nodes']
    loop_b_nodes = data['loop_b_nodes']
    background_nodes = data['background_nodes']

    probe_titles = [
        'Probe 1\n(shared node only)',
        'Probe 2\n(shared + loop A hint)',
        'Probe 3\n(shared + loop B hint)',
    ]
    trajs = [data['traj1'], data['traj2'], data['traj3']]
    means = [
        (data['ma1'], data['mb1'], data['mbg1']),
        (data['ma2'], data['mb2'], data['mbg2']),
        (data['ma3'], data['mb3'], data['mbg3']),
    ]
    bar_labels = ['Loop A\n(1,2,3)', 'Loop B\n(4,5,6)', 'Background\n(7-14)']
    bar_colors = ['steelblue', 'tomato', 'seagreen']

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    fig.suptitle('Loop Resonance Experiment', fontsize=13)

    for col in range(3):
        axes[0][col].set_title(probe_titles[col], fontsize=10)
        traj = trajs[col]

        mean_a = traj[:, loop_a_nodes].mean(axis=1)
        mean_b = traj[:, loop_b_nodes].mean(axis=1)
        mean_bg = traj[:, background_nodes].mean(axis=1) if background_nodes else np.zeros(T_probe)

        axes[0][col].plot(ts, mean_a, label='Loop A (1,2,3)', color='steelblue')
        axes[0][col].plot(ts, mean_b, label='Loop B (4,5,6)', color='tomato')
        axes[0][col].plot(ts, mean_bg, label='Background (7-14)', color='seagreen', alpha=0.7)
        axes[0][col].set_ylim(-0.02, 1.05)
        axes[0][col].set_ylabel('Mean activity', fontsize=9)
        axes[0][col].set_xlabel('Probe timestep', fontsize=8)
        axes[0][col].legend(fontsize=7)
        axes[0][col].grid(True, alpha=0.3)

        ma, mb, mbg = means[col]
        axes[1][col].bar(bar_labels, [ma, mb, mbg], color=bar_colors, alpha=0.8, width=0.5)
        axes[1][col].set_ylabel('Mean activity', fontsize=9)
        axes[1][col].set_ylim(0, 1.1)
        axes[1][col].grid(True, alpha=0.3, axis='y')

    resonance_str = str(data['resonance_supported'])
    txt_color = 'darkgreen' if data['resonance_supported'] else 'firebrick'
    fig.text(
        0.5, 0.01,
        f'Resonance hypothesis supported: {resonance_str}  '
        f'(Probe1 both activated AND Probe2 A selective AND Probe3 B selective)',
        ha='center', fontsize=10, color=txt_color,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = 'images/results_loop_resonance_asymmetric.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_loop_combination_experiment(N=20, seed=42, K=5, T_probe=200):
    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    # Sparse background edges (all pairs), weight 0.01-0.1
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.1:
                G.add_edge(i, j, weight=float(rng.uniform(0.01, 0.1)))

    # Stimulus → loop-entry connections
    G.add_edge(0, 3, weight=0.8)   # food → Loop A entry
    G.add_edge(1, 6, weight=0.5)   # north → Loop B entry
    G.add_edge(2, 9, weight=0.5)   # south → Loop C entry

    # Loop A: 3→4→5→3, weight 0.8
    for src, dst in [(3, 4), (4, 5), (5, 3)]:
        G.add_edge(src, dst, weight=0.8)

    # Loop B: 6→7→8→6, weight 0.05
    for src, dst in [(6, 7), (7, 8), (8, 6)]:
        G.add_edge(src, dst, weight=0.05)

    # Loop C: 9→10→11→9, weight 0.05
    for src, dst in [(9, 10), (10, 11), (11, 9)]:
        G.add_edge(src, dst, weight=0.05)

    # Loop D: 12→13→14→12, weight 0.05 (no stimulus connection)
    for src, dst in [(12, 13), (13, 14), (14, 12)]:
        G.add_edge(src, dst, weight=0.05)

    loop_a_nodes = [3, 4, 5]
    loop_b_nodes = [6, 7, 8]
    loop_c_nodes = [9, 10, 11]
    loop_d_nodes = [12, 13, 14]
    background_nodes = list(range(15, N))

    loop_a_edges = [(3, 4), (4, 5), (5, 3)]
    loop_b_edges = [(6, 7), (7, 8), (8, 6)]
    loop_c_edges = [(9, 10), (10, 11), (11, 9)]
    loop_d_edges = [(12, 13), (13, 14), (14, 12)]

    activity = np.zeros(N)

    def _train_phase(ext, steps):
        nonlocal activity
        for t in range(steps):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            for node, val in ext.items():
                new_activity[node] = val
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
                        if i != j and (i, j) not in existing and rng.random() < 0.005:
                            G.add_edge(i, j, weight=0.05)

    # Phase 1: food + north (T=1500)
    _train_phase({0: 0.8, 1: 0.8, 2: 0.0}, 1500)
    # Phase 2: food + south (T=1500)
    _train_phase({0: 0.8, 1: 0.0, 2: 0.8}, 1500)

    def _mean_loop_weight(edges):
        weights = [G[src][dst]['weight'] for src, dst in edges if G.has_edge(src, dst)]
        return float(np.mean(weights)) if weights else 0.0

    print('=== Loop Combination Experiment ===')
    print()
    print('After training - edge weights:')
    print(
        f'Loop A: {_mean_loop_weight(loop_a_edges):.3f}, '
        f'Loop B: {_mean_loop_weight(loop_b_edges):.3f}, '
        f'Loop C: {_mean_loop_weight(loop_c_edges):.3f}, '
        f'Loop D: {_mean_loop_weight(loop_d_edges):.3f}'
    )
    print()

    def _run_probe(init_vals):
        act = np.zeros(N)
        for node, val in init_vals.items():
            act[node] = val
        trajectories = []
        for _ in range(T_probe):
            new_act = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * act[j] for j in G.predecessors(i))
                new_act[i] = np.tanh(s)
            for node, val in init_vals.items():
                new_act[node] = val
            act = new_act
            trajectories.append(act.copy())
        return np.array(trajectories)  # (T_probe, N)

    def _group_means(traj):
        return (
            float(np.mean(traj[:, loop_a_nodes])),
            float(np.mean(traj[:, loop_b_nodes])),
            float(np.mean(traj[:, loop_c_nodes])),
            float(np.mean(traj[:, loop_d_nodes])),
            float(np.mean(traj[:, background_nodes])) if background_nodes else 0.0,
        )

    probe_specs = [
        ({0: 0.5},              'food only'),
        ({1: 0.5},              'north only'),
        ({2: 0.5},              'south only'),
        ({0: 0.5, 1: 0.5},     'food + north'),
        ({0: 0.5, 2: 0.5},     'food + south'),
    ]

    results = []
    for idx, (init_vals, label) in enumerate(probe_specs, start=1):
        traj = _run_probe(init_vals)
        ma, mb, mc, md, mbg = _group_means(traj)
        results.append((ma, mb, mc, md, mbg))
        print(f'Probe {idx} ({label}):')
        print(f'  Loop A: {ma:.4f}, Loop B: {mb:.4f}, Loop C: {mc:.4f}, '
              f'Loop D: {md:.4f}, Background: {mbg:.4f}')
        if label == 'food + north':
            print('  Expected: A high, B high, C low, D low')
        elif label == 'food + south':
            print('  Expected: A high, C high, B low, D low')
        print()

    ma4, mb4, mc4, md4, _ = results[3]
    ma5, mb5, mc5, md5, _ = results[4]
    combination_detected = (
        ma4 > 0.5 and mb4 > 0.5 and mc4 < 0.3
        and ma5 > 0.5 and mc5 > 0.5 and mb5 < 0.3
    )
    print(f'Combination encoding detected: {combination_detected}')
    print('(True if Probe4: A>0.5 AND B>0.5 AND C<0.3')
    print('     AND Probe5: A>0.5 AND C>0.5 AND B<0.3)')

    return {
        'results': results,
        'combination_detected': combination_detected,
        'loop_a_nodes': loop_a_nodes,
        'loop_b_nodes': loop_b_nodes,
        'loop_c_nodes': loop_c_nodes,
        'loop_d_nodes': loop_d_nodes,
        'background_nodes': background_nodes,
        'T_probe': T_probe,
    }


def plot_loop_combination(data):
    probe_labels = [
        'Probe 1\n(food only)',
        'Probe 2\n(north only)',
        'Probe 3\n(south only)',
        'Probe 4\n(food + north)',
        'Probe 5\n(food + south)',
    ]
    group_labels = [
        'Loop A\n(3,4,5)',
        'Loop B\n(6,7,8)',
        'Loop C\n(9,10,11)',
        'Loop D\n(12,13,14)',
        'Background\n(15-19)',
    ]
    colors = ['steelblue', 'seagreen', 'darkorange', 'tomato', 'gray']

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), squeeze=False)
    fig.suptitle('Loop Combination Experiment', fontsize=13)

    for col, (label, means) in enumerate(zip(probe_labels, data['results'])):
        axes[0][col].set_title(label, fontsize=10)
        axes[0][col].bar(group_labels, list(means), color=colors, alpha=0.8, width=0.6)
        axes[0][col].set_ylim(0, 1.1)
        axes[0][col].grid(True, alpha=0.3, axis='y')
        if col == 0:
            axes[0][col].set_ylabel('Mean activity', fontsize=9)

    detected_str = str(data['combination_detected'])
    txt_color = 'darkgreen' if data['combination_detected'] else 'firebrick'
    fig.text(
        0.5, 0.01,
        f'Combination encoding detected: {detected_str}',
        ha='center', fontsize=10, color=txt_color,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fname = 'images/results_loop_combination.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_association_experiment(N=20, seed=42, K=5, T_probe=200):
    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    # Sparse background edges (all pairs), weight 0.01-0.1
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.1:
                G.add_edge(i, j, weight=float(rng.uniform(0.01, 0.1)))

    # Stimulus connections
    G.add_edge(0, 2, weight=0.5)    # food → Loop Food entry
    G.add_edge(1, 5, weight=0.5)    # north → Loop North entry
    G.add_edge(11, 8, weight=0.5)   # south → Loop South entry

    # Loop Food: 2→3→4→2, weight 0.5
    for src, dst in [(2, 3), (3, 4), (4, 2)]:
        G.add_edge(src, dst, weight=0.5)

    # Loop North: 5→6→7→5, weight 0.5
    for src, dst in [(5, 6), (6, 7), (7, 5)]:
        G.add_edge(src, dst, weight=0.5)

    # Loop South: 8→9→10→8, weight 0.5
    for src, dst in [(8, 9), (9, 10), (10, 8)]:
        G.add_edge(src, dst, weight=0.5)

    loop_food_nodes = [2, 3, 4]
    loop_north_nodes = [5, 6, 7]
    loop_south_nodes = [8, 9, 10]
    background_nodes = list(range(12, N))

    activity = np.zeros(N)

    def _count_cross_edges(nodes_a, nodes_b):
        return sum(
            1 for i in nodes_a for j in nodes_b
            if G.has_edge(i, j) or G.has_edge(j, i)
        )

    def _mean_loop_weight(loop_edges):
        ws = [G[s][d]['weight'] for s, d in loop_edges if G.has_edge(s, d)]
        return float(np.mean(ws)) if ws else 0.0

    def _cross_stats(nodes_a, nodes_b):
        edges = [(i, j) for i in nodes_a for j in nodes_b if G.has_edge(i, j)]
        edges += [(j, i) for i in nodes_a for j in nodes_b if G.has_edge(j, i)]
        n = len(edges)
        mean_w = float(np.mean([G[s][d]['weight'] for s, d in edges])) if edges else 0.0
        return n, mean_w

    loop_food_edges = [(2, 3), (3, 4), (4, 2)]
    loop_north_edges = [(5, 6), (6, 7), (7, 5)]
    loop_south_edges = [(8, 9), (9, 10), (10, 8)]

    def _phase_report(phase_label):
        fn, fn_w = _cross_stats(loop_food_nodes, loop_north_nodes)
        fs, fs_w = _cross_stats(loop_food_nodes, loop_south_nodes)
        print(f'After {phase_label}:')
        print(f'  Loop Food mean weight:  {_mean_loop_weight(loop_food_edges):.3f}')
        print(f'  Loop North mean weight: {_mean_loop_weight(loop_north_edges):.3f}')
        print(f'  Loop South mean weight: {_mean_loop_weight(loop_south_edges):.3f}')
        print(f'  Food-North connections: {fn}  (mean weight: {fn_w:.3f})')
        print(f'  Food-South connections: {fs}  (mean weight: {fs_w:.3f})')
        print()

    def _train_phase(ext, steps):
        nonlocal activity
        for t in range(steps):
            new_activity = np.zeros(N)
            for i in range(N):
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_activity[i] = np.tanh(s)
            for node, val in ext.items():
                new_activity[node] = val
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
                        if i != j and (i, j) not in existing and rng.random() < 0.005:
                            G.add_edge(i, j, weight=0.05)

    def _run_probe_on(graph, init_vals):
        act = np.zeros(N)
        for node, val in init_vals.items():
            act[node] = val
        trajectories = []
        for _ in range(T_probe):
            new_act = np.zeros(N)
            for i in range(N):
                s = sum(graph[j][i]['weight'] * act[j] for j in graph.predecessors(i))
                new_act[i] = np.tanh(s)
            for node, val in init_vals.items():
                new_act[node] = val
            act = new_act
            trajectories.append(act.copy())
        return np.array(trajectories)  # (T_probe, N)

    def _group_means(traj):
        return (
            float(np.mean(traj[:, loop_food_nodes])),
            float(np.mean(traj[:, loop_north_nodes])),
            float(np.mean(traj[:, loop_south_nodes])),
            float(np.mean(traj[:, background_nodes])) if background_nodes else 0.0,
        )

    print('=== Association Experiment ===')
    print()

    # Phase 1: North alone (T=500)
    _train_phase({1: 0.8}, 500)
    _phase_report('Phase 1 (north alone)')

    # Phase 2: Food alone (T=500)
    _train_phase({0: 0.8}, 500)
    _phase_report('Phase 2 (food alone)')

    # Snapshot after Phase 2 for baseline probes
    G_baseline = G.copy()

    # Phase 3: North + Food together (T=3000)
    _train_phase({0: 0.8, 1: 0.8}, 3000)
    _phase_report('Phase 3 (north + food)')

    # Snapshot after Phase 3 for Probe 4
    G_after_p3 = G.copy()

    # Phase 4: South + Food together (T=3000)
    _train_phase({0: 0.8, 11: 0.8}, 3000)
    _phase_report('Phase 4 (south + food)')

    # Probes 1-3: baseline topology (after independent training, before co-occurrence)
    traj1 = _run_probe_on(G_baseline, {0: 0.5})
    mf1, mn1, ms1, mbg1 = _group_means(traj1)
    print('Probe 1 (food only, baseline):')
    print(f'  Food: {mf1:.4f}, North: {mn1:.4f}, South: {ms1:.4f}, Background: {mbg1:.4f}')
    print()

    traj2 = _run_probe_on(G_baseline, {1: 0.5})
    mf2, mn2, ms2, mbg2 = _group_means(traj2)
    print('Probe 2 (north only, baseline):')
    print(f'  Food: {mf2:.4f}, North: {mn2:.4f}, South: {ms2:.4f}, Background: {mbg2:.4f}')
    print()

    traj3 = _run_probe_on(G_baseline, {11: 0.5})
    mf3, mn3, ms3, mbg3 = _group_means(traj3)
    print('Probe 3 (south only, baseline):')
    print(f'  Food: {mf3:.4f}, North: {mn3:.4f}, South: {ms3:.4f}, Background: {mbg3:.4f}')
    print()

    # Probe 4: food only on topology after Phase 3 (north+food co-occurrence)
    traj4 = _run_probe_on(G_after_p3, {0: 0.5})
    mf4, mn4, ms4, mbg4 = _group_means(traj4)
    north_reactivated = mn4 > 0.3
    print('Probe 4 (food only, after north+food training):')
    print(f'  Food: {mf4:.4f}, North: {mn4:.4f}, South: {ms4:.4f}, Background: {mbg4:.4f}')
    print(f'  North reactivated by food: {north_reactivated} (North > 0.3)')
    print()

    # Probe 5: food only on topology after Phase 4 (south+food co-occurrence)
    traj5 = _run_probe_on(G, {0: 0.5})
    mf5, mn5, ms5, mbg5 = _group_means(traj5)
    south_reactivated = ms5 > 0.3
    print('Probe 5 (food only, after south+food training):')
    print(f'  Food: {mf5:.4f}, North: {mn5:.4f}, South: {ms5:.4f}, Background: {mbg5:.4f}')
    print(f'  South reactivated by food: {south_reactivated} (South > 0.3)')
    print()

    association_supported = north_reactivated and south_reactivated
    print(f'Association hypothesis supported: {association_supported}')
    print('(True if Probe4 North > 0.3 AND Probe5 South > 0.3)')

    return {
        'results': [
            (mf1, mn1, ms1, mbg1),
            (mf2, mn2, ms2, mbg2),
            (mf3, mn3, ms3, mbg3),
            (mf4, mn4, ms4, mbg4),
            (mf5, mn5, ms5, mbg5),
        ],
        'north_reactivated': north_reactivated,
        'south_reactivated': south_reactivated,
        'association_supported': association_supported,
        'loop_food_nodes': loop_food_nodes,
        'loop_north_nodes': loop_north_nodes,
        'loop_south_nodes': loop_south_nodes,
        'background_nodes': background_nodes,
        'T_probe': T_probe,
    }


def plot_association(data):
    probe_labels = [
        'Probe 1\n(food, baseline)',
        'Probe 2\n(north, baseline)',
        'Probe 3\n(south, baseline)',
        'Probe 4\n(food, after\nnorth+food)',
        'Probe 5\n(food, after\nsouth+food)',
    ]
    group_labels = [
        'Loop Food\n(2,3,4)',
        'Loop North\n(5,6,7)',
        'Loop South\n(8,9,10)',
        'Background\n(12-19)',
    ]
    colors = ['steelblue', 'seagreen', 'darkorange', 'gray']

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), squeeze=False)
    fig.suptitle('Association Experiment', fontsize=13)

    for col, (label, means) in enumerate(zip(probe_labels, data['results'])):
        axes[0][col].set_title(label, fontsize=10)
        axes[0][col].bar(group_labels, list(means), color=colors, alpha=0.8, width=0.6)
        axes[0][col].set_ylim(0, 1.1)
        axes[0][col].grid(True, alpha=0.3, axis='y')
        if col == 0:
            axes[0][col].set_ylabel('Mean activity', fontsize=9)

    detected_str = str(data['association_supported'])
    txt_color = 'darkgreen' if data['association_supported'] else 'firebrick'
    fig.text(
        0.5, 0.01,
        f'Association hypothesis supported: {detected_str}  '
        f'(Probe4 North > 0.3 AND Probe5 South > 0.3)',
        ha='center', fontsize=10, color=txt_color,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fname = 'images/results_association_long.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


# ─── Session 3: E/I threshold evolution and sparse association ────────────────


def _mutate_genome(genome, rng, mutation_std=0.05):
    return {
        'ei_threshold': float(np.clip(
            genome['ei_threshold'] + rng.normal(0, mutation_std), 0.5, 0.95)),
        'recovery_ratio': float(np.clip(
            genome['recovery_ratio'] + rng.normal(0, mutation_std), 0.1, 0.8)),
        'recovery_delay': int(np.clip(
            genome['recovery_delay'] + int(rng.integers(-20, 21)), 0, 200)),
    }


def _run_episode_ei(G, activity, rng, genome, N=20, K=10,
                    n_propagation_steps=3, temperature=1.0, readout_weights=None):
    """GridWorld episode with E/I isolation dynamics; returns (steps_survived, food_eaten)."""
    from echo_world import GridWorld, _softmax_sample, _hebbian_step
    ei_thr = genome['ei_threshold']
    rec_ratio = genome['recovery_ratio']
    rec_delay = genome['recovery_delay']
    ei_window = 50

    gw = GridWorld()
    row, col = gw.start_pos
    hp = gw.start_hp
    food_available = [True] * len(gw.food_positions)
    food_timers = [0] * len(gw.food_positions)
    food_eaten = 0
    steps_survived = 0

    node_type = np.ones(N, dtype=float)
    act_hist = [deque(maxlen=ei_window) for _ in range(N)]
    rec_timers = np.zeros(N, dtype=int)

    for step in range(gw.max_steps):
        if hp <= 0:
            break
        inp = {
            0: col / (gw.grid_size - 1),
            1: row / (gw.grid_size - 1),
            2: hp / gw.hp_max,
            3: 1.0 if any(food_available) else 0.0,
        }
        for _ in range(n_propagation_steps):
            new_act = np.zeros(N)
            for i in range(N):
                if i in inp:
                    new_act[i] = inp[i]
                elif node_type[i] == 1:
                    s = sum(G[j][i]['weight'] * activity[j]
                            for j in G.predecessors(i) if node_type[j] == 1)
                    new_act[i] = np.tanh(s)
                else:
                    new_act[i] = activity[i] * 0.9
            activity[:] = new_act

        if readout_weights is not None:
            action = _softmax_sample(activity[4:20] @ readout_weights,
                                     temperature=temperature)
        else:
            action = int(rng.integers(0, 5))

        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(gw.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(gw.grid_size - 1, col + 1)
        elif action == 4:
            for idx, (fr, fc) in enumerate(gw.food_positions):
                if row == fr and col == fc and food_available[idx]:
                    hp = min(gw.hp_max, hp + gw.food_value)
                    food_available[idx] = False
                    food_timers[idx] = 0
                    food_eaten += 1
                    break

        hp -= gw.hp_decay
        steps_survived = step + 1
        for idx in range(len(gw.food_positions)):
            if not food_available[idx]:
                food_timers[idx] += 1
                if food_timers[idx] >= gw.food_respawn:
                    food_available[idx] = True
                    food_timers[idx] = 0

        for i in range(4, N):
            act_hist[i].append(float(activity[i]))
            if node_type[i] == -1:
                rec_timers[i] += 1

        if (step + 1) % K == 0:
            _hebbian_step(G, activity, N, rng)
            recent = {i: float(np.mean(act_hist[i])) if act_hist[i] else 0.0
                      for i in range(4, N)}
            exc_cands = [i for i in range(4, N)
                         if node_type[i] == 1 and recent[i] > ei_thr]
            if exc_cands:
                sw = max(exc_cands, key=lambda i: recent[i])
                node_type[sw] = -1
                rec_timers[sw] = 0
            inh_cands = [i for i in range(4, N)
                         if node_type[i] == -1
                         and recent[i] < ei_thr * rec_ratio
                         and rec_timers[i] >= rec_delay]
            if inh_cands:
                sw = min(inh_cands, key=lambda i: recent[i])
                node_type[sw] = 1
                rec_timers[sw] = 0

    return steps_survived, food_eaten


def run_ei_threshold_evolution(n_agents=10, n_generations=30, n_survivors=3,
                                n_episodes_per_agent=3, N=20, K=10,
                                temperature=1.0, seed=42):
    """Evolve E/I threshold genome (ei_threshold, recovery_ratio, recovery_delay) in GridWorld."""
    from echo_world import _make_graph, _mutate_graph
    print('=== E/I Threshold Evolution ===')
    agents = []
    for i in range(n_agents):
        rng_i = np.random.default_rng(seed + 20000 + i)
        G = _make_graph(N, rng_i)
        rw = rng_i.standard_normal((16, 5)) * 0.1
        genome = {
            'ei_threshold': float(rng_i.uniform(0.5, 0.95)),
            'recovery_ratio': float(rng_i.uniform(0.1, 0.8)),
            'recovery_delay': int(rng_i.integers(0, 201)),
        }
        agents.append((G, np.zeros(N), rng_i, rw, genome))

    gen_mean_fitness, gen_best_fitness, gen_best_genome = [], [], []

    for gen in range(n_generations):
        fitnesses = []
        for G, activity, rng, rw, genome in agents:
            total = 0
            for _ in range(n_episodes_per_agent):
                steps, _ = _run_episode_ei(
                    G, activity.copy(), rng, genome,
                    N=N, K=K, temperature=temperature, readout_weights=rw)
                total += steps
            fitnesses.append(total)

        ranked = sorted(range(n_agents), key=lambda i: fitnesses[i], reverse=True)
        mean_fit = float(np.mean(fitnesses))
        best_idx = ranked[0]
        best_genome = agents[best_idx][4]

        gen_mean_fitness.append(mean_fit)
        gen_best_fitness.append(float(fitnesses[best_idx]))
        gen_best_genome.append(dict(best_genome))

        if gen == 0 or (gen + 1) % 5 == 0:
            print(f'Gen {gen + 1:2d}: mean={mean_fit:.0f}  best={fitnesses[best_idx]:.0f}  '
                  f'[thr={best_genome["ei_threshold"]:.3f}, '
                  f'ratio={best_genome["recovery_ratio"]:.3f}, '
                  f'delay={best_genome["recovery_delay"]}]')

        if gen < n_generations - 1:
            survivors = [agents[ranked[i]] for i in range(n_survivors)]
            new_agents = []
            for i in range(n_agents):
                src = i % n_survivors
                G_src, _, _, rw_src, gn_src = survivors[src]
                rng_new = np.random.default_rng(gen * 10000 + i + 30000)
                G_new = _mutate_graph(G_src, N, rng_new, mutation_std=0.05)
                rw_new = rw_src + rng_new.standard_normal((16, 5)) * 0.05
                gn_new = _mutate_genome(gn_src, rng_new)
                new_agents.append((G_new, np.zeros(N), rng_new, rw_new, gn_new))
            agents = new_agents

    best_final = gen_best_genome[-1]
    print(f'\nFinal best genome: thr={best_final["ei_threshold"]:.4f}  '
          f'ratio={best_final["recovery_ratio"]:.4f}  '
          f'delay={best_final["recovery_delay"]}')
    return {
        'gen_mean_fitness': gen_mean_fitness,
        'gen_best_fitness': gen_best_fitness,
        'gen_best_genome': gen_best_genome,
        'best_genome': best_final,
        'n_generations': n_generations,
    }


def plot_ei_evolution(data, fname='images/results_ei_evolution.png'):
    n_gen = data['n_generations']
    gens = np.arange(1, n_gen + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    fig.suptitle('E/I Threshold Evolution in GridWorld', fontsize=13)

    ax = axes[0][0]
    ax.plot(gens, data['gen_best_fitness'], label='Best', color='steelblue')
    ax.plot(gens, data['gen_mean_fitness'], label='Mean', color='steelblue',
            linestyle='--', alpha=0.6)
    ax.set_ylabel('Total steps (3 episodes)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Fitness', fontsize=11)

    ax = axes[0][1]
    ax.plot(gens, [g['ei_threshold'] for g in data['gen_best_genome']],
            color='seagreen', marker='o', ms=3)
    ax.axhline(0.9, color='tomato', linestyle=':', alpha=0.7, label='fixed baseline (0.9)')
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel('ei_threshold', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Evolved ei_threshold', fontsize=11)

    ax = axes[1][0]
    ax.plot(gens, [g['recovery_ratio'] for g in data['gen_best_genome']],
            color='darkorange', marker='o', ms=3)
    ax.axhline(0.5, color='tomato', linestyle=':', alpha=0.7, label='fixed baseline (0.5)')
    ax.set_ylim(0.05, 0.85)
    ax.set_ylabel('recovery_ratio', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Evolved recovery_ratio', fontsize=11)

    ax = axes[1][1]
    ax.plot(gens, [g['recovery_delay'] for g in data['gen_best_genome']],
            color='mediumpurple', marker='o', ms=3)
    ax.axhline(0, color='tomato', linestyle=':', alpha=0.7, label='fixed baseline (0)')
    ax.set_ylim(-5, 205)
    ax.set_ylabel('recovery_delay (steps)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Evolved recovery_delay', fontsize=11)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def _build_association_graph(N, rng):
    """Standard loop-based association graph used in all sparse conditions."""
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.1:
                G.add_edge(i, j, weight=float(rng.uniform(0.01, 0.1)))
    G.add_edge(0, 2, weight=0.5)
    G.add_edge(1, 5, weight=0.5)
    G.add_edge(11, 8, weight=0.5)
    for s, d in [(2, 3), (3, 4), (4, 2)]:
        G.add_edge(s, d, weight=0.5)
    for s, d in [(5, 6), (6, 7), (7, 5)]:
        G.add_edge(s, d, weight=0.5)
    for s, d in [(8, 9), (9, 10), (10, 8)]:
        G.add_edge(s, d, weight=0.5)
    return G


def _probe_graph(graph, init_vals, N, T_probe):
    """Run T_probe steps on frozen graph; return activity trajectory (T_probe, N)."""
    act = np.zeros(N)
    for node, val in init_vals.items():
        act[node] = val
    traj = []
    for _ in range(T_probe):
        new_act = np.zeros(N)
        for i in range(N):
            s = sum(graph[j][i]['weight'] * act[j] for j in graph.predecessors(i))
            new_act[i] = np.tanh(s)
        for node, val in init_vals.items():
            new_act[node] = val
        act = new_act
        traj.append(act.copy())
    return np.array(traj)


def run_sparse_association(genome, N=20, seed=42, K=5, T_probe=200):
    """Association experiment using E/I isolation dynamics from evolved genome (Step 2)."""
    rng = np.random.default_rng(seed)
    G = _build_association_graph(N, rng)

    loop_food = [2, 3, 4]
    loop_north = [5, 6, 7]
    loop_south = [8, 9, 10]
    bg = list(range(12, N))

    ei_thr = genome['ei_threshold']
    rec_ratio = genome['recovery_ratio']
    rec_delay = genome['recovery_delay']
    ei_window = 100

    activity = np.zeros(N)
    node_type = np.ones(N, dtype=float)
    act_hist = [deque(maxlen=ei_window) for _ in range(N)]
    rec_timers = np.zeros(N, dtype=int)

    def _update(ext):
        new_act = np.zeros(N)
        for i in range(N):
            if i in ext:
                new_act[i] = ext[i]
            elif node_type[i] == 1:
                s = sum(G[j][i]['weight'] * activity[j]
                        for j in G.predecessors(i) if node_type[j] == 1)
                new_act[i] = np.tanh(s)
            else:
                new_act[i] = activity[i] * 0.9
        activity[:] = new_act

    def _hebb_ei():
        edges_to_remove = []
        for i, j, d in list(G.edges(data=True)):
            if node_type[i] == 1 and node_type[j] == 1:
                w = d['weight']
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
                    if (i, j) not in existing and rng.random() < 0.005:
                        G.add_edge(i, j, weight=0.05)
        for i in range(N):
            act_hist[i].append(float(activity[i]))
            if node_type[i] == -1:
                rec_timers[i] += 1
        recent = {i: float(np.mean(act_hist[i])) if act_hist[i] else 0.0
                  for i in range(N)}
        exc_cands = [i for i in range(N)
                     if node_type[i] == 1 and recent[i] > ei_thr]
        if exc_cands:
            sw = max(exc_cands, key=lambda i: recent[i])
            node_type[sw] = -1
            rec_timers[sw] = 0
        inh_cands = [i for i in range(N)
                     if node_type[i] == -1
                     and recent[i] < ei_thr * rec_ratio
                     and rec_timers[i] >= rec_delay]
        if inh_cands:
            sw = min(inh_cands, key=lambda i: recent[i])
            node_type[sw] = 1
            rec_timers[sw] = 0

    def _train(ext, steps):
        for t in range(steps):
            _update(ext)
            if (t + 1) % K == 0:
                _hebb_ei()

    def _cross(a, b):
        return sum(1 for i in a for j in b if G.has_edge(i, j) or G.has_edge(j, i))

    def _sparsity():
        return float(np.sum(node_type == -1)) / N

    def _node_act_sparsity():
        return float(np.sum(activity < 0.1)) / N

    print(f'=== Sparse Association [thr={ei_thr:.3f} ratio={rec_ratio:.3f} '
          f'delay={rec_delay}] ===')

    _train({1: 0.8}, 500)
    cross_1 = _cross(loop_food, loop_north)
    sp_1 = _sparsity()
    nact_1 = _node_act_sparsity()

    _train({0: 0.8}, 500)
    cross_2 = _cross(loop_food, loop_north)
    sp_2 = _sparsity()
    nact_2 = _node_act_sparsity()

    G_baseline = G.copy()

    _train({0: 0.8, 1: 0.8}, 3000)
    cross_3 = _cross(loop_food, loop_north)
    sp_3 = _sparsity()
    nact_3 = _node_act_sparsity()

    G_after_p3 = G.copy()

    _train({0: 0.8, 11: 0.8}, 3000)
    cross_4 = _cross(loop_food, loop_south)
    sp_4 = _sparsity()
    nact_4 = _node_act_sparsity()

    def _gm(traj):
        return (
            float(np.mean(traj[:, loop_food])),
            float(np.mean(traj[:, loop_north])),
            float(np.mean(traj[:, loop_south])),
            float(np.mean(traj[:, bg])) if bg else 0.0,
        )

    r1 = _gm(_probe_graph(G_baseline, {0: 0.5}, N, T_probe))
    r2 = _gm(_probe_graph(G_baseline, {1: 0.5}, N, T_probe))
    r3 = _gm(_probe_graph(G_baseline, {11: 0.5}, N, T_probe))
    r4 = _gm(_probe_graph(G_after_p3, {0: 0.5}, N, T_probe))
    r5 = _gm(_probe_graph(G, {0: 0.5}, N, T_probe))

    north_reactivated = r4[1] > 0.3
    south_reactivated = r5[2] > 0.3
    assoc = north_reactivated and south_reactivated

    print(f'Cross-edges food-north: {cross_1} → {cross_2} → {cross_3}')
    print(f'Inh-fraction:  {sp_1:.3f} → {sp_2:.3f} → {sp_3:.3f} → {sp_4:.3f}')
    print(f'Act-sparsity:  {nact_1:.3f} → {nact_2:.3f} → {nact_3:.3f} → {nact_4:.3f}')
    print(f'Probe4 North={r4[1]:.4f}  Probe5 South={r5[2]:.4f}  assoc={assoc}')

    return {
        'results': [r1, r2, r3, r4, r5],
        'north_reactivated': north_reactivated,
        'south_reactivated': south_reactivated,
        'association_supported': assoc,
        'cross_edges': [cross_1, cross_2, cross_3, cross_4],
        'sparsity': [sp_1, sp_2, sp_3, sp_4],
        'act_sparsity': [nact_1, nact_2, nact_3, nact_4],
        'mean_sparsity': float(np.mean([sp_1, sp_2, sp_3, sp_4])),
        'genome': genome,
        'condition': 'evolved_ei',
    }


def plot_sparse_association(data, fname='images/results_sparse_association.png'):
    probe_labels = [
        'Probe 1\n(food, baseline)',
        'Probe 2\n(north, baseline)',
        'Probe 3\n(south, baseline)',
        'Probe 4\n(food after\nnorth+food)',
        'Probe 5\n(food after\nsouth+food)',
    ]
    group_labels = ['Loop Food\n(2,3,4)', 'Loop North\n(5,6,7)',
                    'Loop South\n(8,9,10)', 'Background\n(12-19)']
    colors = ['steelblue', 'seagreen', 'darkorange', 'gray']
    genome = data.get('genome', {})
    ei_thr = genome.get('ei_threshold', '?')
    rec_ratio = genome.get('recovery_ratio', '?')
    rec_delay = genome.get('recovery_delay', '?')
    thr_str = f'{ei_thr:.3f}' if isinstance(ei_thr, float) else str(ei_thr)
    rat_str = f'{rec_ratio:.3f}' if isinstance(rec_ratio, float) else str(rec_ratio)
    title = f'Sparse Association  [thr={thr_str}  ratio={rat_str}  delay={rec_delay}]'

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), squeeze=False)
    fig.suptitle(title, fontsize=12)

    for col, (label, res) in enumerate(zip(probe_labels, data['results'])):
        axes[0][col].set_title(label, fontsize=10)
        axes[0][col].bar(group_labels, list(res), color=colors, alpha=0.8, width=0.6)
        axes[0][col].set_ylim(0, 1.1)
        axes[0][col].grid(True, alpha=0.3, axis='y')
        if col == 0:
            axes[0][col].set_ylabel('Mean activity', fontsize=9)

    txt_color = 'darkgreen' if data['association_supported'] else 'firebrick'
    fig.text(
        0.5, 0.01,
        f'Association: {data["association_supported"]}  '
        f'mean_inh_frac={data["mean_sparsity"]:.3f}  '
        f'cross-edges(food-north after ph3)={data["cross_edges"][2]}',
        ha='center', fontsize=10, color=txt_color,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def _run_association_fixed_ei(N=20, seed=42, K=5, T_probe=200):
    """Association with fixed E/I thresholds (thr=0.9, ratio=0.5, delay=0) — current baseline."""
    fixed_genome = {'ei_threshold': 0.9, 'recovery_ratio': 0.5, 'recovery_delay': 0}
    result = run_sparse_association(fixed_genome, N=N, seed=seed, K=K, T_probe=T_probe)
    result['condition'] = 'fixed_ei'
    return result


def _run_association_wta(k=3, N=20, seed=42, K=5, T_probe=200):
    """Association with k-sparse WTA: top-k internal nodes only active per step."""
    rng = np.random.default_rng(seed)
    G = _build_association_graph(N, rng)

    loop_food = [2, 3, 4]
    loop_north = [5, 6, 7]
    loop_south = [8, 9, 10]
    bg = list(range(12, N))
    activity = np.zeros(N)

    def _update_wta(ext):
        new_act = np.zeros(N)
        for i in range(N):
            if i in ext:
                new_act[i] = ext[i]
            else:
                s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                new_act[i] = np.tanh(s)
        free = sorted([(new_act[i], i) for i in range(N) if i not in ext], reverse=True)
        for _, i in free[k:]:
            new_act[i] = 0.0
        activity[:] = new_act

    def _hebb_wta():
        edges_to_remove = []
        for i, j, d in list(G.edges(data=True)):
            w = d['weight']
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
                if i != j and (i, j) not in existing and rng.random() < 0.005:
                    G.add_edge(i, j, weight=0.05)

    def _train(ext, steps):
        for t in range(steps):
            _update_wta(ext)
            if (t + 1) % K == 0:
                _hebb_wta()

    def _cross(a, b):
        return sum(1 for i in a for j in b if G.has_edge(i, j) or G.has_edge(j, i))

    def _sparsity():
        return float(np.sum(activity < 0.1)) / N

    _train({1: 0.8}, 500)
    cross_1 = _cross(loop_food, loop_north)
    sp_1 = _sparsity()

    _train({0: 0.8}, 500)
    cross_2 = _cross(loop_food, loop_north)
    sp_2 = _sparsity()

    G_baseline = G.copy()

    _train({0: 0.8, 1: 0.8}, 3000)
    cross_3 = _cross(loop_food, loop_north)
    sp_3 = _sparsity()

    G_after_p3 = G.copy()

    _train({0: 0.8, 11: 0.8}, 3000)
    cross_4 = _cross(loop_food, loop_south)
    sp_4 = _sparsity()

    def _gm(traj):
        return (
            float(np.mean(traj[:, loop_food])),
            float(np.mean(traj[:, loop_north])),
            float(np.mean(traj[:, loop_south])),
            float(np.mean(traj[:, bg])) if bg else 0.0,
        )

    r1 = _gm(_probe_graph(G_baseline, {0: 0.5}, N, T_probe))
    r2 = _gm(_probe_graph(G_baseline, {1: 0.5}, N, T_probe))
    r3 = _gm(_probe_graph(G_baseline, {11: 0.5}, N, T_probe))
    r4 = _gm(_probe_graph(G_after_p3, {0: 0.5}, N, T_probe))
    r5 = _gm(_probe_graph(G, {0: 0.5}, N, T_probe))

    north_reactivated = r4[1] > 0.3
    south_reactivated = r5[2] > 0.3
    assoc = north_reactivated and south_reactivated

    return {
        'results': [r1, r2, r3, r4, r5],
        'north_reactivated': north_reactivated,
        'south_reactivated': south_reactivated,
        'association_supported': assoc,
        'cross_edges': [cross_1, cross_2, cross_3, cross_4],
        'sparsity': [sp_1, sp_2, sp_3, sp_4],
        'act_sparsity': [sp_1, sp_2, sp_3, sp_4],
        'mean_sparsity': float(np.mean([sp_1, sp_2, sp_3, sp_4])),
        'k': k,
        'condition': 'wta',
    }


def run_sparse_comparison(evolved_genome, N=20, seed=42, K=5, T_probe=200):
    """Compare 3 sparsity conditions in the association experiment (Step 3)."""
    print('\n=== Sparse Comparison: 3 conditions ===')
    print('Condition 1: Evolved E/I thresholds')
    res_evolved = run_sparse_association(evolved_genome, N=N, seed=seed, K=K, T_probe=T_probe)

    print('\nCondition 2: Fixed E/I (thr=0.9, ratio=0.5, delay=0)')
    res_fixed = _run_association_fixed_ei(N=N, seed=seed, K=K, T_probe=T_probe)

    print('\nCondition 3: k-sparse WTA (k=3)')
    res_wta = _run_association_wta(k=3, N=N, seed=seed, K=K, T_probe=T_probe)

    return {'evolved': res_evolved, 'fixed': res_fixed, 'wta': res_wta}


def plot_sparse_comparison(data, fname='images/results_sparse_comparison.png'):
    conditions = ['evolved', 'fixed', 'wta']
    cond_labels = ['Evolved E/I', 'Fixed E/I\n(thr=0.9)', 'k-sparse WTA\n(k=3)']
    phase_labels = ['Ph1\nnorth', 'Ph2\nfood', 'Ph3\nnorth+food', 'Ph4\nsouth+food']
    phase_colors = ['#6baed6', '#74c476', '#fdae6b', '#9ecae1']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    fig.suptitle('Sparse Comparison: Evolved E/I  vs  Fixed E/I  vs  k-sparse WTA', fontsize=13)
    row_labels = ['Cross-edges (food↔loop)', 'Sparsity (zero/inh fraction)', 'Probe reactivation']

    for col, cond in enumerate(conditions):
        d = data[cond]
        axes[0][col].set_title(cond_labels[col], fontsize=11)

        # Row 0: cross-edges per phase
        axes[0][col].bar(phase_labels, d['cross_edges'], color=phase_colors)
        axes[0][col].set_ylim(0, max(max(d['cross_edges']) + 2, 5))
        axes[0][col].grid(True, alpha=0.3, axis='y')

        # Row 1: sparsity per phase
        axes[1][col].bar(phase_labels, d['sparsity'], color=phase_colors)
        axes[1][col].set_ylim(0, 1.05)
        axes[1][col].grid(True, alpha=0.3, axis='y')

        # Row 2: Probe 4 North and Probe 5 South
        p4_north = d['results'][3][1]
        p5_south = d['results'][4][2]
        bar_vals = [p4_north, p5_south]
        bar_labs = ['P4 North\n(>0.3?)', 'P5 South\n(>0.3?)']
        bar_colors = ['seagreen' if v > 0.3 else 'tomato' for v in bar_vals]
        ax = axes[2][col]
        ax.bar(bar_labs, bar_vals, color=bar_colors, alpha=0.85)
        ax.axhline(0.3, color='black', linestyle=':', linewidth=1)
        ax.set_ylim(0, max(max(bar_vals) * 1.2 if bar_vals else 0.5, 0.5))
        ax.grid(True, alpha=0.3, axis='y')
        assoc_str = str(d['association_supported'])
        assoc_color = 'darkgreen' if d['association_supported'] else 'firebrick'
        ax.set_xlabel(f'assoc={assoc_str}', fontsize=9, color=assoc_color)

        for row in range(3):
            if col == 0:
                axes[row][col].set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


# ─── Session 4: Context interference environment ──────────────────────────────


def _run_context_episode_ei(G, activity, rng, genome, N=20, K=10,
                              n_propagation_steps=3, temperature=1.0,
                              readout_weights=None, mode=None):
    """ContextGridWorld episode with E/I isolation dynamics.
    Returns (steps_survived, food_eaten, mode)."""
    from echo_world import ContextGridWorld, _softmax_sample, _hebbian_step
    gw = ContextGridWorld()
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'
    food_pos = (0, 0) if mode == 'A' else (4, 4)
    food_available = [True]
    food_timers = [0]

    ei_thr = genome['ei_threshold']
    rec_ratio = genome['recovery_ratio']
    rec_delay = genome['recovery_delay']
    ei_window = 50

    row, col = gw.start_pos
    hp = gw.start_hp
    food_eaten = 0
    steps_survived = 0

    node_type = np.ones(N, dtype=float)
    act_hist = [deque(maxlen=ei_window) for _ in range(N)]
    rec_timers = np.zeros(N, dtype=int)

    for step in range(gw.max_steps):
        if hp <= 0:
            break
        inp = {
            0: col / (gw.grid_size - 1),
            1: row / (gw.grid_size - 1),
            2: hp / gw.hp_max,
            3: 1.0 if food_available[0] else 0.0,
        }
        for _ in range(n_propagation_steps):
            new_act = np.zeros(N)
            for i in range(N):
                if i in inp:
                    new_act[i] = inp[i]
                elif node_type[i] == 1:
                    s = sum(G[j][i]['weight'] * activity[j]
                            for j in G.predecessors(i) if node_type[j] == 1)
                    new_act[i] = np.tanh(s)
                else:
                    new_act[i] = activity[i] * 0.9
            activity[:] = new_act

        if readout_weights is not None:
            action = _softmax_sample(activity[4:20] @ readout_weights,
                                     temperature=temperature)
        else:
            action = int(rng.integers(0, 5))

        fr, fc = food_pos
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(gw.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(gw.grid_size - 1, col + 1)
        elif action == 4:
            if row == fr and col == fc and food_available[0]:
                hp = min(gw.hp_max, hp + gw.food_value)
                food_available[0] = False
                food_timers[0] = 0
                food_eaten += 1

        hp -= gw.hp_decay
        steps_survived = step + 1
        if not food_available[0]:
            food_timers[0] += 1
            if food_timers[0] >= gw.food_respawn:
                food_available[0] = True
                food_timers[0] = 0

        for i in range(4, N):
            act_hist[i].append(float(activity[i]))
            if node_type[i] == -1:
                rec_timers[i] += 1

        if (step + 1) % K == 0:
            _hebbian_step(G, activity, N, rng)
            recent = {i: float(np.mean(act_hist[i])) if act_hist[i] else 0.0
                      for i in range(4, N)}
            exc_cands = [i for i in range(4, N)
                         if node_type[i] == 1 and recent[i] > ei_thr]
            if exc_cands:
                sw = max(exc_cands, key=lambda i: recent[i])
                node_type[sw] = -1
                rec_timers[sw] = 0
            inh_cands = [i for i in range(4, N)
                         if node_type[i] == -1
                         and recent[i] < ei_thr * rec_ratio
                         and rec_timers[i] >= rec_delay]
            if inh_cands:
                sw = min(inh_cands, key=lambda i: recent[i])
                node_type[sw] = 1
                rec_timers[sw] = 0

    return steps_survived, food_eaten, mode


def run_context_ei_evolution(n_agents=10, n_generations=30, n_survivors=3,
                               n_episodes_per_agent=3, N=20, K=10,
                               temperature=1.0, seed=42):
    """Evolve E/I genome in ContextGridWorld (A/B context interference). Step 2."""
    from echo_world import _make_graph, _mutate_graph
    print('=== Context E/I Threshold Evolution ===')
    agents = []
    for i in range(n_agents):
        rng_i = np.random.default_rng(seed + 40000 + i)
        G = _make_graph(N, rng_i)
        rw = rng_i.standard_normal((16, 5)) * 0.1
        genome = {
            'ei_threshold': float(rng_i.uniform(0.5, 0.95)),
            'recovery_ratio': float(rng_i.uniform(0.1, 0.8)),
            'recovery_delay': int(rng_i.integers(0, 201)),
        }
        agents.append((G, np.zeros(N), rng_i, rw, genome))

    gen_mean_fitness, gen_best_fitness, gen_best_genome = [], [], []

    for gen in range(n_generations):
        fitnesses = []
        for G, activity, rng, rw, genome in agents:
            total = 0
            for _ in range(n_episodes_per_agent):
                steps, _, _ = _run_context_episode_ei(
                    G, activity.copy(), rng, genome,
                    N=N, K=K, temperature=temperature, readout_weights=rw)
                total += steps
            fitnesses.append(total)

        ranked = sorted(range(n_agents), key=lambda i: fitnesses[i], reverse=True)
        mean_fit = float(np.mean(fitnesses))
        best_idx = ranked[0]
        best_genome = agents[best_idx][4]

        gen_mean_fitness.append(mean_fit)
        gen_best_fitness.append(float(fitnesses[best_idx]))
        gen_best_genome.append(dict(best_genome))

        if gen == 0 or (gen + 1) % 5 == 0:
            print(f'Gen {gen + 1:2d}: mean={mean_fit:.0f}  best={fitnesses[best_idx]:.0f}  '
                  f'[thr={best_genome["ei_threshold"]:.3f}, '
                  f'ratio={best_genome["recovery_ratio"]:.3f}, '
                  f'delay={best_genome["recovery_delay"]}]')

        if gen < n_generations - 1:
            survivors = [agents[ranked[i]] for i in range(n_survivors)]
            new_agents = []
            for i in range(n_agents):
                src = i % n_survivors
                G_src, _, _, rw_src, gn_src = survivors[src]
                rng_new = np.random.default_rng(gen * 10000 + i + 50000)
                G_new = _mutate_graph(G_src, N, rng_new, mutation_std=0.05)
                rw_new = rw_src + rng_new.standard_normal((16, 5)) * 0.05
                gn_new = _mutate_genome(gn_src, rng_new)
                new_agents.append((G_new, np.zeros(N), rng_new, rw_new, gn_new))
            agents = new_agents

    best_final = gen_best_genome[-1]
    print(f'\nFinal best genome: thr={best_final["ei_threshold"]:.4f}  '
          f'ratio={best_final["recovery_ratio"]:.4f}  '
          f'delay={best_final["recovery_delay"]}')
    return {
        'gen_mean_fitness': gen_mean_fitness,
        'gen_best_fitness': gen_best_fitness,
        'gen_best_genome': gen_best_genome,
        'best_genome': best_final,
        'n_generations': n_generations,
    }


def plot_context_ei_evolution(context_data, simple_data=None,
                               fname='images/results_context_ei_evolution.png'):
    """Plot context evolution genome convergence, optionally overlaid with simple-GridWorld results."""
    n_gen = context_data['n_generations']
    gens = np.arange(1, n_gen + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    fig.suptitle('E/I Threshold Evolution: Context GridWorld vs Simple GridWorld', fontsize=13)

    ax = axes[0][0]
    ax.plot(gens, context_data['gen_best_fitness'], label='Context best', color='steelblue')
    ax.plot(gens, context_data['gen_mean_fitness'], label='Context mean',
            color='steelblue', linestyle='--', alpha=0.6)
    if simple_data is not None:
        ns = min(n_gen, simple_data['n_generations'])
        ax.plot(np.arange(1, ns + 1), simple_data['gen_best_fitness'][:ns],
                label='Simple best', color='tomato', linestyle='-.')
    ax.set_ylabel('Total steps (3 episodes)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Fitness', fontsize=11)

    param_specs = [
        ('ei_threshold',   (0, 1),   'axes[0][1]', 'seagreen',    'Evolved ei_threshold',   0.9,   'fixed (0.9)'),
        ('recovery_ratio', (0, 0.9), 'axes[1][0]', 'darkorange',  'Evolved recovery_ratio', 0.5,   'fixed (0.5)'),
        ('recovery_delay', (-5, 205),'axes[1][1]', 'mediumpurple','Evolved recovery_delay', 0,     'fixed (0)'),
    ]

    for param, ylim, ax_expr, color, title, baseline, bl_label in param_specs:
        ax = eval(ax_expr)
        vals_ctx = [g[param] for g in context_data['gen_best_genome']]
        ax.plot(gens, vals_ctx, color=color, marker='o', ms=3, label='Context')
        if simple_data is not None:
            ns = min(n_gen, simple_data['n_generations'])
            vals_sim = [g[param] for g in simple_data['gen_best_genome'][:ns]]
            ax.plot(np.arange(1, ns + 1), vals_sim,
                    color='tomato', marker='s', ms=3, linestyle='-.', label='Simple')
        ax.axhline(baseline, color='gray', linestyle=':', alpha=0.7, label=bl_label)
        ax.set_ylim(*ylim)
        ax.set_ylabel(param, fontsize=10)
        ax.set_xlabel('Generation', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def _collect_context_activity(G, readout_weights, genome, seed, N=20, K=10,
                                temperature=1.0, n_episodes=6, use_ei=True):
    """Run n_episodes in ContextGridWorld; return per-mode mean activity and metrics."""
    from echo_world import _make_graph, _softmax_sample, _hebbian_step, ContextGridWorld
    rng = np.random.default_rng(seed)
    acts_A, acts_B = [], []
    survival_list, sparsity_list = [], []

    for ep in range(n_episodes):
        mode = 'A' if ep % 2 == 0 else 'B'
        rng_ep = np.random.default_rng(seed + ep)
        G_ep = G.copy()
        activity = np.zeros(N)

        if use_ei:
            steps, _, ep_mode = _run_context_episode_ei(
                G_ep, activity, rng_ep, genome,
                N=N, K=K, temperature=temperature,
                readout_weights=readout_weights, mode=mode)
        else:
            # No E/I: use plain context episode from echo_world
            from echo_world import _run_context_episode
            steps, _, _, _, ep_mode = _run_context_episode(
                G_ep, activity, rng_ep, N=N, K=K,
                temperature=temperature, readout_weights=readout_weights,
                mode=mode)

        # Run a fixed-topology measurement pass to collect steady-state activity
        gw = ContextGridWorld()
        food_pos = (0, 0) if mode == 'A' else (4, 4)
        row, col = gw.start_pos
        hp = gw.start_hp
        food_avail = [True]
        food_timer = [0]
        rng_m = np.random.default_rng(seed + 1000 + ep)
        act_snap = np.zeros(N)
        step_acts = []

        for step in range(min(200, gw.max_steps)):
            if hp <= 0:
                break
            inp = {
                0: col / (gw.grid_size - 1),
                1: row / (gw.grid_size - 1),
                2: hp / gw.hp_max,
                3: 1.0 if food_avail[0] else 0.0,
            }
            for _ in range(3):
                new_act = np.zeros(N)
                for i in range(N):
                    if i in inp:
                        new_act[i] = inp[i]
                    else:
                        s = sum(G_ep[j][i]['weight'] * act_snap[j]
                                for j in G_ep.predecessors(i))
                        new_act[i] = np.tanh(s)
                act_snap[:] = new_act
            step_acts.append(act_snap[4:20].copy())

            if readout_weights is not None:
                action = _softmax_sample(act_snap[4:20] @ readout_weights,
                                         temperature=temperature)
            else:
                action = int(rng_m.integers(0, 5))
            fr, fc = food_pos
            if action == 0:
                row = max(0, row - 1)
            elif action == 1:
                row = min(gw.grid_size - 1, row + 1)
            elif action == 2:
                col = max(0, col - 1)
            elif action == 3:
                col = min(gw.grid_size - 1, col + 1)
            elif action == 4:
                if row == fr and col == fc and food_avail[0]:
                    hp = min(gw.hp_max, hp + gw.food_value)
                    food_avail[0] = False
                    food_timer[0] = 0
            hp -= gw.hp_decay
            if not food_avail[0]:
                food_timer[0] += 1
                if food_timer[0] >= gw.food_respawn:
                    food_avail[0] = True
                    food_timer[0] = 0

        mean_act = np.mean(step_acts, axis=0) if step_acts else np.zeros(16)
        sparsity = float(np.mean(mean_act < 0.1))
        survival_list.append(steps)
        sparsity_list.append(sparsity)
        if mode == 'A':
            acts_A.append(mean_act)
        else:
            acts_B.append(mean_act)

    mean_A = np.mean(acts_A, axis=0) if acts_A else np.zeros(16)
    mean_B = np.mean(acts_B, axis=0) if acts_B else np.zeros(16)
    norm_A = np.linalg.norm(mean_A)
    norm_B = np.linalg.norm(mean_B)
    if norm_A > 1e-10 and norm_B > 1e-10:
        cos_dist = 1.0 - float(np.dot(mean_A, mean_B) / (norm_A * norm_B))
    else:
        cos_dist = float('nan')

    return {
        'mean_survival': float(np.mean(survival_list)),
        'mean_sparsity': float(np.mean(sparsity_list)),
        'cosine_dist': cos_dist,
        'mean_A': mean_A,
        'mean_B': mean_B,
        'survival_list': survival_list,
        'sparsity_list': sparsity_list,
    }


def run_context_comparison(simple_genome, context_genome, N=20, K=10,
                            temperature=1.0, seed=42, n_episodes=6):
    """Step 4: Compare 3 conditions in ContextGridWorld."""
    from echo_world import _make_graph
    print('\n=== Context Comparison: 3 conditions ===')
    no_ei_genome = {'ei_threshold': 0.99, 'recovery_ratio': 0.01, 'recovery_delay': 10000}

    conditions = [
        ('Simple genome\n(prev session)',  simple_genome,  True),
        ('Context genome\n(this session)', context_genome, True),
        ('No E/I baseline',                no_ei_genome,   False),
    ]

    rng_base = np.random.default_rng(seed + 60000)
    G_base = _make_graph(N, rng_base)
    rw_base = rng_base.standard_normal((16, 5)) * 0.1

    results = {}
    for cond_name, genome, use_ei in conditions:
        print(f'Evaluating: {cond_name.replace(chr(10), " ")}')
        metrics = _collect_context_activity(
            G_base, rw_base, genome, seed=seed + 60000,
            N=N, K=K, temperature=temperature,
            n_episodes=n_episodes, use_ei=use_ei)
        print(f'  survival={metrics["mean_survival"]:.0f}  '
              f'sparsity={metrics["mean_sparsity"]:.3f}  '
              f'cosine_dist={metrics["cosine_dist"]:.4f}')
        results[cond_name] = metrics

    return results


def plot_context_comparison(data, fname='images/results_context_comparison.png'):
    cond_keys = list(data.keys())
    n_conds = len(cond_keys)
    fig, axes = plt.subplots(2, n_conds, figsize=(5 * n_conds, 10), squeeze=False)
    fig.suptitle('Context GridWorld: 3 Conditions Compared', fontsize=13)

    row_labels = ['Internal node activity (mode A vs B)', 'Summary metrics']
    bar_colors_A = 'steelblue'
    bar_colors_B = 'tomato'

    for col, key in enumerate(cond_keys):
        d = data[key]
        axes[0][col].set_title(key.replace('\n', ' '), fontsize=10)

        # Row 0: mean activity per node in mode A vs B
        nodes = np.arange(16)
        axes[0][col].bar(nodes - 0.2, d['mean_A'], width=0.4,
                         color=bar_colors_A, alpha=0.8, label='Mode A (NW food)')
        axes[0][col].bar(nodes + 0.2, d['mean_B'], width=0.4,
                         color=bar_colors_B, alpha=0.8, label='Mode B (SE food)')
        axes[0][col].set_ylim(0, 1.05)
        axes[0][col].set_xlabel('Internal node (4-19)', fontsize=9)
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3, axis='y')
        cos_str = f'{d["cosine_dist"]:.4f}' if not np.isnan(d['cosine_dist']) else 'nan'
        axes[0][col].set_title(
            f'{key.replace(chr(10), " ")}\ncosine_dist={cos_str}', fontsize=9)

        # Row 1: summary bar (survival, sparsity, cosine_dist)
        metrics = ['Survival\n(steps)', 'Sparsity\n(zero frac)', 'Cosine dist\nA vs B']
        cos_val = d['cosine_dist'] if not np.isnan(d['cosine_dist']) else 0.0
        vals = [d['mean_survival'] / 500.0, d['mean_sparsity'], cos_val]  # normalize survival to 0-1
        bar_c = ['steelblue', 'seagreen', 'darkorange']
        axes[1][col].bar(metrics, vals, color=bar_c, alpha=0.8)
        axes[1][col].set_ylim(0, 1.1)
        axes[1][col].grid(True, alpha=0.3, axis='y')
        axes[1][col].set_title(
            f'survival={d["mean_survival"]:.0f}  sparsity={d["mean_sparsity"]:.3f}',
            fontsize=8)
        for bar_i, (m, v) in enumerate(zip(metrics, vals)):
            axes[1][col].text(bar_i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

        if col == 0:
            axes[0][col].set_ylabel('Mean activity', fontsize=9)
            axes[1][col].set_ylabel('Normalized value', fontsize=9)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


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


def plot_association_sweep(results, fname='images/results_association_sweep.png'):
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
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_association_probe(results, fname='images/results_association_probe.png'):
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
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ─── Session 6: Dynamic (E/I) vs static (low-weight) inhibition ──────────────
#
# Root cause identified in Session 5 sweep:
#   anatomy edge weight=0.5 → tanh(0.5×0.8)=0.38 < Hebbian threshold 0.5
#   → anatomy edges not reinforced → removed by decay within Phase 1.
#
# Fix: anatomy edges marked fixed=True, weight=1.0 (immune to Hebbian decay).
#   tanh(1.0×0.8)=0.664 > 0.5 → anatomy edges ARE reinforced during training.
#   Probe measures loop_north(6,7,8) / loop_south(9,10,11) mean activity
#   instead of stimulus nodes 1/2 (which had arbitrary random connections).

_S6_EI_GENOME  = {'ei_threshold': 0.821, 'recovery_ratio': 0.503, 'recovery_delay': 0}
_S6_BEST_COND  = {'density': 0.2, 'init_weight': 0.05, 'lr': 0.05}
_S6_LOOP_NORTH = [6, 7, 8]
_S6_LOOP_SOUTH = [9, 10, 11]
_S6_LOOP_FOOD  = [3, 4, 5]


def _build_s6_graph(N, rng, initial_density, initial_weight):
    """Association graph for Session 6 with protected anatomy edges.

    Stimulus: 0=food, 1=north, 2=south
    Loop Food:  3→4→5→3   (anatomy: 0→3)
    Loop North: 6→7→8→6   (anatomy: 1→6)
    Loop South: 9→10→11→9 (anatomy: 2→9)
    Background: 12..N-1

    Anatomy edges: fixed=True, weight=1.0 — immune to Hebbian decay.
    Background edges: fixed=False, weight=initial_weight.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < initial_density:
                G.add_edge(i, j, weight=float(initial_weight), fixed=False)
    for s, d in [(0, 3), (1, 6), (2, 9)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    for s, d in [(3, 4), (4, 5), (5, 3)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    for s, d in [(6, 7), (7, 8), (8, 6)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    for s, d in [(9, 10), (10, 11), (11, 9)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    return G


def _s6_make_ei_state(N, ei_window=100):
    node_type  = np.ones(N, dtype=float)
    act_hist   = [deque(maxlen=ei_window) for _ in range(N)]
    rec_timers = np.zeros(N, dtype=int)
    return node_type, act_hist, rec_timers


def _s6_step(G, activity, ext, N, use_ei, node_type):
    """Propagate network one step (no Hebbian). Modifies activity in-place."""
    new_act = np.zeros(N)
    for i in range(N):
        if i in ext:
            new_act[i] = ext[i]
        elif not use_ei or node_type[i] == 1:
            s = sum(G[j][i]['weight'] * activity[j]
                    for j in G.predecessors(i)
                    if not use_ei or node_type[j] == 1)
            new_act[i] = np.tanh(s)
        else:
            new_act[i] = activity[i] * 0.9   # inhibitory decay
    activity[:] = new_act


def _s6_hebb(G, activity, ext, N, use_ei, node_type, act_hist, rec_timers,
              lr, init_w, rng, ei_thr, rec_ratio, rec_delay):
    """Hebbian update + optional E/I switching.  Called every K steps.
    Fixed-anatomy edges (fixed=True) are skipped — they never decay."""
    to_rm = []
    for i, j, d in list(G.edges(data=True)):
        if d.get('fixed', False):
            continue          # protect anatomy edges
        if use_ei and (node_type[i] != 1 or node_type[j] != 1):
            continue
        w = d['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += lr
        w -= 0.01
        if w < 0.01:
            to_rm.append((i, j))
        else:
            G[i][j]['weight'] = min(w, 1.0)
    G.remove_edges_from(to_rm)
    existing = set(G.edges())
    for i in range(N):
        for j in range(N):
            if i != j and (i, j) not in existing and rng.random() < 0.005:
                G.add_edge(i, j, weight=float(init_w), fixed=False)

    if not use_ei:
        return
    for i in range(N):
        act_hist[i].append(float(activity[i]))
        if node_type[i] == -1:
            rec_timers[i] += 1
    recent = {i: float(np.mean(act_hist[i])) if act_hist[i] else 0.0
              for i in range(N)}
    exc_cands = [i for i in range(N)
                 if i not in ext and node_type[i] == 1 and recent[i] > ei_thr]
    if exc_cands:
        sw = max(exc_cands, key=lambda i: recent[i])
        node_type[sw] = -1
        rec_timers[sw] = 0
    inh_cands = [i for i in range(N)
                 if node_type[i] == -1
                 and recent[i] < ei_thr * rec_ratio
                 and rec_timers[i] >= rec_delay]
    if inh_cands:
        sw = min(inh_cands, key=lambda i: recent[i])
        node_type[sw] = 1
        rec_timers[sw] = 0


def _s6_probe(G, activity_snap, N, T_probe=100):
    """Probe with food-only (node0=0.8), frozen graph, pure propagation.
    Returns (mean_loop_north, mean_loop_south) averaged over T_probe steps."""
    act = activity_snap.copy()
    ln_acc, ls_acc = 0.0, 0.0
    for _ in range(T_probe):
        new_act = np.zeros(N)
        new_act[0] = 0.8
        for i in range(1, N):
            s = sum(G[j][i]['weight'] * act[j] for j in G.predecessors(i))
            new_act[i] = np.tanh(s)
        act = new_act
        ln_acc += float(np.mean(act[_S6_LOOP_NORTH]))
        ls_acc += float(np.mean(act[_S6_LOOP_SOUTH]))
    return ln_acc / T_probe, ls_acc / T_probe


def run_ei_vs_static_experiments(N=20, seed=42, K=5, T_probe=100, probe_interval=10):
    """Experiments A & B: overwrite resistance and association switch speed.

    Two conditions — pure Hebbian ('no_ei') vs E/I isolation ('ei') —
    both starting from density=0.2, init_weight=0.05, hebbian_lr=0.05.

    Exp A: train 5 phases, probe (food-only) after each phase.
    Exp B: during Phase 4, probe every probe_interval steps; record when
           mean_south first exceeds mean_north (association switch step).
    """
    density   = _S6_BEST_COND['density']
    init_w    = _S6_BEST_COND['init_weight']
    lr        = _S6_BEST_COND['lr']
    ei_thr    = _S6_EI_GENOME['ei_threshold']
    rec_ratio = _S6_EI_GENOME['recovery_ratio']
    rec_delay = _S6_EI_GENOME['recovery_delay']

    phases = [
        ({1: 0.8},         500,  'North solo'),
        ({0: 0.8},         500,  'Food solo'),
        ({0: 0.8, 1: 0.8}, 500,  'North+Food'),
        ({0: 0.8, 2: 0.8}, 500,  'South+Food'),
        ({0: 0.8, 2: 0.8}, 1000, 'South+Food ×2'),
    ]

    print('=== Session 6: E/I vs static (Exp A & B) ===')
    results = {}

    for cond in ['no_ei', 'ei']:
        use_ei = (cond == 'ei')
        rng = np.random.default_rng(seed)
        G   = _build_s6_graph(N, rng, density, init_w)
        act = np.zeros(N)
        node_type, act_hist, rec_timers = _s6_make_ei_state(N)

        probe_A  = []
        exp_b    = None

        for p_idx, (ext, T, pname) in enumerate(phases):
            if p_idx == 3:    # Phase 4: Exp B — probe at each checkpoint
                b_steps, b_north, b_south = [], [], []
                first_switch = None
                for t in range(T):
                    _s6_step(G, act, ext, N, use_ei, node_type)
                    if (t + 1) % K == 0:
                        _s6_hebb(G, act, ext, N, use_ei, node_type, act_hist,
                                 rec_timers, lr, init_w, rng,
                                 ei_thr, rec_ratio, rec_delay)
                    if (t + 1) % probe_interval == 0:
                        mn, ms = _s6_probe(G, act, N, T_probe)
                        b_steps.append(t + 1)
                        b_north.append(mn)
                        b_south.append(ms)
                        if first_switch is None and ms > mn:
                            first_switch = t + 1
                exp_b = {
                    'steps': b_steps,
                    'north': b_north,
                    'south': b_south,
                    'first_switch': first_switch,
                }
            else:
                for t in range(T):
                    _s6_step(G, act, ext, N, use_ei, node_type)
                    if (t + 1) % K == 0:
                        _s6_hebb(G, act, ext, N, use_ei, node_type, act_hist,
                                 rec_timers, lr, init_w, rng,
                                 ei_thr, rec_ratio, rec_delay)

            mn, ms = _s6_probe(G, act, N, T_probe)
            probe_A.append({'phase': pname, 'mean_north': mn, 'mean_south': ms})
            print(f'  [{cond}] {pname}: loop_north={mn:.4f}  loop_south={ms:.4f}')

        results[cond] = {
            'probe_A': probe_A,
            'exp_b': exp_b,
            'G_final': G.copy(),
            'activity_final': act.copy(),
        }

    return results


def run_ei_vs_static_context(expt_data, N=20, seed=42, n_episodes=10, T_episode=100):
    """Experiment C: context routing test using the Phase5-trained networks.

    Present two context patterns alternately:
      Mode A: {0: 0.8, 1: 0.4}  — food + NW-like hint
      Mode B: {0: 0.8, 2: 0.4}  — food + SE-like hint

    Measure whether loop_north (6,7,8) activates more for mode A and
    loop_south (9,10,11) activates more for mode B.
    """
    loop_north = [6, 7, 8]
    loop_south = [9, 10, 11]
    internal   = list(range(3, N))

    print('\n=== Session 6: E/I vs static (Exp C) ===')
    results = {}

    for cond, cdata in expt_data.items():
        G        = cdata['G_final']
        act_init = cdata['activity_final']

        a_north_list, a_south_list = [], []
        b_north_list, b_south_list = [], []
        mean_A_vecs, mean_B_vecs   = [], []

        rng_ep = np.random.default_rng(seed + 70000)
        modes  = ['A'] * (n_episodes // 2) + ['B'] * (n_episodes - n_episodes // 2)
        rng_ep.shuffle(modes)

        for mode in modes:
            act = act_init.copy()
            ext = {0: 0.8, 1: 0.4} if mode == 'A' else {0: 0.8, 2: 0.4}
            step_vecs = []
            for _ in range(T_episode):
                new_act = np.zeros(N)
                for i in range(N):
                    if i in ext:
                        new_act[i] = ext[i]
                    else:
                        s = sum(G[j][i]['weight'] * act[j]
                                for j in G.predecessors(i))
                        new_act[i] = np.tanh(s)
                act = new_act
                step_vecs.append(act[internal].copy())

            mean_vec = np.mean(step_vecs, axis=0)
            # indices into internal vector: loop_north[k]-3, loop_south[k]-3
            mn = float(np.mean(mean_vec[[n - 3 for n in loop_north]]))
            ms = float(np.mean(mean_vec[[n - 3 for n in loop_south]]))

            if mode == 'A':
                a_north_list.append(mn); a_south_list.append(ms)
                mean_A_vecs.append(mean_vec)
            else:
                b_north_list.append(mn); b_south_list.append(ms)
                mean_B_vecs.append(mean_vec)

        acc_A = float(np.mean(np.array(a_north_list) > np.array(a_south_list)))
        acc_B = float(np.mean(np.array(b_south_list) > np.array(b_north_list)))

        mu_A = np.mean(mean_A_vecs, axis=0) if mean_A_vecs else np.zeros(len(internal))
        mu_B = np.mean(mean_B_vecs, axis=0) if mean_B_vecs else np.zeros(len(internal))
        norm = np.linalg.norm(mu_A) * np.linalg.norm(mu_B)
        cos_dist = 1.0 - float(np.dot(mu_A, mu_B)) / (norm + 1e-10)

        print(f'  [{cond}] acc_A={acc_A:.2f}  acc_B={acc_B:.2f}  cos_dist={cos_dist:.4f}')

        results[cond] = {
            'acc_A': acc_A, 'acc_B': acc_B, 'cos_dist': cos_dist,
            'mean_A_north': float(np.mean(a_north_list)),
            'mean_A_south': float(np.mean(a_south_list)),
            'mean_B_north': float(np.mean(b_north_list)),
            'mean_B_south': float(np.mean(b_south_list)),
            'mean_A_vec': mu_A, 'mean_B_vec': mu_B,
        }

    return results


def plot_ei_vs_static_overwrite(data, fname='images/results_ei_vs_static_overwrite.png'):
    """Plot experiments A and B."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Session 6 — E/I (dynamic) vs pure Hebbian (static): '
                 'overwrite resistance & switch speed', fontsize=11)

    colors = {'no_ei': {'north': 'steelblue', 'south': 'tomato'},
              'ei':    {'north': '#1f77b4',    'south': '#d62728'}}
    alphas = {'no_ei': 1.0, 'ei': 0.55}
    styles = {'no_ei': '-', 'ei': '--'}
    markers = {'no_ei': 'o', 'ei': 's'}
    phase_names = [r['phase'] for r in data['no_ei']['probe_A']]

    # — Ax 0: Exp A — north probe per phase —
    ax = axes[0]
    for cond in ['no_ei', 'ei']:
        probe = data[cond]['probe_A']
        norths = [r['mean_north'] for r in probe]
        souths = [r['mean_south'] for r in probe]
        xs = list(range(len(norths)))
        kw = dict(ls=styles[cond], marker=markers[cond], ms=6, alpha=alphas[cond])
        ax.plot(xs, norths, color=colors[cond]['north'],
                label=f'{cond} north', **kw)
        ax.plot(xs, souths, color=colors[cond]['south'],
                label=f'{cond} south', **kw)
    ax.axhline(0.3, color='gray', lw=0.8, ls=':', label='thr=0.3')
    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Mean loop activity (food-only probe)')
    ax.set_title('Exp A: loop_north/south after each phase')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # — Ax 1: Exp A — south probe per phase —
    ax = axes[1]
    for cond in ['no_ei', 'ei']:
        probe = data[cond]['probe_A']
        norths = [r['mean_north'] for r in probe]
        souths = [r['mean_south'] for r in probe]
        xs = list(range(len(norths)))
        kw = dict(ls=styles[cond], marker=markers[cond], ms=6, alpha=alphas[cond])
        ax.plot(xs, [s - n for s, n in zip(souths, norths)],
                color=colors[cond]['south'],
                label=f'{cond}  south−north')
    ax.axhline(0, color='gray', lw=1.0, ls=':')
    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('mean_south − mean_north (probe)')
    ax.set_title('Exp A: context conflict (south−north margin)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # — Ax 2: Exp B — Phase 4 switch speed —
    ax = axes[2]
    for cond in ['no_ei', 'ei']:
        b = data[cond]['exp_b']
        diff = [s - n for s, n in zip(b['south'], b['north'])]
        sw = b['first_switch']
        lbl = f'{cond} (switch at step {sw})' if sw else f'{cond} (no switch)'
        ax.plot(b['steps'], diff, ls=styles[cond], color='black' if cond == 'no_ei' else 'crimson',
                lw=1.5, label=lbl)
        if sw:
            ax.axvline(sw, ls=':', lw=0.8,
                       color='black' if cond == 'no_ei' else 'crimson')
    ax.axhline(0, color='gray', lw=1.0, ls='--')
    ax.set_xlabel('Training step in Phase 4 (South+Food)')
    ax.set_ylabel('mean_south − mean_north (probe)')
    ax.set_title('Exp B: association switch speed during Phase 4')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_ei_vs_static_context(data, fname='images/results_ei_vs_static_context.png'):
    """Plot experiment C: context routing accuracy and loop activity."""
    conds = list(data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Session 6 — Exp C: context routing after Phase5 training', fontsize=11)

    # — Ax 0: accuracy bars —
    ax = axes[0]
    x  = np.arange(len(conds))
    bw = 0.35
    ax.bar(x - bw/2, [data[c]['acc_A'] for c in conds], bw,
           label='Mode A acc\n(north > south)', color='steelblue', alpha=0.85)
    ax.bar(x + bw/2, [data[c]['acc_B'] for c in conds], bw,
           label='Mode B acc\n(south > north)', color='tomato', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy (fraction of trials)')
    ax.set_title('Context routing accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # — Ax 1: loop activity breakdown —
    ax = axes[1]
    bw = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw
    series  = [
        ('A→North', [data[c]['mean_A_north'] for c in conds], 'steelblue'),
        ('A→South', [data[c]['mean_A_south'] for c in conds], 'lightsteelblue'),
        ('B→North', [data[c]['mean_B_north'] for c in conds], 'lightsalmon'),
        ('B→South', [data[c]['mean_B_south'] for c in conds], 'tomato'),
    ]
    for (lbl, vals, col), off in zip(series, offsets):
        ax.bar(x + off, vals, bw, label=lbl, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel('Mean loop node activity')
    ax.set_title('Loop activity by mode')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # — Ax 2: cosine distance —
    ax = axes[2]
    cos_vals = [data[c]['cos_dist'] for c in conds]
    bars = ax.bar(x, cos_vals, color=['steelblue', 'tomato'], alpha=0.85, width=0.5)
    for bar, val in zip(bars, cos_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel('Cosine distance (A vs B internal activity)')
    ax.set_title('Context separation (cosine dist)')
    ax.set_ylim(0, max(max(cos_vals) * 1.3, 0.05))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=1.5)
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f'Saved {fname}')


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


# ─── Session 8: 世界に問う (ContextGridWorld survival test) ──────────────────

_S8_BEST     = (1000, 200)          # Session 7 best condition (T_phase, switch_interval)
_S8_INTERNAL = list(range(4, 20))  # 16 truly-internal nodes in world context


def _s8_train_best(N, seed):
    """Reproduce Session 7 best-condition training. Returns trained DiGraph."""
    density = _S7_COND['density']
    init_w  = _S7_COND['init_weight']
    lr      = _S7_COND['lr']
    K       = _S7_K
    T_phase, sw = _S8_BEST
    T3 = 2000

    rng = np.random.default_rng(seed)
    G   = _s7_build_graph(N, rng, density, init_w)
    act = np.zeros(N)

    _s7_train_phase(G, act, {0: 0.8, 1: 0.8, 2: 0.0}, T_phase, N, rng, K, lr, init_w)
    _s7_train_phase(G, act, {0: 0.8, 1: 0.0, 2: 0.8}, T_phase, N, rng, K, lr, init_w)

    ext_north = {0: 0.8, 1: 0.8, 2: 0.0}
    ext_south  = {0: 0.8, 1: 0.0, 2: 0.8}
    for t in range(T3):
        ext = ext_north if (t // sw) % 2 == 0 else ext_south
        act[:] = _s7_step(G, act, ext, N)
        if (t + 1) % K == 0:
            _s7_hebb(G, act, N, rng, lr, init_w)
    return G


def _s8_softmax_sample(logits, rng, temperature=1.0):
    """Softmax sampling using local rng for reproducibility."""
    x = logits - np.max(logits)
    p = np.exp(x / temperature)
    p /= p.sum()
    return int(rng.choice(len(logits), p=p))


def _s8_run_episode(G, rng, N, readout_w, topology_frozen, init_weight=0.05, K=10):
    """One ContextGridWorld episode. Returns (steps, food_eaten, mode, mean_internal)."""
    grid_size    = 5
    hp_max       = 200
    hp_decay     = 1
    food_value   = 30
    food_respawn = 40
    max_steps    = 500
    start_hp     = 100

    mode           = 'A' if rng.random() < 0.5 else 'B'
    food_pos       = (0, 0) if mode == 'A' else (4, 4)
    food_available = True
    food_timer     = 0

    row, col   = 2, 2
    hp         = start_hp
    food_eaten = 0
    steps      = 0
    activity   = np.zeros(N)
    acc_int    = np.zeros(len(_S8_INTERNAL))

    for step in range(max_steps):
        if hp <= 0:
            break
        input_vals = {
            0: col / (grid_size - 1),
            1: row / (grid_size - 1),
            2: hp  / hp_max,
            3: 1.0 if food_available else 0.0,
        }
        for _ in range(3):
            new_act = np.zeros(N)
            for i in range(N):
                if i in input_vals:
                    new_act[i] = input_vals[i]
                else:
                    s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                    new_act[i] = np.tanh(s)
            activity[:] = new_act

        acc_int += activity[4:20]

        action = _s8_softmax_sample(activity[4:20] @ readout_w, rng, temperature=1.0)

        fr, fc = food_pos
        if   action == 0: row = max(0, row - 1)
        elif action == 1: row = min(grid_size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(grid_size - 1, col + 1)
        elif action == 4:
            if row == fr and col == fc and food_available:
                hp = min(hp_max, hp + food_value)
                food_available = False
                food_timer     = 0
                food_eaten    += 1

        hp -= hp_decay
        steps = step + 1
        if not food_available:
            food_timer += 1
            if food_timer >= food_respawn:
                food_available = True
                food_timer     = 0

        if not topology_frozen and (step + 1) % K == 0:
            _s7_hebb(G, activity, N, rng, lr=_S7_COND['lr'], init_weight=init_weight)

    mean_int = acc_int / max(steps, 1)
    return steps, food_eaten, mode, mean_int


def _s8_run_agents(cond_label, G_base, rng_main, N, readout_w, topology_frozen,
                   init_weight, n_agents, n_episodes, seed_offset):
    """Run n_agents × n_episodes episodes for one condition."""
    agent_records = []
    for ag in range(n_agents):
        rng_a  = np.random.default_rng(rng_main.integers(0, 2**32) + seed_offset + ag)
        G      = G_base.copy()
        surv_list, food_list, mode_list = [], [], []
        pat_A  = np.zeros(len(_S8_INTERNAL))
        pat_B  = np.zeros(len(_S8_INTERNAL))
        cnt_A = cnt_B = 0
        cos_dists = []
        for ep in range(n_episodes):
            steps, food_eaten, mode, mean_int = _s8_run_episode(
                G, rng_a, N, readout_w, topology_frozen, init_weight)
            surv_list.append(steps)
            food_list.append(food_eaten > 0)
            mode_list.append(mode)
            if mode == 'A':
                pat_A += mean_int; cnt_A += 1
            else:
                pat_B += mean_int; cnt_B += 1
            if cnt_A > 0 and cnt_B > 0:
                a_ = pat_A / cnt_A
                b_ = pat_B / cnt_B
                na = np.linalg.norm(a_)
                nb = np.linalg.norm(b_)
                cd = float(1.0 - np.dot(a_, b_) / (na * nb)) if na > 1e-6 and nb > 1e-6 else float('nan')
            else:
                cd = float('nan')
            cos_dists.append(cd)
        agent_records.append({
            'surv': surv_list, 'food': food_list,
            'mode': mode_list, 'cos_dists': cos_dists,
        })
    print(f'  {cond_label}: mean_surv={np.mean([s for a in agent_records for s in a["surv"]]):.1f}')
    return agent_records


def run_world_test_experiment(N=20, seed=42, n_agents=10, n_episodes=20):
    """Session 8: Compare trained (fixed/dynamic) vs baseline in ContextGridWorld."""
    density  = _S7_COND['density']
    init_w   = _S7_COND['init_weight']

    rng_main = np.random.default_rng(seed)
    readout_w = rng_main.standard_normal((16, 5))

    print('=== Session 8: World test experiment ===')
    print(f'  Training Session 7 best condition (T_phase={_S8_BEST[0]}, sw={_S8_BEST[1]})...')
    base_G = _s8_train_best(N, seed)

    rng_b = np.random.default_rng(seed + 5000)
    base_G3 = _s7_build_graph(N, rng_b, density, init_w)

    r1 = _s8_run_agents('Cond1 trained/frozen',  base_G,  rng_main, N, readout_w, True,  init_w, n_agents, n_episodes, 1000)
    r2 = _s8_run_agents('Cond2 trained/dynamic', base_G,  rng_main, N, readout_w, False, init_w, n_agents, n_episodes, 2000)
    r3 = _s8_run_agents('Cond3 random/frozen',   base_G3, rng_main, N, readout_w, True,  init_w, n_agents, n_episodes, 3000)

    return {
        'results': {1: r1, 2: r2, 3: r3},
        'n_agents': n_agents, 'n_episodes': n_episodes,
    }


def plot_world_test(data, fname='images/results_world_test.png'):
    results    = data['results']
    n_episodes = data['n_episodes']

    labels = ['Trained\n(frozen)', 'Trained\n(dynamic)', 'Random\n(baseline)']
    colors = ['steelblue', 'darkorange', 'gray']

    all_surv = {}
    acc_A    = {}
    acc_B    = {}
    for cond in (1, 2, 3):
        all_surv[cond] = [s for a in results[cond] for s in a['surv']]
        acc_A[cond]    = [a['food'][i] for a in results[cond]
                          for i, m in enumerate(a['mode']) if m == 'A']
        acc_B[cond]    = [a['food'][i] for a in results[cond]
                          for i, m in enumerate(a['mode']) if m == 'B']

    n_agents = data['n_agents']
    ep_cos = {}
    for cond in (1, 2):
        mat = np.full((n_agents, n_episodes), np.nan)
        for ag_i, agent in enumerate(results[cond]):
            for ep_i, cd in enumerate(agent['cos_dists']):
                mat[ag_i, ep_i] = cd
        ep_cos[cond] = np.nanmean(mat, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    fig.suptitle(
        f'Session 8: 世界に問う — trained (S7 best) vs random in ContextGridWorld\n'
        f'(T_phase={_S8_BEST[0]}, switch_interval={_S8_BEST[1]}, '
        f'n_agents={n_agents}, n_episodes={n_episodes})',
        fontsize=10,
    )

    # Row 1: survival boxplot
    ax = axes[0]
    bp = ax.boxplot([all_surv[c] for c in (1, 2, 3)],
                    patch_artist=True, widths=0.5, notch=False)
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    means = [float(np.mean(all_surv[c])) for c in (1, 2, 3)]
    for xi, (m, col) in enumerate(zip(means, colors), 1):
        ax.plot(xi, m, 'D', color=col, zorder=5, markersize=7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Survival Steps')
    ax.set_title('Survival Steps per Episode (diamond = mean)')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 2: accuracy bar (mode A vs B)
    ax = axes[1]
    x  = np.arange(3)
    w  = 0.35
    for ci, cond in enumerate((1, 2, 3)):
        rA = float(np.mean(acc_A[cond])) if acc_A[cond] else 0.0
        rB = float(np.mean(acc_B[cond])) if acc_B[cond] else 0.0
        bar_a = ax.bar(x[ci] - w / 2, rA, width=w, color=colors[ci], alpha=0.9)
        bar_b = ax.bar(x[ci] + w / 2, rB, width=w, color=colors[ci], alpha=0.45, hatch='//')
        ax.text(bar_a[0].get_x() + bar_a[0].get_width() / 2, rA + 0.01,
                f'{rA:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(bar_b[0].get_x() + bar_b[0].get_width() / 2, rB + 0.01,
                f'{rB:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.5, max(float(np.mean(acc_A[c])) for c in (1, 2, 3)
                                 if acc_A[c]) + 0.15))
    ax.set_ylabel('Food Eaten Rate')
    ax.set_title('Correct Rate by Mode (solid=ModeA NW, hatched=ModeB SE)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='silver', label='Mode A (NW food)'),
                        Patch(facecolor='silver', alpha=0.45, hatch='//', label='Mode B (SE food)')],
              fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3: cosine_dist per episode (conditions 1 and 2)
    ax = axes[2]
    ep_x = np.arange(1, n_episodes + 1)
    for cond, col, lbl in [(1, 'steelblue', 'Trained (frozen)'),
                            (2, 'darkorange', 'Trained (dynamic)')]:
        cd = ep_cos[cond]
        valid = ~np.isnan(cd)
        ax.plot(ep_x[valid], cd[valid], color=col, label=lbl, marker='o', markersize=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cosine Distance (Mode A vs B internal patterns)')
    ax.set_title('Internal Pattern Divergence Over Episodes (Conditions 1 & 2 only)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


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

    # enc_data_v2 = run_pattern_encoding_experiment_v2()
    # plot_pattern_encoding_v2(enc_data_v2)
    # plot_pattern_encoding_training_v2(enc_data_v2)
    # plot_pattern_coexistence(enc_data_v2)

    # resonance_data = run_loop_resonance_experiment()
    # plot_loop_resonance(resonance_data)

    # combination_data = run_loop_combination_experiment()
    # plot_loop_combination(combination_data)

    # association_data = run_association_experiment()
    # plot_association(association_data)

    # --- Session 3: E/I threshold evolution ---
    # ei_evo_data = run_ei_threshold_evolution(
    #     n_agents=10, n_generations=30, n_survivors=3,
    #     n_episodes_per_agent=3, N=20, K=10, temperature=1.0, seed=42)
    # plot_ei_evolution(ei_evo_data)
    # sparse_assoc = run_sparse_association(ei_evo_data['best_genome'], seed=42)
    # plot_sparse_association(sparse_assoc)
    # comparison = run_sparse_comparison(ei_evo_data['best_genome'], seed=42)
    # plot_sparse_comparison(comparison)

    # --- Session 4: Context interference environment (results already saved) ---
    # _simple_genome = {'ei_threshold': 0.9500, 'recovery_ratio': 0.2563, 'recovery_delay': 22}
    # context_evo_data = run_context_ei_evolution(
    #     n_agents=10, n_generations=30, n_survivors=3,
    #     n_episodes_per_agent=3, N=20, K=10, temperature=1.0, seed=42)
    # plot_context_ei_evolution(context_evo_data, simple_data=None)
    # context_sparse = run_sparse_association(context_evo_data['best_genome'], seed=42)
    # plot_sparse_association(
    #     context_sparse, fname='images/results_context_sparse_association.png')
    # context_comp = run_context_comparison(
    #     _simple_genome, context_evo_data['best_genome'], seed=42)
    # plot_context_comparison(context_comp)

    # --- Session 5: Association parameter sweep (results already saved) ---
    # sweep_data = run_association_sweep(N=20, seed=42, K=5, T_phase=500, T_probe=200)
    # plot_association_sweep(sweep_data)
    # plot_association_probe(sweep_data)

    # --- Session 6: Dynamic (E/I) vs static (low-weight) inhibition ---
    # s6_ab = run_ei_vs_static_experiments(N=20, seed=42, K=5, T_probe=100,
    #                                       probe_interval=10)
    # plot_ei_vs_static_overwrite(s6_ab)

    # s6_c = run_ei_vs_static_context(s6_ab, N=20, seed=42, n_episodes=10,
    #                                  T_episode=100)
    # plot_ei_vs_static_context(s6_c)

    # --- Session 7: Context-dependent activation patterns ---
    s7_data = run_context_activation_experiment(N=20, seed=42)
    plot_context_activation(s7_data)
    plot_context_topology(s7_data)

    # Prefer non-degenerate conditions (both norms > 1e-4) then highest cos_dist.
    def _s7_score(k):
        p = s7_data['sweep'][k]
        if p['norm_A'] < 1e-4 or p['norm_B'] < 1e-4:
            return -1.0
        return p['cos_dist'] if p['cos_dist'] == p['cos_dist'] else 0.0
    s7_best = max(s7_data['sweep'].keys(), key=_s7_score)
    bp = s7_data['sweep'][s7_best]
    print(f'Best (non-degenerate) condition: T_phase={s7_best[0]}, '
          f'switch_interval={s7_best[1]}, '
          f'cos_dist={bp["cos_dist"]:.4f}  '
          f'‖A‖={bp["norm_A"]:.4f}  ‖B‖={bp["norm_B"]:.4f}')

    s7_ctrl = run_context_control_experiment(s7_best, N=20, seed=42)
    plot_context_control(s7_data, s7_ctrl, s7_best)

    # --- Session 8: 世界に問う ---
    s8_data = run_world_test_experiment(N=20, seed=42, n_agents=10, n_episodes=20)
    plot_world_test(s8_data)
