"""explorations.py — Early unnumbered experiments (echo dynamics, E/I, pattern encoding, loops, association).

Archive of exploratory work that preceded the session-based experiments.
Also includes GridWorld experiments from the original echo_world.py.
"""
import itertools
import os
from collections import deque

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from world_base import (GridWorld, _make_graph, _mutate_graph, _hebbian_step,
                         _softmax_sample, _run_episode, _run_random_episode)


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



# ─── GridWorld experiments (from echo_world.py) ─────────────────────────────

def run_world_experiment(n_agents=10, n_episodes=20, N=25, K=10, temperature=0.5):
    # Input nodes 0-3, internal nodes 4-19, output nodes 20-24
    all_agent_steps = np.zeros((n_agents, n_episodes), dtype=float)
    all_agent_deltas = np.zeros((n_agents, n_episodes), dtype=float)
    all_agent_edges = np.zeros((n_agents, n_episodes), dtype=int)
    all_agent_food = np.zeros((n_agents, n_episodes), dtype=int)

    for seed in range(n_agents):
        rng = np.random.default_rng(seed)
        G = _make_graph(N, rng)
        activity = np.zeros(N)

        for ep in range(n_episodes):
            steps, delta, edges, food = _run_episode(
                G, activity, rng, N=N, K=K, temperature=temperature,
                debug=(seed == 0 and ep == 0)
            )
            all_agent_steps[seed, ep] = steps
            all_agent_deltas[seed, ep] = delta
            all_agent_edges[seed, ep] = edges
            all_agent_food[seed, ep] = food

    # Random baseline with matching seeds
    all_random_steps = np.zeros((n_agents, n_episodes), dtype=float)
    for seed in range(n_agents):
        rng = np.random.default_rng(seed)
        for ep in range(n_episodes):
            all_random_steps[seed, ep] = _run_random_episode(rng)

    # Pearson correlation per agent
    episode_numbers = np.arange(1, n_episodes + 1, dtype=float)
    pearson_episode = []
    pearson_topo = []

    for agent_idx in range(n_agents):
        steps = all_agent_steps[agent_idx]
        cum_deltas = np.cumsum(all_agent_deltas[agent_idx])
        if np.std(steps) < 1e-9 or np.std(cum_deltas) < 1e-9:
            # Constant series — correlation undefined; treat as no correlation
            pearson_episode.append((0.0, 1.0))
            pearson_topo.append((0.0, 1.0))
            continue
        r_ep, p_ep = stats.pearsonr(episode_numbers, steps)
        r_topo, p_topo = stats.pearsonr(cum_deltas, steps)
        pearson_episode.append((float(r_ep), float(p_ep)))
        pearson_topo.append((float(r_topo), float(p_topo)))

    mean_r_ep = float(np.mean([r for r, _ in pearson_episode]))
    mean_p_ep = float(np.mean([p for _, p in pearson_episode]))
    mean_r_topo = float(np.mean([r for r, _ in pearson_topo]))
    mean_p_topo = float(np.mean([p for _, p in pearson_topo]))

    baseline_mean = float(np.mean(all_random_steps))
    baseline_std = float(np.std(all_random_steps))
    echo_mean_overall = float(np.mean(all_agent_steps))

    print(f'Random baseline: mean survival = {baseline_mean:.1f} steps (std={baseline_std:.1f})')
    print()
    print('EchoAgent results:')
    for ep_label in [1, 10, 20]:
        if ep_label <= n_episodes:
            mean_ep = float(np.mean(all_agent_steps[:, ep_label - 1]))
            print(f'  Episode {ep_label}: mean survival = {mean_ep:.1f} steps')
    print(f'  Mean Pearson r (episode vs survival): {mean_r_ep:.4f} (p={mean_p_ep:.4f})')
    print(f'  Mean Pearson r (topology delta vs survival): {mean_r_topo:.4f} (p={mean_p_topo:.4f})')
    print()

    performance_exceeds = echo_mean_overall > baseline_mean
    topo_correlates = mean_r_topo > 0.3 and mean_p_topo < 0.05
    print(f'Performance exceeds baseline: {performance_exceeds}')
    print(f'Topology correlates with performance: {topo_correlates} (r > 0.3 and p < 0.05)')

    return {
        'all_agent_steps': all_agent_steps,
        'all_agent_deltas': all_agent_deltas,
        'all_agent_edges': all_agent_edges,
        'all_agent_food': all_agent_food,
        'all_random_steps': all_random_steps,
        'pearson_episode': pearson_episode,
        'pearson_topo': pearson_topo,
        'mean_r_ep': mean_r_ep, 'mean_p_ep': mean_p_ep,
        'mean_r_topo': mean_r_topo, 'mean_p_topo': mean_p_topo,
        'n_agents': n_agents, 'n_episodes': n_episodes,
    }


def plot_world_results(data, fname='images/results_world.png'):
    n_agents = data['n_agents']
    n_episodes = data['n_episodes']
    episode_numbers = np.arange(1, n_episodes + 1)

    all_agent_steps = data['all_agent_steps']
    all_agent_deltas = data['all_agent_deltas']
    all_random_steps = data['all_random_steps']

    echo_mean = all_agent_steps.mean(axis=0)
    echo_std = all_agent_steps.std(axis=0)
    rand_mean = all_random_steps.mean(axis=0)
    rand_std = all_random_steps.std(axis=0)
    delta_mean = all_agent_deltas.mean(axis=0)
    delta_std = all_agent_deltas.std(axis=0)

    cum_deltas_all = []
    steps_all = []
    for agent_idx in range(n_agents):
        cum_d = np.cumsum(all_agent_deltas[agent_idx])
        cum_deltas_all.extend(cum_d.tolist())
        steps_all.extend(all_agent_steps[agent_idx].tolist())
    cum_deltas_all = np.array(cum_deltas_all)
    steps_all = np.array(steps_all)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), squeeze=False)
    fig.suptitle(
        f'EchoAgent GridWorld  (N=25, K=10, {n_agents} agents × {n_episodes} episodes)',
        fontsize=13,
    )

    # Row 1: survival curves
    ax = axes[0][0]
    ax.plot(episode_numbers, echo_mean, label='EchoAgent', color='steelblue')
    ax.fill_between(episode_numbers,
                    np.maximum(0, echo_mean - echo_std),
                    echo_mean + echo_std,
                    alpha=0.2, color='steelblue')
    ax.plot(episode_numbers, rand_mean, label='Random baseline',
            color='tomato', linestyle='--')
    ax.fill_between(episode_numbers,
                    np.maximum(0, rand_mean - rand_std),
                    rand_mean + rand_std,
                    alpha=0.2, color='tomato')
    ax.set_ylabel('Steps survived', fontsize=10)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_xlim(1, n_episodes)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: topology delta per episode
    ax = axes[1][0]
    ax.plot(episode_numbers, delta_mean, color='seagreen')
    ax.fill_between(episode_numbers,
                    np.maximum(0, delta_mean - delta_std),
                    delta_mean + delta_std,
                    alpha=0.2, color='seagreen')
    ax.set_ylabel('Topology delta  (sum |Δw|)', fontsize=10)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_xlim(1, n_episodes)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Row 3: scatter — cumulative topology delta vs steps survived
    ax = axes[2][0]
    ax.scatter(cum_deltas_all, steps_all, alpha=0.4, s=20, color='steelblue')
    slope, intercept, r_val, p_val, _ = stats.linregress(cum_deltas_all, steps_all)
    x_line = np.linspace(cum_deltas_all.min(), cum_deltas_all.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color='tomato', linewidth=1.5,
            label=f'r = {r_val:.3f},  p = {p_val:.3f}')
    ax.set_xlabel('Cumulative topology delta', fontsize=10)
    ax.set_ylabel('Steps survived', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # fname is a parameter
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def _mutate_graph(G, N, rng, mutation_std=0.05, edge_change_prob=0.05):
    """Return a mutated copy of G."""
    G_new = G.copy()
    for i, j in list(G_new.edges()):
        w = G_new[i][j]['weight'] + rng.normal(0, mutation_std)
        G_new[i][j]['weight'] = float(np.clip(w, 0.01, 1.0))
    existing = set(G_new.edges())
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (i, j) in existing:
                if rng.random() < edge_change_prob:
                    G_new.remove_edge(i, j)
            else:
                if rng.random() < edge_change_prob:
                    G_new.add_edge(i, j, weight=float(rng.uniform(0.01, 1.0)))
    return G_new


def run_evolutionary_experiment(n_generations=20, n_agents=10, n_episodes_per_agent=5,
                                n_survivors=3, mutation_std=0.05, N=25, K=10,
                                temperature=0.5):
    gen_mean_survival = []
    gen_std_survival = []
    gen_best_survival = []
    gen_best_edges = []
    gen_best_weight = []
    gen_best_clustering = []

    # Generation 0: random initial topologies
    agents = []
    for seed in range(n_agents):
        rng = np.random.default_rng(seed + 200)
        G = _make_graph(N, rng)
        agents.append((G, np.zeros(N), rng))

    for gen in range(n_generations):
        survivals = []
        for G, activity, rng in agents:
            total = 0
            for _ in range(n_episodes_per_agent):
                steps, _, _, _ = _run_episode(G, activity, rng, N=N, K=K,
                                              temperature=temperature)
                total += steps
            survivals.append(total)

        ranked_idx = sorted(range(n_agents), key=lambda i: survivals[i], reverse=True)
        mean_surv = float(np.mean(survivals))
        std_surv = float(np.std(survivals))
        best_idx = ranked_idx[0]
        best_surv = float(survivals[best_idx])

        best_G = agents[best_idx][0]
        edge_count = best_G.number_of_edges()
        weights = [d['weight'] for _, _, d in best_G.edges(data=True)]
        mean_weight = float(np.mean(weights)) if weights else 0.0
        clustering = float(nx.average_clustering(best_G))

        gen_mean_survival.append(mean_surv)
        gen_std_survival.append(std_surv)
        gen_best_survival.append(best_surv)
        gen_best_edges.append(edge_count)
        gen_best_weight.append(mean_weight)
        gen_best_clustering.append(clustering)

        if gen + 1 in [1, 10, 20]:
            print(f'Generation {gen + 1}:')
            print(f'  mean survival: {mean_surv:.0f} steps (std={std_surv:.0f})')
            print(f'  best survival: {best_surv:.0f} steps')
            print(f'  best agent edges: {edge_count}, clustering: {clustering:.3f}')

        if gen < n_generations - 1:
            survivor_Gs = [agents[ranked_idx[i]][0] for i in range(n_survivors)]
            new_agents = []
            for i in range(n_survivors):
                rng_new = np.random.default_rng(gen * 10000 + i)
                G_new = _mutate_graph(survivor_Gs[i], N, rng_new, mutation_std=mutation_std)
                new_agents.append((G_new, np.zeros(N), rng_new))
            for i in range(n_agents - n_survivors):
                src = i % n_survivors
                rng_new = np.random.default_rng(gen * 10000 + n_survivors + i)
                G_new = _mutate_graph(survivor_Gs[src], N, rng_new, mutation_std=mutation_std)
                new_agents.append((G_new, np.zeros(N), rng_new))
            agents = new_agents

    # Random baseline
    rand_mean_per_gen = []
    for gen in range(n_generations):
        gen_surv = []
        for seed in range(n_agents):
            rng = np.random.default_rng(seed + gen * 100 + 300)
            total = sum(_run_random_episode(rng) for _ in range(n_episodes_per_agent))
            gen_surv.append(total)
        rand_mean_per_gen.append(float(np.mean(gen_surv)))
    rand_baseline_mean = float(np.mean(rand_mean_per_gen))

    print(f'\nRandom baseline mean survival: {rand_baseline_mean:.0f} steps')

    perf_improvement = gen_mean_survival[-1] > gen_mean_survival[0] * 1.1
    topo_evolved = (gen_best_clustering[0] > 0 and
                    gen_best_clustering[-1] > gen_best_clustering[0] * 1.1)
    print(f'Performance improvement detected: {perf_improvement}')
    print(f'Topology evolved: {topo_evolved}')

    return {
        'gen_mean_survival': gen_mean_survival,
        'gen_std_survival': gen_std_survival,
        'gen_best_survival': gen_best_survival,
        'gen_best_edges': gen_best_edges,
        'gen_best_weight': gen_best_weight,
        'gen_best_clustering': gen_best_clustering,
        'rand_mean_per_gen': rand_mean_per_gen,
        'rand_baseline_mean': rand_baseline_mean,
        'n_generations': n_generations,
    }


def plot_evolution_results(data, fname='images/results_evolution.png'):
    n_gen = data['n_generations']
    gens = np.arange(1, n_gen + 1)
    mean_surv = np.array(data['gen_mean_survival'])
    std_surv = np.array(data['gen_std_survival'])
    best_surv = np.array(data['gen_best_survival'])
    edges = np.array(data['gen_best_edges'], dtype=float)
    clustering = np.array(data['gen_best_clustering'])
    rand_mean = np.array(data['rand_mean_per_gen'])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), squeeze=False)
    fig.suptitle(
        'Evolutionary Selection Experiment\n'
        f'(N=25, K=10, {data["n_generations"]} generations, 10 agents, 5 episodes/agent)',
        fontsize=13,
    )

    # Row 1: mean survival per generation
    ax = axes[0][0]
    ax.plot(gens, mean_surv, label='EchoAgent (selection)', color='steelblue')
    ax.fill_between(gens, np.maximum(0, mean_surv - std_surv), mean_surv + std_surv,
                    alpha=0.2, color='steelblue')
    ax.plot(gens, rand_mean, label='Random baseline', color='tomato', linestyle='--')
    ax.set_ylabel('Total steps survived (5 episodes)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: best agent survival
    ax = axes[1][0]
    ax.plot(gens, best_surv, color='seagreen')
    ax.set_ylabel('Best agent total steps survived', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Row 3: topology metrics on dual axis
    ax = axes[2][0]
    ax2 = ax.twinx()
    ln1 = ax.plot(gens, edges, color='steelblue', label='Edges')
    ln2 = ax2.plot(gens, clustering, color='tomato', linestyle='--', label='Clustering coeff')
    ax.set_ylabel('Number of edges', fontsize=10, color='steelblue')
    ax2.set_ylabel('Clustering coefficient', fontsize=10, color='tomato')
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_evolutionary_readout_experiment(n_generations=20, n_agents=10,
                                        n_episodes_per_agent=5, n_survivors=3,
                                        mutation_std=0.05, N=20, K=10,
                                        temperature=1.0):
    print('Architecture: EchoLoop(N=20) + fixed readout(16→5)')
    print('Readout weights: fixed random, not evolved')
    print()

    gen_mean_survival = []
    gen_std_survival = []
    gen_best_survival = []
    gen_best_edges = []
    gen_best_weight = []
    gen_best_clustering = []

    # Generation 0: random agents with fresh readout weights
    agents = []
    for seed in range(n_agents):
        rng = np.random.default_rng(seed + 400)
        G = _make_graph(N, rng)
        readout_weights = rng.standard_normal((16, 5)) * 0.1
        agents.append((G, np.zeros(N), rng, readout_weights))

    for gen in range(n_generations):
        survivals = []
        for G, activity, rng, readout_weights in agents:
            total = 0
            for _ in range(n_episodes_per_agent):
                steps, _, _, _ = _run_episode(G, activity, rng, N=N, K=K,
                                              temperature=temperature,
                                              readout_weights=readout_weights)
                total += steps
            survivals.append(total)

        ranked_idx = sorted(range(n_agents), key=lambda i: survivals[i], reverse=True)
        mean_surv = float(np.mean(survivals))
        std_surv = float(np.std(survivals))
        best_idx = ranked_idx[0]
        best_surv = float(survivals[best_idx])

        best_G = agents[best_idx][0]
        edge_count = best_G.number_of_edges()
        weights = [d['weight'] for _, _, d in best_G.edges(data=True)]
        mean_weight = float(np.mean(weights)) if weights else 0.0
        clustering = float(nx.average_clustering(best_G))

        gen_mean_survival.append(mean_surv)
        gen_std_survival.append(std_surv)
        gen_best_survival.append(best_surv)
        gen_best_edges.append(edge_count)
        gen_best_weight.append(mean_weight)
        gen_best_clustering.append(clustering)

        if gen == 0 or (gen + 1) % 10 == 0:
            print(f'Generation {gen + 1}:')
            print(f'  mean survival: {mean_surv:.0f} steps (std={std_surv:.0f})')
            print(f'  best survival: {best_surv:.0f} steps')
            print(f'  best agent edges: {edge_count}, clustering: {clustering:.3f}')

        if gen < n_generations - 1:
            survivor_Gs = [agents[ranked_idx[i]][0] for i in range(n_survivors)]
            new_agents = []
            for i in range(n_survivors):
                rng_new = np.random.default_rng(gen * 10000 + i + 7000)
                G_new = _mutate_graph(survivor_Gs[i], N, rng_new, mutation_std=mutation_std)
                rw = rng_new.standard_normal((16, 5)) * 0.1
                new_agents.append((G_new, np.zeros(N), rng_new, rw))
            for i in range(n_agents - n_survivors):
                src = i % n_survivors
                rng_new = np.random.default_rng(gen * 10000 + n_survivors + i + 7000)
                G_new = _mutate_graph(survivor_Gs[src], N, rng_new, mutation_std=mutation_std)
                rw = rng_new.standard_normal((16, 5)) * 0.1
                new_agents.append((G_new, np.zeros(N), rng_new, rw))
            agents = new_agents

    # Random baseline (same GridWorld, uniform actions)
    rand_mean_per_gen = []
    for gen in range(n_generations):
        gen_surv = []
        for seed in range(n_agents):
            rng = np.random.default_rng(seed + gen * 100 + 900)
            total = sum(_run_random_episode(rng) for _ in range(n_episodes_per_agent))
            gen_surv.append(total)
        rand_mean_per_gen.append(float(np.mean(gen_surv)))
    rand_baseline_mean = float(np.mean(rand_mean_per_gen))

    print(f'\nRandom baseline mean survival: {rand_baseline_mean:.0f} steps')

    perf_improvement = gen_mean_survival[-1] > gen_mean_survival[0] * 1.1
    topo_evolved = (gen_best_clustering[0] > 0 and
                    gen_best_clustering[-1] > gen_best_clustering[0] * 1.1)
    print(f'Performance improvement detected: {perf_improvement}')
    print(f'Topology evolved: {topo_evolved}')

    return {
        'gen_mean_survival': gen_mean_survival,
        'gen_std_survival': gen_std_survival,
        'gen_best_survival': gen_best_survival,
        'gen_best_edges': gen_best_edges,
        'gen_best_weight': gen_best_weight,
        'gen_best_clustering': gen_best_clustering,
        'rand_mean_per_gen': rand_mean_per_gen,
        'rand_baseline_mean': rand_baseline_mean,
        'n_generations': n_generations,
    }


def run_coevolution_experiment(n_generations=50, n_agents=10, n_episodes_per_agent=5,
                               n_survivors=3, mutation_std=0.05, N=20, K=10,
                               temperature=1.0):
    print('Architecture: EchoLoop(N=20) + evolvable readout(16→5)')
    print('Both topology and readout weights are evolved')
    print()

    gen_mean_survival = []
    gen_std_survival = []
    gen_best_survival = []
    gen_best_edges = []
    gen_best_weight = []
    gen_best_clustering = []
    gen_best_readout_abs = []

    agents = []
    for seed in range(n_agents):
        rng = np.random.default_rng(seed + 500)
        G = _make_graph(N, rng)
        readout_weights = rng.standard_normal((16, 5)) * 0.1
        agents.append((G, np.zeros(N), rng, readout_weights))

    for gen in range(n_generations):
        survivals = []
        for G, activity, rng, readout_weights in agents:
            total = 0
            for _ in range(n_episodes_per_agent):
                steps, _, _, _ = _run_episode(G, activity, rng, N=N, K=K,
                                              temperature=temperature,
                                              readout_weights=readout_weights)
                total += steps
            survivals.append(total)

        ranked_idx = sorted(range(n_agents), key=lambda i: survivals[i], reverse=True)
        mean_surv = float(np.mean(survivals))
        std_surv = float(np.std(survivals))
        best_idx = ranked_idx[0]
        best_surv = float(survivals[best_idx])

        best_G = agents[best_idx][0]
        best_rw = agents[best_idx][3]
        edge_count = best_G.number_of_edges()
        weights = [d['weight'] for _, _, d in best_G.edges(data=True)]
        mean_weight = float(np.mean(weights)) if weights else 0.0
        clustering = float(nx.average_clustering(best_G))
        mean_abs_readout = float(np.mean(np.abs(best_rw)))

        gen_mean_survival.append(mean_surv)
        gen_std_survival.append(std_surv)
        gen_best_survival.append(best_surv)
        gen_best_edges.append(edge_count)
        gen_best_weight.append(mean_weight)
        gen_best_clustering.append(clustering)
        gen_best_readout_abs.append(mean_abs_readout)

        if gen == 0 or (gen + 1) % 10 == 0:
            print(f'Generation {gen + 1}:')
            print(f'  mean survival: {mean_surv:.0f} steps (std={std_surv:.0f})')
            print(f'  best survival: {best_surv:.0f} steps')
            print(f'  best agent edges: {edge_count}, clustering: {clustering:.3f}')
            print(f'  mean abs readout weight: {mean_abs_readout:.4f}')

        if gen < n_generations - 1:
            survivor_Gs = [agents[ranked_idx[i]][0] for i in range(n_survivors)]
            survivor_rws = [agents[ranked_idx[i]][3] for i in range(n_survivors)]
            new_agents = []
            for i in range(n_survivors):
                rng_new = np.random.default_rng(gen * 10000 + i + 9000)
                G_new = _mutate_graph(survivor_Gs[i], N, rng_new, mutation_std=mutation_std)
                rw_new = survivor_rws[i] + rng_new.standard_normal((16, 5)) * mutation_std
                new_agents.append((G_new, np.zeros(N), rng_new, rw_new))
            for i in range(n_agents - n_survivors):
                src = i % n_survivors
                rng_new = np.random.default_rng(gen * 10000 + n_survivors + i + 9000)
                G_new = _mutate_graph(survivor_Gs[src], N, rng_new, mutation_std=mutation_std)
                rw_new = survivor_rws[src] + rng_new.standard_normal((16, 5)) * mutation_std
                new_agents.append((G_new, np.zeros(N), rng_new, rw_new))
            agents = new_agents

    rand_mean_per_gen = []
    for gen in range(n_generations):
        gen_surv = []
        for seed in range(n_agents):
            rng = np.random.default_rng(seed + gen * 100 + 1200)
            total = sum(_run_random_episode(rng) for _ in range(n_episodes_per_agent))
            gen_surv.append(total)
        rand_mean_per_gen.append(float(np.mean(gen_surv)))
    rand_baseline_mean = float(np.mean(rand_mean_per_gen))

    print(f'\nRandom baseline mean survival: {rand_baseline_mean:.0f} steps')

    perf_improvement = gen_mean_survival[-1] > gen_mean_survival[0] * 1.1
    topo_evolved = (gen_best_clustering[0] > 0 and
                    gen_best_clustering[-1] > gen_best_clustering[0] * 1.1)
    readout_evolved = (gen_best_readout_abs[0] > 0 and
                       gen_best_readout_abs[-1] > gen_best_readout_abs[0] * 1.5)
    print(f'Performance improvement detected: {perf_improvement}')
    print(f'Topology evolved: {topo_evolved}')
    print(f'Readout evolved: {readout_evolved}')

    return {
        'gen_mean_survival': gen_mean_survival,
        'gen_std_survival': gen_std_survival,
        'gen_best_survival': gen_best_survival,
        'gen_best_edges': gen_best_edges,
        'gen_best_weight': gen_best_weight,
        'gen_best_clustering': gen_best_clustering,
        'gen_best_readout_abs': gen_best_readout_abs,
        'rand_mean_per_gen': rand_mean_per_gen,
        'rand_baseline_mean': rand_baseline_mean,
        'n_generations': n_generations,
    }


def plot_coevolution_results(data, fname='images/results_evolution_coevolve.png'):
    n_gen = data['n_generations']
    gens = np.arange(1, n_gen + 1)
    mean_surv = np.array(data['gen_mean_survival'])
    std_surv = np.array(data['gen_std_survival'])
    best_surv = np.array(data['gen_best_survival'])
    edges = np.array(data['gen_best_edges'], dtype=float)
    clustering = np.array(data['gen_best_clustering'])
    readout_abs = np.array(data['gen_best_readout_abs'])
    rand_mean = np.array(data['rand_mean_per_gen'])

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), squeeze=False)
    fig.suptitle(
        'Co-evolution: Topology + Readout\n'
        f'(N=20, K=10, {n_gen} generations, 10 agents, 5 episodes/agent)',
        fontsize=13,
    )

    ax = axes[0][0]
    ax.plot(gens, mean_surv, label='EchoAgent (co-evolve)', color='steelblue')
    ax.fill_between(gens, np.maximum(0, mean_surv - std_surv), mean_surv + std_surv,
                    alpha=0.2, color='steelblue')
    ax.plot(gens, rand_mean, label='Random baseline', color='tomato', linestyle='--')
    ax.set_ylabel('Total steps survived (5 episodes)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.plot(gens, best_surv, color='seagreen')
    ax.set_ylabel('Best agent total steps survived', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2][0]
    ax2 = ax.twinx()
    ln1 = ax.plot(gens, edges, color='steelblue', label='Edges')
    ln2 = ax2.plot(gens, clustering, color='tomato', linestyle='--', label='Clustering coeff')
    ax.set_ylabel('Number of edges', fontsize=10, color='steelblue')
    ax2.set_ylabel('Clustering coefficient', fontsize=10, color='tomato')
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[3][0]
    ax.plot(gens, readout_abs, color='purple')
    ax.set_ylabel('Mean |readout weight|', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_evolution_readout_only(n_generations=50, n_agents=10, n_episodes_per_agent=5,
                               n_survivors=3, mutation_std=0.05, N=20, K=10,
                               temperature=1.0):
    print('=== Control: Readout-only evolution ===')

    gen_mean_survival = []
    gen_std_survival = []
    gen_best_survival = []

    agents = []
    for seed in range(n_agents):
        rng = np.random.default_rng(seed + 600)
        G = _make_graph(N, rng)
        readout_weights = rng.standard_normal((16, 5)) * 0.1
        agents.append((G, np.zeros(N), rng, readout_weights))

    for gen in range(n_generations):
        survivals = []
        for G, activity, rng, readout_weights in agents:
            total = 0
            for _ in range(n_episodes_per_agent):
                steps, _, _, _ = _run_episode(G, activity, rng, N=N, K=K,
                                              temperature=temperature,
                                              readout_weights=readout_weights,
                                              topology_frozen=True)
                total += steps
            survivals.append(total)

        ranked_idx = sorted(range(n_agents), key=lambda i: survivals[i], reverse=True)
        mean_surv = float(np.mean(survivals))
        std_surv = float(np.std(survivals))
        best_surv = float(survivals[ranked_idx[0]])

        gen_mean_survival.append(mean_surv)
        gen_std_survival.append(std_surv)
        gen_best_survival.append(best_surv)

        if gen == 0 or (gen + 1) % 10 == 0:
            print(f'Generation {gen + 1:2d}:  mean={mean_surv:.0f}, best={best_surv:.0f}')

        if gen < n_generations - 1:
            survivor_Gs = [agents[ranked_idx[i]][0] for i in range(n_survivors)]
            survivor_rws = [agents[ranked_idx[i]][3] for i in range(n_survivors)]
            new_agents = []
            for i in range(n_survivors):
                rng_new = np.random.default_rng(gen * 10000 + i + 11000)
                G_new = survivor_Gs[i].copy()
                rw_new = survivor_rws[i] + rng_new.standard_normal((16, 5)) * mutation_std
                new_agents.append((G_new, np.zeros(N), rng_new, rw_new))
            for i in range(n_agents - n_survivors):
                src = i % n_survivors
                rng_new = np.random.default_rng(gen * 10000 + n_survivors + i + 11000)
                G_new = survivor_Gs[src].copy()
                rw_new = survivor_rws[src] + rng_new.standard_normal((16, 5)) * mutation_std
                new_agents.append((G_new, np.zeros(N), rng_new, rw_new))
            agents = new_agents

    rand_mean_per_gen = []
    for gen in range(n_generations):
        gen_surv = []
        for seed in range(n_agents):
            rng = np.random.default_rng(seed + gen * 100 + 1500)
            total = sum(_run_random_episode(rng) for _ in range(n_episodes_per_agent))
            gen_surv.append(total)
        rand_mean_per_gen.append(float(np.mean(gen_surv)))
    rand_baseline_mean = float(np.mean(rand_mean_per_gen))

    return {
        'gen_mean_survival': gen_mean_survival,
        'gen_std_survival': gen_std_survival,
        'gen_best_survival': gen_best_survival,
        'rand_mean_per_gen': rand_mean_per_gen,
        'rand_baseline_mean': rand_baseline_mean,
        'n_generations': n_generations,
    }


def plot_evolution_control(coevo_data, readout_only_data,
                           fname='images/results_evolution_control.png'):
    n_gen = coevo_data['n_generations']
    gens = np.arange(1, n_gen + 1)

    coevo_mean = np.array(coevo_data['gen_mean_survival'])
    coevo_std = np.array(coevo_data['gen_std_survival'])
    coevo_best = np.array(coevo_data['gen_best_survival'])
    ro_mean = np.array(readout_only_data['gen_mean_survival'])
    ro_std = np.array(readout_only_data['gen_std_survival'])
    ro_best = np.array(readout_only_data['gen_best_survival'])
    rand_mean = np.array(coevo_data['rand_mean_per_gen'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), squeeze=False)
    fig.suptitle(
        'Control: Co-evolution vs Readout-only\n'
        f'(N=20, K=10, {n_gen} generations, 10 agents, 5 episodes/agent)',
        fontsize=13,
    )

    ax = axes[0][0]
    ax.plot(gens, coevo_mean, label='Co-evolution (topology + readout)', color='steelblue')
    ax.fill_between(gens, np.maximum(0, coevo_mean - coevo_std), coevo_mean + coevo_std,
                    alpha=0.15, color='steelblue')
    ax.plot(gens, ro_mean, label='Readout only (frozen topology)', color='seagreen')
    ax.fill_between(gens, np.maximum(0, ro_mean - ro_std), ro_mean + ro_std,
                    alpha=0.15, color='seagreen')
    ax.plot(gens, rand_mean, label='Random baseline', color='tomato', linestyle='--')
    ax.set_ylabel('Total steps survived (5 episodes)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.plot(gens, coevo_best, label='Co-evolution', color='steelblue')
    ax.plot(gens, ro_best, label='Readout only', color='seagreen')
    ax.set_ylabel('Best agent total steps survived', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xlim(1, n_gen)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()


def run_world_association_experiment(n_episodes=10, N=25, K=10, seed=42,
                                     temperature=0.5, n_propagation_steps=3):
    """Association experiment: single food at (0,0). Does Hebbian learning link
    north-movement activity with food-proximity activity across episodes?"""
    FOOD_POS = (0, 0)          # fixed north-west only
    GRID = 5
    HP_MAX, HP_DECAY, FOOD_VAL, FOOD_RESPAWN, MAX_STEPS = 200, 1, 30, 30, 500
    START_HP, START_POS = 100, (2, 2)

    rng = np.random.default_rng(seed)
    G = _make_graph(N, rng)
    activity = np.zeros(N)

    north_acts = []        # activity[4:20] whenever action==0
    near_food_acts = []    # activity[4:20] whenever manhattan dist to food < 2
    eat_acts = []          # activity[4:20] whenever eating
    all_acts = []          # activity[4:20] every step (for heatmap)
    episode_boundaries = []

    episode_food_eaten = []
    episode_steps_survived = []

    for ep in range(n_episodes):
        row, col = START_POS
        hp = START_HP
        food_available = True
        food_timer = 0
        food_eaten = 0
        steps_survived = 0

        for step in range(MAX_STEPS):
            if hp <= 0:
                break

            input_vals = {
                0: col / (GRID - 1),
                1: row / (GRID - 1),
                2: hp / HP_MAX,
                3: 1.0 if food_available else 0.0,
            }

            for _ in range(n_propagation_steps):
                new_act = np.zeros(N)
                for i in range(N):
                    if i in input_vals:
                        new_act[i] = input_vals[i]
                    else:
                        s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                        new_act[i] = np.tanh(s)
                activity[:] = new_act

            all_acts.append(activity[4:20].copy())

            action = _softmax_sample(activity[20:25], temperature=temperature)

            if action == 0:
                north_acts.append(activity[4:20].copy())

            dist = abs(row - FOOD_POS[0]) + abs(col - FOOD_POS[1])
            if dist < 2:
                near_food_acts.append(activity[4:20].copy())

            if action == 0:
                row = max(0, row - 1)
            elif action == 1:
                row = min(GRID - 1, row + 1)
            elif action == 2:
                col = max(0, col - 1)
            elif action == 3:
                col = min(GRID - 1, col + 1)
            elif action == 4:
                if row == FOOD_POS[0] and col == FOOD_POS[1] and food_available:
                    hp = min(HP_MAX, hp + FOOD_VAL)
                    food_available = False
                    food_timer = 0
                    food_eaten += 1
                    eat_acts.append(activity[4:20].copy())

            hp -= HP_DECAY
            steps_survived = step + 1

            if not food_available:
                food_timer += 1
                if food_timer >= FOOD_RESPAWN:
                    food_available = True
                    food_timer = 0

            if (step + 1) % K == 0:
                _hebbian_step(G, activity, N, rng)

        episode_food_eaten.append(food_eaten)
        episode_steps_survived.append(steps_survived)
        episode_boundaries.append(len(all_acts))
        print(f'Episode {ep + 1}:  food eaten: {food_eaten},  steps survived: {steps_survived}')

    print()

    all_acts_arr = np.array(all_acts) if all_acts else np.zeros((1, 16))  # (T, 16)

    def _associated(cond_acts, threshold=0.5):
        if not cond_acts:
            return []
        arr = np.array(cond_acts)
        frac = (arr > 0.5).mean(axis=0)
        return [i + 4 for i in range(16) if frac[i] > threshold]

    north_nodes = _associated(north_acts)
    food_nodes  = _associated(near_food_acts)
    overlapping = sorted(set(north_nodes) & set(food_nodes))

    def _cross_edges(a_set, b_set):
        return [(u, v) for u in a_set for v in b_set
                if u != v and (G.has_edge(u, v) or G.has_edge(v, u))]

    cross = _cross_edges(set(north_nodes), set(food_nodes))
    cross_count = len(cross)

    # Baseline: random node-pair sets of same size
    internal = list(range(4, 20))
    n_n = max(1, len(north_nodes))
    n_f = max(1, len(food_nodes))
    baseline_vals = []
    for _ in range(200):
        rn = list(rng.choice(internal, size=n_n, replace=False))
        rf = list(rng.choice(internal, size=n_f, replace=False))
        baseline_vals.append(len(_cross_edges(set(rn), set(rf))))
    baseline_mean = float(np.mean(baseline_vals))

    association_emerged = cross_count > baseline_mean * 1.5

    print('Topology analysis after 10 episodes:')
    print(f'  North-associated nodes: {north_nodes}')
    print(f'  Food-associated nodes:  {food_nodes}')
    print(f'  Overlapping nodes:      {overlapping}')
    print(f'  Cross-edges (north↔food): {cross_count}')
    print(f'  Baseline cross-edges (random pairs): {baseline_mean:.1f}')
    print(f'  Association emerged: {association_emerged} '
          f'(cross-edges > baseline * 1.5)')

    return {
        'all_acts': all_acts_arr,
        'episode_boundaries': episode_boundaries,
        'episode_food_eaten': episode_food_eaten,
        'episode_steps': episode_steps_survived,
        'north_nodes': north_nodes,
        'food_nodes': food_nodes,
        'overlapping': overlapping,
        'cross_count': cross_count,
        'baseline_mean': baseline_mean,
        'association_emerged': association_emerged,
        'G_final': G,
        'N': N,
    }


def plot_world_association(data, fname='images/results_world_association.png'):
    from matplotlib.patches import Patch

    all_acts = data['all_acts']        # (T, 16)
    north_nodes = data['north_nodes']
    food_nodes  = data['food_nodes']
    overlapping = data['overlapping']
    G_final = data['G_final']
    boundaries = data['episode_boundaries']

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), squeeze=False)
    fig.suptitle('World Association Experiment  (food fixed at (0,0))', fontsize=13)

    # Row 1: activity heatmap  (16 nodes × timesteps)
    ax = axes[0][0]
    hmap = all_acts.T          # (16, T)
    T = hmap.shape[1]
    # Downsample columns for display if very long
    MAX_COLS = 1000
    if T > MAX_COLS:
        bin_size = T // MAX_COLS
        hmap = np.array([hmap[:, k*bin_size:(k+1)*bin_size].mean(axis=1)
                         for k in range(MAX_COLS)]).T
        scale = T / MAX_COLS
    else:
        scale = 1.0

    im = ax.imshow(hmap, aspect='auto', origin='lower', cmap='hot',
                   interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='Mean activity')
    ax.set_ylabel('Internal node (4–19)', fontsize=10)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_title('Activity heatmap — internal nodes over all episode steps', fontsize=10)
    ax.set_yticks(range(16))
    ax.set_yticklabels([str(i + 4) for i in range(16)], fontsize=7)
    for b in boundaries[:-1]:
        ax.axvline(b / scale, color='cyan', linewidth=0.8, alpha=0.6)

    # Row 2: topology — internal nodes subgraph
    ax = axes[1][0]
    ax.set_title(
        'Final topology  — blue: north-assoc, red: food-assoc, purple: both  '
        '(thick edges = cross-edges)',
        fontsize=10,
    )

    H = G_final.subgraph(range(4, 20)).copy()
    north_set = set(north_nodes)
    food_set  = set(food_nodes)
    over_set  = set(overlapping)

    node_colors = []
    for n in H.nodes():
        if n in over_set:
            node_colors.append('mediumpurple')
        elif n in north_set:
            node_colors.append('steelblue')
        elif n in food_set:
            node_colors.append('tomato')
        else:
            node_colors.append('lightgray')

    edge_colors = []
    edge_widths = []
    for u, v in H.edges():
        if (u in north_set and v in food_set) or (u in food_set and v in north_set):
            edge_colors.append('mediumpurple')
            edge_widths.append(3.5)
        else:
            edge_colors.append('#cccccc')
            edge_widths.append(0.5)

    pos = nx.spring_layout(H, seed=42)
    nx.draw_networkx(
        H, pos=pos, ax=ax,
        node_color=node_colors, node_size=500,
        edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=12,
        font_size=8, font_color='black',
    )

    legend_elements = [
        Patch(facecolor='steelblue',    label='North-associated'),
        Patch(facecolor='tomato',       label='Food-associated'),
        Patch(facecolor='mediumpurple', label='Overlapping'),
        Patch(facecolor='lightgray',    label='Neither'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    ax.axis('off')

    assoc_str = str(data['association_emerged'])
    txt_color = 'darkgreen' if data['association_emerged'] else 'firebrick'
    fig.text(
        0.5, 0.01,
        f'Association emerged: {assoc_str}  '
        f'(cross-edges={data["cross_count"]}  vs  baseline={data["baseline_mean"]:.1f})',
        ha='center', fontsize=10, color=txt_color,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')
    plt.close()

