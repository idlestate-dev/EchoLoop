"""Session 3: E/I threshold evolution and sparse association."""
import os
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from world_base import GridWorld, ContextGridWorld, _make_graph, _mutate_graph, _hebbian_step, _softmax_sample, _run_episode, _run_context_episode

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
    pass  # imported at module level
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
    pass  # imported at module level
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


def plot_ei_evolution(data, fname='images/session_3/results_ei_evolution.png'):
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


def plot_sparse_association(data, fname='images/session_3/results_sparse_association.png'):
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


def plot_sparse_comparison(data, fname='images/session_3/results_sparse_comparison.png'):
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


