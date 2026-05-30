"""echo_world.py — EchoAgent in GridWorld survival environment.
Core question: Does topology change correlate with performance improvement?
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats


class GridWorld:
    grid_size = 5
    hp_max = 200
    hp_decay = 1
    food_value = 30
    food_respawn = 30
    max_steps = 500
    food_positions = [(0, 0), (4, 4)]
    start_pos = (2, 2)
    start_hp = 100


def _make_graph(N, rng):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=float(rng.random()))
    return G


def _hebbian_step(G, activity, N, rng):
    """Apply Hebbian update. Returns sum of absolute weight changes."""
    delta = 0.0
    edges_to_remove = []
    for i, j, data in list(G.edges(data=True)):
        old_w = data['weight']
        w = old_w
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += 0.05
        w -= 0.01
        if w < 0.01:
            edges_to_remove.append((i, j))
            delta += old_w
        else:
            w = min(w, 1.0)
            G[i][j]['weight'] = w
            delta += abs(w - old_w)
    G.remove_edges_from(edges_to_remove)
    existing = set(G.edges())
    for i in range(N):
        for j in range(N):
            if i != j and (i, j) not in existing and rng.random() < 0.01:
                G.add_edge(i, j, weight=0.05)
                delta += 0.05
    return delta


def _softmax_sample(x, temperature=0.5):
    x = x - np.max(x)
    probs = np.exp(x / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(len(x), p=probs))


def _run_episode(G, activity, rng, N=25, K=10, n_propagation_steps=3,
                 temperature=0.5, readout_weights=None, topology_frozen=False,
                 debug=False):
    """Run one GridWorld episode. Mutates G and activity in place."""
    gw = GridWorld()
    row, col = gw.start_pos
    hp = gw.start_hp
    food_available = [True] * len(gw.food_positions)
    food_timers = [0] * len(gw.food_positions)
    food_eaten = 0
    total_delta = 0.0
    steps_survived = 0

    for step in range(gw.max_steps):
        if hp <= 0:
            break

        x_norm = col / (gw.grid_size - 1)
        y_norm = row / (gw.grid_size - 1)
        hp_norm = hp / gw.hp_max
        food_flag = 1.0 if any(food_available) else 0.0
        input_vals = {0: x_norm, 1: y_norm, 2: hp_norm, 3: food_flag}

        for _ in range(n_propagation_steps):
            new_activity = np.zeros(N)
            for i in range(N):
                if i in input_vals:
                    new_activity[i] = input_vals[i]
                else:
                    s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                    new_activity[i] = np.tanh(s)
            activity[:] = new_activity

        if debug and step == 0:
            n_input_edges = sum(1 for i, j in G.edges() if i in {0, 1, 2, 3})
            print(f'Input node activities: {activity[0:4].round(3)}')
            print(f'Internal node activities (mean): {activity[4:20].mean():.3f}')
            print(f'Number of edges from input nodes (0-3): {n_input_edges}')
            if readout_weights is None:
                n_output_edges = sum(1 for i, j in G.edges() if j in {20, 21, 22, 23, 24})
                print(f'Output node activities: {activity[20:25].round(3)}')
                print(f'Number of edges to output nodes (20-24): {n_output_edges}')
            else:
                scores = activity[4:20] @ readout_weights
                print(f'Readout action scores: {scores.round(3)}')

        if readout_weights is not None:
            action_scores = activity[4:20] @ readout_weights
            action = _softmax_sample(action_scores, temperature=temperature)
        else:
            action = _softmax_sample(activity[20:25], temperature=temperature)

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

        if debug and step % 10 == 0:
            if readout_weights is not None:
                scores = activity[4:20] @ readout_weights
                print(f'Step {step}: pos=({col},{row}), HP={hp}, action={action}, '
                      f'action_scores={scores.round(3)}')
            else:
                print(f'Step {step}: pos=({col},{row}), HP={hp}, action={action}, '
                      f'activity_output={activity[20:25].round(3)}')

        hp -= gw.hp_decay
        steps_survived = step + 1

        for idx in range(len(gw.food_positions)):
            if not food_available[idx]:
                food_timers[idx] += 1
                if food_timers[idx] >= gw.food_respawn:
                    food_available[idx] = True
                    food_timers[idx] = 0

        if not topology_frozen and (step + 1) % K == 0:
            total_delta += _hebbian_step(G, activity, N, rng)

    if debug:
        print(f'Episode end: food_eaten={food_eaten}, steps={steps_survived}')

    return steps_survived, total_delta, G.number_of_edges(), food_eaten


def _run_random_episode(rng):
    """Run one episode with a uniformly random agent."""
    gw = GridWorld()
    row, col = gw.start_pos
    hp = gw.start_hp
    food_available = [True] * len(gw.food_positions)
    food_timers = [0] * len(gw.food_positions)
    steps_survived = 0

    for step in range(gw.max_steps):
        if hp <= 0:
            break

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
                    break

        hp -= gw.hp_decay
        steps_survived = step + 1

        for idx in range(len(gw.food_positions)):
            if not food_available[idx]:
                food_timers[idx] += 1
                if food_timers[idx] >= gw.food_respawn:
                    food_available[idx] = True
                    food_timers[idx] = 0

    return steps_survived


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


# ─── Session 4: Context interference environment ──────────────────────────────


class ContextGridWorld:
    """GridWorld where food position depends on episode context (A=NW, B=SE)."""
    grid_size = 5
    hp_max = 200
    hp_decay = 1
    food_value = 30
    food_respawn = 40
    max_steps = 500
    start_pos = (2, 2)
    start_hp = 100


def _run_context_episode(G, activity, rng, N=20, K=10, n_propagation_steps=3,
                          temperature=1.0, readout_weights=None,
                          topology_frozen=False, mode=None):
    """ContextGridWorld episode. mode='A' → food at (0,0), 'B' → food at (4,4), None → random."""
    gw = ContextGridWorld()
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'
    food_pos = (0, 0) if mode == 'A' else (4, 4)
    food_available = [True]
    food_timers = [0]

    row, col = gw.start_pos
    hp = gw.start_hp
    food_eaten = 0
    total_delta = 0.0
    steps_survived = 0

    for step in range(gw.max_steps):
        if hp <= 0:
            break

        input_vals = {
            0: col / (gw.grid_size - 1),
            1: row / (gw.grid_size - 1),
            2: hp / gw.hp_max,
            3: 1.0 if food_available[0] else 0.0,
        }

        for _ in range(n_propagation_steps):
            new_activity = np.zeros(N)
            for i in range(N):
                if i in input_vals:
                    new_activity[i] = input_vals[i]
                else:
                    s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                    new_activity[i] = np.tanh(s)
            activity[:] = new_activity

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

        if not topology_frozen and (step + 1) % K == 0:
            total_delta += _hebbian_step(G, activity, N, rng)

    return steps_survived, total_delta, G.number_of_edges(), food_eaten, mode


if __name__ == '__main__':
    assoc_data = run_world_association_experiment()
    plot_world_association(assoc_data)

    # coevo_data = run_coevolution_experiment()
    # plot_coevolution_results(coevo_data)

    # readout_only_data = run_evolution_readout_only()

    # print('\n=== Comparison at generation 50 ===')
    # print(f'Co-evolution mean survival:  {coevo_data["gen_mean_survival"][-1]:.0f}')
    # print(f'Readout-only mean survival:  {readout_only_data["gen_mean_survival"][-1]:.0f}')
    # print(f'Random baseline:             {coevo_data["rand_baseline_mean"]:.0f}')
    # hebbian_contributes = (coevo_data['gen_mean_survival'][-1] >
    #                        readout_only_data['gen_mean_survival'][-1] * 1.1)
    # print(f'Hebbian topology change contributes: {hebbian_contributes}')

    # plot_evolution_control(coevo_data, readout_only_data)
