"""Session 4: Context interference environment (E/I evolution)."""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from world_base import GridWorld, ContextGridWorld, _make_graph, _mutate_graph, _hebbian_step, _softmax_sample, _run_episode, _run_context_episode

# ─── Session 4: Context interference environment ──────────────────────────────


def _run_context_episode_ei(G, activity, rng, genome, N=20, K=10,
                              n_propagation_steps=3, temperature=1.0,
                              readout_weights=None, mode=None):
    """ContextGridWorld episode with E/I isolation dynamics.
    Returns (steps_survived, food_eaten, mode)."""
    pass  # imported at module level
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
    pass  # imported at module level
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
                               fname='images/session_4/results_context_ei_evolution.png'):
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
    pass  # imported at module level
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
            pass  # imported at module level
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
    pass  # imported at module level
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


def plot_context_comparison(data, fname='images/session_4/results_context_comparison.png'):
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


