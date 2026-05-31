"""Session 10: アウトプットノードによる身体化

Node layout (N=20):
  node 0-3  : input  — x, y, HP, food_flag (injected each step)
  node 4-8  : output — argmax(activity[4:9]) selects action 0-4
  node 9-19 : internal — 11 purely recurrent nodes

Action mapping (follows world_base.py):
  action 0 → North (row-1)   node4
  action 1 → South (row+1)   node5
  action 2 → West  (col-1)   node6
  action 3 → East  (col+1)   node7
  action 4 → Eat             node8

Old arch baseline (N=20):
  node 0-3  : input
  node 4-19 : internal (16 nodes) → evolved readout_w (16×5) → softmax action

Experiments:
  A  Single agent, T=10000 continuous steps; snapshot per 1000-step window
  B  Evolution: 50 gen × 10 agents × 5 episodes — new arch vs old arch vs random
  C  Best evolved new-arch agent: pre-food output-node patterns vs food direction
"""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# ─── Constants ────────────────────────────────────────────────────────────────
_N         = 20
_K         = 10          # Hebbian every K steps
_DENSITY   = 0.2
_INIT_W    = 0.05
_LR        = 0.05
_GRID      = 5
_HP_MAX    = 200
_HP_DECAY  = 1
_FOOD_VAL  = 30
_RESPAWN   = 30
_FOOD_POS  = [(0, 0), (4, 4)]   # (row, col): NW, SE
_N_PROP    = 3
_MAX_STEPS = 500
_ACTION_NAMES = ['North', 'South', 'West', 'East', 'Eat']


# ─── Graph utilities ──────────────────────────────────────────────────────────

def _s10_build_graph(rng):
    G = nx.DiGraph()
    G.add_nodes_from(range(_N))
    for i in range(_N):
        for j in range(_N):
            if i != j and rng.random() < _DENSITY:
                G.add_edge(i, j, weight=_INIT_W)
    return G


def _s10_get_W(G):
    """N×N weight matrix, W[i,j] = weight of edge i→j."""
    W = np.zeros((_N, _N))
    for i, j, d in G.edges(data=True):
        W[i, j] = d['weight']
    return W


def _s10_propagate(W, activity, inp4):
    """One propagation step. inp4: sensor values for nodes 0-3."""
    new_act = np.tanh(W.T @ activity)
    new_act[:4] = inp4
    return new_act


def _s10_hebb(G, W, activity, rng):
    """Hebbian update: modifies G and W in place."""
    to_remove = []
    for i, j, data in list(G.edges(data=True)):
        w = data['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += _LR
        w -= 0.01
        if w < 0.01:
            to_remove.append((i, j))
            W[i, j] = 0.0
        else:
            w = min(w, 1.0)
            G[i][j]['weight'] = w
            W[i, j] = w
    G.remove_edges_from(to_remove)
    existing = set(G.edges())
    for i in range(_N):
        for j in range(_N):
            if i != j and (i, j) not in existing and rng.random() < 0.01:
                G.add_edge(i, j, weight=_INIT_W)
                W[i, j] = _INIT_W


def _s10_mutate(G, rng, mutation_std=0.05, edge_change_prob=0.05):
    """Return a mutated copy of G."""
    G_new = G.copy()
    for i, j in list(G_new.edges()):
        w = float(G_new[i][j]['weight']) + rng.normal(0, mutation_std)
        G_new[i][j]['weight'] = float(np.clip(w, 0.01, 1.0))
    existing = set(G_new.edges())
    for i in range(_N):
        for j in range(_N):
            if i == j:
                continue
            if (i, j) in existing:
                if rng.random() < edge_change_prob:
                    G_new.remove_edge(i, j)
            else:
                if rng.random() < edge_change_prob:
                    G_new.add_edge(i, j, weight=float(rng.uniform(0.01, 1.0)))
    return G_new


# ─── World step ───────────────────────────────────────────────────────────────

def _s10_world_step(row, col, action, food_avail, hp):
    ate = -1
    if   action == 0: row = max(0, row - 1)
    elif action == 1: row = min(_GRID - 1, row + 1)
    elif action == 2: col = max(0, col - 1)
    elif action == 3: col = min(_GRID - 1, col + 1)
    elif action == 4:
        for idx, (fr, fc) in enumerate(_FOOD_POS):
            if row == fr and col == fc and food_avail[idx]:
                hp  = min(_HP_MAX, hp + _FOOD_VAL)
                ate = idx
                break
    return row, col, hp, ate


def _s10_inp4(row, col, hp, food_avail):
    return np.array([
        col / (_GRID - 1),
        row / (_GRID - 1),
        hp  / _HP_MAX,
        1.0 if any(food_avail) else 0.0,
    ])


# ─── Episode runners ──────────────────────────────────────────────────────────

def _s10_run_ep_new(G, W, rng, use_hebb=True):
    """New-arch episode. Modifies G and W in-place via Hebbian. Returns steps."""
    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0

    for step in range(_MAX_STEPS):
        if hp <= 0:
            break
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, _s10_inp4(row, col, hp, food_avail))
        action = int(np.argmax(activity[4:9]))
        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
        hp -= _HP_DECAY
        steps = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0
        if use_hebb and (step + 1) % _K == 0:
            _s10_hebb(G, W, activity, rng)
    return steps


def _s10_run_ep_new_tracked(G, W, rng, use_hebb=False):
    """New-arch episode with full tracking. Returns (steps, food_count, pre_food dict)."""
    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    food_count = 0
    pre_food   = {0: [], 1: []}
    act_buf    = deque(maxlen=20)

    for step in range(_MAX_STEPS):
        if hp <= 0:
            break
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, _s10_inp4(row, col, hp, food_avail))
        act_buf.append(activity[4:9].copy())
        action = int(np.argmax(activity[4:9]))
        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            food_count += 1
            if len(act_buf) >= 5:
                pre_food[ate].append(np.mean(list(act_buf), axis=0))
        hp -= _HP_DECAY
        steps = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0
        if use_hebb and (step + 1) % _K == 0:
            _s10_hebb(G, W, activity, rng)
    return steps, food_count, pre_food


def _s10_softmax_sample(logits, rng, T=1.0):
    x = logits - np.max(logits)
    p = np.exp(x / T)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _s10_run_ep_old(G, W, readout_w, rng):
    """Old-arch episode: softmax(activity[4:20] @ readout_w). Modifies G/W via Hebbian."""
    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0

    for step in range(_MAX_STEPS):
        if hp <= 0:
            break
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, _s10_inp4(row, col, hp, food_avail))
        action = _s10_softmax_sample(activity[4:20] @ readout_w, rng)
        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
        hp -= _HP_DECAY
        steps = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0
        if (step + 1) % _K == 0:
            _s10_hebb(G, W, activity, rng)
    return steps


def _s10_run_ep_random(rng):
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    for step in range(_MAX_STEPS):
        if hp <= 0:
            break
        action = int(rng.integers(0, 5))
        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
        hp -= _HP_DECAY
        steps = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0
    return steps


# ─── Experiment A: Single agent long-term ─────────────────────────────────────

def run_single_agent(seed=42, T_total=10000, window=1000):
    """Single new-arch agent, T_total steps. G/activity persist across resets."""
    rng = np.random.default_rng(seed)
    G   = _s10_build_graph(rng)
    W   = _s10_get_W(G)

    # Accumulators
    win_action_counts = np.zeros(5, dtype=int)
    win_food          = 0
    win_output_sum    = np.zeros(5)

    windows_action  = []   # (n_windows, 5) normalized frequency
    windows_food    = []
    windows_output  = []   # (n_windows, 5) mean activity of output nodes
    windows_edges   = []

    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    deaths     = 0

    for t in range(T_total):
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, _s10_inp4(row, col, hp, food_avail))

        action = int(np.argmax(activity[4:9]))
        win_action_counts[action] += 1
        win_output_sum += activity[4:9]

        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            win_food += 1

        hp -= _HP_DECAY
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (t + 1) % _K == 0:
            _s10_hebb(G, W, activity, rng)

        if hp <= 0:
            deaths += 1
            row, col   = 2, 2
            hp         = 100
            food_avail = [True, True]
            food_timer = [0, 0]
            # G, W, activity persist

        if (t + 1) % window == 0:
            total_actions = win_action_counts.sum()
            windows_action.append(win_action_counts / total_actions
                                  if total_actions > 0 else np.ones(5) / 5)
            windows_food.append(win_food)
            windows_output.append(win_output_sum / window)
            windows_edges.append(G.number_of_edges())
            win_action_counts = np.zeros(5, dtype=int)
            win_food          = 0
            win_output_sum    = np.zeros(5)

    print(f'  deaths={deaths}, total_food={sum(windows_food)}')
    return {
        'windows_action': np.array(windows_action),   # (n_win, 5)
        'windows_food':   windows_food,
        'windows_output': np.array(windows_output),   # (n_win, 5)
        'windows_edges':  windows_edges,
        'T_total':        T_total,
        'window':         window,
        'G_final':        G,
        'W_final':        W,
    }


# ─── Experiment B: Evolution ───────────────────────────────────────────────────

def _s10_evolve_new(seed, n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_agents):
        G = _s10_build_graph(rng)
        pop.append((G, _s10_get_W(G)))

    gen_means, gen_bests = [], []
    best_G = best_W = None

    for gen in range(n_gen):
        fitnesses = []
        for G, W in pop:
            rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
            total  = sum(_s10_run_ep_new(G, W,
                         np.random.default_rng(int(rng_ag.integers(0, 2**32))))
                         for _ in range(n_ep))
            fitnesses.append(total / n_ep)

        gen_means.append(float(np.mean(fitnesses)))
        gen_bests.append(float(np.max(fitnesses)))

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:n_surv]]
        best_G, best_W = survivors[0]

        new_pop = list(survivors)
        while len(new_pop) < n_agents:
            G_p, _ = survivors[int(rng.integers(0, n_surv))]
            G_c    = _s10_mutate(G_p, rng)
            new_pop.append((G_c, _s10_get_W(G_c)))
        pop = new_pop

        if (gen + 1) % 10 == 0:
            print(f'    gen {gen+1:3d}: mean={gen_means[-1]:.1f}, best={gen_bests[-1]:.1f}')

    return {'gen_means': gen_means, 'gen_bests': gen_bests,
            'best_G': best_G, 'best_W': best_W}


def _s10_evolve_old(seed, n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_agents):
        G  = _s10_build_graph(rng)
        rw = rng.standard_normal((16, 5)) * 0.5
        pop.append((G, _s10_get_W(G), rw))

    gen_means, gen_bests = [], []

    for gen in range(n_gen):
        fitnesses = []
        for G, W, rw in pop:
            rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
            total  = sum(_s10_run_ep_old(G, W, rw,
                         np.random.default_rng(int(rng_ag.integers(0, 2**32))))
                         for _ in range(n_ep))
            fitnesses.append(total / n_ep)

        gen_means.append(float(np.mean(fitnesses)))
        gen_bests.append(float(np.max(fitnesses)))

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:n_surv]]

        new_pop = list(survivors)
        while len(new_pop) < n_agents:
            G_p, _, rw_p = survivors[int(rng.integers(0, n_surv))]
            G_c  = _s10_mutate(G_p, rng)
            rw_c = rw_p + rng.standard_normal(rw_p.shape) * 0.1
            new_pop.append((G_c, _s10_get_W(G_c), rw_c))
        pop = new_pop

        if (gen + 1) % 10 == 0:
            print(f'    gen {gen+1:3d}: mean={gen_means[-1]:.1f}, best={gen_bests[-1]:.1f}')

    return {'gen_means': gen_means, 'gen_bests': gen_bests}


def _s10_random_baseline(seed, n_gen=50, n_agents=10, n_ep=5):
    rng = np.random.default_rng(seed)
    gen_means, gen_bests = [], []
    for _ in range(n_gen):
        fitnesses = [
            sum(_s10_run_ep_random(np.random.default_rng(int(rng.integers(0, 2**32))))
                for _ in range(n_ep)) / n_ep
            for _ in range(n_agents)
        ]
        gen_means.append(float(np.mean(fitnesses)))
        gen_bests.append(float(np.max(fitnesses)))
    return {'gen_means': gen_means, 'gen_bests': gen_bests}


def run_evolution(seed=42, n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    print('  [new arch — output nodes]...')
    new_data = _s10_evolve_new(seed,     n_gen, n_agents, n_ep, n_surv)
    print('  [old arch — co-evolution with readout]...')
    old_data = _s10_evolve_old(seed + 1, n_gen, n_agents, n_ep, n_surv)
    print('  [random baseline]...')
    rnd_data = _s10_random_baseline(seed + 2, n_gen, n_agents, n_ep)
    return {'new': new_data, 'old': old_data, 'rnd': rnd_data, 'n_gen': n_gen}


# ─── Experiment C: Action patterns ────────────────────────────────────────────

def run_action_patterns(best_G, best_W, seed=42, n_ep=30):
    """Probe best evolved agent (topology frozen) for pre-food output patterns."""
    rng      = np.random.default_rng(seed)
    pre_food = {0: [], 1: []}

    for _ in range(n_ep):
        rng_ep = np.random.default_rng(int(rng.integers(0, 2**32)))
        _, _, pf = _s10_run_ep_new_tracked(best_G, best_W, rng_ep, use_hebb=False)
        for fi in (0, 1):
            pre_food[fi].extend(pf[fi])

    print(f'  Food0 patterns: {len(pre_food[0])}, Food1 patterns: {len(pre_food[1])}')
    return {'pre_food': pre_food}


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_single_agent(data, fname='images/session_10/results_s10_single_agent.png'):
    act_mat    = data['windows_action']    # (n_win, 5)
    food_list  = data['windows_food']
    out_mat    = data['windows_output']    # (n_win, 5)
    edge_list  = data['windows_edges']
    window     = data['window']
    n_win      = len(food_list)
    xs         = [window * (i + 1) for i in range(n_win)]

    colors_act = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    colors_out = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle(
        'Session 10 Exp A: Single Agent, T=10000 continuous steps\n'
        'New arch: output nodes 4-8, argmax action selection',
        fontsize=10,
    )

    # Panel 1: Action frequency
    ax = axes[0]
    for i, (name, c) in enumerate(zip(_ACTION_NAMES, colors_act)):
        ax.plot(xs, act_mat[:, i], label=name, color=c, marker='o', markersize=4)
    ax.axhline(0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='chance')
    ax.set_ylabel('Action Frequency')
    ax.set_title('Action Distribution per 1000-step Window')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2: Food eaten + edge count
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    ax2a.bar(xs, food_list, width=window * 0.6, color='gold', alpha=0.7, label='Food eaten')
    ax2b.plot(xs, edge_list, color='slategray', marker='s', markersize=4,
              linestyle='--', label='Edge count')
    ax2a.set_ylabel('Food Eaten (per window)', color='goldenrod')
    ax2b.set_ylabel('Edge Count', color='slategray')
    ax2a.set_title('Food Eaten and Edge Count per Window')
    lines_a, labs_a = ax2a.get_legend_handles_labels()
    lines_b, labs_b = ax2b.get_legend_handles_labels()
    ax2a.legend(lines_a + lines_b, labs_a + labs_b, fontsize=8)
    ax2a.grid(True, alpha=0.3)

    # Panel 3: Output node mean activity
    ax = axes[2]
    node_labels = [f'node{4+i} ({_ACTION_NAMES[i]})' for i in range(5)]
    for i, (lbl, c) in enumerate(zip(node_labels, colors_out)):
        ax.plot(xs, out_mat[:, i], label=lbl, color=c, marker='o', markersize=4)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Mean Activity')
    ax.set_xlabel('Total Steps')
    ax.set_title('Output Node (4-8) Mean Activity per Window\n'
                 '(differentiation = output nodes have learned distinct roles)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_evolution(data, fname='images/session_10/results_s10_evolution.png'):
    n_gen  = data['n_gen']
    xs     = np.arange(1, n_gen + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Session 10 Exp B: Evolution (50 gen × 10 agents × 5 episodes)\n'
        'New arch (output nodes) vs Old arch (co-evolution+readout) vs Random',
        fontsize=10,
    )

    cond_map = [
        ('new', 'New arch\n(output nodes, argmax)', 'steelblue'),
        ('old', 'Old arch\n(co-evo readout, softmax)', 'darkorange'),
        ('rnd', 'Random\nbaseline', 'gray'),
    ]

    for ax, (key, ylabel) in zip(axes, [('gen_means', 'Mean Survival Steps'),
                                         ('gen_bests', 'Best Survival Steps')]):
        for cond_key, label, color in cond_map:
            vals = data[cond_key][key]
            ax.plot(xs, vals, label=label, color=color, linewidth=2)
            # Smoothed trend
            if len(vals) > 5:
                smooth = np.convolve(vals, np.ones(5) / 5, mode='valid')
                ax.plot(xs[4:], smooth, color=color, linewidth=3, alpha=0.35)
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' per Generation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_action_patterns(data, fname='images/session_10/results_s10_action_pattern.png'):
    pre_food     = data['pre_food']
    node_labels  = [f'node{4+i}\n({_ACTION_NAMES[i]})' for i in range(5)]
    food_labels  = ['Food@NW (0,0)\nGoal: North+West', 'Food@SE (4,4)\nGoal: South+East']
    good_actions = [[0, 2], [1, 3]]   # expected output node indices for NW vs SE

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Session 10 Exp C: Pre-food Output Node Activity\n'
        'Best evolved agent (frozen topology), output nodes 4-8',
        fontsize=10,
    )

    # Heatmap data: 2 rows (NW, SE) × 5 cols (node4-8)
    hm_data = np.zeros((2, 5))
    n_count = [0, 0]
    for fi in (0, 1):
        if pre_food[fi]:
            hm_data[fi] = np.mean(pre_food[fi], axis=0)
            n_count[fi] = len(pre_food[fi])

    colors = ['steelblue', 'darkorange']
    for ax, food_idx, food_label, color, good in zip(
            axes, (0, 1), food_labels, colors, good_actions):
        patterns = pre_food[food_idx]
        if not patterns:
            ax.text(0.5, 0.5, 'No food-eat events\nduring probe', ha='center',
                    va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(food_label)
            continue
        mean_act  = np.mean(patterns, axis=0)
        std_act   = np.std(patterns,  axis=0)
        x         = np.arange(5)
        bars      = ax.bar(x, mean_act, color=color, alpha=0.7,
                           yerr=std_act, capsize=5,
                           error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
        for i in good:
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3.0)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(node_labels, fontsize=8)
        ax.set_ylabel('Mean Output Node Activity\n(t-20 to t before eating)')
        ax.set_title(f'{food_label}\nn={len(patterns)} patterns')
        ax.text(0.5, 0.96, 'Gold border = expected active node',
                ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Also add heatmap inset
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 2.5))
    ax2.imshow(hm_data, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(node_labels, fontsize=8)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([f'NW food (n={n_count[0]})', f'SE food (n={n_count[1]})'],
                        fontsize=8)
    ax2.set_title('Output node activity heatmap (NW vs SE food)')
    for (r, c), val in np.ndenumerate(hm_data):
        ax2.text(c, r, f'{val:.2f}', ha='center', va='center', fontsize=9)
    plt.tight_layout()
    hm_fname = fname.replace('.png', '_heatmap.png')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(hm_fname, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {hm_fname}')

    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 10: アウトプットノードによる身体化 ===')

    print('\n[Exp A] Single agent, T=10000...')
    expA = run_single_agent(seed=42, T_total=10000, window=1000)
    print(f'  Action dist (final window): '
          f'{dict(zip(_ACTION_NAMES, expA["windows_action"][-1].round(3)))}')
    print(f'  Food per window: {expA["windows_food"]}')
    plot_single_agent(expA)

    print('\n[Exp B] Evolution (50 gen × 10 agents × 5 ep)...')
    expB = run_evolution(seed=42, n_gen=50, n_agents=10, n_ep=5, n_surv=3)
    for key, label in [('new', 'New'), ('old', 'Old'), ('rnd', 'Random')]:
        means = expB[key]['gen_means']
        print(f'  {label}: gen1={means[0]:.1f}, gen25={means[24]:.1f}, '
              f'gen50={means[-1]:.1f}')
    plot_evolution(expB)

    print('\n[Exp C] Action patterns (best new-arch agent, frozen, 30 ep)...')
    best_G = expB['new']['best_G']
    best_W = expB['new']['best_W']
    expC   = run_action_patterns(best_G, best_W, seed=42, n_ep=30)
    plot_action_patterns(expC)

    print('\nDone.')
