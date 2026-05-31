"""Session 9: 世界がトポロジーを彫刻するメカニズム

Experiments:
  A  Topology convergence — single agent, T=5000 steps continuous,
     topology snapshots every 500 steps (topology + activity inherit across resets)
  B  Survivor vs non-survivor topology — 20 agents × 500 steps
  C  Experience trace — pre-food activity probed against trained topology
"""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
_N        = 20          # total nodes: 0-3 input, 4-19 internal
_K        = 10          # Hebbian update every K steps
_DENSITY  = 0.2
_INIT_W   = 0.05
_LR       = 0.05

_GRID     = 5
_HP_MAX   = 200
_HP_DECAY = 1
_FOOD_VAL = 30
_RESPAWN  = 30
_FOOD_POS = [(0, 0), (4, 4)]  # (row, col): NW=top-left, SE=bottom-right
_N_PROP   = 3   # propagation steps per world step
_TEMP     = 1.0


# ─── Graph utilities ──────────────────────────────────────────────────────────

def _s9_build_graph(rng):
    G = nx.DiGraph()
    G.add_nodes_from(range(_N))
    for i in range(_N):
        for j in range(_N):
            if i != j and rng.random() < _DENSITY:
                G.add_edge(i, j, weight=_INIT_W)
    return G


def _s9_propagate(G, activity, input_vals):
    new_act = np.zeros(_N)
    for i in range(_N):
        if i in input_vals:
            new_act[i] = input_vals[i]
        else:
            s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
            new_act[i] = np.tanh(s)
    return new_act


def _s9_hebb(G, activity, rng):
    to_remove = []
    for i, j, data in list(G.edges(data=True)):
        w = data['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += _LR
        w -= 0.01
        if w < 0.01:
            to_remove.append((i, j))
        else:
            G[i][j]['weight'] = min(w, 1.0)
    G.remove_edges_from(to_remove)
    existing = set(G.edges())
    for i in range(_N):
        for j in range(_N):
            if i != j and (i, j) not in existing and rng.random() < 0.01:
                G.add_edge(i, j, weight=_INIT_W)


def _s9_softmax(logits, rng):
    x = logits - np.max(logits)
    p = np.exp(x / _TEMP)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _s9_snapshot(G):
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    n_edges = G.number_of_edges()
    if n_edges > 0:
        w_mean  = float(np.mean(weights))
        w_std   = float(np.std(weights))
        ug      = G.to_undirected()
        clust   = float(nx.average_clustering(ug))
        n_cycles = sum(1 for c in nx.simple_cycles(G) if len(c) <= 5)
    else:
        w_mean = w_std = clust = 0.0
        n_cycles = 0
    return {
        'n_edges':    n_edges,
        'w_mean':     w_mean,
        'w_std':      w_std,
        'clustering': clust,
        'n_cycles':   n_cycles,
    }


def _s9_step_world(row, col, action, food_avail, hp):
    """Apply one action. Returns updated (row, col, hp, ate_food_idx or -1)."""
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


# ─── Experiment A: Topology convergence ───────────────────────────────────────

def run_topology_convergence(seed=42, T_total=5000, snapshot_interval=500):
    """Single agent run: topology and activity persist across resets.

    Returns snapshots every snapshot_interval steps, plus pre-food and
    pre-death internal activity patterns.
    """
    rng       = np.random.default_rng(seed)
    G         = _s9_build_graph(rng)
    act       = np.zeros(_N)
    readout_w = rng.standard_normal((16, 5))

    snapshots  = []
    pre_food   = {0: [], 1: []}
    pre_death  = []
    act_buf    = deque(maxlen=100)  # rolling window of internal activity

    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    next_snap  = snapshot_interval
    deaths     = 0
    food_count = [0, 0]

    for t in range(T_total):
        inp = {
            0: col / (_GRID - 1),
            1: row / (_GRID - 1),
            2: hp  / _HP_MAX,
            3: 1.0 if any(food_avail) else 0.0,
        }
        for _ in range(_N_PROP):
            act = _s9_propagate(G, act, inp)

        act_buf.append(act[4:20].copy())

        action = _s9_softmax(act[4:20] @ readout_w, rng)
        row, col, hp, ate = _s9_step_world(row, col, action, food_avail, hp)

        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            food_count[ate] += 1
            if len(act_buf) >= 10:
                pre_food[ate].append(np.mean(list(act_buf), axis=0))

        hp -= _HP_DECAY

        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (t + 1) % _K == 0:
            _s9_hebb(G, act, rng)

        if hp <= 0:
            if len(act_buf) >= 10:
                pre_death.append(np.mean(list(act_buf), axis=0))
            deaths += 1
            # Reset world state; topology (G) and activity (act) persist
            row, col   = 2, 2
            hp         = 100
            food_avail = [True, True]
            food_timer = [0, 0]
            act_buf.clear()

        if t + 1 >= next_snap:
            snapshots.append(_s9_snapshot(G))
            next_snap += snapshot_interval

    print(f'  deaths={deaths}, food0={food_count[0]}, food1={food_count[1]}, '
          f'pre_food0={len(pre_food[0])}, pre_food1={len(pre_food[1])}, '
          f'pre_death={len(pre_death)}')
    return {
        'snapshots':         snapshots,
        'pre_food':          pre_food,
        'pre_death':         pre_death,
        'G_final':           G,
        'T_total':           T_total,
        'snapshot_interval': snapshot_interval,
    }


# ─── Experiment B: Survivor topology ──────────────────────────────────────────

def _s9_run_single_episode(G, rng, readout_w):
    """500-step episode. Returns (steps_survived, pre_food, pre_death)."""
    act        = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    act_buf    = deque(maxlen=100)
    pre_food   = {0: [], 1: []}
    pre_death  = []
    steps      = 0

    for step in range(500):
        if hp <= 0:
            break

        inp = {
            0: col / (_GRID - 1),
            1: row / (_GRID - 1),
            2: hp  / _HP_MAX,
            3: 1.0 if any(food_avail) else 0.0,
        }
        for _ in range(_N_PROP):
            act = _s9_propagate(G, act, inp)

        act_buf.append(act[4:20].copy())

        action = _s9_softmax(act[4:20] @ readout_w, rng)
        row, col, hp, ate = _s9_step_world(row, col, action, food_avail, hp)

        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            if len(act_buf) >= 10:
                pre_food[ate].append(np.mean(list(act_buf), axis=0))

        hp -= _HP_DECAY
        steps = step + 1

        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (step + 1) % _K == 0:
            _s9_hebb(G, act, rng)

    if hp <= 0 and len(act_buf) >= 10:
        pre_death.append(np.mean(list(act_buf), axis=0))

    return steps, pre_food, pre_death


def run_survivor_topology(seed=42, n_agents=20):
    """Run n_agents for 500 steps each. Compare top-5 vs bottom-5."""
    rng       = np.random.default_rng(seed)
    readout_w = rng.standard_normal((16, 5))

    agents = []
    for ag in range(n_agents):
        rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
        G      = _s9_build_graph(rng_ag)
        steps, pre_food, pre_death = _s9_run_single_episode(G, rng_ag, readout_w)
        agents.append({
            'steps':     steps,
            'G':         G,
            'pre_food':  pre_food,
            'pre_death': pre_death,
            'snap':      _s9_snapshot(G),
        })

    agents_sorted = sorted(agents, key=lambda a: a['steps'], reverse=True)
    return {
        'agents':    agents_sorted,
        'top5':      agents_sorted[:5],
        'bot5':      agents_sorted[-5:],
        'readout_w': readout_w,
    }


# ─── Experiment C: Experience trace ───────────────────────────────────────────

def run_experience_trace(survivor_data):
    """Probe top-5 agents' topologies with their pre-food activity patterns.

    For each agent × food location: inject the mean pre-food internal pattern,
    propagate 20 steps with neutral position input, read out action distribution.
    """
    readout_w    = survivor_data['readout_w']
    top5         = survivor_data['top5']
    neutral_inp  = {0: 0.5, 1: 0.5, 2: 0.5, 3: 1.0}
    n_probe_steps = 20

    results = {0: [], 1: []}

    for agent in top5:
        G = agent['G']
        for food_idx in (0, 1):
            patterns = agent['pre_food'][food_idx]
            if not patterns:
                continue
            probe_internal = np.mean(patterns, axis=0)  # shape (16,)

            # Set initial activity: neutral inputs + probe internal state
            probe_act = np.zeros(_N)
            for k, v in neutral_inp.items():
                probe_act[k] = v
            probe_act[4:20] = probe_internal

            for _ in range(n_probe_steps):
                probe_act = _s9_propagate(G, probe_act, neutral_inp)

            logits = probe_act[4:20] @ readout_w
            x      = logits - np.max(logits)
            probs  = np.exp(x / _TEMP)
            probs /= probs.sum()
            results[food_idx].append(probs)

    return {'results': results, 'action_names': ['North', 'South', 'West', 'East', 'Eat']}


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_topology_convergence(data, fname='images/session_9/results_topology_convergence.png'):
    snaps = data['snapshots']
    si    = data['snapshot_interval']
    xs    = [si * (i + 1) for i in range(len(snaps))]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12))
    fig.suptitle(
        'Session 9 Exp A: Topology Convergence\n'
        'single agent, T=5000 steps, topology+activity persist across resets',
        fontsize=10,
    )

    metrics = [
        ('n_edges',    'Edge Count',                  'steelblue'),
        ('w_mean',     'Mean Edge Weight',             'darkorange'),
        ('clustering', 'Clustering Coefficient',       'seagreen'),
        ('n_cycles',   'Short Directed Cycles (≤ 5)', 'crimson'),
    ]
    for ax, (key, ylabel, color) in zip(axes, metrics):
        vals = [s[key] for s in snaps]
        ax.plot(xs, vals, color=color, marker='o', markersize=5, linewidth=1.8)
        if key == 'w_mean':
            stds = [s['w_std'] for s in snaps]
            ax.fill_between(
                xs,
                [v - sd for v, sd in zip(vals, stds)],
                [v + sd for v, sd in zip(vals, stds)],
                alpha=0.2, color=color, label='±1 std',
            )
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Total Steps')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_survivor_topology(data, fname='images/session_9/results_survivor_topology.png'):
    top5 = data['top5']
    bot5 = data['bot5']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Session 9 Exp B: Survivor vs Non-survivor Topology\n'
        f'N=20 agents × 500 steps | top-5 vs bottom-5',
        fontsize=10,
    )

    # Panel 1: Survival steps bar
    ax = axes[0, 0]
    top_steps = [a['steps'] for a in top5]
    bot_steps = [a['steps'] for a in bot5]
    ax.bar(range(5),      top_steps, color='steelblue', alpha=0.8, label='Top 5')
    ax.bar(range(5, 10),  bot_steps, color='firebrick', alpha=0.8, label='Bottom 5')
    ax.axhline(np.mean(top_steps), color='steelblue', linestyle='--', linewidth=1)
    ax.axhline(np.mean(bot_steps), color='firebrick', linestyle='--', linewidth=1)
    ax.set_ylabel('Steps Survived')
    ax.set_title('Survival Steps')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Edge weight distribution
    ax = axes[0, 1]
    top_w = [d['weight'] for a in top5 for _, _, d in a['G'].edges(data=True)]
    bot_w = [d['weight'] for a in bot5 for _, _, d in a['G'].edges(data=True)]
    bins  = np.linspace(0, 1, 21)
    ax.hist(top_w, bins=bins, alpha=0.6, color='steelblue', label='Top 5',    density=True)
    ax.hist(bot_w, bins=bins, alpha=0.6, color='firebrick', label='Bottom 5', density=True)
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Density')
    ax.set_title('Edge Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Input node (0-3) mean outgoing weight heatmap
    ax = axes[1, 0]
    inp_mat = np.zeros((10, 4))  # [top5, bot5] × 4 input nodes
    for ai, a in enumerate(top5 + bot5):
        for src in range(4):
            ws = [a['G'][src][j]['weight'] for j in a['G'].successors(src)]
            inp_mat[ai, src] = float(np.mean(ws)) if ws else 0.0
    im = ax.imshow(inp_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=max(inp_mat.max(), 0.01))
    ax.set_xticks(range(4))
    ax.set_xticklabels(['x (col)', 'y (row)', 'HP', 'food_flag'], fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'T{i+1}' for i in range(5)] + [f'B{i+1}' for i in range(5)], fontsize=8)
    ax.axhline(4.5, color='white', linewidth=2.5)
    ax.set_title('Input Node Mean Outgoing Weight\n(T=Top, B=Bottom)')
    plt.colorbar(im, ax=ax)

    # Panel 4: Topology metrics comparison (bar)
    ax = axes[1, 1]
    metric_keys   = ['n_edges', 'clustering', 'n_cycles']
    metric_labels = ['Edges', 'Clustering', 'Cycles (≤5)']
    x  = np.arange(len(metric_keys))
    w  = 0.35
    top_vals = [np.mean([a['snap'][k] for a in top5]) for k in metric_keys]
    bot_vals = [np.mean([a['snap'][k] for a in bot5]) for k in metric_keys]
    ax.bar(x - w/2, top_vals, width=w, color='steelblue', alpha=0.8, label='Top 5')
    ax.bar(x + w/2, bot_vals, width=w, color='firebrick', alpha=0.8, label='Bottom 5')
    for xi, (tv, bv) in enumerate(zip(top_vals, bot_vals)):
        ax.text(xi - w/2, tv + 0.01 * max(top_vals + bot_vals),
                f'{tv:.2f}', ha='center', va='bottom', fontsize=8, color='steelblue')
        ax.text(xi + w/2, bv + 0.01 * max(top_vals + bot_vals),
                f'{bv:.2f}', ha='center', va='bottom', fontsize=8, color='firebrick')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_title('Topology Metrics (mean over 5 agents)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_experience_trace(data, fname='images/session_9/results_experience_trace.png'):
    results      = data['results']
    action_names = data['action_names']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        'Session 9 Exp C: Experience Trace\n'
        'Pre-food activity (mean over t-100→t) probed → action distribution',
        fontsize=10,
    )

    food_meta = [
        (0, 'Food at (0,0) NW',  'steelblue',  [0, 2], 'North + West expected'),
        (1, 'Food at (4,4) SE',  'darkorange', [1, 3], 'South + East expected'),
    ]

    for ax, (food_idx, title, color, good_actions, note) in zip(axes, food_meta):
        probs_list = results[food_idx]
        if not probs_list:
            ax.text(0.5, 0.5, 'No data\n(no food eaten here in top-5)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        mean_probs = np.mean(probs_list, axis=0)
        std_probs  = np.std(probs_list,  axis=0)
        x          = np.arange(5)

        bars = ax.bar(x, mean_probs, color=color, alpha=0.7,
                      yerr=std_probs, capsize=5,
                      error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
        for i in good_actions:
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3.0)

        ax.axhline(0.2, color='gray', linestyle='--', linewidth=1.2, label='chance (0.2)')
        ax.set_xticks(x)
        ax.set_xticklabels(action_names)
        ax.set_ylim(0, min(0.6, mean_probs.max() + std_probs.max() + 0.1))
        ax.set_ylabel('Action Probability')
        ax.set_title(f'{title}\n({note})')
        ax.text(0.5, 0.96, f'n={len(probs_list)} patterns | gold border = expected actions',
                ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 9: 世界がトポロジーを彫刻するメカニズム ===')

    print('\n[Exp A] Topology convergence (T=5000)...')
    expA = run_topology_convergence(seed=42, T_total=5000, snapshot_interval=500)
    snaps = expA['snapshots']
    print(f'  Snapshots taken: {len(snaps)}')
    print(f'  Final: edges={snaps[-1]["n_edges"]}, w_mean={snaps[-1]["w_mean"]:.3f}, '
          f'clustering={snaps[-1]["clustering"]:.3f}, cycles={snaps[-1]["n_cycles"]}')
    plot_topology_convergence(expA)

    print('\n[Exp B] Survivor topology (n_agents=20)...')
    expB = run_survivor_topology(seed=42, n_agents=20)
    print(f'  Top-5 steps:    {[a["steps"] for a in expB["top5"]]}')
    print(f'  Bottom-5 steps: {[a["steps"] for a in expB["bot5"]]}')
    plot_survivor_topology(expB)

    print('\n[Exp C] Experience trace (probe top-5 agents)...')
    expC = run_experience_trace(expB)
    for fi in (0, 1):
        plist = expC['results'][fi]
        if plist:
            mp = np.mean(plist, axis=0)
            print(f'  Food{fi}: n={len(plist)}, mean_probs={mp.round(3)}')
        else:
            print(f'  Food{fi}: no patterns collected')
    plot_experience_trace(expC)

    print('\nDone.')
