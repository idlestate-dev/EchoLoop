"""Session 8: 世界に問う — ContextGridWorld survival test."""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from session_7_context_activation import (
    _S7_COND, _S7_K, _S7_INTERNAL,
    _s7_build_graph, _s7_step, _s7_hebb, _s7_train_phase,
)

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


def plot_world_test(data, fname='images/session_8/results_world_test.png'):
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
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


