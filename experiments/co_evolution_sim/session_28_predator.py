"""Session 28: 捕食者の導入による神経系の複雑化

カンブリア爆発の計算論的検証:
  「食料を取る」+「捕食者から逃げる」という相反する圧力が
  文脈依存的な行動を必然にするか

ネットワーク変更 (Session 27 → 28):
  入力ノード: 4 → 5 (predator_flag 追加)
  総ノード数: 20 → 21
  出力/内部の範囲: [4:9]/[9:20] → [5:10]/[10:21]
  tau_s/tau_i/tau_o はSession 27最良値に固定 (94/43/34)

実験:
  A  捕食者あり vs なしの比較 (50世代進化)
  B  睡眠様状態の観察 (T=2000連続実行)
  C  文脈依存的な行動の計測 (4文脈)
  D  捕食者速度スイープ [1,2,4,8]

出力:
  images/session_28/results_s28_evolution.png
  images/session_28/results_s28_sleep_pattern.png
  images/session_28/results_s28_context.png
  images/session_28/results_s28_predator_speed.png
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import _jp_font  # noqa: F401 — sets Japanese font in matplotlib rcParams

from session_10_embodied_output import _N_PROP, _K, _INIT_W, _LR
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_18_ratio_evolution import (
    _EP_INIT_MAX, _AR_INIT_MAX,
    _EP_MUT_STD, _AR_MUT_STD,
    _ACTIVITY_NOISE,
    _N_SURV,
)
from session_27_tm_resources import _s27_update_resources

# ── Constants ──────────────────────────────────────────────────────────────────

_S28_N          = 21
_S28_INP_START  = 0
_S28_INP_END    = 5   # nodes 0-4: x, y, HP, food_flag, predator_flag
_S28_OUT_START  = 5
_S28_OUT_END    = 10  # nodes 5-9: N/S/W/E/Eat
_S28_INT_START  = 10
_S28_INT_END    = 21  # nodes 10-20: internal

_S28_DENSITY    = 0.2
_S28_MUT_STD    = 0.05
_S28_EDGE_CHNG  = 0.05
_S28_HEBB_LR    = float(_LR)
_S28_HEBB_DECAY = 0.01

# Fixed tau values from Session 27 best result
_S28_TAU_S      = 94
_S28_TAU_I      = 43
_S28_TAU_O      = 34

# Depletion evolution params
_S28_DEPL_LO      = 0.0
_S28_DEPL_HI      = 0.5
_S28_DEPL_MUT_STD = 0.02

# World params
_S28_GRID          = 5
_S28_HP_START      = 100
_S28_HP_MAX        = 200
_S28_HP_DECAY      = 1
_S28_FOOD_VALUE    = 30
_S28_FOOD_RESPAWN  = 40
_S28_MAX_STEPS     = 500
_S28_N_FOODS       = 2
_S28_PRED_DAMAGE   = 50
_S28_PRED_SPEED    = 2    # predator moves every 2 steps
_S28_FOOD_RESOURCE = 0.5  # resource boost fraction when eating
_S28_FOOD_DIST     = 2    # Manhattan distance threshold for food_flag
_S28_PRED_DIST     = 1    # Manhattan distance threshold for predator_flag

# Evolution params
_S28_N_GEN        = 50
_S28_N_AGENTS     = 10
_S28_N_EP         = 5
_S28_N_SURV       = 3
_S28_SEED         = 42
_S28_MR           = 0.01
_S28_ACT_NOISE    = float(_ACTIVITY_NOISE)
_S28_T_CONSOL     = 0
_S28_ACT_THRESH   = 0.1
_S28_N_GEN_D      = 30   # reduced generations for Exp D speed sweep

# Sleep observation
_S28_SLEEP_T      = 2000
_S28_SLEEP_CHUNK  = 100
_S28_CONTEXT_T    = 100  # steps for context measurement


# ── Network primitives ─────────────────────────────────────────────────────────

def _s28_build_graph(rng):
    """Build a 21-node directed graph with random edges."""
    G = nx.DiGraph()
    G.add_nodes_from(range(_S28_N))
    for i in range(_S28_N):
        for j in range(_S28_N):
            if i != j and rng.random() < _S28_DENSITY:
                G.add_edge(i, j, weight=_INIT_W)
    return G


def _s28_get_W(G):
    """N×N weight matrix from graph G. W[i,j] = weight of edge i→j."""
    n = G.number_of_nodes()
    W = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        W[i, j] = d['weight']
    return W


def _s28_propagate(W, activity, inp5):
    """One propagation step. Injects inp5 (5 values) into nodes 0-4."""
    new_act = np.tanh(W.T @ activity)
    new_act[:5] = inp5
    return new_act


def _s28_mutate_graph(G, rng):
    """Return a mutated copy of G (weight perturbation + edge flip)."""
    G_new = G.copy()
    for i, j in list(G_new.edges()):
        w = float(G_new[i][j]['weight']) + rng.normal(0, _S28_MUT_STD)
        G_new[i][j]['weight'] = float(np.clip(w, 0.01, 1.0))
    existing = set(G_new.edges())
    for i in range(_S28_N):
        for j in range(_S28_N):
            if i == j:
                continue
            if (i, j) in existing:
                if rng.random() < _S28_EDGE_CHNG:
                    G_new.remove_edge(i, j)
            else:
                if rng.random() < _S28_EDGE_CHNG:
                    G_new.add_edge(i, j, weight=float(rng.uniform(0.01, 1.0)))
    return G_new


def _s28_hebb(G, W, activity, rng, edge_add_prob, activity_ratio):
    """Hebbian update for 21-node network (same logic as _s18_hebb)."""
    to_remove = []
    for i, j, data in list(G.edges(data=True)):
        w = data['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += _S28_HEBB_LR
        w -= _S28_HEBB_DECAY
        if w < 0.01:
            to_remove.append((i, j))
            W[i, j] = 0.0
        else:
            w = min(w, 1.0)
            G[i][j]['weight'] = w
            W[i, j] = w
    G.remove_edges_from(to_remove)

    if edge_add_prob <= 0.0:
        return

    if rng.random() < edge_add_prob:
        if rng.random() < activity_ratio:
            act_pos = np.clip(activity, 0, None)
            if act_pos.sum() > 1e-9:
                probs = act_pos / act_pos.sum()
                src = int(rng.choice(_S28_N, p=probs))
                dst = int(rng.choice(_S28_N, p=probs))
            else:
                src = int(rng.integers(0, _S28_N))
                dst = int(rng.integers(0, _S28_N))
        else:
            src = int(rng.integers(0, _S28_N))
            dst = int(rng.integers(0, _S28_N))
        if src != dst and not G.has_edge(src, dst):
            G.add_edge(src, dst, weight=_INIT_W)
            W[src, dst] = _INIT_W


# ── Resource helpers ───────────────────────────────────────────────────────────

def _s28_make_tau_arr(tau_s=_S28_TAU_S, tau_i=_S28_TAU_I, tau_o=_S28_TAU_O):
    """Node-type-specific tau_rec array for 21-node network."""
    tau = np.ones(_S28_N, dtype=float)
    tau[_S28_INP_START:_S28_INP_END] = float(tau_s)
    tau[_S28_OUT_START:_S28_OUT_END] = float(tau_o)
    tau[_S28_INT_START:_S28_INT_END] = float(tau_i)
    return tau


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s28_make_genome(rng):
    """Create a random genome for PredatorGridWorld."""
    G = _s28_build_graph(rng)
    return {
        'G':              G,
        'W':              _s28_get_W(G),
        'depletion_rate': float(rng.uniform(_S28_DEPL_LO, _S28_DEPL_HI)),
        'edge_add_prob':  float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio': float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate': _S28_MR,
    }


def _s28_mutate_genome(genome, rng):
    """Return a mutated copy of the genome."""
    G_new = _s28_mutate_graph(genome['G'], rng)
    return {
        'G':              G_new,
        'W':              _s28_get_W(G_new),
        'depletion_rate': float(np.clip(
            genome['depletion_rate'] + rng.normal(0, _S28_DEPL_MUT_STD),
            _S28_DEPL_LO, _S28_DEPL_HI)),
        'edge_add_prob':  float(np.clip(
            genome['edge_add_prob'] + rng.normal(0, _EP_MUT_STD),
            0.0, _EP_INIT_MAX)),
        'activity_ratio': float(np.clip(
            genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD),
            0.0, _AR_INIT_MAX)),
        'metabolic_rate': genome['metabolic_rate'],
    }


# ── World helpers ──────────────────────────────────────────────────────────────

def _s28_inp5(row, col, hp, food_positions, food_avail, pred_pos):
    """Compute 5-input signal vector.

    node0: x position (col/4)
    node1: y position (row/4)
    node2: HP normalized
    node3: food_flag  (1.0 if any uneaten food within Manhattan ≤ 2)
    node4: pred_flag  (1.0 if predator within Manhattan ≤ 1)
    """
    food_flag = 0.0
    for (fr, fc), avail in zip(food_positions, food_avail):
        if avail and abs(row - fr) + abs(col - fc) <= _S28_FOOD_DIST:
            food_flag = 1.0
            break

    pr, pc = pred_pos
    pred_flag = 1.0 if abs(row - pr) + abs(col - pc) <= _S28_PRED_DIST else 0.0

    return np.array([
        col / (_S28_GRID - 1),
        row / (_S28_GRID - 1),
        np.clip(hp / _S28_HP_START, 0.0, 1.5),
        food_flag,
        pred_flag,
    ])


def _s28_init_foods(rng, agent_row=2, agent_col=2):
    """Initialize _S28_N_FOODS food items at random positions (not on agent)."""
    candidates = [
        (r, c) for r in range(_S28_GRID) for c in range(_S28_GRID)
        if not (r == agent_row and c == agent_col)
    ]
    idxs = rng.choice(len(candidates), size=_S28_N_FOODS, replace=False)
    return [list(candidates[i]) for i in idxs]


def _s28_init_pred(rng, agent_row=2, agent_col=2, min_dist=2):
    """Initialize predator at Manhattan distance ≥ min_dist from agent."""
    candidates = [
        (r, c) for r in range(_S28_GRID) for c in range(_S28_GRID)
        if abs(r - agent_row) + abs(c - agent_col) >= min_dist
    ]
    idx = int(rng.integers(0, len(candidates)))
    return list(candidates[idx])


def _s28_pred_step(pred_pos, rng):
    """Move predator one step in a random direction within the grid."""
    pr, pc = pred_pos
    d = int(rng.integers(0, 4))
    if d == 0:   pr = max(0, pr - 1)
    elif d == 1: pr = min(_S28_GRID - 1, pr + 1)
    elif d == 2: pc = max(0, pc - 1)
    else:        pc = min(_S28_GRID - 1, pc + 1)
    return [pr, pc]


# ── Episode runner ─────────────────────────────────────────────────────────────

def _s28_run_ep(G, W, genome, rng,
                n_predators=1, predator_speed=_S28_PRED_SPEED,
                record_activity=False, record_resources=False,
                record_encounters=False,
                freeze_pred=False):
    """PredatorGridWorld episode with TM resource model.

    Modifies G and W in place via Hebbian learning.

    Returns dict:
      steps, food, pred_hits,
      act_recs (list of eff arrays or None),
      res_recs (list of resources arrays or None),
      enc_steps (list of encounter timesteps or None).
    """
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S28_N)
    activity  = np.zeros(_S28_N)

    row, col  = 2, 2
    hp        = float(_S28_HP_START)

    food_positions = _s28_init_foods(rng)
    food_avail     = [True] * _S28_N_FOODS
    food_timer     = [0]   * _S28_N_FOODS

    pred_positions = []
    for _ in range(n_predators):
        pred_positions.append(_s28_init_pred(rng))

    steps     = 0
    food      = 0
    pred_hits = 0

    act_recs  = [] if record_activity   else None
    res_recs  = [] if record_resources  else None
    enc_steps = [] if record_encounters else None

    for step in range(_S28_MAX_STEPS):
        if hp <= 0:
            break

        # Move predator(s) and check collision
        for pi in range(len(pred_positions)):
            if not freeze_pred and step % predator_speed == 0:
                pred_positions[pi] = _s28_pred_step(pred_positions[pi], rng)
            if pred_positions[pi][0] == row and pred_positions[pi][1] == col:
                hp -= _S28_PRED_DAMAGE
                pred_hits += 1
                if record_encounters:
                    enc_steps.append(step)

        # Compute input
        pred_ref = pred_positions[0] if pred_positions else [-99, -99]
        inp5 = _s28_inp5(row, col, hp, food_positions, food_avail, pred_ref)

        # Propagate network
        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)

        # Effective activity with resources
        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        # Update resources (TM model inherited from S27)
        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())
        if record_resources:
            res_recs.append(resources.copy())

        # HP costs
        hp -= _S28_HP_DECAY
        hp -= metabolic_rate * float(np.sum(eff))

        # Agent action
        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(_S28_GRID - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(_S28_GRID - 1, col + 1)
        elif action == 4:
            for fi in range(_S28_N_FOODS):
                if (food_avail[fi]
                        and row == food_positions[fi][0]
                        and col == food_positions[fi][1]):
                    hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                    resources = np.clip(
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources), 0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    food += 1
                    break

        steps = step + 1

        # Food respawn
        for fi in range(_S28_N_FOODS):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        # Hebbian learning
        if (step + 1) % _K == 0:
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, _S28_T_CONSOL)

    return {
        'steps':     steps,
        'food':      food,
        'pred_hits': pred_hits,
        'act_recs':  act_recs,
        'res_recs':  res_recs,
        'enc_steps': enc_steps,
    }


# ── Evolution ──────────────────────────────────────────────────────────────────

def _s28_evolve(n_predators=1, predator_speed=_S28_PRED_SPEED,
                seed=_S28_SEED, n_gen=_S28_N_GEN):
    """Evolve genomes in PredatorGridWorld.

    Returns (best_genome, history_dict).
    """
    rng = np.random.default_rng(seed + 28000 + n_predators * 100 + predator_speed)
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits',
        'gen_mean_active', 'gen_depl', 'gen_edge_count',
    )}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s28_run_ep(
                    g['G'], g['W'], g, ep_rng,
                    n_predators=n_predators,
                    predator_speed=predator_speed,
                    record_activity=True)
                total += res['steps']
                ep_food.append(res['food'])
                ep_hits.append(res['pred_hits'])
                if res['act_recs']:
                    arr = np.array(res['act_recs'])
                    ep_active.append(
                        float(np.sum(np.mean(arr, axis=0) > _S28_ACT_THRESH)))
            fitnesses.append(total / _S28_N_EP)
            g['_ep_food']   = float(np.mean(ep_food))
            g['_ep_hits']   = float(np.mean(ep_hits))
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0
            g['_ep_edges']  = float(g['G'].number_of_edges())

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]

        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_pred_hits'].append(bg['_ep_hits'])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_depl'].append(bg['depletion_rate'])
        hist['gen_edge_count'].append(bg['_ep_edges'])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_S28_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _S28_N_AGENTS:
            parent = survivors[int(rng.integers(0, _S28_N_SURV))]
            new_pop.append(_s28_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            pred_str = f'hits={bg["_ep_hits"]:.2f}/ep  ' if n_predators > 0 else ''
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'{pred_str}'
                  f'depl={bg["depletion_rate"]:.3f}  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        for k in ('_ep_food', '_ep_hits', '_ep_active', '_ep_edges'):
            g.pop(k, None)

    return pop[0], hist


# ── Experiment A: predator vs no-predator ─────────────────────────────────────

def run_exp_a_evolution(seed=_S28_SEED):
    """Evolve under 2 conditions: no predator vs 1 predator.

    Returns (no_pred_best, no_pred_hist, pred_best, pred_hist).
    """
    print(f'\n  [Exp A] 進化比較 ({_S28_N_GEN}世代, {_S28_N_AGENTS}個体, {_S28_N_EP}エピソード)')

    print('  条件1: 捕食者なし')
    no_pred_best, no_pred_hist = _s28_evolve(
        n_predators=0, predator_speed=_S28_PRED_SPEED, seed=seed)
    print(f'  → steps={no_pred_hist["gen_best_steps"][-1]:.1f}  '
          f'depl={no_pred_best["depletion_rate"]:.4f}  '
          f'active={no_pred_hist["gen_mean_active"][-1]:.1f}')

    print('  条件2: 捕食者あり')
    pred_best, pred_hist = _s28_evolve(
        n_predators=1, predator_speed=_S28_PRED_SPEED, seed=seed)
    print(f'  → steps={pred_hist["gen_best_steps"][-1]:.1f}  '
          f'depl={pred_best["depletion_rate"]:.4f}  '
          f'active={pred_hist["gen_mean_active"][-1]:.1f}  '
          f'hits={pred_hist["gen_pred_hits"][-1]:.2f}/ep')

    return no_pred_best, no_pred_hist, pred_best, pred_hist


def plot_exp_a_evolution(no_pred_best, no_pred_hist, pred_best, pred_hist,
                          fname='images/session_28/results_s28_evolution.png'):
    xs = np.arange(1, len(no_pred_hist['gen_best_steps']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Session 28 Exp A: 捕食者あり vs なしの進化比較\n'
        f'PredatorGridWorld  ({_S28_N_GEN}世代, {_S28_N_AGENTS}個体, '
        f'{_S28_N_EP}ep/個体)',
        fontsize=12,
    )

    # Panel 1: survival steps
    ax = axes[0][0]
    ax.plot(xs, no_pred_hist['gen_best_steps'], color='steelblue', linewidth=2,
            label='捕食者なし')
    ax.plot(xs, pred_hist['gen_best_steps'], color='tomato', linewidth=2,
            label='捕食者あり')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best steps / ep')
    ax.set_title('生存ステップ数の推移')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: depletion_rate
    ax = axes[0][1]
    ax.plot(xs, no_pred_hist['gen_depl'], color='steelblue', linewidth=2,
            label='捕食者なし')
    ax.plot(xs, pred_hist['gen_depl'], color='tomato', linewidth=2,
            label='捕食者あり')
    ax.set_xlabel('Generation')
    ax.set_ylabel('depletion_rate')
    ax.set_title('depletion_rate の収束\n(高=活動コスト大=睡眠様状態出やすい)')
    ax.set_ylim(-0.01, _S28_DEPL_HI + 0.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: active nodes
    ax = axes[0][2]
    ax.plot(xs, no_pred_hist['gen_mean_active'], color='steelblue', linewidth=2,
            label='捕食者なし')
    ax.plot(xs, pred_hist['gen_mean_active'], color='tomato', linewidth=2,
            label='捕食者あり')
    ax.axhline(_S28_N, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label=f'最大{_S28_N}ノード')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Active nodes (mean)')
    ax.set_title('活動ノード数の推移')
    ax.set_ylim(0, _S28_N + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: food/ep
    ax = axes[1][0]
    ax.plot(xs, no_pred_hist['gen_food_count'], color='steelblue', linewidth=2,
            label='捕食者なし')
    ax.plot(xs, pred_hist['gen_food_count'], color='tomato', linewidth=2,
            label='捕食者あり')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Food / episode')
    ax.set_title('食料獲得数の推移')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: predator hits/ep (only with_predator condition)
    ax = axes[1][1]
    ax.plot(xs, pred_hist['gen_pred_hits'], color='tomato', linewidth=2,
            label='捕食者ヒット')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Predator hits / episode')
    ax.set_title('捕食者ヒット数の推移\n(減少 = 回避学習)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: summary
    ax = axes[1][2]
    no_pred_steps = no_pred_hist['gen_best_steps'][-1]
    pred_steps    = pred_hist['gen_best_steps'][-1]
    summary = (
        f'最終世代サマリー\n\n'
        f'【捕食者なし】\n'
        f'  生存 = {no_pred_steps:.1f} steps\n'
        f'  depl = {no_pred_best["depletion_rate"]:.4f}\n'
        f'  active = {no_pred_hist["gen_mean_active"][-1]:.1f} nodes\n\n'
        f'【捕食者あり】\n'
        f'  生存 = {pred_steps:.1f} steps\n'
        f'  depl = {pred_best["depletion_rate"]:.4f}\n'
        f'  active = {pred_hist["gen_mean_active"][-1]:.1f} nodes\n'
        f'  hits = {pred_hist["gen_pred_hits"][-1]:.2f}/ep\n\n'
        f'depl差 = {pred_best["depletion_rate"] - no_pred_best["depletion_rate"]:+.4f}\n'
        f'(正 = 捕食圧がdepletion_rate増加)'
    )
    ax.text(0.5, 0.5, summary, transform=ax.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.set_title('実験サマリー')
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment B: sleep pattern observation ───────────────────────────────────

def _s28_run_long(G, W, genome, rng, n_predators=1,
                   predator_speed=_S28_PRED_SPEED, T=_S28_SLEEP_T):
    """T-step continuous run with chunk-level recording.

    HP resets when it hits 0 to maintain T steps of observation.
    Returns dict: chunks, food_total, pred_hits_total, encounter_steps.
    """
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S28_N)
    activity  = np.zeros(_S28_N)

    row, col  = 2, 2
    hp        = float(_S28_HP_START)

    food_positions = _s28_init_foods(rng)
    food_avail     = [True] * _S28_N_FOODS
    food_timer     = [0]   * _S28_N_FOODS

    pred_positions = []
    for _ in range(n_predators):
        pred_positions.append(_s28_init_pred(rng))

    food_total  = 0
    pred_hits   = 0
    enc_steps   = []

    chunks      = []
    chunk_acts  = []
    chunk_ress  = []

    for step in range(T):
        if hp <= 0:
            hp = float(_S28_HP_START)

        for pi in range(len(pred_positions)):
            if step % predator_speed == 0:
                pred_positions[pi] = _s28_pred_step(pred_positions[pi], rng)
            if pred_positions[pi][0] == row and pred_positions[pi][1] == col:
                hp -= _S28_PRED_DAMAGE
                pred_hits += 1
                enc_steps.append(step)

        pred_ref = pred_positions[0] if pred_positions else [-99, -99]
        inp5 = _s28_inp5(row, col, hp, food_positions, food_avail, pred_ref)

        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)
        chunk_acts.append(eff.copy())
        chunk_ress.append(resources.copy())

        hp -= _S28_HP_DECAY
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(_S28_GRID - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(_S28_GRID - 1, col + 1)
        elif action == 4:
            for fi in range(_S28_N_FOODS):
                if (food_avail[fi]
                        and row == food_positions[fi][0]
                        and col == food_positions[fi][1]):
                    hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                    resources = np.clip(
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources), 0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    food_total += 1
                    break

        for fi in range(_S28_N_FOODS):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        if (step + 1) % _K == 0:
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

        if (step + 1) % _S28_SLEEP_CHUNK == 0:
            arr_a = np.array(chunk_acts)
            arr_r = np.array(chunk_ress)
            chunks.append({
                't_end':             step + 1,
                'sensory_activity':  float(np.mean(arr_a[:, _S28_INP_START:_S28_INP_END])),
                'output_activity':   float(np.mean(arr_a[:, _S28_OUT_START:_S28_OUT_END])),
                'internal_activity': float(np.mean(arr_a[:, _S28_INT_START:_S28_INT_END])),
                'resource_mean':     float(np.mean(arr_r)),
                'resource_sensory':  float(np.mean(arr_r[:, _S28_INP_START:_S28_INP_END])),
                'resource_internal': float(np.mean(arr_r[:, _S28_INT_START:_S28_INT_END])),
                'enc_in_chunk':      sum(1 for e in enc_steps if step + 1 - _S28_SLEEP_CHUNK < e <= step + 1),
            })
            chunk_acts = []
            chunk_ress = []

    return {
        'chunks':        chunks,
        'food_total':    food_total,
        'pred_hits':     pred_hits,
        'enc_steps':     enc_steps,
    }


def run_exp_b_sleep_pattern(pred_best, seed=_S28_SEED):
    """T=2000の連続実行で睡眠様状態と捕食者遭遇パターンを観察。

    Returns dict with chunks, food_total, pred_hits, enc_steps.
    """
    print(f'\n  [Exp B] 睡眠様状態の観察 (T={_S28_SLEEP_T}ステップ連続実行)')
    G_copy = pred_best['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 28100)
    res    = _s28_run_long(G_copy, W_copy, pred_best, rng)
    n_c    = len(res['chunks'])
    if n_c > 0:
        print(f'  → chunks={n_c}  food={res["food_total"]}  '
              f'pred_hits={res["pred_hits"]}  '
              f'encounters={len(res["enc_steps"])}')
        c0, cf = res['chunks'][0], res['chunks'][-1]
        print(f'     最初100step: sens={c0["sensory_activity"]:.3f}  '
              f'int={c0["internal_activity"]:.3f}  '
              f'res_s={c0["resource_sensory"]:.3f}')
        print(f'     最後100step: sens={cf["sensory_activity"]:.3f}  '
              f'int={cf["internal_activity"]:.3f}  '
              f'res_s={cf["resource_sensory"]:.3f}')
    return res


def plot_exp_b_sleep_pattern(exp_b,
                               fname='images/session_28/results_s28_sleep_pattern.png'):
    chunks = exp_b['chunks']
    if not chunks:
        print('Warning: no chunks recorded for Exp B')
        return

    t_vals  = [c['t_end']             for c in chunks]
    sens    = [c['sensory_activity']  for c in chunks]
    out     = [c['output_activity']   for c in chunks]
    intern  = [c['internal_activity'] for c in chunks]
    res_all = [c['resource_mean']     for c in chunks]
    res_sen = [c['resource_sensory']  for c in chunks]
    res_int = [c['resource_internal'] for c in chunks]
    enc_cnt = [c['enc_in_chunk']      for c in chunks]

    enc_steps = exp_b['enc_steps']

    mid_idx        = len(chunks) // 2
    half1_avg_sens = float(np.mean(sens[:mid_idx]))   if mid_idx > 0 else 0.0
    half2_avg_sens = float(np.mean(sens[mid_idx:]))   if mid_idx < len(sens) else 0.0
    sens_drop      = half1_avg_sens - half2_avg_sens

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'Session 28 Exp B: 睡眠様状態の観察 (T={_S28_SLEEP_T}ステップ)\n'
        f'捕食者あり環境  enc={len(enc_steps)}回  '
        f'sens Δ={sens_drop:+.3f}',
        fontsize=12,
    )

    # Panel 1: layer activity
    ax = axes[0][0]
    ax.plot(t_vals, sens,   color='steelblue', linewidth=2, marker='o', markersize=4,
            label='感覚器 (node0-4)')
    ax.plot(t_vals, out,    color='tomato',    linewidth=2, marker='s', markersize=4,
            label='出力   (node5-9)')
    ax.plot(t_vals, intern, color='green',     linewidth=2, marker='^', markersize=4,
            label='内部   (node10-20)')
    # Overlay encounter markers
    if enc_steps:
        for e in enc_steps:
            ax.axvline(e, color='red', alpha=0.15, linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean activity')
    ax.set_title('層別平均活動の時系列\n(赤縦線=捕食者遭遇)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: resource time series
    ax = axes[0][1]
    ax.plot(t_vals, res_all, color='purple',    linewidth=2, marker='o', markersize=4,
            label='全ノード平均')
    ax.plot(t_vals, res_sen, color='steelblue', linewidth=2, linestyle='--', marker='s',
            markersize=4, label='感覚器資源')
    ax.plot(t_vals, res_int, color='green',     linewidth=2, linestyle='--', marker='^',
            markersize=4, label='内部資源')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean resources')
    ax.set_title('資源量の時系列')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: encounters per chunk
    ax = axes[1][0]
    ax.bar(t_vals, enc_cnt, width=_S28_SLEEP_CHUNK * 0.8, color='tomato', alpha=0.7,
           label='遭遇回数/チャンク')
    ax.set_xlabel('Step')
    ax.set_ylabel('Encounters per chunk')
    ax.set_title(f'捕食者遭遇頻度 (100stepごと)\n合計 {len(enc_steps)}回遭遇')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: summary
    ax = axes[1][1]
    if mid_idx > 0:
        half1_int = float(np.mean([c['internal_activity'] for c in chunks[:mid_idx]]))
        half2_int = float(np.mean([c['internal_activity'] for c in chunks[mid_idx:]]))
        int_drop  = half1_int - half2_int
    else:
        half1_int = half2_int = int_drop = 0.0

    summary = (
        f'2段階パターン分析\n\n'
        f'前半 (〜{t_vals[mid_idx-1] if mid_idx > 0 else "?"}step)\n'
        f'  感覚器活動: {half1_avg_sens:.3f}\n'
        f'  内部活動:   {half1_int:.3f}\n\n'
        f'後半 ({t_vals[mid_idx] if mid_idx < len(t_vals) else "?"}step〜)\n'
        f'  感覚器活動: {half2_avg_sens:.3f}\n'
        f'  内部活動:   {half2_int:.3f}\n\n'
        f'感覚器の変化: Δ={sens_drop:+.3f}\n'
        f'内部の変化:   Δ={int_drop:+.3f}\n\n'
        f'捕食者遭遇: {len(enc_steps)}回\n'
        f'食料獲得:   {exp_b["food_total"]}個'
    )
    color = 'lightgreen' if sens_drop > 0.05 else 'lightyellow'
    ax.text(0.5, 0.5, summary, transform=ax.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.85))
    ax.set_title('睡眠様状態パターン判定')
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment C: context-dependent behavior ──────────────────────────────────

_S28_CONTEXTS = [
    {'food_near': True,  'pred_near': False, 'label': '食料近\n捕食者遠'},
    {'food_near': True,  'pred_near': True,  'label': '食料近\n捕食者近'},
    {'food_near': False, 'pred_near': True,  'label': '食料遠\n捕食者近'},
    {'food_near': False, 'pred_near': False, 'label': '食料遠\n捕食者遠'},
]
_S28_ACTION_NAMES = ['北', '南', '西', '東', '食事']


def _s28_measure_context(G, W, genome, context, rng, T=_S28_CONTEXT_T):
    """Run T steps with forced context inputs, return mean output and action dist.

    The environment is held constant (predator frozen, food/agent at fixed positions).
    """
    food_flag = 1.0 if context['food_near'] else 0.0
    pred_flag = 1.0 if context['pred_near'] else 0.0
    # Fixed agent position center, HP=70%
    inp5 = np.array([0.5, 0.5, 0.7, food_flag, pred_flag])

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S28_N)
    activity  = np.zeros(_S28_N)
    depletion_rate = genome['depletion_rate']

    out_recs     = []
    action_count = np.zeros(5)

    for _ in range(T):
        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)
        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)
        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)
        out_vec = eff[_S28_OUT_START:_S28_OUT_END].copy()
        out_recs.append(out_vec)
        action_count[int(np.argmax(out_vec))] += 1
        activity = eff.copy()

    return {
        'mean_output': np.mean(out_recs, axis=0),
        'action_dist': action_count / T,
    }


def _s28_cosine_dist(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))


def run_exp_c_context(pred_best, seed=_S28_SEED, T=_S28_CONTEXT_T):
    """4文脈での出力パターンを計測し、cosine距離行列を返す。

    Returns dict: context_results (list), cosine_matrix (4×4), mean_cosine_dist.
    """
    print(f'\n  [Exp C] 文脈依存的な行動の計測 (T={T} steps/文脈)')
    G_copy = pred_best['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 28200)

    results = []
    for ctx in _S28_CONTEXTS:
        r = _s28_measure_context(G_copy, W_copy, pred_best, ctx, rng, T=T)
        results.append(r)
        dominant_action = _S28_ACTION_NAMES[int(np.argmax(r['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}]  '
              f'output={r["mean_output"].round(3)}  '
              f'主行動={dominant_action}  '
              f'({r["action_dist"][int(np.argmax(r["action_dist"]))]:.0%})')

    # 4×4 cosine distance matrix
    n_ctx = len(_S28_CONTEXTS)
    cos_mat = np.zeros((n_ctx, n_ctx))
    for i in range(n_ctx):
        for j in range(n_ctx):
            cos_mat[i, j] = _s28_cosine_dist(
                results[i]['mean_output'], results[j]['mean_output'])

    # Upper-triangle pairs (i < j)
    pairs = [(i, j) for i in range(n_ctx) for j in range(i + 1, n_ctx)]
    mean_cd = float(np.mean([cos_mat[i, j] for i, j in pairs]))
    print(f'  → mean cosine dist = {mean_cd:.4f}  '
          f'(最大={max(cos_mat[i,j] for i,j in pairs):.4f}  '
          f'最小={min(cos_mat[i,j] for i,j in pairs):.4f})')

    return {
        'context_results':  results,
        'cosine_matrix':    cos_mat,
        'mean_cosine_dist': mean_cd,
    }


def plot_exp_c_context(exp_c,
                        fname='images/session_28/results_s28_context.png'):
    results  = exp_c['context_results']
    cos_mat  = exp_c['cosine_matrix']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Session 28 Exp C: 文脈依存的な行動の計測\n'
        f'4文脈 × 5出力ノード  mean cosine dist={exp_c["mean_cosine_dist"]:.4f}',
        fontsize=12,
    )

    # Panel 1: output activity heatmap
    ax = axes[0]
    out_matrix = np.array([r['mean_output'] for r in results])  # (4, 5)
    im = ax.imshow(out_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'node{i+5}\n({a})' for i, a in enumerate(_S28_ACTION_NAMES)],
                        fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=9)
    ax.set_title('出力ノード平均活動\n(文脈 × 出力ノード)')
    for i in range(n_ctx):
        for j in range(5):
            ax.text(j, i, f'{out_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if out_matrix[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: cosine distance matrix
    ax = axes[1]
    im2 = ax.imshow(cos_mat, cmap='Blues', vmin=0, vmax=max(cos_mat.max(), 0.01),
                    aspect='auto')
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=8)
    ax.set_title('Cosine距離行列\n(大=文脈間の出力パターンが異なる)')
    for i in range(n_ctx):
        for j in range(n_ctx):
            ax.text(j, i, f'{cos_mat[i, j]:.3f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if cos_mat[i, j] > cos_mat.max() * 0.6 else 'black')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 3: action distribution stacked bar
    ax = axes[2]
    action_mat = np.array([r['action_dist'] for r in results])  # (4, 5)
    colors_act = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    bottoms = np.zeros(n_ctx)
    xs = np.arange(n_ctx)
    for ai, (color, name) in enumerate(zip(colors_act, _S28_ACTION_NAMES)):
        ax.bar(xs, action_mat[:, ai], bottom=bottoms, color=color, alpha=0.85,
               label=name, width=0.6)
        bottoms += action_mat[:, ai]
    ax.set_xticks(xs)
    ax.set_xticklabels(ctx_lbls, fontsize=9)
    ax.set_ylabel('Action probability')
    ax.set_title('文脈別行動分布\n(文脈依存的な行動 = 文脈間で分布が異なる)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment D: predator speed sweep ────────────────────────────────────────

_S28_PRED_SPEEDS = [1, 2, 4, 8]


def run_exp_d_speed_sweep(seed=_S28_SEED, n_gen=_S28_N_GEN_D):
    """捕食者速度スイープ: 各速度で進化→文脈依存性を計測。

    Returns list of dicts: speed, best_genome, hist, mean_cosine_dist, best_steps.
    """
    print(f'\n  [Exp D] 捕食者速度スイープ '
          f'(speeds={_S28_PRED_SPEEDS}, {n_gen}世代/速度)')
    sweep_results = []

    for speed in _S28_PRED_SPEEDS:
        print(f'\n  速度 {speed} (毎{speed}ステップ移動):')
        best, hist = _s28_evolve(
            n_predators=1, predator_speed=speed, seed=seed, n_gen=n_gen)

        G_copy = best['G'].copy()
        W_copy = _s28_get_W(G_copy)
        rng_c  = np.random.default_rng(seed + 28300 + speed)
        ctx_results = []
        for ctx in _S28_CONTEXTS:
            r = _s28_measure_context(G_copy, W_copy, best, ctx, rng_c)
            ctx_results.append(r)

        n_ctx  = len(_S28_CONTEXTS)
        pairs  = [(i, j) for i in range(n_ctx) for j in range(i + 1, n_ctx)]
        vecs   = [r['mean_output'] for r in ctx_results]
        mcd    = float(np.mean([_s28_cosine_dist(vecs[i], vecs[j]) for i, j in pairs]))

        print(f'  → best_steps={hist["gen_best_steps"][-1]:.1f}  '
              f'depl={best["depletion_rate"]:.3f}  '
              f'mean_cosine_dist={mcd:.4f}')

        sweep_results.append({
            'speed':            speed,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'depl':             best['depletion_rate'],
            'pred_hits':        hist['gen_pred_hits'][-1],
        })

    return sweep_results


def plot_exp_d_speed_sweep(sweep_results,
                            fname='images/session_28/results_s28_predator_speed.png'):
    speeds  = [r['speed']            for r in sweep_results]
    mcds    = [r['mean_cosine_dist'] for r in sweep_results]
    steps   = [r['best_steps']       for r in sweep_results]
    depls   = [r['depl']             for r in sweep_results]
    hits    = [r['pred_hits']        for r in sweep_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Session 28 Exp D: 捕食者速度と文脈依存性の関係\n'
        f'speeds={_S28_PRED_SPEEDS}  ({_S28_N_GEN_D}世代/条件)',
        fontsize=12,
    )
    xs     = np.arange(len(speeds))
    xlbls  = [f'speed={s}\n(毎{s}step)' for s in speeds]

    # Panel 1: mean cosine distance (context dependence)
    ax = axes[0][0]
    bars = ax.bar(xs, mcds, color='steelblue', alpha=0.85, edgecolor='white')
    for xi, v in enumerate(mcds):
        ax.text(xi, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Mean pairwise cosine dist')
    ax.set_title('文脈依存性の強さ\n(高い = 文脈間で出力パターンが大きく異なる)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: survival steps
    ax = axes[0][1]
    ax.bar(xs, steps, color='forestgreen', alpha=0.85, edgecolor='white')
    for xi, v in enumerate(steps):
        ax.text(xi, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Best steps / ep')
    ax.set_title('生存ステップ数\n(捕食者が速いほど生存が難しい?)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: depletion rate
    ax = axes[1][0]
    ax.bar(xs, depls, color='purple', alpha=0.85, edgecolor='white')
    for xi, v in enumerate(depls):
        ax.text(xi, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('depletion_rate')
    ax.set_ylim(0, _S28_DEPL_HI + 0.05)
    ax.set_title('depletion_rate の収束値')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: predator hits
    ax = axes[1][1]
    ax.bar(xs, hits, color='tomato', alpha=0.85, edgecolor='white')
    for xi, v in enumerate(hits):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Predator hits / ep')
    ax.set_title('捕食者ヒット数\n(速いほどヒット多い?)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 28: 捕食者の導入による神経系の複雑化 ===')
    print(f'ノード構成: {_S28_N}ノード '
          f'(入力{_S28_INP_END}+出力{_S28_OUT_END-_S28_OUT_START}+'
          f'内部{_S28_INT_END-_S28_INT_START})')
    print(f'固定tau: tau_s={_S28_TAU_S}  tau_i={_S28_TAU_I}  tau_o={_S28_TAU_O} '
          f'(Session 27最良値)')
    print(f'進化対象: depletion_rate, edge_add_prob, activity_ratio')
    print()

    # Experiment A
    print('[Exp A] 捕食者あり vs なしの比較')
    no_pred_best, no_pred_hist, pred_best, pred_hist = run_exp_a_evolution(
        seed=_S28_SEED)
    plot_exp_a_evolution(no_pred_best, no_pred_hist, pred_best, pred_hist)

    # Experiment B
    print('\n[Exp B] 睡眠様状態の観察 (T=2000ステップ)')
    exp_b = run_exp_b_sleep_pattern(pred_best, seed=_S28_SEED)
    plot_exp_b_sleep_pattern(exp_b)

    # Experiment C
    print('\n[Exp C] 文脈依存的な行動の計測')
    exp_c = run_exp_c_context(pred_best, seed=_S28_SEED)
    plot_exp_c_context(exp_c)

    # Experiment D
    print('\n[Exp D] 捕食者速度スイープ')
    sweep = run_exp_d_speed_sweep(seed=_S28_SEED)
    plot_exp_d_speed_sweep(sweep)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 28 Summary ===')
    print()

    print('[A] 捕食者あり vs なし')
    depl_diff = pred_best['depletion_rate'] - no_pred_best['depletion_rate']
    print(f'  捕食者なし: steps={no_pred_hist["gen_best_steps"][-1]:.1f}  '
          f'depl={no_pred_best["depletion_rate"]:.4f}')
    print(f'  捕食者あり: steps={pred_hist["gen_best_steps"][-1]:.1f}  '
          f'depl={pred_best["depletion_rate"]:.4f}')
    print(f'  depletion_rate差: {depl_diff:+.4f}  '
          f'({"✓ 捕食圧がdepletion_rate増加" if depl_diff > 0 else "✗ 差なし/逆"})')

    print('\n[B] 睡眠様状態')
    if exp_b['chunks']:
        mid   = len(exp_b['chunks']) // 2
        s1    = float(np.mean([c['sensory_activity']  for c in exp_b['chunks'][:mid]]))
        s2    = float(np.mean([c['sensory_activity']  for c in exp_b['chunks'][mid:]]))
        delta = s1 - s2
        print(f'  感覚器活動: 前半={s1:.3f} → 後半={s2:.3f}  Δ={delta:+.3f}')
        print(f'  捕食者遭遇: {len(exp_b["enc_steps"])}回  '
              f'食料獲得: {exp_b["food_total"]}個')
        print(f'  判定 (Δ>0.05): {"✓ 睡眠様活動低下" if delta > 0.05 else "✗ 未検出"}')

    print('\n[C] 文脈依存性')
    print(f'  mean cosine dist = {exp_c["mean_cosine_dist"]:.4f}  '
          f'({"✓ 文脈分離あり" if exp_c["mean_cosine_dist"] > 0.05 else "✗ 分離弱い"})')
    for i, (ctx, res) in enumerate(zip(_S28_CONTEXTS, exp_c['context_results'])):
        dom = _S28_ACTION_NAMES[int(np.argmax(res['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}]: '
              f'主行動={dom} ({res["action_dist"].max():.0%})')

    print('\n[D] 捕食者速度スイープ')
    for r in sweep:
        print(f'  speed={r["speed"]}: '
              f'mcd={r["mean_cosine_dist"]:.4f}  '
              f'steps={r["best_steps"]:.1f}  '
              f'depl={r["depl"]:.3f}')
    best_speed = max(sweep, key=lambda r: r['mean_cosine_dist'])
    print(f'  → 最大文脈依存性: speed={best_speed["speed"]}  '
          f'mcd={best_speed["mean_cosine_dist"]:.4f}')

    print('\nDone.')
