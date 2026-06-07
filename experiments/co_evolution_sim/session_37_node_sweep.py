"""Session 37: ノード数（内部ノード数）スイープ

Session 36の診断:
  pred_distをどう変えても C0-C1差は改善しない。
  → ネットワーク容量（N=21）の限界の可能性。
  → 「食料フラグ」と「捕食者フラグ」を同時に統合して
    行動を切り替える計算能力が不足している可能性。

変更点:
  内部ノード数だけ増やす。入力(5)・出力(5)は固定。
  N=21(内部11) → N=25(内部15) → N=30(内部20) → N=40(内部30)

  入力・出力が固定なので、Session 36の環境設定がそのまま使える。
  差分は_s28_make_genome, _s28_make_tau_arr の N だけ。

固定条件:
  grid=8, hp_decay=5, food_dist=2, pred_dist=1
  pursuit_prob=0.9
  （Session 36でC0-C1差が最もマシだったpred_dist=1を使用）

実験:
  A: ノード数スイープ [21, 25, 30, 40]
     → C0-C1食事率差、mcd、進化vs固定の変化を確認
  B: ベスト条件で複数seed確認（42〜46）

出力:
  images/session_37/results_s37_node_sweep.png
  images/session_37/results_s37_multiseed.png
"""

import os
from dataclasses import dataclass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import _N_PROP, _K, _INIT_W, _LR
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_28_predator import (
    _S28_INP_START, _S28_INP_END,   # 0-4（固定）
    _S28_OUT_START, _S28_OUT_END,   # 5-9（固定）
    _S28_DENSITY, _S28_MUT_STD, _S28_EDGE_CHNG,
    _S28_TAU_S, _S28_TAU_I, _S28_TAU_O,
    _S28_DEPL_LO, _S28_DEPL_HI, _S28_DEPL_MUT_STD,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_MR, _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_ACTION_NAMES,
    _s28_get_W, _s28_propagate,
    _s28_hebb, _s28_mutate_genome,
)
from session_31_grid_sweep import (
    WorldConfig, _s31_init_foods, _s31_init_pred,
)
from session_33_eat_range import _S33_CFG
from session_34_pursuit_x_s33 import (
    _S34_HP_DECAY, _S34_PRED_SPEED,
    _s34_pred_step,
)
from session_36_pred_dist_sweep import (
    _s36_inp5, _run_ep_context_log, aggregate_context_actions,
)

# ── Session 37 定数 ────────────────────────────────────────────────────────────

_S37_SEED       = _S28_SEED
_S37_N_GEN      = _S28_N_GEN
_S37_PURSUIT    = 0.9
_S37_HP_DECAY   = _S34_HP_DECAY   # 5
_S37_N_SEEDS    = 5
_S37_SEEDS      = list(range(42, 42 + _S37_N_SEEDS))
_S37_T_LONG     = 2000
_S37_N_TRIALS   = 20
_S37_PRED_DIST  = 1   # Session 36でC0-C1差が最もマシだった値

_S37_NODE_SIZES = [21, 25, 30, 40]   # スイープ対象

# 入力・出力ノードは固定
_N_INP = 5   # nodes 0-4
_N_OUT = 5   # nodes 5-9
_N_FIXED = _N_INP + _N_OUT   # 10

# ベースのWorldConfig（pred_distだけ差し替え）
_S37_CFG = WorldConfig(
    grid      = _S33_CFG.grid,
    max_steps = _S33_CFG.max_steps,
    hp_start  = _S33_CFG.hp_start,
    n_foods   = _S33_CFG.n_foods,
    food_dist = _S33_CFG.food_dist,
    pred_dist = _S37_PRED_DIST,
)


# ── ノード数に依存するヘルパー ────────────────────────────────────────────────

def _make_tau_arr(n: int) -> np.ndarray:
    """N個のノードのtau_rec配列を作る。
    nodes 0-4: tau_s（感覚器）
    nodes 5-9: tau_o（出力）
    nodes 10+: tau_i（内部）
    """
    tau = np.ones(n, dtype=float)
    tau[0:5]  = float(_S28_TAU_S)
    tau[5:10] = float(_S28_TAU_O)
    tau[10:]  = float(_S28_TAU_I)
    return tau


def _make_genome(n: int, rng) -> dict:
    """N個のノードのゲノムを作る。"""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < _S28_DENSITY:
                G.add_edge(i, j, weight=float(_INIT_W))
    W = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        W[i, j] = d['weight']
    return {
        'G':              G,
        'W':              W,
        'n':              n,
        'depletion_rate': float(rng.uniform(_S28_DEPL_LO, _S28_DEPL_HI)),
        'edge_add_prob':  float(rng.uniform(0.0, 0.1)),
        'activity_ratio': float(rng.uniform(0.0, 0.6)),
        'metabolic_rate': _S28_MR,
    }


def _mutate_genome(genome: dict, rng) -> dict:
    """ゲノムを突然変異させる。nは引き継ぐ。"""
    n   = genome['n']
    G   = genome['G'].copy()
    # エッジ重みの変異
    for i, j in list(G.edges()):
        w = float(G[i][j]['weight']) + rng.normal(0, _S28_MUT_STD)
        G[i][j]['weight'] = float(np.clip(w, 0.01, 1.0))
    # エッジの追加・削除
    existing = set(G.edges())
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) in existing:
                if rng.random() < _S28_EDGE_CHNG:
                    G.remove_edge(i, j)
            else:
                if rng.random() < _S28_EDGE_CHNG * _S28_DENSITY:
                    G.add_edge(i, j, weight=float(_INIT_W))
    W = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        W[i, j] = d['weight']
    return {
        'G':              G,
        'W':              W,
        'n':              n,
        'depletion_rate': float(np.clip(
            genome['depletion_rate'] + rng.normal(0, _S28_DEPL_MUT_STD),
            _S28_DEPL_LO, _S28_DEPL_HI)),
        'edge_add_prob':  float(np.clip(
            genome['edge_add_prob']  + rng.normal(0, 0.01), 0.0, 0.1)),
        'activity_ratio': float(np.clip(
            genome['activity_ratio'] + rng.normal(0, 0.05), 0.0, 0.6)),
        'metabolic_rate': genome['metabolic_rate'],
    }


def _propagate(W: np.ndarray, activity: np.ndarray,
               inp5: np.ndarray) -> np.ndarray:
    """1ステップの伝播。入力ノード(0-4)にinp5を注入。"""
    new_act = np.tanh(W.T @ activity)
    new_act[:5] = inp5
    return new_act


def _hebb(G, W, eff, rng, edge_add_prob, activity_ratio):
    """Hebbianルール（session_28と同じロジック、nだけ変わる）。"""
    n = W.shape[0]
    lr         = float(_LR)
    decay      = 0.01
    act_thresh = float(_S28_ACT_THRESH)
    ar         = float(activity_ratio)
    ep         = float(edge_add_prob)

    for i, j, d in list(G.edges(data=True)):
        if eff[i] > act_thresh and eff[j] > act_thresh:
            d['weight'] = float(np.clip(d['weight'] + lr, 0.01, 1.0))
            W[i, j]     = d['weight']
        else:
            new_w = d['weight'] - decay
            if new_w < 0.01:
                G.remove_edge(i, j)
                W[i, j] = 0.0
            else:
                d['weight'] = new_w
                W[i, j]     = new_w

    # エッジ追加
    active_nodes = [i for i in range(n) if eff[i] > act_thresh]
    n_add = int(rng.binomial(max(1, len(active_nodes)), ep))
    for _ in range(n_add):
        if rng.random() < ar and len(active_nodes) >= 2:
            src = int(rng.choice(active_nodes))
            dst = int(rng.choice(active_nodes))
        else:
            src = int(rng.integers(0, n))
            dst = int(rng.integers(0, n))
        if src != dst and not G.has_edge(src, dst):
            G.add_edge(src, dst, weight=float(_INIT_W))
            W[src, dst] = float(_INIT_W)


# ── エピソードランナー ─────────────────────────────────────────────────────────

def _s37_run_ep(cfg: WorldConfig, G, W, genome, rng,
                pursuit_prob: float   = _S37_PURSUIT,
                hp_decay: float       = _S37_HP_DECAY,
                predator_speed: int   = _S34_PRED_SPEED,
                record_activity: bool = False):
    """Session 36と同じ、Nだけ可変。"""
    n              = genome['n']
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _make_tau_arr(n)
    resources = np.ones(n)
    activity  = np.zeros(n)

    center   = cfg.grid // 2
    row, col = center, center
    hp       = float(cfg.hp_start)

    food_positions = _s31_init_foods(cfg, rng, row, col)
    food_avail     = [True] * cfg.n_foods
    food_timer     = [0]   * cfg.n_foods
    pred_pos       = _s31_init_pred(cfg, rng, row, col)
    pred_resources = 1.0
    pred_dormant   = False

    steps = food = pred_hits = 0
    act_recs = [] if record_activity else None

    for step in range(cfg.max_steps):
        if hp <= 0:
            break

        if step % predator_speed == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                pursuit_prob, pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            pred_hits += 1

        inp5 = _s36_inp5(cfg, row, col, hp,
                         food_positions, food_avail, pred_pos)

        for _ in range(_N_PROP):
            activity = _propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, n), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())

        hp -= hp_decay
        hp -= metabolic_rate * float(np.sum(eff))

        # 行動は出力ノード(5-9)のargmax
        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(cfg.grid - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(cfg.grid - 1, col + 1)
        elif action == 4:
            for fi in range(cfg.n_foods):
                fr, fc = food_positions[fi]
                if (food_avail[fi]
                        and abs(row - fr) + abs(col - fc) <= cfg.food_dist):
                    hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                    resources = np.clip(
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources),
                        0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    food += 1
                    break

        steps = step + 1

        for fi in range(cfg.n_foods):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        if (step + 1) % _K == 0:
            _hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, _S28_T_CONSOL)

    return {'steps': steps, 'food': food,
            'pred_hits': pred_hits, 'act_recs': act_recs}


# ── エピソード中の文脈別行動ログ ──────────────────────────────────────────────

def _s37_run_ep_context_log(cfg, G, W, genome, rng,
                             pursuit_prob=_S37_PURSUIT,
                             hp_decay=_S37_HP_DECAY,
                             T=_S37_T_LONG):
    """文脈別行動を記録するエピソードランナー（N可変版）。"""
    n              = genome['n']
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _make_tau_arr(n)
    resources = np.ones(n)
    activity  = np.zeros(n)

    center   = cfg.grid // 2
    row, col = center, center
    hp       = float(cfg.hp_start)

    food_positions = _s31_init_foods(cfg, rng, row, col)
    food_avail     = [True] * cfg.n_foods
    food_timer     = [0]   * cfg.n_foods
    pred_pos       = _s31_init_pred(cfg, rng, row, col)
    pred_resources = 1.0
    pred_dormant   = False

    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    log     = []

    for step in range(T):
        if hp <= 0:
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s31_init_foods(cfg, rng, row, col)
            food_avail     = [True] * cfg.n_foods
            food_timer     = [0]   * cfg.n_foods
            pred_pos       = _s31_init_pred(cfg, rng, row, col)
            pred_resources = 1.0
            pred_dormant   = False
            resources      = np.ones(n)
            activity       = np.zeros(n)

        if step % _S34_PRED_SPEED == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                pursuit_prob, pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE

        inp5      = _s36_inp5(cfg, row, col, hp,
                              food_positions, food_avail, pred_pos)
        food_flag = int(inp5[3])
        pred_flag = int(inp5[4])
        ctx_idx   = ctx_map[(food_flag, pred_flag)]

        for _ in range(_N_PROP):
            activity = _propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, n), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, _make_tau_arr(n), depletion_rate)

        hp -= hp_decay
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        log.append((ctx_idx, action))

        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(cfg.grid - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(cfg.grid - 1, col + 1)
        elif action == 4:
            for fi in range(cfg.n_foods):
                fr, fc = food_positions[fi]
                if (food_avail[fi]
                        and abs(row - fr) + abs(col - fc) <= cfg.food_dist):
                    hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                    resources = np.clip(
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources),
                        0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    break

        for fi in range(cfg.n_foods):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        if (step + 1) % _K == 0:
            _hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    return log


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s37_evolve(cfg: WorldConfig, n: int,
                pursuit_prob: float = _S37_PURSUIT,
                hp_decay: float     = _S37_HP_DECAY,
                seed: int           = _S37_SEED,
                n_gen: int          = _S37_N_GEN):
    rng = np.random.default_rng(seed + 37000 + n * 10)
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s37_run_ep(
                    cfg, g['G'], g['W'], g, ep_rng,
                    pursuit_prob=pursuit_prob,
                    hp_decay=hp_decay,
                    record_activity=True)
                total += res['steps']
                ep_food.append(res['food'])
                ep_hits.append(res['pred_hits'])
                if res['act_recs']:
                    arr = np.array(res['act_recs'])
                    ep_active.append(
                        float(np.sum(
                            np.mean(arr, axis=0) > _S28_ACT_THRESH)))
            fitnesses.append(total / _S28_N_EP)
            g['_ep_food']   = float(np.mean(ep_food))
            g['_ep_hits']   = float(np.mean(ep_hits))
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]
        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_pred_hits'].append(bg['_ep_hits'])
        hist['gen_mean_active'].append(bg['_ep_active'])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_S28_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _S28_N_AGENTS:
            parent = survivors[int(rng.integers(0, _S28_N_SURV))]
            new_pop.append(_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'hits={bg["_ep_hits"]:.2f}/ep  '
                  f'active={bg["_ep_active"]:.1f}/{n}')

    for g in pop:
        for k in ('_ep_food', '_ep_hits', '_ep_active'):
            g.pop(k, None)

    return pop[0], hist


# ── 固定戦略の確認 ────────────────────────────────────────────────────────────

def _s37_check_fixed(cfg, pursuit_prob=_S37_PURSUIT,
                     hp_decay=_S37_HP_DECAY,
                     seed=_S37_SEED, n_trials=_S37_N_TRIALS):
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 37300 + trial * 10 + action_idx)
            center   = cfg.grid // 2
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s31_init_foods(cfg, rng, row, col)
            food_avail     = [True] * cfg.n_foods
            food_timer     = [0]   * cfg.n_foods
            pred_pos       = _s31_init_pred(cfg, rng, row, col)
            pred_resources = 1.0
            pred_dormant   = False

            for step in range(cfg.max_steps):
                if hp <= 0:
                    break
                if step % _S34_PRED_SPEED == 0:
                    pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                        cfg, pred_pos, [row, col],
                        pursuit_prob, pred_resources, pred_dormant, rng)
                if pred_pos[0] == row and pred_pos[1] == col:
                    hp -= _S28_PRED_DAMAGE
                hp -= hp_decay

                if action_idx == 0:   row = max(0, row - 1)
                elif action_idx == 1: row = min(cfg.grid - 1, row + 1)
                elif action_idx == 2: col = max(0, col - 1)
                elif action_idx == 3: col = min(cfg.grid - 1, col + 1)
                elif action_idx == 4:
                    for fi in range(cfg.n_foods):
                        fr, fc = food_positions[fi]
                        if (food_avail[fi]
                                and abs(row-fr)+abs(col-fc) <= cfg.food_dist):
                            hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                            food_avail[fi] = False
                            food_timer[fi] = 0
                            break

                for fi in range(cfg.n_foods):
                    if not food_avail[fi]:
                        food_timer[fi] += 1
                        if food_timer[fi] >= _S28_FOOD_RESPAWN:
                            food_avail[fi] = True
                            food_timer[fi] = 0

            total_steps.append(step + 1)
        results[action_name] = float(np.mean(total_steps))
    return results


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_node_sweep(sweep_results,
                    fname='images/session_37/results_s37_node_sweep.png'):
    ns      = [r['n']            for r in sweep_results]
    c0_eat  = [r['c0_eat_rate']  for r in sweep_results]
    c1_eat  = [r['c1_eat_rate']  for r in sweep_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eat, c1_eat)]
    steps   = [r['best_steps']   for r in sweep_results]
    foods   = [r['food']         for r in sweep_results]
    hits    = [r['pred_hits']    for r in sweep_results]
    ef_diff = [r['evo_vs_fixed'] for r in sweep_results]
    c1_ns   = [r['c1_steps']     for r in sweep_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 37: ノード数スイープ\n'
        f'grid={_S37_CFG.grid}x{_S37_CFG.grid}  '
        f'pred_dist={_S37_PRED_DIST}  '
        f'hp_decay={_S37_HP_DECAY}  pp={_S37_PURSUIT}  {_S37_N_GEN}世代',
        fontsize=12,
    )
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(ns)))
    xlbls  = [f'N={n}\n(内部{n-10})' for n in ns]

    # Panel 1: C0-C1食事率差【核心】
    ax = axes[0][0]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(ns)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.03, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 34b基準(+3%)')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1【文脈依存の核心】\n(正=捕食者がいると食事を控える)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (d, c0, c1) in enumerate(zip(diffs, c0_eat, c1_eat)):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.012,
                f'{d:+.0%}\nC0={c0:.0%}/C1={c1:.0%}',
                ha='center', fontsize=8)

    # Panel 2: 進化 - 固定戦略の差
    ax = axes[0][1]
    bar_c2 = ['seagreen' if d > 0 else 'tomato' for d in ef_diff]
    ax.bar(range(len(ns)), ef_diff, color=bar_c2, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Evolution - Fixed (steps)')
    ax.set_title('進化 - 固定戦略\n(正=文脈依存が有効)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(ef_diff):
        ax.text(i, d + 0.3 if d >= 0 else d - 1.5,
                f'{d:+.1f}', ha='center', fontsize=10)

    # Panel 3: C1経験ステップ数
    ax = axes[0][2]
    ax.bar(range(len(ns)), c1_ns,
           color=[colors[i] for i in range(len(ns))],
           alpha=0.85, edgecolor='white')
    ax.axhline(80, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(80)')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('C1経験ステップ数')
    ax.set_title('C1（食料近・捕食者近）の経験量\n(多いほど学習機会がある)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(c1_ns):
        ax.text(i, v + 3, f'{v}', ha='center', fontsize=9)

    # Panel 4: 生存ステップ
    ax = axes[1][0]
    ax.bar(range(len(ns)), steps,
           color=[colors[i] for i in range(len(ns))],
           alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Mean steps / ep')
    ax.set_title('生存ステップ数\n(Nが増えて改善するか)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(steps):
        ax.text(i, v + 0.3, f'{v:.0f}', ha='center', fontsize=9)

    # Panel 5: food/ep と hits/ep
    ax = axes[1][1]
    x = np.arange(len(ns))
    w = 0.35
    ax.bar(x - w/2, foods, width=w, color='seagreen', alpha=0.85,
           label='food/ep', edgecolor='white')
    ax.bar(x + w/2, hits,  width=w, color='tomato',   alpha=0.85,
           label='hits/ep', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Count / ep')
    ax.set_title('食料獲得 vs 捕食者ヒット')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: サマリー
    ax = axes[1][2]
    lines = ['ノード数別サマリー\n',
             f'{"N":>4}  {"int":>4}  {"C0eat":>6}  '
             f'{"C1eat":>6}  {"diff":>6}  {"steps":>7}  {"ef":>6}']
    for r in sweep_results:
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        lines.append(
            f'{r["n"]:>4}  {r["n"]-10:>4}  '
            f'{r["c0_eat_rate"]:>6.0%}  {r["c1_eat_rate"]:>6.0%}  '
            f'{d:>+6.0%}  {r["best_steps"]:>7.1f}  '
            f'{r["evo_vs_fixed"]:>+6.1f}')
    ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('数値サマリー')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_multiseed(multi_results, best_n,
                   fname='images/session_37/results_s37_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    c1_ns   = [r['c1_steps']    for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 37: 複数seed確認（N={best_n}）\n'
        f'pp={_S37_PURSUIT}  hp_decay={_S37_HP_DECAY}  T={_S37_T_LONG}',
        fontsize=13,
    )

    ax = axes[0]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, c0_eats, width=w, color='royalblue', alpha=0.85,
           label='C0(食料近・捕食者遠)', edgecolor='white')
    ax.bar(x + w/2, c1_eats, width=w, color='tomato',    alpha=0.85,
           label='C1(食料近・捕食者近)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('食事行動率')
    ax.set_title('C0 vs C1 の食事率')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (c0, c1) in enumerate(zip(c0_eats, c1_eats)):
        ax.text(i - w/2, c0 + 0.005, f'{c0:.0%}', ha='center', fontsize=8)
        ax.text(i + w/2, c1 + 0.005, f'{c1:.0%}', ha='center', fontsize=8)

    ax = axes[1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(seeds)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.03, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 34b基準(+3%)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title(f'食事率差 C0-C1（N={best_n}）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.01,
                f'{d:+.0%}', ha='center', fontsize=9)

    ax = axes[2]
    ax.bar(range(len(seeds)), c1_ns, color='darkorange', alpha=0.85,
           edgecolor='white')
    ax.axhline(80, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(80)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('Steps')
    ax.set_title('C1経験ステップ数')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(c1_ns):
        ax.text(i, v + 2, f'{v}', ha='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S37_CFG

    print('=== Session 37: ノード数スイープ ===')
    print(f'grid={cfg.grid}x{cfg.grid}  pred_dist={cfg.pred_dist}  '
          f'food_dist={cfg.food_dist}')
    print(f'hp_decay={_S37_HP_DECAY}  pp={_S37_PURSUIT}  '
          f'n_gen={_S37_N_GEN}')
    print(f'node_sizes={_S37_NODE_SIZES}')
    print()
    print('ノード構成:')
    for n in _S37_NODE_SIZES:
        print(f'  N={n}: 入力{_N_INP} + 出力{_N_OUT} + 内部{n-_N_FIXED}')
    print()

    # 固定戦略（Nに依存しないので一度だけ計測）
    print('[固定戦略確認]（Nに依存しない）')
    fixed = _s37_check_fixed(cfg)
    best_fixed = max(fixed.values())
    best_fname = max(fixed, key=fixed.get)
    for a, v in fixed.items():
        print(f'  固定「{a}」: {v:.1f}steps')
    print(f'  最良固定: {best_fname}={best_fixed:.0f}steps')
    print()

    sweep_results = []

    for n in _S37_NODE_SIZES:
        print(f'\n{"="*60}')
        print(f'N={n}  内部ノード={n-_N_FIXED}')
        print(f'{"="*60}')

        print('\n  [進化]')
        best, hist = _s37_evolve(cfg, n=n, seed=_S37_SEED)
        steps = hist['gen_best_steps'][-1]
        food  = hist['gen_food_count'][-1]
        hits  = hist['gen_pred_hits'][-1]
        print(f'  → steps={steps:.1f}  food={food:.2f}/ep  hits={hits:.2f}/ep  '
              f'vs固定={steps-best_fixed:+.1f}')

        print(f'\n  [エピソード中の文脈別行動 T={_S37_T_LONG}]')
        rng_ep = np.random.default_rng(_S37_SEED + 37200)
        G_ep   = best['G'].copy()
        _n_ep  = best['n']
        W_ep   = np.zeros((_n_ep, _n_ep))
        for _i, _j, _d in G_ep.edges(data=True):
            W_ep[_i, _j] = _d['weight']
        log_ep = _s37_run_ep_context_log(
            cfg, G_ep, W_ep, best, rng_ep, T=_S37_T_LONG)
        counts_ep, fracs_ep, totals_ep = aggregate_context_actions(log_ep)

        c0_eat = fracs_ep[0, 4]
        c1_eat = fracs_ep[1, 4]
        c1_n   = totals_ep[1]
        print(f'  C0食事率={c0_eat:.0%}  C1食事率={c1_eat:.0%}  '
              f'差={c0_eat-c1_eat:+.0%}  C1n={c1_n}')
        for c in range(4):
            if totals_ep[c] > 0:
                dom = int(np.argmax(fracs_ep[c]))
                print(f'  C{c}: {totals_ep[c]}steps  '
                      f'主行動={_S28_ACTION_NAMES[dom]}({fracs_ep[c,dom]:.0%})  '
                      f'食事={fracs_ep[c,4]:.0%}')

        sweep_results.append({
            'n':            n,
            'best_steps':   steps,
            'food':         food,
            'pred_hits':    hits,
            'c0_eat_rate':  c0_eat,
            'c1_eat_rate':  c1_eat,
            'c1_steps':     c1_n,
            'evo_vs_fixed': steps - best_fixed,
        })

    plot_node_sweep(sweep_results)

    # ベストN（C0-C1差が最大）で複数seed確認
    best_r = max(sweep_results,
                 key=lambda r: r['c0_eat_rate'] - r['c1_eat_rate'])
    best_n = best_r['n']
    print(f'\n[Exp B] ベストN={best_n}で複数seed確認')
    print(f'  C0-C1差={best_r["c0_eat_rate"]-best_r["c1_eat_rate"]:+.0%}')

    multi_results = []
    for seed in _S37_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, _ = _s37_evolve(cfg, n=best_n, seed=seed)
        rng_s = np.random.default_rng(seed + 37200)
        G_s   = best_s['G'].copy()
        _n_s  = best_s['n']
        W_s   = np.zeros((_n_s, _n_s))
        for _i, _j, _d in G_s.edges(data=True):
            W_s[_i, _j] = _d['weight']
        log_s = _s37_run_ep_context_log(
            cfg, G_s, W_s, best_s, rng_s, T=_S37_T_LONG)
        _, fracs_s, totals_s = aggregate_context_actions(log_s)
        c0 = fracs_s[0, 4]
        c1 = fracs_s[1, 4]
        n1 = totals_s[1]
        print(f'    C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={n1}')
        multi_results.append({
            'seed': seed, 'c0_eat_rate': c0,
            'c1_eat_rate': c1, 'c1_steps': n1,
        })

    plot_multiseed(multi_results, best_n)

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 37 Summary ===')
    print()
    print(f'固定戦略ベスト: {best_fname}={best_fixed:.0f}steps')
    print()
    print('ノード数別:')
    for r in sweep_results:
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        print(f'  N={r["n"]}(内部{r["n"]-10}): '
              f'C0eat={r["c0_eat_rate"]:.0%}  '
              f'C1eat={r["c1_eat_rate"]:.0%}  '
              f'diff={d:+.0%}  '
              f'evo_fixed={r["evo_vs_fixed"]:+.1f}  '
              f'steps={r["best_steps"]:.0f}')

    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    print()
    print(f'ベストN={best_n} 複数seed (n={len(_S37_SEEDS)}):')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S37_SEEDS)} seeds')
    print()
    print('--- 判断基準 ---')
    print('Nが増えるとC0-C1差が大きくなる → 容量が効いている')
    print('Nに関係なく差が出ない → 設計の問題（環境・計測方法）')
    print()
    print('Done.')
