"""Session 36: pred_dist スイープ

Session 35の診断:
  pred_dist=1（隣接1マスのみ）では捕食者を認識してから逃げる猶予がない。
  → pred_flagへの反応を学習する意味がない。
  → 現実の生物は視覚・嗅覚で遠方から捕食者を認識する。

変更点（Session 35からの差分は1箇所）:
  WorldConfig の pred_dist を変更するだけ。
  _s31_inp5 の pred_flag 計算がそれに連動する。

pred_distの意味（grid=8の場合）:
  pred_dist=1: 隣接1マス（4セル/64 = 6%）← 現状
  pred_dist=2: 距離≤2  （12セル/64 = 19%）
  pred_dist=3: 距離≤3  （24セル/64 = 38%）
  pred_dist=4: 距離≤4  （36セル/64 = 56%）

  pred_dist が大きい = 遠方から捕食者を認識できる
  → 「捕食者が来る前に逃げ始める」ことができる
  → pred_flagへの反応が有意義になる

固定条件:
  grid=8, hp_decay=5, food_dist=2
  pursuit_prob=0.9（Session 34 best）
  spawn_mode=near（Session 35で試した）→ anywhereに戻す
  （spawn_modeの効果より pred_dist の効果を見たい）

実験:
  A: pred_dist スイープ [1, 2, 3, 4]
     → C1経験頻度、C0-C1食事率差、mcdの変化
  B: ベスト条件で複数seed確認（42〜46）

出力:
  images/session_36/results_s36_pred_dist_sweep.png
  images/session_36/results_s36_multiseed.png
"""

import os
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_28_predator import (
    _S28_N, _S28_OUT_START, _S28_OUT_END,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_MR, _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_ACTION_NAMES,
    _s28_get_W, _s28_propagate,
    _s28_hebb, _s28_make_tau_arr, _s28_make_genome, _s28_mutate_genome,
)
from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import (
    WorldConfig,
    _s31_init_foods, _s31_init_pred,
)
from session_33_eat_range import _S33_CFG
from session_34_pursuit_x_s33 import (
    _S34_HP_DECAY, _S34_PRED_SPEED,
    _S34_PRED_DEPLETION, _S34_PRED_RECOVERY,
    _S34_PRED_DORMANT_LO, _S34_PRED_DORMANT_HI,
    _s34_pred_step,
)
from session_35_near_spawn import aggregate_context_actions

# ── Session 36 定数 ────────────────────────────────────────────────────────────

_S36_SEED        = _S28_SEED
_S36_N_GEN       = _S28_N_GEN
_S36_PURSUIT     = 0.9
_S36_HP_DECAY    = _S34_HP_DECAY   # 5
_S36_PRED_SPEED  = _S34_PRED_SPEED # 2
_S36_N_SEEDS     = 5
_S36_SEEDS       = list(range(42, 42 + _S36_N_SEEDS))
_S36_T_LONG      = 2000
_S36_N_TRIALS    = 20

# スイープ対象
_S36_PRED_DISTS  = [1, 2, 3, 4]

# ベースとなるWorldConfig（pred_distだけ差し替える）
_S36_CFG_BASE    = _S33_CFG   # grid=8, food_dist=2, pred_dist=1


def _make_cfg(pred_dist: int) -> WorldConfig:
    """pred_distだけ変えたWorldConfigを作る。"""
    base = _S36_CFG_BASE
    return WorldConfig(
        grid      = base.grid,
        max_steps = base.max_steps,
        hp_start  = base.hp_start,
        n_foods   = base.n_foods,
        food_dist = base.food_dist,
        pred_dist = pred_dist,      # ここだけ変える
    )


# ── inp5（pred_distをcfgから取る） ────────────────────────────────────────────

def _s36_inp5(cfg: WorldConfig, row, col, hp,
              food_positions, food_avail, pred_pos):
    """_s31_inp5 と同じだが pred_dist を cfg から取る（自明だが明示）。"""
    food_flag = 0.0
    for (fr, fc), av in zip(food_positions, food_avail):
        if av and abs(row - fr) + abs(col - fc) <= cfg.food_dist:
            food_flag = 1.0
            break
    pr, pc    = pred_pos
    pred_flag = 1.0 if abs(row - pr) + abs(col - pc) <= cfg.pred_dist else 0.0
    return np.array([
        col / (cfg.grid - 1),
        row / (cfg.grid - 1),
        np.clip(hp / cfg.hp_start, 0.0, 1.5),
        food_flag,
        pred_flag,
    ])


# ── エピソードランナー ─────────────────────────────────────────────────────────

def _s36_run_ep(cfg: WorldConfig, G, W, genome, rng,
                pursuit_prob: float   = _S36_PURSUIT,
                hp_decay: float       = _S36_HP_DECAY,
                predator_speed: int   = _S36_PRED_SPEED,
                record_activity: bool = False):
    """Session 34と同じ、inp5だけ_s36_inp5に差し替え。"""
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S28_N)
    activity  = np.zeros(_S28_N)

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
            activity = _s28_propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())

        hp -= hp_decay
        hp -= metabolic_rate * float(np.sum(eff))

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
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, _S28_T_CONSOL)

    return {'steps': steps, 'food': food,
            'pred_hits': pred_hits, 'act_recs': act_recs}


# ── エピソード中の文脈別行動ログ ──────────────────────────────────────────────

def _run_ep_context_log(cfg: WorldConfig, G, W, genome, rng,
                        pursuit_prob: float = _S36_PURSUIT,
                        hp_decay: float     = _S36_HP_DECAY,
                        T: int              = _S36_T_LONG):
    """エピソードを走らせて（文脈, 行動）を記録。死亡時リセット。"""
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S28_N)
    activity  = np.zeros(_S28_N)

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
            resources      = np.ones(_S28_N)
            activity       = np.zeros(_S28_N)

        if step % _S36_PRED_SPEED == 0:
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
            activity = _s28_propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, _s28_make_tau_arr(), depletion_rate)

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
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    return log


# ── C1発生率の事前計測 ────────────────────────────────────────────────────────

def measure_c1_rate(cfg: WorldConfig, pursuit_prob: float,
                    seed: int, n_ep: int = 20):
    """ランダム行動でC1（食料近・捕食者近）の経験頻度を計測。"""
    rng = np.random.default_rng(seed + 36900)
    ctx_counts = defaultdict(int)
    total = 0
    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}

    for _ in range(n_ep):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        center   = cfg.grid // 2
        row, col = center, center
        hp       = float(cfg.hp_start)
        food_positions = _s31_init_foods(cfg, ep_rng, row, col)
        food_avail     = [True] * cfg.n_foods
        food_timer     = [0]   * cfg.n_foods
        pred_pos       = _s31_init_pred(cfg, ep_rng, row, col)
        pred_resources = 1.0
        pred_dormant   = False

        for step in range(cfg.max_steps):
            if hp <= 0:
                break
            if step % _S36_PRED_SPEED == 0:
                pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                    cfg, pred_pos, [row, col],
                    pursuit_prob, pred_resources, pred_dormant, ep_rng)
            if pred_pos[0] == row and pred_pos[1] == col:
                hp -= _S28_PRED_DAMAGE
            hp -= _S36_HP_DECAY

            inp5      = _s36_inp5(cfg, row, col, hp,
                                  food_positions, food_avail, pred_pos)
            food_flag = int(inp5[3])
            pred_flag = int(inp5[4])
            ctx_counts[ctx_map[(food_flag, pred_flag)]] += 1
            total += 1

            action = int(ep_rng.integers(0, 5))
            if action == 0:   row = max(0, row - 1)
            elif action == 1: row = min(cfg.grid - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(cfg.grid - 1, col + 1)
            for fi in range(cfg.n_foods):
                if not food_avail[fi]:
                    food_timer[fi] += 1
                    if food_timer[fi] >= _S28_FOOD_RESPAWN:
                        food_avail[fi] = True
                        food_timer[fi] = 0

    rates = {c: ctx_counts[c] / total for c in range(4)} if total > 0 else {}
    return rates, total


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s36_evolve(cfg: WorldConfig,
                pursuit_prob: float = _S36_PURSUIT,
                hp_decay: float     = _S36_HP_DECAY,
                seed: int           = _S36_SEED,
                n_gen: int          = _S36_N_GEN):
    rng = np.random.default_rng(seed + 36000 + cfg.pred_dist * 100)
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s36_run_ep(
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
                        float(np.sum(np.mean(arr, axis=0) > _S28_ACT_THRESH)))
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
            new_pop.append(_s28_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'hits={bg["_ep_hits"]:.2f}/ep  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        for k in ('_ep_food', '_ep_hits', '_ep_active'):
            g.pop(k, None)

    return pop[0], hist


# ── 固定戦略の確認 ────────────────────────────────────────────────────────────

def _s36_check_simple_strategy(cfg: WorldConfig,
                                pursuit_prob: float = _S36_PURSUIT,
                                hp_decay: float     = _S36_HP_DECAY,
                                seed: int           = _S36_SEED,
                                n_trials: int       = _S36_N_TRIALS):
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 36300 + trial * 10 + action_idx)
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
                if step % _S36_PRED_SPEED == 0:
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
                                and abs(row - fr) + abs(col - fc) <= cfg.food_dist):
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

def plot_pred_dist_sweep(sweep_results, c1_rates_pre,
                         fname='images/session_36/results_s36_pred_dist_sweep.png'):
    dists   = [r['pred_dist']     for r in sweep_results]
    c1_pre  = [c1_rates_pre[d]    for d in dists]
    c1_ep   = [r['c1_rate_ep']    for r in sweep_results]
    c0_eat  = [r['c0_eat_rate']   for r in sweep_results]
    c1_eat  = [r['c1_eat_rate']   for r in sweep_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eat, c1_eat)]
    steps   = [r['best_steps']    for r in sweep_results]
    foods   = [r['food']          for r in sweep_results]
    hits    = [r['pred_hits']     for r in sweep_results]
    bf_diff = [r['evo_vs_fixed']  for r in sweep_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 36: pred_dist スイープ\n'
        f'grid={_S36_CFG_BASE.grid}x{_S36_CFG_BASE.grid}  '
        f'food_dist={_S36_CFG_BASE.food_dist}  '
        f'pp={_S36_PURSUIT}  hp_decay={_S36_HP_DECAY}  {_S36_N_GEN}世代',
        fontsize=12,
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dists)))
    xlbls  = [f'pred_dist={d}\n(≤{d}マス)' for d in dists]

    # Panel 1: C1発生率（事前 vs エピソード中）
    ax = axes[0][0]
    x = np.arange(len(dists))
    w = 0.35
    ax.bar(x - w/2, c1_pre, width=w, color='tomato',    alpha=0.85,
           label='事前(ランダム行動)', edgecolor='white')
    ax.bar(x + w/2, c1_ep,  width=w, color='steelblue', alpha=0.85,
           label='エピソード中(進化個体)', edgecolor='white')
    ax.axhline(0.04, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(4%)')
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel('C1発生率')
    ax.set_title('C1（食料近・捕食者近）の経験頻度\npred_distが大きいほど増えるはず')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (pre, ep) in enumerate(zip(c1_pre, c1_ep)):
        ax.text(i - w/2, pre + 0.005, f'{pre:.0%}', ha='center', fontsize=8)
        ax.text(i + w/2, ep  + 0.005, f'{ep:.0%}',  ha='center', fontsize=8)

    # Panel 2: C0-C1食事率差【核心】
    ax = axes[0][1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(dists)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.03, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 34b基準(+3%)')
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1【文脈依存の核心】\n(正=捕食者がいると食事を控える)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (d, c0, c1) in enumerate(zip(diffs, c0_eat, c1_eat)):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.012,
                f'{d:+.0%}\nC0={c0:.0%}/C1={c1:.0%}',
                ha='center', fontsize=7.5)

    # Panel 3: 進化 - 固定戦略の差
    ax = axes[0][2]
    bar_colors2 = ['seagreen' if d > 0 else 'tomato' for d in bf_diff]
    ax.bar(range(len(dists)), bf_diff, color=bar_colors2, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel('Evolution - Fixed (steps)')
    ax.set_title('進化 - 固定戦略\n(正=文脈依存が有効に機能)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(bf_diff):
        ax.text(i, d + 0.3 if d >= 0 else d - 1.5,
                f'{d:+.1f}', ha='center', fontsize=9)

    # Panel 4: 生存ステップと食料
    ax = axes[1][0]
    x = np.arange(len(dists))
    w = 0.35
    ax.bar(x - w/2, steps, width=w, color='steelblue', alpha=0.85,
           label='生存steps', edgecolor='white')
    ax2 = ax.twinx()
    ax2.bar(x + w/2, foods, width=w, color='seagreen', alpha=0.85,
            label='food/ep', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel('Steps', color='steelblue')
    ax2.set_ylabel('Food/ep', color='seagreen')
    ax.set_title('生存ステップ vs 食料獲得')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: pred_hits/ep
    ax = axes[1][1]
    ax.bar(range(len(dists)), hits,
           color=[colors[i] for i in range(len(dists))],
           alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数\n(pred_distが大きいと早めに逃げてhitsが減るはず)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(hits):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    # Panel 6: サマリー
    ax = axes[1][2]
    lines = ['pred_dist別サマリー\n',
             f'{"dist":>5}  {"C1%":>6}  {"C0eat":>6}  '
             f'{"C1eat":>6}  {"diff":>6}  {"steps":>7}  {"hits":>6}']
    for r, pre in zip(sweep_results, c1_pre):
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        lines.append(
            f'{r["pred_dist"]:>5}  {pre:>6.0%}  '
            f'{r["c0_eat_rate"]:>6.0%}  {r["c1_eat_rate"]:>6.0%}  '
            f'{d:>+6.0%}  {r["best_steps"]:>7.1f}  {r["pred_hits"]:>6.2f}')
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


def plot_multiseed(multi_results,
                   fname='images/session_36/results_s36_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    c1_ns   = [r['c1_steps']    for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 36: 複数seed確認（ベストpred_dist）\n'
        f'pp={_S36_PURSUIT}  hp_decay={_S36_HP_DECAY}  T={_S36_T_LONG}',
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
    ax.set_title('C0 vs C1 の食事率\n(C0>C1 なら文脈依存あり)')
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
    ax.set_title('食事率の差 C0-C1\n(正かつ+3%超えで前進)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.01,
                f'{d:+.0%}', ha='center', fontsize=9)

    ax = axes[2]
    ax.bar(range(len(seeds)), c1_ns, color='darkorange', alpha=0.85,
           edgecolor='white')
    ax.axhline(80, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(80steps)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('Steps')
    ax.set_title('C1経験ステップ数\n(多いほど学習機会がある)')
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
    print('=== Session 36: pred_dist スイープ ===')
    print(f'grid={_S36_CFG_BASE.grid}x{_S36_CFG_BASE.grid}  '
          f'food_dist={_S36_CFG_BASE.food_dist}  '
          f'hp_decay={_S36_HP_DECAY}  pp={_S36_PURSUIT}')
    print(f'pred_dists={_S36_PRED_DISTS}')
    print()

    # pred_distの意味を表示
    g = _S36_CFG_BASE.grid
    total = g * g
    print('pred_distの認識範囲:')
    for d in _S36_PRED_DISTS:
        # Manhattan距離≤dのセル数（自分を除く）
        n_cells = sum(1 for r in range(g) for c in range(g)
                      if 0 < abs(r - g//2) + abs(c - g//2) <= d)
        print(f'  pred_dist={d}: 認識範囲≈{n_cells}セル/{total}セル '
              f'({n_cells/total:.0%})')
    print()

    # ── 事前確認: C1発生率 ─────────────────────────────────────────────────
    print('[事前確認] pred_dist別のC1発生率（ランダム行動）')
    c1_rates_pre = {}
    for d in _S36_PRED_DISTS:
        cfg   = _make_cfg(d)
        rates, n = measure_c1_rate(cfg, _S36_PURSUIT, seed=_S36_SEED)
        c1_rates_pre[d] = rates.get(1, 0.0)
        print(f'  pred_dist={d}: '
              f'C0={rates.get(0,0):.0%}  C1={rates.get(1,0):.0%}  '
              f'C2={rates.get(2,0):.0%}  C3={rates.get(3,0):.0%}  '
              f'({n}steps)')
    print()

    # ── Exp A: スイープ ────────────────────────────────────────────────────
    sweep_results = []
    for d in _S36_PRED_DISTS:
        cfg = _make_cfg(d)
        print(f'\n{"="*60}')
        print(f'pred_dist={d}  C1率={c1_rates_pre[d]:.0%}')
        print(f'{"="*60}')

        print('\n  [進化]')
        best, hist = _s36_evolve(cfg, seed=_S36_SEED)
        steps = hist['gen_best_steps'][-1]
        food  = hist['gen_food_count'][-1]
        hits  = hist['gen_pred_hits'][-1]
        print(f'  → steps={steps:.1f}  food={food:.2f}/ep  hits={hits:.2f}/ep')

        print(f'\n  [固定戦略確認]')
        strategy   = _s36_check_simple_strategy(cfg, seed=_S36_SEED)
        best_fixed = max(strategy.values())
        best_fname = max(strategy, key=strategy.get)
        diff_fixed = steps - best_fixed
        print(f'  最良固定: {best_fname}={best_fixed:.0f}steps  差={diff_fixed:+.1f}')

        print(f'\n  [エピソード中の文脈別行動 T={_S36_T_LONG}]')
        rng_ep = np.random.default_rng(_S36_SEED + 36200)
        G_ep   = best['G'].copy()
        W_ep   = _s28_get_W(G_ep)
        log_ep = _run_ep_context_log(
            cfg, G_ep, W_ep, best, rng_ep, T=_S36_T_LONG)
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
            'pred_dist':    d,
            'best_steps':   steps,
            'food':         food,
            'pred_hits':    hits,
            'c0_eat_rate':  c0_eat,
            'c1_eat_rate':  c1_eat,
            'c1_rate_ep':   c1_n / len(log_ep) if log_ep else 0.0,
            'c1_steps':     c1_n,
            'evo_vs_fixed': diff_fixed,
            'strategy':     strategy,
        })

    plot_pred_dist_sweep(sweep_results, c1_rates_pre)

    # ── Exp B: ベストpred_distで複数seed ──────────────────────────────────
    # C0-C1差が最大のpred_distを選ぶ
    best_sweep = max(sweep_results,
                     key=lambda r: r['c0_eat_rate'] - r['c1_eat_rate'])
    best_d = best_sweep['pred_dist']
    best_cfg = _make_cfg(best_d)
    print(f'\n[Exp B] ベストpred_dist={best_d}で複数seed確認')
    print(f'  C0-C1差={best_sweep["c0_eat_rate"]-best_sweep["c1_eat_rate"]:+.0%}')

    multi_results = []
    for seed in _S36_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, hist_s = _s36_evolve(best_cfg, seed=seed)
        rng_s = np.random.default_rng(seed + 36200)
        G_s   = best_s['G'].copy()
        W_s   = _s28_get_W(G_s)
        log_s = _run_ep_context_log(
            best_cfg, G_s, W_s, best_s, rng_s, T=_S36_T_LONG)
        counts_s, fracs_s, totals_s = aggregate_context_actions(log_s)
        c0 = fracs_s[0, 4]
        c1 = fracs_s[1, 4]
        n1 = totals_s[1]
        print(f'    C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={n1}')
        multi_results.append({
            'seed': seed, 'c0_eat_rate': c0,
            'c1_eat_rate': c1, 'c1_steps': n1,
        })

    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 36 Summary ===')
    print()
    print('pred_dist別:')
    for r in sweep_results:
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        print(f'  pred_dist={r["pred_dist"]}: '
              f'C1%={c1_rates_pre[r["pred_dist"]]:.0%}  '
              f'C0eat={r["c0_eat_rate"]:.0%}  '
              f'C1eat={r["c1_eat_rate"]:.0%}  '
              f'diff={d:+.0%}  '
              f'evo_fixed={r["evo_vs_fixed"]:+.1f}  '
              f'steps={r["best_steps"]:.0f}')

    print()
    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    print(f'ベストpred_dist={best_d} 複数seed (n={len(_S36_SEEDS)}):')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S36_SEEDS)} seeds')
    print()
    print('--- 判断基準 ---')
    print('pred_distが大きくなるとC1率・C0-C1差が増加 → 視覚範囲が効いている')
    print('変化なし → ネットワーク容量の限界の可能性')
    print()
    print('Done.')
