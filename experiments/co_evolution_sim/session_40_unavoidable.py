"""Session 40: 回避不可能なトレードオフ

Session 39/39bの診断:
  「文脈依存が必要な状況（C1）をエージェント自身が回避する」
  → 予防的回避が文脈依存判断を不要にしていた

  これは適応的な振る舞いだが、
  文脈依存行動が進化するには
  「回避できないトレードオフ」が必要

設計:
  食料を1箇所に固定（グリッド中央付近）
  捕食者は食料セル上に固定スポーン + 疲労でランダムウォーク期間あり
  → 「食べに行く = 捕食者リスク」が常に成立
  → 捕食者が疲れて休眠している隙に食べに行く必要がある
  → 「捕食者が近いか遠いか」を読む必要が生まれる

  Session 29で観察した「捕食者疲労による休眠」と自然に繋がる:
    捕食者が活動中 → pred_flag=1 → 逃げる
    捕食者が休眠中 → pred_flag=0 → 食べに行く
    これがまさに文脈依存行動

具体的な変更点（Session 38からの差分）:
  1. food_positions を固定（グリッド中央の1箇所）
  2. 捕食者を食料セル上に固定スポーン
  3. 捕食者はpursuit_prob=0.9でエージェントを追跡
     → 疲弊すると休眠（ランダムウォーク）→ 回復したら再追跡
     → この休眠中がエージェントの「食べるチャンス」

  hp_decay=3（食べないとすぐ死ぬ圧力も必要）
  N=25, grid=8

判断基準:
  C1発生率が上がるか（食料と捕食者が常に近い）
  C0-C1差が改善するか（休眠中に食べる行動が学習されるか）

出力:
  images/session_40/results_s40_c1_rate.png
  images/session_40/results_s40_multiseed.png
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_28_predator import (
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_OUT_START, _S28_OUT_END,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_ACTION_NAMES,
    _s28_get_W,
)
from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import WorldConfig
from session_34_pursuit_x_s33 import (
    _S34_PRED_SPEED,
    _S34_PRED_DEPLETION, _S34_PRED_RECOVERY,
    _S34_PRED_DORMANT_LO, _S34_PRED_DORMANT_HI,
    _s34_pred_step,
)
from session_36_pred_dist_sweep import _s36_inp5
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate, _hebb,
    aggregate_context_actions,
)
from session_38_long_episode import (
    _S38_N, _run_context_log,
)

# ── Session 40 定数 ────────────────────────────────────────────────────────────

_S40_SEED      = _S28_SEED
_S40_N_GEN     = _S28_N_GEN
_S40_N         = _S38_N         # 25
_S40_PURSUIT   = 0.9
_S40_HP_DECAY  = 3              # 食べないとすぐ死ぬ圧力
_S40_T_LONG    = 2000
_S40_N_SEEDS   = 5
_S40_SEEDS     = list(range(42, 42 + _S40_N_SEEDS))
_S40_N_TRIALS  = 20

# grid=8, food_dist=2, pred_dist=1 は維持
_S40_GRID      = 8
_S40_HP_START  = 256
_S40_MAX_STEPS = 1280
_S40_N_FOODS   = 1              # 食料を1箇所に減らす
_S40_FOOD_DIST = 2
_S40_PRED_DIST = 1

_S40_CFG = WorldConfig(
    grid      = _S40_GRID,
    max_steps = _S40_MAX_STEPS,
    hp_start  = _S40_HP_START,
    n_foods   = _S40_N_FOODS,
    food_dist = _S40_FOOD_DIST,
    pred_dist = _S40_PRED_DIST,
)

# 食料の固定位置（グリッド中央からずらす）
_S40_FOOD_POS = [_S40_GRID // 4, _S40_GRID // 4]    # (2, 2) for grid=8


# ── 世界ヘルパー（食料固定版） ─────────────────────────────────────────────────

def _s40_init_foods():
    """食料を固定位置に配置。"""
    return [list(_S40_FOOD_POS)]


def _s40_init_pred_on_food():
    """捕食者を食料セル上にスポーン。"""
    return list(_S40_FOOD_POS)


def _s40_inp5(cfg: WorldConfig, row, col, hp,
              food_positions, food_avail, pred_pos):
    """_s36_inp5と同じ（food_dist, pred_distをcfgから取る）。"""
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

def _s40_run_ep(cfg: WorldConfig, G, W, genome, rng,
                hp_decay: float       = _S40_HP_DECAY,
                pursuit_prob: float   = _S40_PURSUIT,
                predator_speed: int   = _S34_PRED_SPEED,
                record_activity: bool = False):
    """食料固定 + 捕食者食料上スポーンのエピソード。

    Session 38からの差分:
      食料位置: ランダム → 固定（_S40_FOOD_POS）
      捕食者初期位置: ランダム → 食料セル上
      食料数: 2 → 1
    """
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

    # 食料固定、捕食者を食料セル上にスポーン
    food_positions = _s40_init_foods()
    food_avail     = [True]
    food_timer     = [0]
    pred_pos       = _s40_init_pred_on_food()
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

        inp5 = _s40_inp5(cfg, row, col, hp,
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


# ── 文脈別行動ログ ────────────────────────────────────────────────────────────

def _s40_run_context_log(cfg, G, W, genome, rng,
                          hp_decay=_S40_HP_DECAY,
                          pursuit_prob=_S40_PURSUIT,
                          T=_S40_T_LONG):
    """文脈別行動を記録。食料固定版。"""
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

    food_positions = _s40_init_foods()
    food_avail     = [True]
    food_timer     = [0]
    pred_pos       = _s40_init_pred_on_food()
    pred_resources = 1.0
    pred_dormant   = False

    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    log     = []
    act_log = []

    for step in range(T):
        if hp <= 0:
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s40_init_foods()
            food_avail     = [True]
            food_timer     = [0]
            pred_pos       = _s40_init_pred_on_food()
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

        inp5      = _s40_inp5(cfg, row, col, hp,
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

        act_log.append(int(np.sum(eff > _S28_ACT_THRESH)))

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

    return log, act_log


# ── C1発生率の事前計測 ────────────────────────────────────────────────────────

def measure_c1_rate(cfg: WorldConfig, seed: int, n_ep: int = 50):
    """固定食料環境でのC1発生率を計測（ランダム行動）。"""
    rng     = np.random.default_rng(seed + 40900)
    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    ctx_counts = defaultdict(int)
    total = 0

    for _ in range(n_ep):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        center   = cfg.grid // 2
        row, col = center, center
        hp       = float(cfg.hp_start)
        food_positions = _s40_init_foods()
        food_avail     = [True]
        food_timer     = [0]
        pred_pos       = _s40_init_pred_on_food()
        pred_resources = 1.0
        pred_dormant   = False

        for step in range(cfg.max_steps):
            if hp <= 0:
                break
            if step % _S34_PRED_SPEED == 0:
                pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                    cfg, pred_pos, [row, col],
                    _S40_PURSUIT, pred_resources, pred_dormant, ep_rng)
            if pred_pos[0] == row and pred_pos[1] == col:
                hp -= _S28_PRED_DAMAGE
            hp -= _S40_HP_DECAY

            inp5      = _s40_inp5(cfg, row, col, hp,
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

    rates = {c: ctx_counts[c] / total for c in range(4)} if total else {}
    return rates, total


# ── 固定戦略の確認 ────────────────────────────────────────────────────────────

def _check_fixed(cfg, hp_decay=_S40_HP_DECAY,
                 seed=_S40_SEED, n_trials=_S40_N_TRIALS):
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 40300 + trial * 10 + action_idx)
            center   = cfg.grid // 2
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s40_init_foods()
            food_avail     = [True]
            food_timer     = [0]
            pred_pos       = _s40_init_pred_on_food()
            pred_resources = 1.0
            pred_dormant   = False

            for step in range(cfg.max_steps):
                if hp <= 0:
                    break
                if step % _S34_PRED_SPEED == 0:
                    pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                        cfg, pred_pos, [row, col],
                        _S40_PURSUIT, pred_resources, pred_dormant, rng)
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
        print(f'    固定「{action_name}」: {results[action_name]:.1f}steps')
    return results


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s40_evolve(cfg: WorldConfig, n: int,
                hp_decay: float = _S40_HP_DECAY,
                seed: int       = _S40_SEED,
                n_gen: int      = _S40_N_GEN):
    rng = np.random.default_rng(seed + 40000)
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s40_run_ep(
                    cfg, g['G'], g['W'], g, ep_rng,
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


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_multiseed(multi_results,
                   fname='images/session_40/results_s40_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    c1_ns   = [r['c1_steps']    for r in multi_results]
    actives = [r['mean_active'] for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 40: 食料固定+捕食者常駐\n'
        f'N={_S40_N}  hp_decay={_S40_HP_DECAY}  pp={_S40_PURSUIT}  '
        f'food_pos={_S40_FOOD_POS}  T={_S40_T_LONG}',
        fontsize=12,
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
    ax.set_title('C0 vs C1 の食事率\n(C0>C1 = 捕食者がいると食事を控える)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (c0, c1) in enumerate(zip(c0_eats, c1_eats)):
        ax.text(i - w/2, c0 + 0.005, f'{c0:.0%}', ha='center', fontsize=9)
        ax.text(i + w/2, c1 + 0.005, f'{c1:.0%}', ha='center', fontsize=9)

    ax = axes[1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(seeds)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.02, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 38基準(+2%)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1\n回避不可能なトレードオフで改善するか？')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.012,
                f'{d:+.0%}', ha='center', fontsize=11, fontweight='bold')

    ax = axes[2]
    ax.bar(range(len(seeds)), c1_ns, color='darkorange', alpha=0.85,
           edgecolor='white')
    ax.axhline(80, color='gray', linestyle='--', linewidth=1.5,
               label='Session 38基準(80steps)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('Steps')
    ax.set_title('C1経験ステップ数\n(食料固定で大幅増加するはず)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(c1_ns):
        ax.text(i, v + 3, f'{v}', ha='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S40_CFG

    print('=== Session 40: 食料固定+捕食者常駐 ===')
    print(f'N={_S40_N}  grid={cfg.grid}x{cfg.grid}  '
          f'hp_decay={_S40_HP_DECAY}  pp={_S40_PURSUIT}')
    print(f'食料固定位置: {_S40_FOOD_POS}  食料数: {cfg.n_foods}')
    print(f'捕食者: food上スポーン + 追跡型(pp={_S40_PURSUIT})')
    print()

    # ── 事前確認: C1発生率 ─────────────────────────────────────────────────
    print('[事前確認] C1発生率（ランダム行動 50ep）')
    rates, n_steps = measure_c1_rate(cfg, seed=_S40_SEED)
    c1_rate = rates.get(1, 0.0)
    print(f'  C0={rates.get(0,0):.0%}  C1={c1_rate:.0%}  '
          f'C2={rates.get(2,0):.0%}  C3={rates.get(3,0):.0%}  '
          f'({n_steps}steps)')
    print(f'  Session 38比較: 通常C1=5% → 固定食料C1={c1_rate:.0%}')
    print()

    # ── 固定戦略確認 ───────────────────────────────────────────────────────
    print('[固定戦略確認]')
    fixed     = _check_fixed(cfg)
    best_fix  = max(fixed.values())
    best_fn   = max(fixed, key=fixed.get)
    print(f'  最良固定: {best_fn}={best_fix:.0f}steps')
    print()

    # ── 複数seed実験 ──────────────────────────────────────────────────────
    print(f'[実験] seeds={_S40_SEEDS}')
    multi_results = []

    for seed in _S40_SEEDS:
        print(f'\n{"="*50}')
        print(f'seed={seed}')

        best, hist = _s40_evolve(cfg, _S40_N, seed=seed)
        steps   = hist['gen_best_steps'][-1]
        food    = hist['gen_food_count'][-1]
        hits    = hist['gen_pred_hits'][-1]
        actv    = hist['gen_mean_active'][-1]
        ef_diff = steps - best_fix
        print(f'  → steps={steps:.1f}  food={food:.2f}/ep  '
              f'hits={hits:.2f}/ep  active={actv:.1f}  '
              f'vs固定={ef_diff:+.1f}')

        rng_ep = np.random.default_rng(seed + 40200)
        G_ep   = best['G'].copy()
        W_ep   = best['W'].copy()
        log_ep, act_ep = _s40_run_context_log(
            cfg, G_ep, W_ep, best, rng_ep, T=_S40_T_LONG)
        _, fracs_ep, totals_ep = aggregate_context_actions(log_ep)

        c0 = fracs_ep[0, 4]
        c1 = fracs_ep[1, 4]
        n1 = totals_ep[1]
        ma = float(np.mean(act_ep))
        print(f'  C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={n1}  active={ma:.1f}')
        for c in range(4):
            if totals_ep[c] > 0:
                dom = int(np.argmax(fracs_ep[c]))
                print(f'  C{c}: {totals_ep[c]}steps  '
                      f'主行動={_S28_ACTION_NAMES[dom]}'
                      f'({fracs_ep[c,dom]:.0%})  '
                      f'食事={fracs_ep[c,4]:.0%}')

        multi_results.append({
            'seed':        seed,
            'c0_eat_rate': c0,
            'c1_eat_rate': c1,
            'c1_steps':    n1,
            'best_steps':  steps,
            'evo_fixed':   ef_diff,
            'mean_active': ma,
        })

    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 40 Summary ===')
    print()
    print(f'C1発生率: {c1_rate:.0%}（Session 38比較: 通常=5%）')
    print(f'固定戦略ベスト: {best_fn}={best_fix:.0f}steps')
    print()
    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    print(f'複数seed (n={len(_S40_SEEDS)}):')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S40_SEEDS)} seeds')
    print(f'  Session 38基準: mean=+2%  C0>C1: 4/5')
    print()

    improved = np.mean(diffs) > 0.05 and n_pos >= 4
    same     = abs(np.mean(diffs)) <= 0.02
    print('--- 判断 ---')
    if improved:
        print('→ 仮説B支持: 回避不可能なトレードオフで文脈依存が出た')
        print('  十分な経験があればHebbianでも文脈依存学習は可能')
    elif same:
        print('→ 仮説A支持: 回避不可能にしても文脈依存が出ない')
        print('  アーキテクチャ的な限界')
    else:
        print(f'→ 中間的な結果 (mean={np.mean(diffs):+.0%})')
    print()
    print('Done.')
