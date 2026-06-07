"""Session 35: 捕食者を食料の近くにスポーン

Session 34bの診断:
  C1（食料近・捕食者近）の経験 = 4% (80steps/2000)
  → トレードオフを経験する機会が少なすぎる
  → 「捕食者がいると食事を控える」の差が+3%と小さい

解決:
  捕食者の初期スポーン位置を「食料の近く」に変更する。
  追跡ロジック（pp=0.9）はSession 34と同じ。
  スポーン位置だけ変える。

  _s35_init_pred_near_food():
    食料のいずれかから距離≤food_dist の範囲内にスポーン
    ただしエージェントの初期位置は避ける

  これにより:
    エピソード開始直後からC1（食料近・捕食者近）状態が発生
    → トレードオフの経験頻度が上がる
    → 「捕食者がいると食事を控える」が学習される機会が増える

  生態学的対応:
    捕食者が食料場所の近くにいる（待ち伏せ型）
    + エージェントを追跡する（能動型）
    の組み合わせ

Session 34からの変更点:
  _s34_run_ep の pred_pos 初期化部分だけ変更
  _s35_init_pred_near_food() を新規追加

固定条件:
  pursuit_prob=0.9（Session 34 best）
  hp_decay=5, grid=8x8, food_dist=2

実験:
  A: スポーン範囲スイープ [food_dist, food_dist*2, anywhere]
     → C1経験頻度とmcdの変化を確認
  B: ベスト条件でエピソード中の文脈別行動を計測（Session 34b と同じ手法）
  C: 複数seed（42〜46）で再現確認

出力:
  images/session_35/results_s35_spawn_sweep.png
  images/session_35/results_s35_context.png
  images/session_35/results_s35_multiseed.png
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_28_predator import (
    _S28_N, _S28_OUT_START, _S28_OUT_END,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_MR, _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_ACTION_NAMES,
    _s28_get_W, _s28_propagate,
    _s28_hebb, _s28_make_tau_arr, _s28_make_genome, _s28_mutate_genome,
    _s28_cosine_dist, _s28_measure_context,
)
from session_31_grid_sweep import (
    WorldConfig,
    _s31_init_foods, _s31_init_pred, _s31_inp5,
)
from session_33_eat_range import _S33_CFG
from session_34_pursuit_x_s33 import (
    _S34_HP_DECAY, _S34_PRED_SPEED,
    _S34_PRED_DEPLETION, _S34_PRED_RECOVERY,
    _S34_PRED_DORMANT_LO, _S34_PRED_DORMANT_HI,
    _s34_pred_step,
)
from session_34b_in_episode import (
    _run_ep_with_context_log,
    aggregate_context_actions,
    _CTX_LABELS,
)

# ── Session 35 定数 ────────────────────────────────────────────────────────────

_S35_SEED         = _S28_SEED
_S35_N_GEN        = _S28_N_GEN
_S35_PURSUIT      = 0.9          # Session 34 best
_S35_HP_DECAY     = _S34_HP_DECAY  # 5
_S35_CFG          = _S33_CFG       # grid=8, food_dist=2
_S35_N_CONTEXT    = 25
_S35_CONTEXT_T    = 100
_S35_N_TRIALS     = 20
_S35_T_LONG       = 2000
_S35_N_SEEDS      = 5
_S35_SEEDS        = list(range(42, 42 + _S35_N_SEEDS))

# スポーン範囲スイープ
# 'near':    食料から距離≤food_dist にスポーン（新設計）
# 'medium':  食料から距離≤food_dist*2 にスポーン
# 'anywhere': 従来通りランダム（Session 34と同じ）
_S35_SPAWN_MODES  = ['near', 'medium', 'anywhere']


# ── 捕食者の初期配置（食料の近く） ───────────────────────────────────────────

def _s35_init_pred_near_food(cfg: WorldConfig, rng,
                              food_positions,
                              agent_row: int, agent_col: int,
                              spawn_dist: int):
    """食料から距離≤spawn_dist の範囲内に捕食者をスポーン。

    食料が複数ある場合はランダムに1つを選んでその近くにスポーン。
    エージェントの初期位置は避ける。

    spawn_dist=0 の場合は食料と同じセルにスポーン（最強）。

    Returns: [row, col]
    """
    # 食料のいずれかから距離≤spawn_dist のセルを候補にする
    candidates = []
    for fr, fc in food_positions:
        for r in range(cfg.grid):
            for c in range(cfg.grid):
                dist_food  = abs(r - fr) + abs(c - fc)
                dist_agent = abs(r - agent_row) + abs(c - agent_col)
                if dist_food <= spawn_dist and dist_agent >= 1:
                    candidates.append((r, c))

    # 重複除去
    candidates = list(set(candidates))

    if len(candidates) == 0:
        # 候補がない場合（spawn_distが小さすぎ）はランダム配置にフォールバック
        return _s31_init_pred(cfg, rng, agent_row, agent_col, min_dist=1)

    idx = int(rng.integers(0, len(candidates)))
    return list(candidates[idx])


def _spawn_dist_for_mode(cfg: WorldConfig, mode: str) -> int:
    """スポーンモードからspawn_distを計算。"""
    if mode == 'near':
        return cfg.food_dist          # ≤food_dist（=2）
    elif mode == 'medium':
        return cfg.food_dist * 2      # ≤food_dist*2（=4）
    else:  # 'anywhere'
        return cfg.grid * 2           # 実質無制限（ランダム配置と同じ）


# ── エピソードランナー（スポーン位置変更版） ──────────────────────────────────

def _s35_run_ep(cfg: WorldConfig, G, W, genome, rng,
                spawn_mode: str        = 'near',
                pursuit_prob: float    = _S35_PURSUIT,
                hp_decay: float        = _S35_HP_DECAY,
                predator_speed: int    = _S34_PRED_SPEED,
                record_activity: bool  = False):
    """Session 34との差分: pred_posの初期化だけ変更。"""
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

    # ── ここだけSession 34と違う ──────────────────────────────────────────────
    spawn_dist = _spawn_dist_for_mode(cfg, spawn_mode)
    if spawn_mode == 'anywhere':
        pred_pos = _s31_init_pred(cfg, rng, row, col)
    else:
        pred_pos = _s35_init_pred_near_food(
            cfg, rng, food_positions, row, col, spawn_dist)
    # ─────────────────────────────────────────────────────────────────────────

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

        inp5 = _s31_inp5(cfg, row, col, hp, food_positions, food_avail, pred_pos)

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


# ── C1経験頻度の事前計測 ─────────────────────────────────────────────────────

def measure_c1_rate(cfg: WorldConfig, spawn_mode: str,
                    pursuit_prob: float, seed: int, n_ep: int = 20):
    """ランダム行動エージェントでC1（食料近・捕食者近）の経験頻度を計測。"""
    rng = np.random.default_rng(seed + 35900)
    ctx_counts = defaultdict(int)
    total = 0

    for _ in range(n_ep):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        center   = cfg.grid // 2
        row, col = center, center
        hp       = float(cfg.hp_start)
        food_positions = _s31_init_foods(cfg, ep_rng, row, col)
        food_avail     = [True] * cfg.n_foods
        food_timer     = [0]   * cfg.n_foods
        spawn_dist = _spawn_dist_for_mode(cfg, spawn_mode)
        if spawn_mode == 'anywhere':
            pred_pos = _s31_init_pred(cfg, ep_rng, row, col)
        else:
            pred_pos = _s35_init_pred_near_food(
                cfg, ep_rng, food_positions, row, col, spawn_dist)
        pred_resources = 1.0
        pred_dormant   = False

        for step in range(cfg.max_steps):
            if hp <= 0:
                break
            if step % _S34_PRED_SPEED == 0:
                pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                    cfg, pred_pos, [row, col],
                    pursuit_prob, pred_resources, pred_dormant, ep_rng)
            if pred_pos[0] == row and pred_pos[1] == col:
                hp -= _S28_PRED_DAMAGE
            hp -= _S35_HP_DECAY

            inp5 = _s31_inp5(cfg, row, col, hp,
                             food_positions, food_avail, pred_pos)
            food_flag = int(inp5[3])
            pred_flag = int(inp5[4])
            ctx_map   = {(1,0):0, (1,1):1, (0,1):2, (0,0):3}
            ctx_idx   = ctx_map[(food_flag, pred_flag)]
            ctx_counts[ctx_idx] += 1
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

def _s35_evolve(cfg: WorldConfig, spawn_mode: str,
                pursuit_prob: float = _S35_PURSUIT,
                hp_decay: float     = _S35_HP_DECAY,
                seed: int           = _S35_SEED,
                n_gen: int          = _S35_N_GEN):
    rng = np.random.default_rng(
        seed + 35000 + _S35_SPAWN_MODES.index(spawn_mode) * 100)
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s35_run_ep(
                    cfg, g['G'], g['W'], g, ep_rng,
                    spawn_mode=spawn_mode,
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


# ── エピソード中の文脈別行動計測（Session 34bと同じ手法） ────────────────────

def _run_ep_s35_context_log(cfg, G, W, genome, rng, spawn_mode,
                             pursuit_prob=_S35_PURSUIT,
                             hp_decay=_S35_HP_DECAY,
                             T=_S35_T_LONG):
    """_run_ep_with_context_log のSession 35版（スポーン位置変更対応）。"""
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

    spawn_dist = _spawn_dist_for_mode(cfg, spawn_mode)
    if spawn_mode == 'anywhere':
        pred_pos = _s31_init_pred(cfg, rng, row, col)
    else:
        pred_pos = _s35_init_pred_near_food(
            cfg, rng, food_positions, row, col, spawn_dist)

    pred_resources = 1.0
    pred_dormant   = False

    ctx_map = {(1,0):0, (1,1):1, (0,1):2, (0,0):3}
    context_action_log = []
    total_food = total_hits = 0

    for step in range(T):
        if hp <= 0:
            # リセット
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s31_init_foods(cfg, rng, row, col)
            food_avail     = [True] * cfg.n_foods
            food_timer     = [0]   * cfg.n_foods
            spawn_dist = _spawn_dist_for_mode(cfg, spawn_mode)
            if spawn_mode == 'anywhere':
                pred_pos = _s31_init_pred(cfg, rng, row, col)
            else:
                pred_pos = _s35_init_pred_near_food(
                    cfg, rng, food_positions, row, col, spawn_dist)
            pred_resources = 1.0
            pred_dormant   = False
            resources      = np.ones(_S28_N)
            activity       = np.zeros(_S28_N)

        if step % _S34_PRED_SPEED == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                pursuit_prob, pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            total_hits += 1

        inp5 = _s31_inp5(cfg, row, col, hp, food_positions, food_avail, pred_pos)
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
        context_action_log.append((ctx_idx, action))

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
                    total_food += 1
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

    return context_action_log, {'food': total_food, 'hits': total_hits}


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_spawn_sweep(sweep_results, c1_rates_pre,
                     fname='images/session_35/results_s35_spawn_sweep.png'):
    modes  = [r['spawn_mode']      for r in sweep_results]
    c1_pre = [c1_rates_pre[m]      for m in modes]
    c1_ep  = [r['c1_rate_ep']      for r in sweep_results]
    c0_eat = [r['c0_eat_rate']     for r in sweep_results]
    c1_eat = [r['c1_eat_rate']     for r in sweep_results]
    diffs  = [c0 - c1 for c0, c1 in zip(c0_eat, c1_eat)]
    steps  = [r['best_steps']      for r in sweep_results]
    foods  = [r['food']            for r in sweep_results]
    hits   = [r['pred_hits']       for r in sweep_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 35: 捕食者スポーン位置スイープ\n'
        f'pp={_S35_PURSUIT}  hp_decay={_S35_HP_DECAY}  '
        f'grid={_S35_CFG.grid}x{_S35_CFG.grid}  {_S35_N_GEN}世代',
        fontsize=12,
    )
    colors = ['tomato', 'steelblue', 'gray']
    mode_labels = ['near\n(食料から≤2)', 'medium\n(食料から≤4)', 'anywhere\n(ランダム)']

    # Panel 1: C1経験率（事前計測とエピソード中）
    ax = axes[0][0]
    x = np.arange(len(modes))
    w = 0.35
    ax.bar(x - w/2, c1_pre, width=w, color='tomato', alpha=0.85,
           label='事前計測(ランダム行動)', edgecolor='white')
    ax.bar(x + w/2, c1_ep,  width=w, color='steelblue', alpha=0.85,
           label='エピソード中(進化個体)', edgecolor='white')
    ax.axhline(0.04, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(4%)')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_ylabel('C1発生率')
    ax.set_title('C1（食料近・捕食者近）の経験頻度\n[核心] nearで上がるか？')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (pre, ep) in enumerate(zip(c1_pre, c1_ep)):
        ax.text(i - w/2, pre + 0.003, f'{pre:.0%}', ha='center', fontsize=8)
        ax.text(i + w/2, ep  + 0.003, f'{ep:.0%}',  ha='center', fontsize=8)

    # Panel 2: C0-C1食事率の差【文脈依存の核心指標】
    ax = axes[0][1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(modes)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.03, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(+3%)')
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1【文脈依存の核心】\n(正=捕食者がいると食事を控える)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (d, c0, c1) in enumerate(zip(diffs, c0_eat, c1_eat)):
        ax.text(i, d + 0.003 if d >= 0 else d - 0.008,
                f'{d:+.0%}\n(C0={c0:.0%},C1={c1:.0%})',
                ha='center', fontsize=8)

    # Panel 3: food/ep と pred_hits/ep
    ax = axes[0][2]
    x = np.arange(len(modes))
    w = 0.35
    ax.bar(x - w/2, foods, width=w, color='seagreen', alpha=0.85,
           label='food/ep', edgecolor='white')
    ax.bar(x + w/2, hits,  width=w, color='tomato',   alpha=0.85,
           label='hits/ep', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_ylabel('Count / ep')
    ax.set_title('食料獲得 vs 捕食者ヒット\n(nearで両方増えるか)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: 生存ステップ
    ax = axes[1][0]
    ax.bar(range(len(modes)), steps, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_ylabel('Mean steps / ep')
    ax.set_title('生存ステップ数')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(steps):
        ax.text(i, v + 0.3, f'{v:.0f}', ha='center', fontsize=10)

    # Panel 5: C0 vs C1 の食事率（絶対値）
    ax = axes[1][1]
    x = np.arange(len(modes))
    w = 0.35
    ax.bar(x - w/2, c0_eat, width=w, color='royalblue', alpha=0.85,
           label='C0(食料近・捕食者遠)', edgecolor='white')
    ax.bar(x + w/2, c1_eat, width=w, color='tomato',    alpha=0.85,
           label='C1(食料近・捕食者近)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_ylabel('食事行動率')
    ax.set_title('C0 vs C1 の食事率\n(C0>C1 なら文脈依存あり)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (c0, c1) in enumerate(zip(c0_eat, c1_eat)):
        ax.text(i - w/2, c0 + 0.005, f'{c0:.0%}', ha='center', fontsize=8)
        ax.text(i + w/2, c1 + 0.005, f'{c1:.0%}', ha='center', fontsize=8)

    # Panel 6: サマリーテキスト
    ax = axes[1][2]
    lines = ['スポーンモード別サマリー\n',
             f'{"mode":>8}  {"C1%":>6}  {"C0eat":>6}  '
             f'{"C1eat":>6}  {"diff":>6}  {"steps":>7}']
    for r, pre in zip(sweep_results, c1_pre):
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        lines.append(
            f'{r["spawn_mode"]:>8}  {pre:>6.0%}  '
            f'{r["c0_eat_rate"]:>6.0%}  {r["c1_eat_rate"]:>6.0%}  '
            f'{d:>+6.0%}  {r["best_steps"]:>7.1f}')
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
                   fname='images/session_35/results_s35_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    c1_ns   = [r['c1_steps']    for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 35: 複数seed確認（nearスポーン）\n'
        f'pp={_S35_PURSUIT}  hp_decay={_S35_HP_DECAY}  '
        f'T={_S35_T_LONG}  seeds={seeds}',
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
    ax.set_title('食事率の差 C0-C1\n(正かつSession 34b超え=改善)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.003 if d >= 0 else d - 0.008,
                f'{d:+.0%}', ha='center', fontsize=9)

    ax = axes[2]
    ax.bar(range(len(seeds)), c1_ns, color='darkorange', alpha=0.85,
           edgecolor='white')
    ax.axhline(80, color='gray', linestyle='--', linewidth=1.5,
               label='Session 34b基準(80steps)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('Steps')
    ax.set_title('C1経験ステップ数\n(Session 34b基準80より増えたか)')
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
    cfg = _S35_CFG

    print('=== Session 35: 捕食者スポーン位置スイープ ===')
    print(f'grid={cfg.grid}x{cfg.grid}  hp_decay={_S35_HP_DECAY}  '
          f'food_dist={cfg.food_dist}  pp={_S35_PURSUIT}')
    print(f'spawn_modes={_S35_SPAWN_MODES}')
    print()

    # ── 事前計測: C1発生率 ─────────────────────────────────────────────────
    print('[事前確認] スポーンモード別のC1発生率（ランダム行動）')
    c1_rates_pre = {}
    for mode in _S35_SPAWN_MODES:
        rates, n = measure_c1_rate(
            cfg, mode, _S35_PURSUIT, seed=_S35_SEED)
        c1_rates_pre[mode] = rates.get(1, 0.0)
        print(f'  {mode:>8}: C1={c1_rates_pre[mode]:.1%}  '
              f'C0={rates.get(0,0):.1%}  '
              f'C2={rates.get(2,0):.1%}  '
              f'C3={rates.get(3,0):.1%}  ({n}steps)')
    print()

    # ── Exp A: スポーンモードスイープ ────────────────────────────────────────
    sweep_results = []
    for mode in _S35_SPAWN_MODES:
        print(f'\n{"="*60}')
        print(f'spawn_mode={mode}  C1率={c1_rates_pre[mode]:.1%}')
        print(f'{"="*60}')

        print('\n  [進化]')
        best, hist = _s35_evolve(cfg, spawn_mode=mode, seed=_S35_SEED)
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'hits={hist["gen_pred_hits"][-1]:.2f}/ep')

        print(f'\n  [エピソード中の文脈別行動計測 T={_S35_T_LONG}]')
        rng_ep = np.random.default_rng(_S35_SEED + 35200)
        G_ep   = best['G'].copy()
        W_ep   = _s28_get_W(G_ep)
        log_ep, stats_ep = _run_ep_s35_context_log(
            cfg, G_ep, W_ep, best, rng_ep,
            spawn_mode=mode, T=_S35_T_LONG)
        counts_ep, fracs_ep, totals_ep = aggregate_context_actions(log_ep)

        c0_eat = fracs_ep[0, 4]
        c1_eat = fracs_ep[1, 4]
        c1_n   = totals_ep[1]
        diff   = c0_eat - c1_eat
        print(f'  C0食事率={c0_eat:.0%}  C1食事率={c1_eat:.0%}  '
              f'差={diff:+.0%}  C1n={c1_n}')
        for c in range(4):
            if totals_ep[c] > 0:
                dom = int(np.argmax(fracs_ep[c]))
                print(f'  C{c}: {totals_ep[c]}steps  '
                      f'主行動={_S28_ACTION_NAMES[dom]}({fracs_ep[c,dom]:.0%})  '
                      f'食事={fracs_ep[c,4]:.0%}')

        sweep_results.append({
            'spawn_mode':   mode,
            'best_genome':  best,
            'hist':         hist,
            'best_steps':   hist['gen_best_steps'][-1],
            'food':         hist['gen_food_count'][-1],
            'pred_hits':    hist['gen_pred_hits'][-1],
            'c0_eat_rate':  c0_eat,
            'c1_eat_rate':  c1_eat,
            'c1_rate_ep':   c1_n / len(log_ep) if log_ep else 0.0,
            'c1_steps':     c1_n,
            'fracs':        fracs_ep,
            'totals':       totals_ep,
        })

    plot_spawn_sweep(sweep_results, c1_rates_pre)

    # ── Exp B: nearモードで複数seed確認 ──────────────────────────────────────
    print(f'\n[Exp B] nearモードで複数seed確認 (seeds={_S35_SEEDS})')
    multi_results = []
    for seed in _S35_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, hist_s = _s35_evolve(
            cfg, spawn_mode='near', seed=seed)
        rng_s = np.random.default_rng(seed + 35200)
        G_s   = best_s['G'].copy()
        W_s   = _s28_get_W(G_s)
        log_s, _ = _run_ep_s35_context_log(
            cfg, G_s, W_s, best_s, rng_s,
            spawn_mode='near', T=_S35_T_LONG)
        counts_s, fracs_s, totals_s = aggregate_context_actions(log_s)
        c0_eat = fracs_s[0, 4]
        c1_eat = fracs_s[1, 4]
        c1_n   = totals_s[1]
        print(f'    C0食事率={c0_eat:.0%}  C1食事率={c1_eat:.0%}  '
              f'差={c0_eat-c1_eat:+.0%}  C1n={c1_n}')
        multi_results.append({
            'seed':        seed,
            'c0_eat_rate': c0_eat,
            'c1_eat_rate': c1_eat,
            'c1_steps':    c1_n,
        })

    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 35 Summary ===')
    print()
    print('スポーンモード別:')
    for r in sweep_results:
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        print(f'  {r["spawn_mode"]:>8}: '
              f'C1率={c1_rates_pre[r["spawn_mode"]]:.0%}  '
              f'C0eat={r["c0_eat_rate"]:.0%}  '
              f'C1eat={r["c1_eat_rate"]:.0%}  '
              f'diff={d:+.0%}  '
              f'steps={r["best_steps"]:.0f}')

    print()
    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    print(f'nearモード 複数seed (n={len(_S35_SEEDS)}):')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S35_SEEDS)} seeds')
    print()
    print('--- 判断基準 ---')
    print('nearでC1率が上がり、かつC0-C1差がSession 34b(+3%)を超えれば前進')
    print('差が変わらない → スポーン位置より別の問題（ネットワーク容量等）')
    print()
    print('Done.')
