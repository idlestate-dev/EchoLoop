"""Session 38: hp_decay=1 でエピソードを長くする

Session 37の問題:
  hp_decay=5 → 平均生存28steps（max_steps=1280の2%）
  → 睡眠サイクルが完結する前に死んでいる
  → TM資源モデルが「省エネ戦略」への進化圧になっている
  → active ノードが 2〜6個しかない

Session 10-14（睡眠が出た設定）との比較:
  HP_DECAY = 1/step
  食料を取りながら 100〜200steps 生存
  → 睡眠サイクルが完結する時間的余裕がある

変更点（Session 37からの差分）:
  hp_decay = 5 → 1 に戻す
  それ以外は全て Session 37 と同じ

  hp_decay=1 での生存時間の見込み:
    food_value=30 → 食料1個で+30steps
    hp_start=256  → 食料なし生存=256steps（max_stepsの20%）
    食料を定期的に取れば長時間生存可能

目的:
  1. active ノード数が増えるか？
     → 増えれば「リソース枯渇が原因だった」と確認できる
  2. C0-C1食事率差は改善するか？
     → 改善すれば「覚醒中の文脈統合」が可能になった
  3. 睡眠様状態（リソース枯渇→回復サイクル）と
     文脈依存行動が共存するか？
     → Session 12-14の発見と繋がる

固定条件:
  N=25（Session 37のベスト）
  grid=8, food_dist=2, pred_dist=1
  pursuit_prob=0.9

実験:
  A: hp_decay スイープ [1, 2, 3, 5]
     → active ノード数と C0-C1差の変化を確認
     ※ 5 は Session 37 の参照値として含める
  B: hp_decay=1 での複数seed確認（42〜46）
  C: hp_decay=1 での長時間観察（T=5000）
     → リソース枯渇→回復サイクルが出るか

出力:
  images/session_38/results_s38_decay_sweep.png
  images/session_38/results_s38_multiseed.png
  images/session_38/results_s38_long_obs.png
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_28_predator import (
    _S28_OUT_START, _S28_OUT_END,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_ACTION_NAMES,
)
from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import (
    WorldConfig, _s31_init_foods, _s31_init_pred,
)
from session_33_eat_range import _S33_CFG
from session_34_pursuit_x_s33 import (
    _S34_PRED_SPEED, _s34_pred_step,
)
from session_36_pred_dist_sweep import _s36_inp5
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate, _hebb,
    aggregate_context_actions,
)

# ── Session 38 定数 ────────────────────────────────────────────────────────────

_S38_SEED       = _S28_SEED
_S38_N_GEN      = _S28_N_GEN
_S38_PURSUIT    = 0.9
_S38_N          = 25            # Session 37のベスト
_S38_PRED_DIST  = 1
_S38_N_SEEDS    = 5
_S38_SEEDS      = list(range(42, 42 + _S38_N_SEEDS))
_S38_T_LONG     = 2000
_S38_T_OBS      = 5000          # 長時間観察用
_S38_N_TRIALS   = 20

# hp_decayスイープ（5はSession 37参照値）
_S38_HP_DECAYS  = [1, 2, 3, 5]

# WorldConfig（pred_distだけ差し替え）
_S38_CFG = WorldConfig(
    grid      = _S33_CFG.grid,
    max_steps = _S33_CFG.max_steps,
    hp_start  = _S33_CFG.hp_start,
    n_foods   = _S33_CFG.n_foods,
    food_dist = _S33_CFG.food_dist,
    pred_dist = _S38_PRED_DIST,
)


# ── エピソードランナー ─────────────────────────────────────────────────────────

def _s38_run_ep(cfg: WorldConfig, G, W, genome, rng,
                hp_decay: float       = 1.0,
                pursuit_prob: float   = _S38_PURSUIT,
                predator_speed: int   = _S34_PRED_SPEED,
                record_activity: bool = False):
    """Session 37と同じ、hp_decayだけパラメータ化。"""
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

def _run_context_log(cfg, G, W, genome, rng,
                     hp_decay=1.0,
                     pursuit_prob=_S38_PURSUIT,
                     T=_S38_T_LONG):
    """文脈別行動を記録。死亡時リセット。"""
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
    act_log = []   # active ノード数の記録

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

        n_active = int(np.sum(eff > _S28_ACT_THRESH))
        act_log.append(n_active)

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


# ── 長時間観察（リソース枯渇→回復サイクルの確認） ─────────────────────────────

def _run_long_obs(cfg, G, W, genome, rng,
                  hp_decay=1.0, pursuit_prob=_S38_PURSUIT,
                  T=_S38_T_OBS, chunk=100):
    """T_OBS ステップを走らせてリソース枯渇→回復サイクルを観察。
    死亡時リセットして観察を継続。

    Returns list of chunk-level records.
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

    food_positions = _s31_init_foods(cfg, rng, row, col)
    food_avail     = [True] * cfg.n_foods
    food_timer     = [0]   * cfg.n_foods
    pred_pos       = _s31_init_pred(cfg, rng, row, col)
    pred_resources = 1.0
    pred_dormant   = False

    chunks    = []
    chunk_res_sens = []   # 感覚器リソース
    chunk_res_int  = []   # 内部リソース
    chunk_act      = []   # active ノード数

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

        inp5 = _s36_inp5(cfg, row, col, hp,
                         food_positions, food_avail, pred_pos)

        for _ in range(_N_PROP):
            activity = _propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, n), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, _make_tau_arr(n), depletion_rate)

        # 感覚器(0-4)・内部(10+)のリソース
        chunk_res_sens.append(float(np.mean(resources[:5])))
        chunk_res_int.append(float(np.mean(resources[10:])))
        chunk_act.append(int(np.sum(eff > _S28_ACT_THRESH)))

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

        # チャンク集計
        if (step + 1) % chunk == 0:
            chunks.append({
                'step':         step + 1,
                'res_sensory':  float(np.mean(chunk_res_sens)),
                'res_internal': float(np.mean(chunk_res_int)),
                'mean_active':  float(np.mean(chunk_act)),
            })
            chunk_res_sens = []
            chunk_res_int  = []
            chunk_act      = []

    return chunks


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s38_evolve(cfg: WorldConfig, n: int, hp_decay: float,
                pursuit_prob: float = _S38_PURSUIT,
                seed: int           = _S38_SEED,
                n_gen: int          = _S38_N_GEN):
    rng = np.random.default_rng(seed + 38000 + n * 10 + int(hp_decay * 10))
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s38_run_ep(
                    cfg, g['G'], g['W'], g, ep_rng,
                    hp_decay=hp_decay,
                    pursuit_prob=pursuit_prob,
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


# ── 固定戦略 ──────────────────────────────────────────────────────────────────

def _check_fixed(cfg, hp_decay, pursuit_prob=_S38_PURSUIT,
                 seed=_S38_SEED, n_trials=_S38_N_TRIALS):
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 38300 + trial * 10 + action_idx)
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

def plot_decay_sweep(sweep_results,
                     fname='images/session_38/results_s38_decay_sweep.png'):
    decays   = [r['hp_decay']       for r in sweep_results]
    c0_eat   = [r['c0_eat_rate']    for r in sweep_results]
    c1_eat   = [r['c1_eat_rate']    for r in sweep_results]
    diffs    = [c0 - c1 for c0, c1 in zip(c0_eat, c1_eat)]
    steps    = [r['best_steps']     for r in sweep_results]
    foods    = [r['food']           for r in sweep_results]
    hits     = [r['pred_hits']      for r in sweep_results]
    actives  = [r['mean_active']    for r in sweep_results]
    ef_diffs = [r['evo_vs_fixed']   for r in sweep_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 38: hp_decay スイープ（N={_S38_N}, エピソード長回復）\n'
        f'grid={_S38_CFG.grid}x{_S38_CFG.grid}  pred_dist={_S38_PRED_DIST}  '
        f'pp={_S38_PURSUIT}  {_S38_N_GEN}世代',
        fontsize=12,
    )
    colors = ['seagreen', 'steelblue', 'darkorange', 'tomato']
    xlbls  = [f'decay={d}' for d in decays]

    # Panel 1: active ノード数【今回の核心】
    ax = axes[0][0]
    ax.bar(range(len(decays)), actives, color=colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(4.0, color='gray', linestyle='--', linewidth=1.5,
               label='Session 37参照(≈4)')
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel(f'Mean active nodes / {_S38_N}')
    ax.set_title('活動ノード数【核心】\nhp_decay=1で増えるか？')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(actives):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10,
                fontweight='bold')

    # Panel 2: C0-C1食事率差
    ax = axes[0][1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(decays)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.03, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 34b基準(+3%)')
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1\n(正=捕食者がいると食事を控える)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (d, c0, c1) in enumerate(zip(diffs, c0_eat, c1_eat)):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.012,
                f'{d:+.0%}\nC0={c0:.0%}/C1={c1:.0%}',
                ha='center', fontsize=8)

    # Panel 3: 進化 - 固定戦略の差
    ax = axes[0][2]
    bar_c = ['seagreen' if d > 0 else 'tomato' for d in ef_diffs]
    ax.bar(range(len(decays)), ef_diffs, color=bar_c, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Evolution - Fixed (steps)')
    ax.set_title('進化 - 固定戦略')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(ef_diffs):
        ax.text(i, d + 1 if d >= 0 else d - 3,
                f'{d:+.1f}', ha='center', fontsize=10)

    # Panel 4: 生存ステップ
    ax = axes[1][0]
    ax.bar(range(len(decays)), steps, color=colors, alpha=0.85,
           edgecolor='white')
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel('Mean steps / ep')
    ax.set_title('生存ステップ数\n(decay=1で大幅増加するはず)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(steps):
        ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=9)

    # Panel 5: food/ep と hits/ep
    ax = axes[1][1]
    x = np.arange(len(decays))
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
    lines = ['hp_decay別サマリー\n',
             f'{"decay":>6}  {"active":>7}  {"C0eat":>6}  '
             f'{"C1eat":>6}  {"diff":>6}  {"steps":>7}  {"ef":>7}']
    for r in sweep_results:
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        lines.append(
            f'{r["hp_decay"]:>6}  {r["mean_active"]:>7.1f}  '
            f'{r["c0_eat_rate"]:>6.0%}  {r["c1_eat_rate"]:>6.0%}  '
            f'{d:>+6.0%}  {r["best_steps"]:>7.1f}  '
            f'{r["evo_vs_fixed"]:>+7.1f}')
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
                   fname='images/session_38/results_s38_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    actives = [r['mean_active'] for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 38: 複数seed確認（hp_decay=1, N={_S38_N}）\n'
        f'pp={_S38_PURSUIT}  T={_S38_T_LONG}',
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
    ax.set_title('食事率差 C0-C1')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.01,
                f'{d:+.0%}', ha='center', fontsize=9)

    ax = axes[2]
    ax.bar(range(len(seeds)), actives, color='steelblue', alpha=0.85,
           edgecolor='white')
    ax.axhline(4.0, color='gray', linestyle='--', linewidth=1.5,
               label='Session 37参照(≈4)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel(f'Mean active nodes / {_S38_N}')
    ax.set_title('活動ノード数')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(actives):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_long_obs(chunks,
                  fname='images/session_38/results_s38_long_obs.png'):
    """長時間観察：リソース枯渇→回復サイクルの可視化。"""
    t_vals   = [c['step']         for c in chunks]
    res_s    = [c['res_sensory']  for c in chunks]
    res_i    = [c['res_internal'] for c in chunks]
    actives  = [c['mean_active']  for c in chunks]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(
        f'Session 38: 長時間観察（T={_S38_T_OBS}, hp_decay=1, N={_S38_N}）\n'
        f'リソース枯渇→回復サイクルと活動ノード数の推移',
        fontsize=13,
    )

    ax = axes[0]
    ax.plot(t_vals, res_s, color='tomato',    linewidth=2,
            label='感覚器リソース(nodes 0-4)')
    ax.plot(t_vals, res_i, color='steelblue', linewidth=2,
            label='内部リソース(nodes 10+)')
    ax.set_ylabel('Mean resources')
    ax.set_ylim(0, 1.05)
    ax.set_title('リソース量の推移\n(感覚器が先に枯渇すれば睡眠様状態)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_vals, actives, color='seagreen', linewidth=2,
            label='active nodes')
    ax.axhline(4.0, color='gray', linestyle='--', linewidth=1.5,
               label='Session 37参照(≈4)')
    ax.set_xlabel('Step')
    ax.set_ylabel(f'Mean active nodes / {_S38_N}')
    ax.set_title('活動ノード数の推移\n(リソース枯渇時に減少すれば睡眠と連動)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S38_CFG

    print('=== Session 38: hp_decay=1 でエピソードを長くする ===')
    print(f'N={_S38_N}  grid={cfg.grid}x{cfg.grid}  '
          f'pred_dist={cfg.pred_dist}  pp={_S38_PURSUIT}')
    print(f'hp_start={cfg.hp_start}  food_value={_S28_FOOD_VALUE}  '
          f'max_steps={cfg.max_steps}')
    print()
    print('hp_decayの意味:')
    for d in _S38_HP_DECAYS:
        survive = int(cfg.hp_start / d)
        frac    = survive / cfg.max_steps
        print(f'  decay={d}: 食料なし生存={survive}steps({frac:.0%})')
    print()

    sweep_results = []

    for hp_decay in _S38_HP_DECAYS:
        print(f'\n{"="*60}')
        print(f'hp_decay={hp_decay}')
        print(f'{"="*60}')

        # 固定戦略
        fixed     = _check_fixed(cfg, hp_decay)
        best_fix  = max(fixed.values())
        best_fn   = max(fixed, key=fixed.get)

        print('\n  [進化]')
        best, hist = _s38_evolve(cfg, _S38_N, hp_decay, seed=_S38_SEED)
        steps   = hist['gen_best_steps'][-1]
        food    = hist['gen_food_count'][-1]
        hits    = hist['gen_pred_hits'][-1]
        actv    = hist['gen_mean_active'][-1]
        ef_diff = steps - best_fix
        print(f'  → steps={steps:.1f}  food={food:.2f}/ep  '
              f'hits={hits:.2f}/ep  active={actv:.1f}  '
              f'vs固定({best_fn})={ef_diff:+.1f}')

        print(f'\n  [文脈別行動計測 T={_S38_T_LONG}]')
        rng_ep = np.random.default_rng(_S38_SEED + 38200)
        G_ep   = best['G'].copy()
        W_ep   = best['W'].copy()
        log_ep, act_log = _run_context_log(
            cfg, G_ep, W_ep, best, rng_ep,
            hp_decay=hp_decay, T=_S38_T_LONG)
        counts_ep, fracs_ep, totals_ep = aggregate_context_actions(log_ep)

        c0_eat = fracs_ep[0, 4]
        c1_eat = fracs_ep[1, 4]
        c1_n   = totals_ep[1]
        mean_act = float(np.mean(act_log))
        print(f'  C0食事率={c0_eat:.0%}  C1食事率={c1_eat:.0%}  '
              f'差={c0_eat-c1_eat:+.0%}  C1n={c1_n}  '
              f'平均active={mean_act:.1f}/{_S38_N}')
        for c in range(4):
            if totals_ep[c] > 0:
                dom = int(np.argmax(fracs_ep[c]))
                print(f'  C{c}: {totals_ep[c]}steps  '
                      f'主行動={_S28_ACTION_NAMES[dom]}'
                      f'({fracs_ep[c,dom]:.0%})  '
                      f'食事={fracs_ep[c,4]:.0%}')

        sweep_results.append({
            'hp_decay':     hp_decay,
            'best_steps':   steps,
            'food':         food,
            'pred_hits':    hits,
            'mean_active':  actv,
            'c0_eat_rate':  c0_eat,
            'c1_eat_rate':  c1_eat,
            'c1_steps':     c1_n,
            'evo_vs_fixed': ef_diff,
            'best_genome':  best,
        })

    plot_decay_sweep(sweep_results)

    # ── Exp B: hp_decay=1 で複数seed ──────────────────────────────────────
    print(f'\n[Exp B] hp_decay=1 で複数seed確認')
    r1 = next(r for r in sweep_results if r['hp_decay'] == 1)
    print(f'  seed=42: C0-C1差={r1["c0_eat_rate"]-r1["c1_eat_rate"]:+.0%}  '
          f'active={r1["mean_active"]:.1f}')

    multi_results = []
    for seed in _S38_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, hist_s = _s38_evolve(cfg, _S38_N, hp_decay=1, seed=seed)
        rng_s = np.random.default_rng(seed + 38200)
        G_s   = best_s['G'].copy()
        W_s   = best_s['W'].copy()
        log_s, act_s = _run_context_log(
            cfg, G_s, W_s, best_s, rng_s,
            hp_decay=1, T=_S38_T_LONG)
        _, fracs_s, totals_s = aggregate_context_actions(log_s)
        c0 = fracs_s[0, 4]
        c1 = fracs_s[1, 4]
        n1 = totals_s[1]
        ma = float(np.mean(act_s))
        print(f'    C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={n1}  active={ma:.1f}')
        multi_results.append({
            'seed': seed, 'c0_eat_rate': c0,
            'c1_eat_rate': c1, 'c1_steps': n1,
            'mean_active': ma,
        })

    plot_multiseed(multi_results)

    # ── Exp C: hp_decay=1 で長時間観察 ────────────────────────────────────
    print(f'\n[Exp C] hp_decay=1 で長時間観察（T={_S38_T_OBS}）')
    best1 = r1['best_genome']
    rng_obs = np.random.default_rng(_S38_SEED + 38500)
    G_obs   = best1['G'].copy()
    W_obs   = best1['W'].copy()
    chunks  = _run_long_obs(
        cfg, G_obs, W_obs, best1, rng_obs,
        hp_decay=1, T=_S38_T_OBS)
    plot_long_obs(chunks)

    mid = len(chunks) // 2
    res_s_h1 = float(np.mean([c['res_sensory']  for c in chunks[:mid]]))
    res_s_h2 = float(np.mean([c['res_sensory']  for c in chunks[mid:]]))
    res_i_h1 = float(np.mean([c['res_internal'] for c in chunks[:mid]]))
    res_i_h2 = float(np.mean([c['res_internal'] for c in chunks[mid:]]))
    act_h1   = float(np.mean([c['mean_active']  for c in chunks[:mid]]))
    act_h2   = float(np.mean([c['mean_active']  for c in chunks[mid:]]))
    print(f'  感覚器リソース: 前半={res_s_h1:.3f} → 後半={res_s_h2:.3f}  '
          f'Δ={res_s_h2-res_s_h1:+.3f}')
    print(f'  内部リソース:   前半={res_i_h1:.3f} → 後半={res_i_h2:.3f}  '
          f'Δ={res_i_h2-res_i_h1:+.3f}')
    print(f'  活動ノード数:   前半={act_h1:.1f}  → 後半={act_h2:.1f}')

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 38 Summary ===')
    print()
    print('hp_decay別:')
    for r in sweep_results:
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        print(f'  decay={r["hp_decay"]}: '
              f'active={r["mean_active"]:.1f}  '
              f'C0eat={r["c0_eat_rate"]:.0%}  '
              f'C1eat={r["c1_eat_rate"]:.0%}  '
              f'diff={d:+.0%}  '
              f'evo_fixed={r["evo_vs_fixed"]:+.1f}  '
              f'steps={r["best_steps"]:.0f}')

    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    acts  = [r['mean_active'] for r in multi_results]
    print()
    print(f'hp_decay=1 複数seed (n={len(_S38_SEEDS)}):')
    print(f'  活動ノード数: mean={np.mean(acts):.1f}  std={np.std(acts):.1f}')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S38_SEEDS)} seeds')
    print()
    print('--- 判断基準 ---')
    print('hp_decay=1 で active ノード数が増加 → リソース枯渇が原因だった')
    print('active 増加しても C0-C1差が変わらない → 容量以外の問題')
    print('長時間観察で感覚器リソースが先に枯渇 → 睡眠様状態との共存が見える')
    print()
    print('Done.')
