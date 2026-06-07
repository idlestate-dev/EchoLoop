"""Session 34: 追跡型捕食者 × Session 33 環境

動機:
  Session 33b の診断:
    pred_flag が立つ経験がエピソード中にほとんどない
    → 捕食者フラグへの反応が学習されない
    → 「捕食者が近い時は逃げる」が出ない

  解決:
    捕食者をエージェントに向けて追跡させる（Session 29の設計）
    + Session 33 の環境（decay=5、食事範囲≤food_dist）を維持

  これにより:
    「逃げ続けると餓死（decay=5）」
    「食べに行くと捕食者が追いかけてくる」
    というトレードオフが成立するはず

Session 29との違い:
  Session 29: grid=5, decay=1, 食事=距離0のみ
  Session 34: grid=8, decay=5, 食事=距離≤2  ← Session 33の環境

パラメータ:
  pursuit_prob スイープ: [0.0, 0.3, 0.6, 0.9]
    0.0 = ランダムウォーク（Session 33と同じ）
    0.6 = Session 29 best
    0.9 = より強い追跡
  hp_decay = 5 固定（Session 33 best）
  pred_resources/dormancy は Session 29 と同じ設計

実験:
  A: pursuit_prob スイープ（50世代進化 × 4条件）
     → pred_flag 発生頻度、food、mcdの変化を確認
  B: ベスト条件での文脈依存行動の詳細計測
  C: 固定戦略の排除確認

出力:
  images/session_34/results_s34_sweep.png
  images/session_34/results_s34_context.png
"""

import os
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
    _S28_CONTEXTS, _S28_ACTION_NAMES,
    _s28_get_W, _s28_propagate,
    _s28_hebb, _s28_make_tau_arr, _s28_make_genome, _s28_mutate_genome,
    _s28_cosine_dist, _s28_measure_context,
)
from session_31_grid_sweep import (
    WorldConfig,
    _s31_init_foods, _s31_init_pred, _s31_inp5,
)
from session_33_eat_range import _S33_CFG, _S33_FOOD_BIAS

# ── Session 34 定数 ────────────────────────────────────────────────────────────

_S34_SEED          = _S28_SEED
_S34_N_GEN         = _S28_N_GEN
_S34_PRED_SPEED    = 2
_S34_N_CONTEXT     = 25
_S34_CONTEXT_T     = 100
_S34_N_TRIALS      = 20

_S34_HP_DECAY      = 5       # Session 33 best
_S34_FOOD_BIAS     = 0.0     # 捕食者はエージェントを追跡するので食料バイアス不要
_S34_CFG           = _S33_CFG  # grid=8, food_dist=2

# pursuit_prob スイープ
_S34_PURSUIT_PROBS = [0.0, 0.3, 0.6, 0.9]

# 捕食者の疲労パラメータ（Session 29と同じ）
_S34_PRED_DEPLETION  = 0.05
_S34_PRED_RECOVERY   = 0.02
_S34_PRED_DORMANT_LO = 0.2
_S34_PRED_DORMANT_HI = 0.8


# ── 追跡型捕食者の移動 ────────────────────────────────────────────────────────

def _s34_pred_step(cfg: WorldConfig, pred_pos, agent_pos,
                   pursuit_prob, pred_resources, pred_dormant, rng):
    """Session 29の追跡ロジック + WorldConfigのgridサイズ対応版。

    pursuit_prob の確率でエージェントに向かって1歩近づく。
    追跡するたびに捕食者の資源が枯渇 → 休眠（ランダムウォーク）。

    Returns (new_pos, new_pred_resources, new_pred_dormant).
    """
    pr, pc = pred_pos
    ar, ac = agent_pos

    if pred_dormant:
        is_pursuing = False
    else:
        is_pursuing = bool(rng.random() < pursuit_prob)

    if is_pursuing:
        dr = int(np.sign(ar - pr))
        dc = int(np.sign(ac - pc))
        if dr == 0 and dc == 0:
            is_pursuing = False  # 同セルならランダムへ
        elif dr == 0:
            pc = int(np.clip(pc + dc, 0, cfg.grid - 1))
        elif dc == 0:
            pr = int(np.clip(pr + dr, 0, cfg.grid - 1))
        else:
            if rng.random() < 0.5:
                pr = int(np.clip(pr + dr, 0, cfg.grid - 1))
            else:
                pc = int(np.clip(pc + dc, 0, cfg.grid - 1))

    if not is_pursuing:
        d = int(rng.integers(0, 4))
        if d == 0:   pr = max(0, pr - 1)
        elif d == 1: pr = min(cfg.grid - 1, pr + 1)
        elif d == 2: pc = max(0, pc - 1)
        else:        pc = min(cfg.grid - 1, pc + 1)

    # 捕食者資源の更新
    if is_pursuing:
        pred_resources -= _S34_PRED_DEPLETION
    else:
        pred_resources += _S34_PRED_RECOVERY
    pred_resources = float(np.clip(pred_resources, 0.0, 1.0))

    # 休眠遷移（ヒステリシス）
    if pred_resources < _S34_PRED_DORMANT_LO:
        pred_dormant = True
    elif pred_resources > _S34_PRED_DORMANT_HI:
        pred_dormant = False

    return [pr, pc], pred_resources, pred_dormant


# ── エピソードランナー ─────────────────────────────────────────────────────────

def _s34_run_ep(cfg: WorldConfig, G, W, genome, rng,
                pursuit_prob: float,
                hp_decay: float        = _S34_HP_DECAY,
                predator_speed: int    = _S34_PRED_SPEED,
                record_activity: bool  = False,
                record_pred_flag: bool = False):
    """追跡型捕食者 + Session 33の食事範囲緩和を組み合わせたエピソード。

    Session 33との差分:
      捕食者移動: food_bias → pursuit_prob（エージェント追跡）
      捕食者状態: pred_resources, pred_dormant を管理
    """
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
    act_recs      = [] if record_activity  else None
    pred_flag_recs = [] if record_pred_flag else None

    for step in range(cfg.max_steps):
        if hp <= 0:
            break

        # 追跡型捕食者の移動
        if step % predator_speed == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                pursuit_prob, pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            pred_hits += 1

        inp5 = _s31_inp5(cfg, row, col, hp, food_positions, food_avail, pred_pos)

        if record_pred_flag:
            pred_flag_recs.append(float(inp5[4]))  # node4 = pred_flag

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
            # Session 33と同じ: 距離≤food_dist で食べられる
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

    return {
        'steps':          steps,
        'food':           food,
        'pred_hits':      pred_hits,
        'act_recs':       act_recs,
        'pred_flag_recs': pred_flag_recs,
    }


# ── pred_flag 発生頻度の計測 ──────────────────────────────────────────────────

def measure_pred_flag_rate(cfg: WorldConfig, pursuit_prob: float,
                            seed: int, n_ep: int = 20):
    """エピソード中のpred_flag発生頻度を計測。
    ランダム行動エージェント（ネットワーク使用なし）で計測。
    """
    rng        = np.random.default_rng(seed + 34900)
    total_steps = 0
    pred_flag_steps = 0

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
            if step % _S34_PRED_SPEED == 0:
                pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                    cfg, pred_pos, [row, col],
                    pursuit_prob, pred_resources, pred_dormant, ep_rng)
            if pred_pos[0] == row and pred_pos[1] == col:
                hp -= _S28_PRED_DAMAGE
            hp -= _S34_HP_DECAY
            inp5 = _s31_inp5(cfg, row, col, hp,
                             food_positions, food_avail, pred_pos)
            pred_flag_steps += int(inp5[4])
            total_steps     += 1
            # ランダム行動
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

    rate = pred_flag_steps / total_steps if total_steps > 0 else 0.0
    return rate, total_steps


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s34_evolve(cfg: WorldConfig, pursuit_prob: float,
                hp_decay: float = _S34_HP_DECAY,
                seed: int = _S34_SEED, n_gen: int = _S34_N_GEN):
    rng = np.random.default_rng(seed + 34000 + int(pursuit_prob * 100))
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s34_run_ep(
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


# ── 文脈依存性の計測 ──────────────────────────────────────────────────────────

def _s34_measure_context_dep(genome, seed):
    G_copy = genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 34200)

    results = []
    for ctx in _S28_CONTEXTS:
        samples = []
        for _ in range(_S34_N_CONTEXT):
            r = _s28_measure_context(
                G_copy, W_copy, genome, ctx, rng, T=_S34_CONTEXT_T)
            samples.append(r['mean_output'])
        mean_out     = np.mean(samples, axis=0)
        action_count = np.zeros(5)
        for s in samples:
            action_count[int(np.argmax(s))] += 1
        results.append({
            'mean_output':    mean_out,
            'action_dist':    action_count / _S34_N_CONTEXT,
            'output_samples': np.array(samples),
        })

    n_ctx   = len(_S28_CONTEXTS)
    pairs   = [(i, j) for i in range(n_ctx) for j in range(i + 1, n_ctx)]
    cos_mat = np.zeros((n_ctx, n_ctx))
    for i in range(n_ctx):
        for j in range(n_ctx):
            cos_mat[i, j] = _s28_cosine_dist(
                results[i]['mean_output'], results[j]['mean_output'])
    mcd = float(np.mean([cos_mat[i, j] for i, j in pairs]))

    p_values = {}
    for (i, j) in pairs:
        si = results[i]['output_samples'].max(axis=1)
        sj = results[j]['output_samples'].max(axis=1)
        _, p = scipy_stats.ttest_ind(si, sj)
        p_values[(i, j)] = float(p)

    return {'context_results': results, 'cosine_matrix': cos_mat,
            'mean_cosine_dist': mcd, 'p_values': p_values}


# ── 固定戦略の確認 ────────────────────────────────────────────────────────────

def _s34_check_simple_strategy(cfg: WorldConfig, pursuit_prob: float,
                                hp_decay: float = _S34_HP_DECAY,
                                seed: int = _S34_SEED,
                                n_trials: int = _S34_N_TRIALS):
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 34300 + trial * 10 + action_idx)
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
        print(f'    固定「{action_name}」: {results[action_name]:.1f}steps')
    return results


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_sweep(sweep_results, pred_flag_rates,
               fname='images/session_34/results_s34_sweep.png'):
    probs      = [r['pursuit_prob']      for r in sweep_results]
    mcds       = [r['mean_cosine_dist']  for r in sweep_results]
    steps      = [r['best_steps']        for r in sweep_results]
    foods      = [r['food']              for r in sweep_results]
    hits       = [r['pred_hits']         for r in sweep_results]
    best_fixed = [max(r['strategy'].values()) for r in sweep_results]
    diffs      = [s - f for s, f in zip(steps, best_fixed)]
    pfrates    = [pred_flag_rates[p] for p in probs]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 34: 追跡型捕食者 x Session 33環境\n'
        f'grid={_S34_CFG.grid}x{_S34_CFG.grid}  '
        f'hp_decay={_S34_HP_DECAY}  food_dist={_S34_CFG.food_dist}  '
        f'{_S34_N_GEN}世代  seed={_S34_SEED}',
        fontsize=12,
    )
    colors = ['steelblue', 'seagreen', 'tomato', 'darkorange']

    # Panel 1: pred_flag発生率【今回の核心】
    ax = axes[0][0]
    ax.bar(range(len(probs)), pfrates, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0.067, color='gray', linestyle='--', linewidth=1.5,
               label='Session 33b基準(6.7%)')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'pp={p}' for p in probs])
    ax.set_ylabel('pred_flag発生率')
    ax.set_title('pred_flag発生率【核心】\n(捕食者がエージェント隣接する頻度)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(pfrates):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=10,
                fontweight='bold')

    # Panel 2: food/ep
    ax = axes[0][1]
    ax.bar(range(len(probs)), foods, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0.40, color='gray', linestyle='--', linewidth=1.5,
               label='Session 33 decay=5')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'pp={p}' for p in probs])
    ax.set_ylabel('Food / ep')
    ax.set_title('食料獲得数\n(灰色=Session 33 decay=5 参照)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(foods):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    # Panel 3: 進化 - 固定戦略の差
    ax = axes[0][2]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(probs)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'pp={p}' for p in probs])
    ax.set_ylabel('Evolution - Fixed (steps)')
    ax.set_title('進化 - 固定戦略\n(正=文脈依存が有効)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + (0.3 if d >= 0 else -1.5),
                f'{d:+.1f}', ha='center', fontsize=10)

    # Panel 4: mcd
    ax = axes[1][0]
    ax.plot(probs, mcds, 'o-', color='purple', linewidth=2, markersize=10)
    ax.axhline(0.1201, color='gray', linestyle='--', linewidth=1.5,
               label='Session 33 decay=5 best (0.1201)')
    for p, m in zip(probs, mcds):
        ax.text(p, m + 0.003, f'{m:.4f}', ha='center', fontsize=9)
    ax.set_xlabel('pursuit_prob')
    ax.set_ylabel('mean cosine dist')
    ax.set_title('文脈依存性（mcd）\n灰色=Session 33参照値')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: pred_hits
    ax = axes[1][1]
    ax.bar(range(len(probs)), hits, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'pp={p}' for p in probs])
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数\n(増加=捕食者が実際に近づいている)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(hits):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    # Panel 6: サマリー
    ax = axes[1][2]
    bf_names = [max(r['strategy'], key=r['strategy'].get) for r in sweep_results]
    lines = ['pursuit_prob別サマリー\n',
             f'{"pp":>4}  {"pflag":>7}  {"food":>6}  {"steps":>7}  '
             f'{"fixed":>7}  {"diff":>6}  {"mcd":>8}']
    for r, pf, bfn in zip(sweep_results, pfrates, bf_names):
        bf = max(r['strategy'].values())
        d  = r['best_steps'] - bf
        lines.append(
            f'{r["pursuit_prob"]:>4}  {pf:>7.1%}  '
            f'{r["food"]:>6.2f}  {r["best_steps"]:>7.1f}  '
            f'{bf:>7.1f}  {d:>+6.1f}  '
            f'{r["mean_cosine_dist"]:>8.4f}')
    ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('数値サマリー')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context(exp_b, pursuit_prob,
                 fname='images/session_34/results_s34_context.png'):
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 34: 文脈依存行動（ベスト pp={pursuit_prob}）\n'
        f'grid={_S34_CFG.grid}x{_S34_CFG.grid}  '
        f'hp_decay={_S34_HP_DECAY}  food_dist={_S34_CFG.food_dist}  '
        f'mcd={exp_b["mean_cosine_dist"]:.4f}',
        fontsize=12,
    )

    ax = axes[0]
    out_mat = np.array([r['mean_output'] for r in results])
    vmax = max(out_mat.max(), 0.01)
    im = ax.imshow(out_mat, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels(
        [f'node{i+5}\n({a})' for i, a in enumerate(_S28_ACTION_NAMES)],
        fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=9)
    ax.set_title('出力ノード平均活動')
    for i in range(n_ctx):
        for j in range(5):
            ax.text(j, i, f'{out_mat[i,j]:.3f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if out_mat[i,j] > vmax * 0.6 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    vmax2 = max(cos_mat.max(), 0.01)
    im2 = ax.imshow(cos_mat, cmap='Blues', vmin=0, vmax=vmax2, aspect='auto')
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=8)
    ax.set_title('Cosine距離行列')
    pairs = [(i, j) for i in range(n_ctx) for j in range(i + 1, n_ctx)]
    for i in range(n_ctx):
        for j in range(n_ctx):
            pair = (min(i, j), max(i, j))
            p    = exp_b['p_values'].get(pair, float('nan'))
            sig  = '*' if p < 0.05 else ''
            ax.text(j, i, f'{cos_mat[i,j]:.3f}{sig}',
                    ha='center', va='center', fontsize=7,
                    color='white' if cos_mat[i,j] > vmax2 * 0.6 else 'black')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    ax = axes[2]
    act_mat  = np.array([r['action_dist'] for r in results])
    colors_a = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    bottoms  = np.zeros(n_ctx)
    for ai, (aname, col) in enumerate(zip(_S28_ACTION_NAMES, colors_a)):
        bars = ax.bar(range(n_ctx), act_mat[:, ai], bottom=bottoms,
                      color=col, alpha=0.85, label=aname, edgecolor='white')
        for bi, (b, v) in enumerate(zip(bottoms, act_mat[:, ai])):
            if v > 0.1:
                ax.text(bi, b + v/2, f'{v:.0%}',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        bottoms += act_mat[:, ai]
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=9)
    ax.set_ylabel('Action ratio')
    ax.set_title('文脈別行動分布\n期待: C0=食事↑, C1=食事↓')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S34_CFG

    print('=== Session 34: 追跡型捕食者 x Session 33環境 ===')
    print(f'grid={cfg.grid}x{cfg.grid}  hp_decay={_S34_HP_DECAY}  '
          f'food_dist={cfg.food_dist}  n_gen={_S34_N_GEN}')
    print(f'pursuit_probs={_S34_PURSUIT_PROBS}')
    print(f'pred_depletion={_S34_PRED_DEPLETION}  '
          f'pred_recovery={_S34_PRED_RECOVERY}')
    print()

    # pred_flag発生率を事前計測（進化前に環境レベルで確認）
    print('[事前確認] pred_flag発生率（ランダム行動、20ep）')
    pred_flag_rates = {}
    for pp in _S34_PURSUIT_PROBS:
        rate, n_steps = measure_pred_flag_rate(cfg, pp, seed=_S34_SEED)
        pred_flag_rates[pp] = rate
        print(f'  pp={pp}: pred_flag={rate:.1%}  ({n_steps}steps)')
    print()

    sweep_results = []

    for pp in _S34_PURSUIT_PROBS:
        print(f'\n{"="*60}')
        print(f'pursuit_prob={pp}  pred_flag率={pred_flag_rates[pp]:.1%}')
        print(f'{"="*60}')

        print('\n  [進化]')
        best, hist = _s34_evolve(cfg, pursuit_prob=pp, seed=_S34_SEED)
        ctx_data   = _s34_measure_context_dep(best, seed=_S34_SEED)
        mcd        = ctx_data['mean_cosine_dist']
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'mcd={mcd:.4f}')

        print('\n  [固定戦略確認]')
        strategy   = _s34_check_simple_strategy(cfg, pursuit_prob=pp,
                                                seed=_S34_SEED)
        best_fixed = max(strategy.values())
        best_fname = max(strategy, key=strategy.get)
        diff       = hist['gen_best_steps'][-1] - best_fixed
        print(f'  → 最良固定: {best_fname}={best_fixed:.0f}steps  '
              f'差={diff:+.1f}steps')

        sweep_results.append({
            'pursuit_prob':     pp,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'food':             hist['gen_food_count'][-1],
            'pred_hits':        hist['gen_pred_hits'][-1],
            'ctx_data':         ctx_data,
            'strategy':         strategy,
        })

    plot_sweep(sweep_results, pred_flag_rates)

    # ベスト条件（mcd最大）で文脈依存行動の詳細
    best_r = max(sweep_results, key=lambda r: r['mean_cosine_dist'])
    print(f'\n[Exp B] mcd最大: pp={best_r["pursuit_prob"]}  '
          f'mcd={best_r["mean_cosine_dist"]:.4f}')
    for ctx, res in zip(_S28_CONTEXTS, best_r['ctx_data']['context_results']):
        dom = _S28_ACTION_NAMES[int(np.argmax(res['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}] -> {dom} '
              f'({res["action_dist"][int(np.argmax(res["action_dist"]))]:.0%})')
    plot_context(best_r['ctx_data'], best_r['pursuit_prob'])

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 34 Summary ===')
    print()
    print(f'{"pp":>4}  {"pflag":>7}  {"food":>6}  {"steps":>7}  '
          f'{"fixed":>7}  {"diff":>6}  {"mcd":>8}  {"hits":>6}')
    for r in sweep_results:
        bf   = max(r['strategy'].values())
        diff = r['best_steps'] - bf
        print(f'{r["pursuit_prob"]:>4}  '
              f'{pred_flag_rates[r["pursuit_prob"]]:>7.1%}  '
              f'{r["food"]:>6.2f}  '
              f'{r["best_steps"]:>7.1f}  '
              f'{bf:>7.1f}  '
              f'{diff:>+6.1f}  '
              f'{r["mean_cosine_dist"]:>8.4f}  '
              f'{r["pred_hits"]:>6.2f}')

    print()
    print('--- 観察してから判断すること ---')
    print('1. pursuit_probが上がるとpred_flag率は上がるか？')
    print('   → 上がれば「捕食者に近づかれる経験」が増えている')
    print('2. pred_flag率が上がるとmcdは上がるか？')
    print('   → 上がれば捕食者フラグへの反応が学習されている')
    print('3. 期待パターン（C0=食事↑, C1=食事↓）は出るか？')
    print('4. 進化個体が固定戦略を上回るか？')
    print()
    print('Done.')
