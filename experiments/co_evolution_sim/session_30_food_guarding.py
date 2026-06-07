"""Session 30: 食料周辺に捕食者が留まる環境

動機:
  Session 29bの根本問題 = 「食べないと死ぬ」と「捕食者を避けないと死ぬ」が
  同時に成立していない。「全方向に逃げ続ける」戦略で生き延びられる。

設計変更（Session 29からの差分）:
  捕食者の移動に「最寄りの食料に引き寄せられる」バイアスを追加する。
  food_bias_prob で確率的に最寄り食料へ1歩近づく。それ以外はランダムウォーク。

  これにより:
    食料の周辺 = 危険ゾーン（捕食者がいる）
    食料から離れた場所 = 安全（でもHPが減る）
    → HPが低くなったら危険を冒して食べに行く必要が生まれる
    → 「捕食者が近いか遠いか」を読む必要が環境的に保証される

  生態学的対応: 待ち伏せ型捕食者（ライオンが水場を狙う等）

パラメータ:
  food_bias_prob: 捕食者が食料方向へ向かう確率 [0.0=完全ランダム, 1.0=常に最寄り食料へ]
  スイープ: [0.0, 0.3, 0.5, 0.7] で文脈依存性の変化を確認

実験:
  A: food_bias_prob スイープ（50世代進化 × 4条件）
     → 文脈依存性（mean_cosine_dist）がどの条件で最大か
  B: ベスト条件での文脈依存行動の計測（4文脈 × 25サンプル）
  C: 「全方向同じ行動」戦略は本当に排除されているか確認
     → food_bias=0 vs best での「単純戦略生存率」を比較

出力:
  images/session_30/results_s30_sweep.png
  images/session_30/results_s30_context.png
  images/session_30/results_s30_strategy_check.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import _N_PROP, _K, _INIT_W, _LR
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_28_predator import (
    _S28_N, _S28_INP_START, _S28_INP_END,
    _S28_OUT_START, _S28_OUT_END, _S28_INT_START, _S28_INT_END,
    _S28_DENSITY, _S28_MUT_STD, _S28_EDGE_CHNG,
    _S28_HEBB_LR, _S28_HEBB_DECAY,
    _S28_TAU_S, _S28_TAU_I, _S28_TAU_O,
    _S28_DEPL_LO, _S28_DEPL_HI, _S28_DEPL_MUT_STD,
    _S28_GRID, _S28_HP_START, _S28_HP_MAX, _S28_HP_DECAY,
    _S28_FOOD_VALUE, _S28_FOOD_RESPAWN, _S28_MAX_STEPS,
    _S28_N_FOODS, _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_FOOD_DIST, _S28_PRED_DIST,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_MR, _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_CONTEXTS, _S28_ACTION_NAMES,
    _s28_build_graph, _s28_get_W, _s28_propagate, _s28_mutate_graph,
    _s28_hebb, _s28_make_tau_arr, _s28_make_genome, _s28_mutate_genome,
    _s28_inp5, _s28_init_foods, _s28_init_pred, _s28_cosine_dist,
    _s28_measure_context,
)

# ── Session 30 定数 ────────────────────────────────────────────────────────────

_S30_SEED            = _S28_SEED
_S30_N_GEN           = _S28_N_GEN        # 50世代
_S30_PRED_SPEED      = 2                 # Session 29と同じ
_S30_N_CONTEXT       = 25               # サンプル数/文脈
_S30_CONTEXT_T       = 100              # 計測ステップ数

# food_bias_prob スイープ
_S30_FOOD_BIAS_PROBS = [0.0, 0.3, 0.5, 0.7]


# ── 捕食者移動：食料バイアスあり ──────────────────────────────────────────────

def _s30_pred_step(pred_pos, food_positions, food_avail, food_bias_prob, rng):
    """捕食者を1ステップ移動する。

    food_bias_prob の確率で最寄りの食料（利用可能なもの）に1歩近づく。
    それ以外はランダムウォーク。

    Returns new_pos (list[int, int]).
    """
    pr, pc = pred_pos

    # 利用可能な食料の中で最寄りを探す
    avail_foods = [
        fp for fp, av in zip(food_positions, food_avail) if av
    ]

    move_toward_food = (
        len(avail_foods) > 0
        and rng.random() < food_bias_prob
    )

    if move_toward_food:
        # Manhattan距離で最寄りの食料を選ぶ
        nearest = min(avail_foods,
                      key=lambda fp: abs(fp[0] - pr) + abs(fp[1] - pc))
        fr, fc = nearest
        dr = int(np.sign(fr - pr))
        dc = int(np.sign(fc - pc))
        # 行方向か列方向かランダムに選んで1歩
        if dr == 0 and dc == 0:
            move_toward_food = False  # 同じセルならランダムへ
        elif dr == 0:
            pc = int(np.clip(pc + dc, 0, _S28_GRID - 1))
        elif dc == 0:
            pr = int(np.clip(pr + dr, 0, _S28_GRID - 1))
        else:
            if rng.random() < 0.5:
                pr = int(np.clip(pr + dr, 0, _S28_GRID - 1))
            else:
                pc = int(np.clip(pc + dc, 0, _S28_GRID - 1))

    if not move_toward_food:
        d = int(rng.integers(0, 4))
        if d == 0:   pr = max(0, pr - 1)
        elif d == 1: pr = min(_S28_GRID - 1, pr + 1)
        elif d == 2: pc = max(0, pc - 1)
        else:        pc = min(_S28_GRID - 1, pc + 1)

    return [pr, pc]


# ── エピソードランナー ─────────────────────────────────────────────────────────

def _s30_run_ep(G, W, genome, rng,
                food_bias_prob=0.5,
                predator_speed=_S30_PRED_SPEED,
                record_activity=False):
    """食料バイアス付き捕食者のエピソード。

    Session 28/29のエピソードランナーと同じ構造。
    捕食者移動だけ _s30_pred_step に差し替え。
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

    pred_pos  = _s28_init_pred(rng)

    steps     = 0
    food      = 0
    pred_hits = 0
    act_recs  = [] if record_activity else None

    for step in range(_S28_MAX_STEPS):
        if hp <= 0:
            break

        # 捕食者移動（食料バイアスあり）
        if step % predator_speed == 0:
            pred_pos = _s30_pred_step(
                pred_pos, food_positions, food_avail, food_bias_prob, rng)

        # 捕食者衝突
        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            pred_hits += 1

        # 入力計算
        inp5 = _s28_inp5(row, col, hp, food_positions, food_avail, pred_pos)

        # ネットワーク伝播
        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())

        hp -= _S28_HP_DECAY
        hp -= metabolic_rate * float(np.sum(eff))

        # 行動
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
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources),
                        0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    food += 1
                    break

        steps = step + 1

        for fi in range(_S28_N_FOODS):
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
        'steps':     steps,
        'food':      food,
        'pred_hits': pred_hits,
        'act_recs':  act_recs,
    }


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s30_evolve(food_bias_prob=0.5, seed=_S30_SEED, n_gen=_S30_N_GEN):
    """food_bias_prob 固定で50世代進化。"""
    rng = np.random.default_rng(seed + 30000 + int(food_bias_prob * 100))
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits',
        'gen_mean_active',
    )}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s30_run_ep(
                    g['G'], g['W'], g, ep_rng,
                    food_bias_prob=food_bias_prob,
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

def _s30_measure_context_dep(genome, food_bias_prob, seed):
    """4文脈でのmean_cosine_distを計測。Session 29のExp Bと同じ構造。"""
    G_copy = genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 30200)

    results = []
    for ctx in _S28_CONTEXTS:
        samples = []
        for _ in range(_S30_N_CONTEXT):
            r = _s28_measure_context(
                G_copy, W_copy, genome, ctx, rng, T=_S30_CONTEXT_T)
            samples.append(r['mean_output'])
        mean_out     = np.mean(samples, axis=0)
        action_count = np.zeros(5)
        for s in samples:
            action_count[int(np.argmax(s))] += 1
        results.append({
            'mean_output': mean_out,
            'action_dist': action_count / _S30_N_CONTEXT,
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

    # p値（ペアごとのt検定）
    p_values = {}
    for (i, j) in pairs:
        si = results[i]['output_samples'].max(axis=1)
        sj = results[j]['output_samples'].max(axis=1)
        _, p = scipy_stats.ttest_ind(si, sj)
        p_values[(i, j)] = float(p)

    return {
        'context_results':  results,
        'cosine_matrix':    cos_mat,
        'mean_cosine_dist': mcd,
        'p_values':         p_values,
    }


# ── Exp C: 単純戦略の排除確認 ─────────────────────────────────────────────────

def _s30_check_simple_strategy(food_bias_prob, seed, n_trials=20):
    """「常に同じ方向に動き続ける」固定戦略の生存ステップ数を計測。

    文脈依存が必要かどうかの環境レベルの確認。
    固定戦略（常に北/南/西/東/食事）を持つ疑似ゲノムで評価する。

    Returns dict: action_name → mean_steps
    """
    # 固定戦略の疑似ゲノム
    # W行列を細工して特定の出力ノードだけ常に高活性になるようにする
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 30300 + trial * 10 + action_idx)

            # 固定行動エピソード（ネットワーク不使用）
            row, col = 2, 2
            hp = float(_S28_HP_START)
            food_positions = _s28_init_foods(rng)
            food_avail     = [True] * _S28_N_FOODS
            food_timer     = [0]   * _S28_N_FOODS
            pred_pos       = _s28_init_pred(rng)

            for step in range(_S28_MAX_STEPS):
                if hp <= 0:
                    break
                if step % _S30_PRED_SPEED == 0:
                    pred_pos = _s30_pred_step(
                        pred_pos, food_positions, food_avail, food_bias_prob, rng)
                if pred_pos[0] == row and pred_pos[1] == col:
                    hp -= _S28_PRED_DAMAGE
                hp -= _S28_HP_DECAY

                action = action_idx
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
                            food_avail[fi] = False
                            food_timer[fi] = 0
                            break

                for fi in range(_S28_N_FOODS):
                    if not food_avail[fi]:
                        food_timer[fi] += 1
                        if food_timer[fi] >= _S28_FOOD_RESPAWN:
                            food_avail[fi] = True
                            food_timer[fi] = 0

            total_steps.append(step + 1)

        results[action_name] = float(np.mean(total_steps))
        print(f'  固定戦略「{action_name}」: mean_steps={results[action_name]:.1f}')

    return results


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_sweep(sweep, fname='images/session_30/results_s30_sweep.png'):
    """Exp A: food_bias_prob スイープの結果を可視化。"""
    probs  = [r['food_bias_prob']     for r in sweep]
    mcds   = [r['mean_cosine_dist']   for r in sweep]
    steps  = [r['best_steps']         for r in sweep]
    foods  = [r['food']               for r in sweep]
    hits   = [r['pred_hits']          for r in sweep]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Session 30 Exp A: food_bias_prob スイープ\n'
        f'{_S30_N_GEN}世代 × {_S28_N_AGENTS}個体 × {_S28_N_EP}ep/個体',
        fontsize=13,
    )
    colors = ['steelblue', 'seagreen', 'tomato', 'darkorange']

    ax = axes[0][0]
    bars = ax.bar(range(len(probs)), mcds, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'bias={p}' for p in probs])
    ax.set_ylabel('mean cosine dist')
    ax.set_title('文脈依存性\n（高い = 文脈間の出力差が大きい）')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mcds):
        ax.text(i, v + 0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[0][1]
    ax.bar(range(len(probs)), steps, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'bias={p}' for p in probs])
    ax.set_ylabel('Best steps / ep')
    ax.set_title('生存ステップ数')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(steps):
        ax.text(i, v + 0.5, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1][0]
    ax.bar(range(len(probs)), foods, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'bias={p}' for p in probs])
    ax.set_ylabel('Food / ep')
    ax.set_title('食料獲得数\n（増加 = 食べに行く必要が生まれている）')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(foods):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1][1]
    ax.bar(range(len(probs)), hits, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f'bias={p}' for p in probs])
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数\n（適切なら低下するはず）')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(hits):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context(exp_b, food_bias_prob,
                 fname='images/session_30/results_s30_context.png'):
    """Exp B: ベスト条件での文脈依存行動を可視化。"""
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 30 Exp B: 文脈依存行動の計測\n'
        f'food_bias_prob={food_bias_prob}  '
        f'mean cosine dist={exp_b["mean_cosine_dist"]:.4f}',
        fontsize=13,
    )

    # 出力活動ヒートマップ
    ax = axes[0]
    out_mat = np.array([r['mean_output'] for r in results])
    im = ax.imshow(out_mat, cmap='hot', vmin=0, vmax=out_mat.max() * 1.1,
                   aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels(
        [f'node{i+5}\n({a})' for i, a in enumerate(_S28_ACTION_NAMES)],
        fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=9)
    ax.set_title('出力ノード平均活動')
    for i in range(n_ctx):
        for j in range(5):
            ax.text(j, i, f'{out_mat[i, j]:.3f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if out_mat[i, j] > out_mat.max() * 0.6
                    else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # コサイン距離行列
    ax = axes[1]
    vmax = max(cos_mat.max(), 0.01)
    im2 = ax.imshow(cos_mat, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
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
                    color='white' if cos_mat[i, j] > vmax * 0.6 else 'black')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # 行動分布積み上げ棒グラフ
    ax = axes[2]
    act_mat = np.array([r['action_dist'] for r in results])
    colors_a = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    bottoms  = np.zeros(n_ctx)
    for ai, (aname, col) in enumerate(zip(_S28_ACTION_NAMES, colors_a)):
        ax.bar(range(n_ctx), act_mat[:, ai], bottom=bottoms,
               color=col, alpha=0.85, label=aname, edgecolor='white')
        bottoms += act_mat[:, ai]
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=9)
    ax.set_ylabel('行動割合')
    ax.set_title('文脈別行動分布')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_strategy_check(strategy_results_by_bias,
                        fname='images/session_30/results_s30_strategy_check.png'):
    """Exp C: 固定戦略の生存ステップ数を food_bias_prob 別に比較。"""
    biases  = sorted(strategy_results_by_bias.keys())
    actions = _S28_ACTION_NAMES
    n_bias  = len(biases)
    n_act   = len(actions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 30 Exp C: 固定戦略の生存ステップ数\n'
        '（food_bias_prob が上がると単純戦略の優位性が崩れるか？）',
        fontsize=13,
    )

    # 左: バイアス別の固定戦略生存ステップ
    ax = axes[0]
    x = np.arange(n_bias)
    width = 0.15
    colors_a = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    for ai, (aname, col) in enumerate(zip(actions, colors_a)):
        vals = [strategy_results_by_bias[b][aname] for b in biases]
        ax.bar(x + ai * width, vals, width, label=aname,
               color=col, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'bias={b}' for b in biases])
    ax.set_ylabel('Mean steps')
    ax.set_title('固定戦略の生存ステップ数\n（低いほど「単純戦略では生き延びられない」）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 右: 最良固定戦略 vs bias=0 の比較
    ax = axes[1]
    best_steps = [max(strategy_results_by_bias[b].values()) for b in biases]
    worst_steps = [min(strategy_results_by_bias[b].values()) for b in biases]
    ax.plot(biases, best_steps,  'o-', color='tomato',    linewidth=2,
            label='最良固定戦略', markersize=8)
    ax.plot(biases, worst_steps, 's--', color='steelblue', linewidth=2,
            label='最悪固定戦略', markersize=8)
    ax.set_xlabel('food_bias_prob')
    ax.set_ylabel('Mean steps')
    ax.set_title('food_bias_prob と単純戦略生存率\n'
                 '（bias増加で最良固定戦略の生存が下がれば環境設計成功）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for b, s in zip(biases, best_steps):
        ax.text(b, s + 0.5, f'{s:.0f}', ha='center', fontsize=9, color='tomato')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 30: 食料周辺に捕食者が留まる環境 ===')
    print(f'food_bias_probs: {_S30_FOOD_BIAS_PROBS}')
    print(f'グリッド: {_S28_GRID}×{_S28_GRID}  '
          f'食料数: {_S28_N_FOODS}  最大ステップ: {_S28_MAX_STEPS}')
    print()

    # ── Exp A: food_bias_prob スイープ ────────────────────────────────────────
    print('[Exp A] food_bias_prob スイープ')
    sweep = []
    for fb in _S30_FOOD_BIAS_PROBS:
        print(f'\n  food_bias_prob={fb}:')
        best, hist = _s30_evolve(food_bias_prob=fb, seed=_S30_SEED)
        ctx_data   = _s30_measure_context_dep(best, fb, seed=_S30_SEED)
        mcd        = ctx_data['mean_cosine_dist']
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'mcd={mcd:.4f}')
        sweep.append({
            'food_bias_prob':   fb,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'food':             hist['gen_food_count'][-1],
            'pred_hits':        hist['gen_pred_hits'][-1],
            'ctx_data':         ctx_data,
        })
    plot_sweep(sweep)

    # ── Exp B: ベスト条件で文脈依存行動の詳細計測 ─────────────────────────────
    best_r = max(sweep, key=lambda r: r['mean_cosine_dist'])
    print(f'\n[Exp B] ベスト条件: food_bias_prob={best_r["food_bias_prob"]}  '
          f'mcd={best_r["mean_cosine_dist"]:.4f}')
    for ctx, res in zip(_S28_CONTEXTS, best_r['ctx_data']['context_results']):
        dom = _S28_ACTION_NAMES[int(np.argmax(res['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}] → {dom} '
              f'({res["action_dist"][int(np.argmax(res["action_dist"]))]:.0%})')
    plot_context(best_r['ctx_data'], best_r['food_bias_prob'])

    # ── Exp C: 固定戦略の排除確認 ─────────────────────────────────────────────
    print('\n[Exp C] 固定戦略の生存ステップ確認')
    strategy_results = {}
    for fb in _S30_FOOD_BIAS_PROBS:
        print(f'\n  food_bias_prob={fb}:')
        strategy_results[fb] = _s30_check_simple_strategy(
            fb, seed=_S30_SEED, n_trials=20)
    plot_strategy_check(strategy_results)

    # ── サマリー ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 30 Summary ===')
    print()
    print('food_bias_prob別 mean_cosine_dist:')
    for r in sweep:
        mark = '★' if r == best_r else '  '
        print(f'  {mark}bias={r["food_bias_prob"]}: '
              f'mcd={r["mean_cosine_dist"]:.4f}  '
              f'steps={r["best_steps"]:.0f}  '
              f'food={r["food"]:.2f}/ep  '
              f'hits={r["pred_hits"]:.2f}/ep')

    print()
    print('固定戦略の生存（best固定戦略 / bias別）:')
    for fb in _S30_FOOD_BIAS_PROBS:
        best_fixed = max(strategy_results[fb].values())
        best_name  = max(strategy_results[fb], key=strategy_results[fb].get)
        print(f'  bias={fb}: 最良={best_name}で{best_fixed:.0f}steps')

    print()
    print('--- 観察してから判断すること ---')
    print('1. food_biasが上がると固定戦略の生存は下がるか？')
    print('   → 下がれば「文脈を読まないと損」な環境になっている')
    print('2. food_biasが上がるとmcdは上がるか？')
    print('   → 上がれば環境圧力と文脈依存性が連動している')
    print('3. food_countは増えているか？')
    print('   → 増えていれば「食べに行く必要」が実際に生まれている')
    print()
    print('Done.')
