"""Session 33: 食事範囲の緩和

Session 32の問題:
  「即食可能 & 捕食者隣接」= 総ステップの0.61%
  → トレードオフが発生する機会がほぼなく、文脈依存が進化できない

変更点（Session 32からの差分は1箇所）:
  食事の成立条件を「同一セル（距離=0）」から「距離≤food_dist」に緩和する。
  food_flagが立つ範囲内（≤food_dist）で action=4 を取れば食べられる。

  これにより:
    「知覚トレードオフ（both flags=6.7%）」が
    「行動トレードオフ」に直結する
    → 食料が見えている & 捕食者が近い → 食べるか逃げるか
    が実際の選択肢になる

  生態学的対応:
    動物が「食料の近くで食べる」のは自然な行動
    （同一セルまで行く必要はない）

固定条件:
  grid=8×8, food_bias=0.5（Session 30/31固定）
  hp_decay スイープ: [1, 3, 5, 8]（Session 32と同じ）
  → 「食事範囲緩和」の効果だけを見る

確認すること:
  1. food/epは増えるか？（緩和で食べやすくなるはず）
  2. pred_hits/epは増えるか？（食べに行くと捕食者リスクが上がるはず）
  3. 進化個体が固定戦略を上回るか？
  4. mcdは上がるか？

出力:
  images/session_33/results_s33_sweep.png
  images/session_33/results_s33_context.png
"""

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_28_predator import (
    _S28_N, _S28_OUT_START, _S28_OUT_END,
    _S28_TAU_S, _S28_TAU_I, _S28_TAU_O,
    _S28_DEPL_LO, _S28_DEPL_HI, _S28_DEPL_MUT_STD,
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
    _s31_pred_step, _s31_measure_context_dep,
)

# ── Session 33 定数 ────────────────────────────────────────────────────────────

_S33_SEED       = _S28_SEED
_S33_N_GEN      = _S28_N_GEN
_S33_PRED_SPEED = 2
_S33_N_CONTEXT  = 25
_S33_CONTEXT_T  = 100
_S33_N_TRIALS   = 20

_S33_GRID       = 8
_S33_FOOD_BIAS  = 0.5
_S33_HP_DECAYS  = [1, 3, 5, 8]

_S33_CFG        = WorldConfig.from_grid(_S33_GRID)


# ── エピソードランナー（食事範囲緩和版） ──────────────────────────────────────

def _s33_run_ep(cfg: WorldConfig, G, W, genome, rng,
                hp_decay: float,
                food_bias_prob: float = _S33_FOOD_BIAS,
                predator_speed: int   = _S33_PRED_SPEED,
                record_activity: bool = False):
    """Session 32からの変更点: 食事判定を距離=0 → 距離≤food_dist に緩和。"""
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

    steps = food = pred_hits = 0
    act_recs = [] if record_activity else None

    for step in range(cfg.max_steps):
        if hp <= 0:
            break

        if step % predator_speed == 0:
            pred_pos = _s31_pred_step(
                cfg, pred_pos, food_positions, food_avail, food_bias_prob, rng)

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
            # ── ここだけSession 32と違う ──────────────────────────────────────
            # 旧: row == food_positions[fi][0] and col == food_positions[fi][1]
            # 新: Manhattan距離 ≤ food_dist なら食べられる
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
            # ─────────────────────────────────────────────────────────────────

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


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s33_evolve(cfg: WorldConfig, hp_decay: float,
                seed: int = _S33_SEED, n_gen: int = _S33_N_GEN):
    rng = np.random.default_rng(seed + 33000 + int(hp_decay * 100))
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s33_run_ep(
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

def _s33_check_simple_strategy(cfg: WorldConfig, hp_decay: float,
                                seed: int = _S33_SEED,
                                n_trials: int = _S33_N_TRIALS):
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 33300 + trial * 10 + action_idx)
            center   = cfg.grid // 2
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s31_init_foods(cfg, rng, row, col)
            food_avail     = [True] * cfg.n_foods
            food_timer     = [0]   * cfg.n_foods
            pred_pos       = _s31_init_pred(cfg, rng, row, col)

            for step in range(cfg.max_steps):
                if hp <= 0:
                    break
                if step % _S33_PRED_SPEED == 0:
                    pred_pos = _s31_pred_step(
                        cfg, pred_pos, food_positions, food_avail,
                        _S33_FOOD_BIAS, rng)
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

def plot_sweep(sweep_results,
               fname='images/session_33/results_s33_sweep.png'):
    decays     = [r['hp_decay']          for r in sweep_results]
    mcds       = [r['mean_cosine_dist']  for r in sweep_results]
    steps      = [r['best_steps']        for r in sweep_results]
    foods      = [r['food']              for r in sweep_results]
    hits       = [r['pred_hits']         for r in sweep_results]
    best_fixed = [max(r['strategy'].values()) for r in sweep_results]
    best_fname = [max(r['strategy'], key=r['strategy'].get)
                  for r in sweep_results]
    diffs      = [s - f for s, f in zip(steps, best_fixed)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 33: 食事範囲緩和（distance<=food_dist）+ hp_decay スイープ\n'
        f'grid={_S33_GRID}x{_S33_GRID}  food_bias={_S33_FOOD_BIAS}  '
        f'food_dist={_S33_CFG.food_dist}  {_S33_N_GEN}世代  seed={_S33_SEED}',
        fontsize=12,
    )
    colors = ['steelblue', 'seagreen', 'tomato', 'darkorange']

    # Panel 1: food/ep【最重要】
    ax = axes[0][0]
    ax.bar(range(len(decays)), foods, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('Food / ep')
    ax.set_title('Food/ep [最重要]\n(Session 32と比べて増えているか)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(foods):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=11,
                fontweight='bold')

    # Panel 2: pred_hits/ep
    ax = axes[0][1]
    ax.bar(range(len(decays)), hits, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('Pred hits/ep\n(食べに行くとリスクが上がるはず)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(hits):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    # Panel 3: 進化 vs 固定戦略の差
    ax = axes[0][2]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(decays)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('Evolution - Fixed (steps)')
    ax.set_title('進化 - 固定戦略\n(正=進化が有効、文脈依存が機能)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (d, fn, bf) in enumerate(zip(diffs, best_fname, best_fixed)):
        ax.text(i, d + (0.3 if d >= 0 else -1.5),
                f'{d:+.1f}\n(vs {fn}={bf:.0f})',
                ha='center', fontsize=8)

    # Panel 4: mcd
    ax = axes[1][0]
    ax.plot(decays, mcds, 'o-', color='purple', linewidth=2, markersize=10)
    # Session 32との比較ライン
    s32_mcds = [0.0010, 0.0005, 0.0001, 0.0247]
    ax.plot(decays, s32_mcds, 's--', color='gray', linewidth=1.5,
            markersize=7, alpha=0.6, label='Session 32 (参照)')
    for d, m in zip(decays, mcds):
        ax.text(d, m + 0.0005, f'{m:.4f}', ha='center', fontsize=9)
    ax.set_xlabel('hp_decay')
    ax.set_ylabel('mean cosine dist')
    ax.set_title('文脈依存性（mcd）\n灰色=Session 32参照値')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: 生存ステップ比較
    ax = axes[1][1]
    x = np.arange(len(decays))
    w = 0.35
    ax.bar(x - w/2, steps,      width=w, color='steelblue', alpha=0.85,
           label='進化個体', edgecolor='white')
    ax.bar(x + w/2, best_fixed, width=w, color='tomato',    alpha=0.85,
           label='最良固定戦略', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('Mean steps / ep')
    ax.set_title('生存ステップ数（進化 vs 固定戦略）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (s, f) in enumerate(zip(steps, best_fixed)):
        ax.text(i - w/2, s + 0.5, f'{s:.0f}', ha='center', fontsize=8,
                color='steelblue')
        ax.text(i + w/2, f + 0.5, f'{f:.0f}', ha='center', fontsize=8,
                color='tomato')

    # Panel 6: Session 32との比較サマリー
    ax = axes[1][2]
    s32_foods = [0.40, 0.20, 0.00, 0.20]
    s32_hits  = [1.00, 0.00, 0.00, 0.00]
    lines = ['Session 32 vs 33 比較\n',
             f'{"decay":>6}  {"food32":>7}  {"food33":>7}  '
             f'{"hits32":>7}  {"hits33":>7}']
    for d, f32, f33, h32, h33 in zip(
            decays, s32_foods, foods, s32_hits, hits):
        fd = f33 - f32
        hd = h33 - h32
        lines.append(
            f'{d:>6}  {f32:>7.2f}  {f33:>7.2f}({fd:+.2f})  '
            f'{h32:>7.2f}  {h33:>7.2f}({hd:+.2f})')
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('Session 32との差分')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context(exp_b, hp_decay,
                 fname='images/session_33/results_s33_context.png'):
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 33: 文脈依存行動（food最大条件: hp_decay={hp_decay}）\n'
        f'grid={_S33_GRID}x{_S33_GRID}  food_dist={_S33_CFG.food_dist}  '
        f'mcd={exp_b["mean_cosine_dist"]:.4f}',
        fontsize=13,
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
    pairs = [(i, j) for i in range(n_ctx) for j in range(i+1, n_ctx)]
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
        ax.bar(range(n_ctx), act_mat[:, ai], bottom=bottoms,
               color=col, alpha=0.85, label=aname, edgecolor='white')
        bottoms += act_mat[:, ai]
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=9)
    ax.set_ylabel('Action ratio')
    ax.set_title('文脈別行動分布')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S33_CFG

    print('=== Session 33: 食事範囲緩和 + hp_decay スイープ ===')
    print(f'grid={_S33_GRID}x{_S33_GRID}  food_dist={cfg.food_dist}  '
          f'food_bias={_S33_FOOD_BIAS}')
    print(f'hp_start={cfg.hp_start}  food_value={_S28_FOOD_VALUE}  '
          f'max_steps={cfg.max_steps}')
    print(f'食事条件: Manhattan距離 <= {cfg.food_dist} で action=4 が成立')
    print(f'（Session 32: 距離=0 のみ → Session 33: 距離<={cfg.food_dist}）')
    print()

    sweep_results = []

    for hp_decay in _S33_HP_DECAYS:
        print(f'\n{"="*60}')
        print(f'hp_decay={hp_decay}  '
              f'(食料なし生存={int(cfg.hp_start/hp_decay)}steps, '
              f'{int(cfg.hp_start/hp_decay/cfg.max_steps*100)}%)')
        print(f'{"="*60}')

        print('\n  [進化]')
        best, hist = _s33_evolve(cfg, hp_decay=hp_decay, seed=_S33_SEED)
        ctx_data   = _s31_measure_context_dep(best, seed=_S33_SEED)
        mcd        = ctx_data['mean_cosine_dist']
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'mcd={mcd:.4f}')

        print('\n  [固定戦略確認]')
        strategy   = _s33_check_simple_strategy(cfg, hp_decay=hp_decay,
                                                seed=_S33_SEED)
        best_fixed = max(strategy.values())
        best_fname = max(strategy, key=strategy.get)
        diff       = hist['gen_best_steps'][-1] - best_fixed
        print(f'  → 最良固定: {best_fname}={best_fixed:.0f}steps  '
              f'差={diff:+.1f}steps')

        sweep_results.append({
            'hp_decay':         hp_decay,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'food':             hist['gen_food_count'][-1],
            'pred_hits':        hist['gen_pred_hits'][-1],
            'ctx_data':         ctx_data,
            'strategy':         strategy,
        })

    plot_sweep(sweep_results)

    # ベスト条件（food最大）で文脈依存行動の詳細
    best_r = max(sweep_results, key=lambda r: r['food'])
    print(f'\n[Exp B] food最大: hp_decay={best_r["hp_decay"]}  '
          f'food={best_r["food"]:.2f}  mcd={best_r["mean_cosine_dist"]:.4f}')
    for ctx, res in zip(_S28_CONTEXTS, best_r['ctx_data']['context_results']):
        dom = _S28_ACTION_NAMES[int(np.argmax(res['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}] -> {dom} '
              f'({res["action_dist"][int(np.argmax(res["action_dist"]))]:.0%})')
    plot_context(best_r['ctx_data'], best_r['hp_decay'])

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 33 Summary ===')
    print()
    print(f'{"decay":>6}  {"food":>6}  {"steps":>7}  '
          f'{"fixed":>7}  {"diff":>7}  {"mcd":>8}  {"hits":>6}')
    for r in sweep_results:
        bf   = max(r['strategy'].values())
        diff = r['best_steps'] - bf
        print(f'{r["hp_decay"]:>6}  '
              f'{r["food"]:>6.2f}  '
              f'{r["best_steps"]:>7.1f}  '
              f'{bf:>7.1f}  '
              f'{diff:>+7.1f}  '
              f'{r["mean_cosine_dist"]:>8.4f}  '
              f'{r["pred_hits"]:>6.2f}')

    print()
    print('--- 観察してから判断すること ---')
    print('1. food/epはSession 32より増えたか？')
    print('   → 食事範囲緩和の効果確認')
    print('2. pred_hits/epは増えたか？')
    print('   → 食べに行くリスクが実際に生じているか')
    print('3. 進化が固定戦略を上回るdecayはどこか？')
    print('4. mcdはSession 32より上がったか？')
    print()
    print('Done.')
