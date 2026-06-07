"""Session 32: hp_decay スイープ

Session 31の問題:
  food=0.00/ep が多い → 食料を一度も取らずに生き延びる戦略が選択されている。
  hp_start=256（grid=8）、hp_decay=1/step の場合、
  食料なしで256steps生き延びられる = エピソード(1280steps)の20%。
  → 「食べないと死ぬ」圧力が弱すぎる。

変更点:
  grid=8×8 に固定（Session 31でmcdが唯一上がった）。
  food_bias_prob=0.5 に固定（Session 30 best）。
  hp_decay だけスイープ: [1, 3, 5, 8]

  hp_decayとfood_valueの関係：
    hp_decay=1: 食料なし生存 = 256steps (エピソードの20%)
    hp_decay=3: 食料なし生存 = 85steps  (エピソードの 7%)
    hp_decay=5: 食料なし生存 = 51steps  (エピソードの 4%)
    hp_decay=8: 食料なし生存 = 32steps  (エピソードの 3%)
    ※ food_value=30 は固定なので、食料1個で回復できるステップ数も変わる

確認すること（優先順位順）:
  1. food/epが増えているか？（「食べないと死ぬ」圧力が機能しているか）
  2. 進化個体が固定戦略を上回るか？（文脈依存が有効な環境か）
  3. mcdが上がるか？（ネットワークレベルで文脈依存が出ているか）

出力:
  images/session_32/results_s32_decay_sweep.png
  images/session_32/results_s32_context.png
  images/session_32/results_s32_strategy_check.png
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

# ── Session 32 定数 ────────────────────────────────────────────────────────────

_S32_SEED        = _S28_SEED
_S32_N_GEN       = _S28_N_GEN
_S32_PRED_SPEED  = 2
_S32_N_CONTEXT   = 25
_S32_CONTEXT_T   = 100
_S32_N_TRIALS    = 20

_S32_GRID        = 8
_S32_FOOD_BIAS   = 0.5           # Session 30/31 固定
_S32_HP_DECAYS   = [1, 3, 5, 8]  # スイープ対象

# grid=8 の固定パラメータ（hp_decay以外）
_S32_CFG_BASE    = WorldConfig.from_grid(_S32_GRID)


def _decay_summary(hp_decay: float, cfg: WorldConfig) -> str:
    """hp_decayの設定とその意味を表示する文字列。"""
    survive_steps = int(cfg.hp_start / hp_decay)
    survive_frac  = survive_steps / cfg.max_steps
    food_equiv    = _S28_FOOD_VALUE / hp_decay  # 食料1個で何steps生き延びられるか
    return (f'hp_decay={hp_decay}  '
            f'食料なし生存={survive_steps}steps({survive_frac:.0%})  '
            f'food_value={_S28_FOOD_VALUE}→+{food_equiv:.0f}steps分')


# ── エピソードランナー（hp_decayを引数で受け取る） ────────────────────────────

def _s32_run_ep(cfg: WorldConfig, G, W, genome, rng,
                hp_decay: float,
                food_bias_prob: float = _S32_FOOD_BIAS,
                predator_speed: int   = _S32_PRED_SPEED,
                record_activity: bool = False):
    """hp_decayをパラメータとして受け取るエピソードランナー。
    session_31の_s31_run_epとの差分はhp_decayだけ。
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

        # hp_decayをパラメータで制御
        hp -= hp_decay
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(cfg.grid - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(cfg.grid - 1, col + 1)
        elif action == 4:
            for fi in range(cfg.n_foods):
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

def _s32_evolve(cfg: WorldConfig, hp_decay: float,
                food_bias_prob: float = _S32_FOOD_BIAS,
                seed: int = _S32_SEED, n_gen: int = _S32_N_GEN):
    """hp_decay固定で進化。"""
    rng = np.random.default_rng(
        seed + 32000 + int(hp_decay * 100))
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s32_run_ep(
                    cfg, g['G'], g['W'], g, ep_rng,
                    hp_decay=hp_decay,
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


# ── 固定戦略の確認 ────────────────────────────────────────────────────────────

def _s32_check_simple_strategy(cfg: WorldConfig, hp_decay: float,
                                food_bias_prob: float = _S32_FOOD_BIAS,
                                seed: int = _S32_SEED,
                                n_trials: int = _S32_N_TRIALS):
    """固定戦略の生存ステップ数を計測。"""
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 32300 + trial * 10 + action_idx)
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
                if step % _S32_PRED_SPEED == 0:
                    pred_pos = _s31_pred_step(
                        cfg, pred_pos, food_positions, food_avail,
                        food_bias_prob, rng)
                if pred_pos[0] == row and pred_pos[1] == col:
                    hp -= _S28_PRED_DAMAGE
                hp -= hp_decay  # ここだけ変わる

                if action_idx == 0:   row = max(0, row - 1)
                elif action_idx == 1: row = min(cfg.grid - 1, row + 1)
                elif action_idx == 2: col = max(0, col - 1)
                elif action_idx == 3: col = min(cfg.grid - 1, col + 1)
                elif action_idx == 4:
                    for fi in range(cfg.n_foods):
                        if (food_avail[fi]
                                and row == food_positions[fi][0]
                                and col == food_positions[fi][1]):
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

def plot_decay_sweep(sweep_results,
                     fname='images/session_32/results_s32_decay_sweep.png'):
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
        f'Session 32: hp_decay スイープ\n'
        f'grid={_S32_GRID}×{_S32_GRID}  food_bias={_S32_FOOD_BIAS}  '
        f'{_S32_N_GEN}世代  seed={_S32_SEED}',
        fontsize=13,
    )
    colors = ['steelblue', 'seagreen', 'tomato', 'darkorange']

    # Panel 1: food/ep（最重要 — 食べる必要が生まれているか）
    ax = axes[0][0]
    ax.bar(range(len(decays)), foods, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('Food / ep')
    ax.set_title('食料獲得数【最重要】\n（増加＝「食べないと死ぬ」圧力が機能）')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(foods):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    # Panel 2: 進化 vs 固定戦略
    ax = axes[0][1]
    x = np.arange(len(decays))
    w = 0.35
    ax.bar(x - w/2, steps,      width=w, color='steelblue', alpha=0.85,
           label='進化個体', edgecolor='white')
    ax.bar(x + w/2, best_fixed, width=w, color='tomato',    alpha=0.85,
           label='最良固定戦略', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('Mean steps / ep')
    ax.set_title('生存ステップ数\n（進化 vs 固定戦略）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (s, f, fn) in enumerate(zip(steps, best_fixed, best_fname)):
        ax.text(i - w/2, s + 1, f'{s:.0f}', ha='center', fontsize=8,
                color='steelblue')
        ax.text(i + w/2, f + 1, f'{f:.0f}\n({fn})', ha='center', fontsize=7,
                color='tomato')

    # Panel 3: 進化 − 固定戦略の差
    ax = axes[0][2]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(decays)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels([f'decay={d}' for d in decays])
    ax.set_ylabel('進化 − 固定戦略 (steps)')
    ax.set_title('進化個体の優位性\n（正＝文脈依存が有効に機能）')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + (0.5 if d >= 0 else -1.5),
                f'{d:+.0f}', ha='center', fontsize=10)

    # Panel 4: mcd
    ax = axes[1][0]
    ax.plot(decays, mcds, 'o-', color='purple', linewidth=2, markersize=10)
    for d, m in zip(decays, mcds):
        ax.text(d, m + 0.0003, f'{m:.4f}', ha='center', fontsize=9)
    ax.set_xlabel('hp_decay')
    ax.set_ylabel('mean cosine dist')
    ax.set_title('文脈依存性（mcd）\n（上がればネットワーク内部でも分化）')
    ax.grid(True, alpha=0.3)

    # Panel 5: pred_hits
    ax = axes[1][1]
    ax.plot(decays, hits, '^-', color='darkorange', linewidth=2, markersize=10)
    for d, h in zip(decays, hits):
        ax.text(d, h + 0.02, f'{h:.2f}', ha='center', fontsize=9)
    ax.set_xlabel('hp_decay')
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数\n（食べに行くと捕食者に当たるはず）')
    ax.grid(True, alpha=0.3)

    # Panel 6: 設計サマリー
    ax = axes[1][2]
    cfg = _S32_CFG_BASE
    lines = [f'grid={cfg.grid}×{cfg.grid}  food_bias={_S32_FOOD_BIAS}\n']
    for d in decays:
        survive = int(cfg.hp_start / d)
        frac    = survive / cfg.max_steps
        fval    = _S28_FOOD_VALUE / d
        lines.append(
            f'decay={d}: 食料なし={survive}steps({frac:.0%})  '
            f'food+{fval:.0f}steps分'
        )
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('設計パラメータ')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context(exp_b, hp_decay,
                 fname='images/session_32/results_s32_context.png'):
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 32 Exp B: 文脈依存行動（ベスト hp_decay={hp_decay}）\n'
        f'grid={_S32_GRID}×{_S32_GRID}  '
        f'mean cosine dist={exp_b["mean_cosine_dist"]:.4f}',
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
                    color='white' if out_mat[i,j] > vmax*0.6 else 'black')
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
                    color='white' if cos_mat[i,j] > vmax2*0.6 else 'black')
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
    ax.set_ylabel('行動割合')
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
    cfg = _S32_CFG_BASE

    print('=== Session 32: hp_decay スイープ ===')
    print(f'grid={_S32_GRID}×{_S32_GRID}  food_bias={_S32_FOOD_BIAS}  '
          f'n_gen={_S32_N_GEN}  seed={_S32_SEED}')
    print(f'hp_start={cfg.hp_start}  food_value={_S28_FOOD_VALUE}  '
          f'max_steps={cfg.max_steps}')
    print()
    print('hp_decayの意味:')
    for d in _S32_HP_DECAYS:
        print(f'  {_decay_summary(d, cfg)}')
    print()

    sweep_results = []

    for hp_decay in _S32_HP_DECAYS:
        print(f'\n{"="*60}')
        print(f'hp_decay={hp_decay}')
        print(f'{"="*60}')

        print('\n  [進化]')
        best, hist = _s32_evolve(cfg, hp_decay=hp_decay, seed=_S32_SEED)
        ctx_data   = _s31_measure_context_dep(best, seed=_S32_SEED)
        mcd        = ctx_data['mean_cosine_dist']
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'mcd={mcd:.4f}')

        print('\n  [固定戦略確認]')
        strategy   = _s32_check_simple_strategy(cfg, hp_decay=hp_decay,
                                                seed=_S32_SEED)
        best_fixed = max(strategy.values())
        best_fname = max(strategy, key=strategy.get)
        diff       = hist['gen_best_steps'][-1] - best_fixed
        print(f'  → 最良固定: {best_fname}={best_fixed:.0f}steps  '
              f'差={diff:+.0f}steps')

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

    plot_decay_sweep(sweep_results)

    # ベスト条件で文脈依存行動の詳細
    # mcdよりもfoodとdiffを優先してベストを選ぶ
    best_r = max(sweep_results, key=lambda r: r['food'])
    print(f'\n[Exp B] food最大のベスト条件: hp_decay={best_r["hp_decay"]}  '
          f'food={best_r["food"]:.2f}  mcd={best_r["mean_cosine_dist"]:.4f}')
    for ctx, res in zip(_S28_CONTEXTS, best_r['ctx_data']['context_results']):
        dom = _S28_ACTION_NAMES[int(np.argmax(res['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}] → {dom} '
              f'({res["action_dist"][int(np.argmax(res["action_dist"]))]:.0%})')
    plot_context(best_r['ctx_data'], best_r['hp_decay'])

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 32 Summary ===')
    print()
    print(f'{"decay":>6}  {"food":>6}  {"steps":>7}  '
          f'{"fixed":>7}  {"diff":>6}  {"mcd":>8}  {"hits":>6}')
    for r in sweep_results:
        bf   = max(r['strategy'].values())
        diff = r['best_steps'] - bf
        print(f'{r["hp_decay"]:>6}  '
              f'{r["food"]:>6.2f}  '
              f'{r["best_steps"]:>7.1f}  '
              f'{bf:>7.1f}  '
              f'{diff:>+6.1f}  '
              f'{r["mean_cosine_dist"]:>8.4f}  '
              f'{r["pred_hits"]:>6.2f}')

    print()
    print('--- 観察してから判断すること ---')
    print('1. food/epは増えているか？')
    print('   → 増えていれば「食べないと死ぬ」圧力が機能している')
    print('2. 進化個体が固定戦略を上回るdecayはどこか？')
    print('   → そのdecayが「文脈依存が有効になる最小圧力」')
    print('3. foodが増えてもmcdは上がらないか？')
    print('   → 上がらなければ「食べに行く行動」は出ても'
          '「文脈を読む」は別の問題')
    print()
    print('Done.')
