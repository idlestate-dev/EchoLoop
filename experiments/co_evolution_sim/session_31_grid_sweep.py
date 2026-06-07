"""Session 31: グリッドサイズスイープ

Session 30の問題:
  5×5グリッドが小さすぎて「食料=危険ゾーン」と「安全ゾーン」が
  空間的に分離できなかった。固定戦略が排除されなかった。

変更点:
  グリッドサイズ（5/8/10/12）をスイープする。
  グリッドに比例して変えるパラメータ:
    MAX_STEPS  = grid^2 * 20   （面積に比例）
    HP_START   = grid^2 * 4    （長いエピソードに対応）
    FOOD_DIST  = grid // 3     （検知範囲を相対的に維持）
    PRED_DIST  = max(1, grid // 5)
    N_FOODS    = max(2, grid // 3)  （密度を相対的に維持）
  変えないパラメータ:
    FOOD_VALUE, HP_DECAY, PRED_DAMAGE, PRED_SPEED, tau値

  food_bias_prob は Session 30 best の 0.5 に固定。
  （グリッドサイズの効果だけを見る）

実験:
  A: グリッドサイズスイープ（各サイズで50世代進化）
     → mean_cosine_dist と固定戦略生存ステップを比較
  B: ベストサイズでの文脈依存行動の詳細計測（4文脈）
  C: 固定戦略の排除確認（各グリッドサイズで）

出力:
  images/session_31/results_s31_grid_sweep.png
  images/session_31/results_s31_context.png
  images/session_31/results_s31_strategy_check.png
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import _N_PROP, _K, _INIT_W, _LR
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_28_predator import (
    _S28_N,
    _S28_OUT_START, _S28_OUT_END,
    _S28_DENSITY, _S28_MUT_STD, _S28_EDGE_CHNG,
    _S28_TAU_S, _S28_TAU_I, _S28_TAU_O,
    _S28_DEPL_LO, _S28_DEPL_HI, _S28_DEPL_MUT_STD,
    _S28_HP_MAX, _S28_HP_DECAY,
    _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_MR, _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_CONTEXTS, _S28_ACTION_NAMES,
    _s28_get_W, _s28_propagate,
    _s28_hebb, _s28_make_tau_arr, _s28_make_genome, _s28_mutate_genome,
    _s28_cosine_dist, _s28_measure_context,
)

# ── Session 31 定数 ────────────────────────────────────────────────────────────

_S31_SEED         = _S28_SEED
_S31_N_GEN        = _S28_N_GEN   # 50世代
_S31_PRED_SPEED   = 2
_S31_N_CONTEXT    = 25
_S31_CONTEXT_T    = 100
_S31_FOOD_BIAS    = 0.5           # Session 30 best に固定
_S31_N_TRIALS     = 20            # 固定戦略確認のtrials数

_S31_GRID_SIZES   = [5, 8, 10, 12]


# ── WorldConfig: グリッドサイズに依存するパラメータを一箇所に集約 ──────────────

@dataclass
class WorldConfig:
    grid:       int
    max_steps:  int
    hp_start:   int
    n_foods:    int
    food_dist:  int
    pred_dist:  int

    @classmethod
    def from_grid(cls, grid: int) -> 'WorldConfig':
        return cls(
            grid      = grid,
            max_steps = grid * grid * 20,
            hp_start  = grid * grid * 4,
            n_foods   = max(2, grid // 3),
            food_dist = max(1, grid // 3),
            pred_dist = max(1, grid // 5),
        )

    def summary(self) -> str:
        return (f'grid={self.grid}  max_steps={self.max_steps}  '
                f'hp_start={self.hp_start}  n_foods={self.n_foods}  '
                f'food_dist={self.food_dist}  pred_dist={self.pred_dist}')


# ── 世界ヘルパー（WorldConfigを引数に取る版） ──────────────────────────────────

def _s31_init_foods(cfg: WorldConfig, rng, agent_row=None, agent_col=None):
    """食料をランダム配置。エージェント位置を避ける。"""
    center = cfg.grid // 2
    ar = agent_row if agent_row is not None else center
    ac = agent_col if agent_col is not None else center
    candidates = [
        (r, c) for r in range(cfg.grid) for c in range(cfg.grid)
        if not (r == ar and c == ac)
    ]
    idxs = rng.choice(len(candidates), size=cfg.n_foods, replace=False)
    return [list(candidates[i]) for i in idxs]


def _s31_init_pred(cfg: WorldConfig, rng, agent_row=None, agent_col=None,
                   min_dist=2):
    """捕食者をランダム配置。エージェントからmin_dist以上離す。"""
    center = cfg.grid // 2
    ar = agent_row if agent_row is not None else center
    ac = agent_col if agent_col is not None else center
    candidates = [
        (r, c) for r in range(cfg.grid) for c in range(cfg.grid)
        if abs(r - ar) + abs(c - ac) >= min_dist
    ]
    idx = int(rng.integers(0, len(candidates)))
    return list(candidates[idx])


def _s31_inp5(cfg: WorldConfig, row, col, hp,
              food_positions, food_avail, pred_pos):
    """5入力ベクトルを計算。Session 28と同じ構造、グリッドサイズだけ変わる。"""
    food_flag = 0.0
    for (fr, fc), av in zip(food_positions, food_avail):
        if av and abs(row - fr) + abs(col - fc) <= cfg.food_dist:
            food_flag = 1.0
            break
    pr, pc = pred_pos
    pred_flag = 1.0 if abs(row - pr) + abs(col - pc) <= cfg.pred_dist else 0.0
    return np.array([
        col / (cfg.grid - 1),
        row / (cfg.grid - 1),
        np.clip(hp / cfg.hp_start, 0.0, 1.5),
        food_flag,
        pred_flag,
    ])


def _s31_pred_step(cfg: WorldConfig, pred_pos, food_positions, food_avail,
                   food_bias_prob, rng):
    """捕食者移動（食料バイアスあり）。Session 30と同じロジック。"""
    pr, pc = pred_pos
    avail_foods = [fp for fp, av in zip(food_positions, food_avail) if av]
    move_toward = len(avail_foods) > 0 and rng.random() < food_bias_prob

    if move_toward:
        nearest = min(avail_foods,
                      key=lambda fp: abs(fp[0] - pr) + abs(fp[1] - pc))
        fr, fc  = nearest
        dr, dc  = int(np.sign(fr - pr)), int(np.sign(fc - pc))
        if dr == 0 and dc == 0:
            move_toward = False
        elif dr == 0:
            pc = int(np.clip(pc + dc, 0, cfg.grid - 1))
        elif dc == 0:
            pr = int(np.clip(pr + dr, 0, cfg.grid - 1))
        else:
            if rng.random() < 0.5:
                pr = int(np.clip(pr + dr, 0, cfg.grid - 1))
            else:
                pc = int(np.clip(pc + dc, 0, cfg.grid - 1))

    if not move_toward:
        d = int(rng.integers(0, 4))
        if d == 0:   pr = max(0, pr - 1)
        elif d == 1: pr = min(cfg.grid - 1, pr + 1)
        elif d == 2: pc = max(0, pc - 1)
        else:        pc = min(cfg.grid - 1, pc + 1)

    return [pr, pc]


# ── エピソードランナー ─────────────────────────────────────────────────────────

def _s31_run_ep(cfg: WorldConfig, G, W, genome, rng,
                food_bias_prob=_S31_FOOD_BIAS,
                predator_speed=_S31_PRED_SPEED,
                record_activity=False):
    """WorldConfigを受け取るエピソードランナー。"""
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
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())

        hp -= _S28_HP_DECAY
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

def _s31_evolve(cfg: WorldConfig, food_bias_prob=_S31_FOOD_BIAS,
                seed=_S31_SEED, n_gen=_S31_N_GEN):
    """WorldConfig を使って進化。"""
    rng = np.random.default_rng(seed + 31000 + cfg.grid * 100)
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s31_run_ep(
                    cfg, g['G'], g['W'], g, ep_rng,
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

def _s31_measure_context_dep(genome, seed):
    """4文脈でのmean_cosine_dist。_s28_measure_contextをそのまま使う。
    （文脈計測は固定入力なのでグリッドサイズに依存しない）
    """
    G_copy = genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 31200)

    results = []
    for ctx in _S28_CONTEXTS:
        samples = []
        for _ in range(_S31_N_CONTEXT):
            r = _s28_measure_context(
                G_copy, W_copy, genome, ctx, rng, T=_S31_CONTEXT_T)
            samples.append(r['mean_output'])
        mean_out     = np.mean(samples, axis=0)
        action_count = np.zeros(5)
        for s in samples:
            action_count[int(np.argmax(s))] += 1
        results.append({
            'mean_output':    mean_out,
            'action_dist':    action_count / _S31_N_CONTEXT,
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

def _s31_check_simple_strategy(cfg: WorldConfig, food_bias_prob=_S31_FOOD_BIAS,
                                seed=_S31_SEED, n_trials=_S31_N_TRIALS):
    """固定戦略（常に同じ行動）の生存ステップ数を計測。"""
    results = {}
    for action_idx, action_name in enumerate(_S28_ACTION_NAMES):
        total_steps = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + 31300 + trial * 10 + action_idx)
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
                if step % _S31_PRED_SPEED == 0:
                    pred_pos = _s31_pred_step(
                        cfg, pred_pos, food_positions, food_avail,
                        food_bias_prob, rng)
                if pred_pos[0] == row and pred_pos[1] == col:
                    hp -= _S28_PRED_DAMAGE
                hp -= _S28_HP_DECAY

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

def plot_grid_sweep(sweep_results,
                    fname='images/session_31/results_s31_grid_sweep.png'):
    """グリッドサイズ別の結果を4パネルで可視化。"""
    grids  = [r['grid']             for r in sweep_results]
    mcds   = [r['mean_cosine_dist'] for r in sweep_results]
    steps  = [r['best_steps']       for r in sweep_results]
    foods  = [r['food']             for r in sweep_results]
    hits   = [r['pred_hits']        for r in sweep_results]
    # 固定戦略の最良値
    best_fixed = [max(r['strategy'].values()) for r in sweep_results]
    best_fixed_name = [
        max(r['strategy'], key=r['strategy'].get) for r in sweep_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 31: グリッドサイズスイープ\n'
        f'food_bias_prob={_S31_FOOD_BIAS}  {_S31_N_GEN}世代  '
        f'seed={_S31_SEED}',
        fontsize=13,
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grids)))

    # Panel 1: mcd vs grid
    ax = axes[0][0]
    ax.plot(grids, mcds, 'o-', color='tomato', linewidth=2, markersize=10)
    for g, m in zip(grids, mcds):
        ax.text(g, m + 0.0003, f'{m:.4f}', ha='center', fontsize=9)
    ax.set_xlabel('Grid size')
    ax.set_ylabel('mean cosine dist')
    ax.set_title('文脈依存性\n（高いほど文脈間の行動差が大きい）')
    ax.grid(True, alpha=0.3)

    # Panel 2: 固定戦略 vs 進化個体の生存ステップ
    ax = axes[0][1]
    x = np.arange(len(grids))
    w = 0.35
    ax.bar(x - w/2, steps,      width=w, color='steelblue', alpha=0.85,
           label='進化個体', edgecolor='white')
    ax.bar(x + w/2, best_fixed, width=w, color='tomato',    alpha=0.85,
           label='最良固定戦略', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}×{g}' for g in grids])
    ax.set_ylabel('Mean steps / ep')
    ax.set_title('生存ステップ数\n（進化 vs 固定戦略の差が重要）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (s, f, fn) in enumerate(zip(steps, best_fixed, best_fixed_name)):
        ax.text(i - w/2, s + 1, f'{s:.0f}', ha='center', fontsize=8,
                color='steelblue')
        ax.text(i + w/2, f + 1, f'{f:.0f}\n({fn})', ha='center', fontsize=7,
                color='tomato')

    # Panel 3: 進化個体 - 固定戦略（差が正なら進化が勝っている）
    ax = axes[0][2]
    diffs = [s - f for s, f in zip(steps, best_fixed)]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(grids)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([f'{g}×{g}' for g in grids])
    ax.set_ylabel('進化 − 固定戦略 (steps)')
    ax.set_title('進化個体の優位性\n（正=進化が勝ち、文脈依存が有効）')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + (0.5 if d >= 0 else -1.5),
                f'{d:+.0f}', ha='center', fontsize=9)

    # Panel 4: food/ep vs grid
    ax = axes[1][0]
    ax.plot(grids, foods, 's-', color='seagreen', linewidth=2, markersize=10)
    for g, f in zip(grids, foods):
        ax.text(g, f + 0.02, f'{f:.2f}', ha='center', fontsize=9)
    ax.set_xlabel('Grid size')
    ax.set_ylabel('Food / ep')
    ax.set_title('食料獲得数\n（増加=食べに行く必要が生まれている）')
    ax.grid(True, alpha=0.3)

    # Panel 5: pred_hits vs grid
    ax = axes[1][1]
    ax.plot(grids, hits, '^-', color='darkorange', linewidth=2, markersize=10)
    for g, h in zip(grids, hits):
        ax.text(g, h + 0.02, f'{h:.2f}', ha='center', fontsize=9)
    ax.set_xlabel('Grid size')
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数\n（適切に回避できているか）')
    ax.grid(True, alpha=0.3)

    # Panel 6: WorldConfigのサマリーテキスト
    ax = axes[1][2]
    lines = ['WorldConfig per grid size\n']
    for r in sweep_results:
        cfg = r['cfg']
        lines.append(
            f'{cfg.grid}×{cfg.grid}: '
            f'steps={cfg.max_steps}  hp={cfg.hp_start}  '
            f'foods={cfg.n_foods}  '
            f'food_d={cfg.food_dist}  pred_d={cfg.pred_dist}'
        )
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('パラメータ設定')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_context(exp_b, grid,
                 fname='images/session_31/results_s31_context.png'):
    """ベストグリッドでの文脈依存行動を可視化。"""
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 31 Exp B: 文脈依存行動（ベストグリッド={grid}×{grid}）\n'
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
            ax.text(j, i, f'{out_mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if out_mat[i, j] > vmax * 0.6 else 'black')
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
                    color='white' if cos_mat[i, j] > vmax2 * 0.6 else 'black')
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


def plot_strategy_check(sweep_results,
                        fname='images/session_31/results_s31_strategy_check.png'):
    """グリッドサイズ別の固定戦略生存ステップを可視化。"""
    grids   = [r['grid'] for r in sweep_results]
    actions = _S28_ACTION_NAMES
    n_grid  = len(grids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 31: 固定戦略の生存ステップ vs グリッドサイズ\n'
        '（固定戦略が排除されるほど文脈依存が必要な環境になっている）',
        fontsize=13,
    )

    ax = axes[0]
    x = np.arange(n_grid)
    width = 0.15
    colors_a = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    for ai, (aname, col) in enumerate(zip(actions, colors_a)):
        vals = [r['strategy'][aname] for r in sweep_results]
        # max_stepsで正規化して比較しやすくする
        norm_vals = [v / r['cfg'].max_steps for v, r in zip(vals, sweep_results)]
        ax.bar(x + ai * width, norm_vals, width, label=aname,
               color=col, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'{g}×{g}' for g in grids])
    ax.set_ylabel('生存率 (steps / max_steps)')
    ax.set_title('固定戦略の生存率（max_stepsで正規化）\n'
                 '（低いほど固定戦略が通用しない環境）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    best_rates = [
        max(r['strategy'].values()) / r['cfg'].max_steps
        for r in sweep_results
    ]
    ax.plot(grids, best_rates, 'o-', color='tomato', linewidth=2, markersize=10)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1,
               label='生存率50%')
    ax.set_xlabel('Grid size')
    ax.set_ylabel('最良固定戦略の生存率')
    ax.set_title('グリッドサイズと「単純戦略排除」の関係\n'
                 '（下がるほど文脈を読む必要がある環境）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for g, r in zip(grids, best_rates):
        ax.text(g, r + 0.01, f'{r:.2f}', ha='center', fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 31: グリッドサイズスイープ ===')
    print(f'Grid sizes: {_S31_GRID_SIZES}')
    print(f'food_bias_prob={_S31_FOOD_BIAS} 固定（Session 30の結果から）')
    print()

    # WorldConfigを事前に表示
    print('WorldConfig:')
    for g in _S31_GRID_SIZES:
        cfg = WorldConfig.from_grid(g)
        print(f'  {cfg.summary()}')
    print()

    sweep_results = []

    for grid in _S31_GRID_SIZES:
        cfg = WorldConfig.from_grid(grid)
        print(f'\n{"="*60}')
        print(f'Grid {grid}×{grid}')
        print(f'{"="*60}')
        print(f'  {cfg.summary()}')

        # Exp A: 進化
        print('\n  [進化]')
        best, hist = _s31_evolve(cfg, seed=_S31_SEED)
        ctx_data   = _s31_measure_context_dep(best, seed=_S31_SEED)
        mcd        = ctx_data['mean_cosine_dist']
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'mcd={mcd:.4f}')

        # Exp C: 固定戦略
        print('\n  [固定戦略確認]')
        strategy = _s31_check_simple_strategy(cfg, seed=_S31_SEED)
        best_fixed = max(strategy.values())
        best_name  = max(strategy, key=strategy.get)
        print(f'  → 最良固定戦略: {best_name}で{best_fixed:.0f}steps '
              f'(生存率={best_fixed/cfg.max_steps:.2f})')

        sweep_results.append({
            'grid':             grid,
            'cfg':              cfg,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'food':             hist['gen_food_count'][-1],
            'pred_hits':        hist['gen_pred_hits'][-1],
            'ctx_data':         ctx_data,
            'strategy':         strategy,
        })

    # 可視化
    plot_grid_sweep(sweep_results)
    plot_strategy_check(sweep_results)

    # Exp B: ベストグリッドで詳細計測
    best_r = max(sweep_results, key=lambda r: r['mean_cosine_dist'])
    print(f'\n[Exp B] ベストグリッド: {best_r["grid"]}×{best_r["grid"]}  '
          f'mcd={best_r["mean_cosine_dist"]:.4f}')
    for ctx, res in zip(_S28_CONTEXTS, best_r['ctx_data']['context_results']):
        dom = _S28_ACTION_NAMES[int(np.argmax(res['action_dist']))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}] → {dom} '
              f'({res["action_dist"][int(np.argmax(res["action_dist"]))]:.0%})')
    plot_context(best_r['ctx_data'], best_r['grid'])

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 31 Summary ===')
    print()
    print(f'{"grid":>6}  {"mcd":>8}  {"steps":>7}  '
          f'{"food":>6}  {"hits":>6}  {"best_fixed":>10}  {"survival":>8}')
    for r in sweep_results:
        bf   = max(r['strategy'].values())
        bfn  = max(r['strategy'], key=r['strategy'].get)
        rate = bf / r['cfg'].max_steps
        print(f'{r["grid"]:>4}×{r["grid"]:<2}  '
              f'{r["mean_cosine_dist"]:>8.4f}  '
              f'{r["best_steps"]:>7.1f}  '
              f'{r["food"]:>6.2f}  '
              f'{r["pred_hits"]:>6.2f}  '
              f'{bfn}({bf:.0f}):>10  '
              f'{rate:>8.2f}')

    print()
    print('--- 観察してから判断すること ---')
    print('1. グリッドが大きくなると固定戦略の生存率は下がるか？')
    print('   → 下がれば「文脈を読む必要がある環境」が実現している')
    print('2. グリッドが大きくなるとmcdは上がるか？')
    print('   → 上がれば環境設計と文脈依存性が連動している')
    print('3. 進化個体が固定戦略を上回るグリッドサイズはどこか？')
    print('   → そのサイズが「文脈依存が有効な最小環境」')
    print()
    print('Done.')
