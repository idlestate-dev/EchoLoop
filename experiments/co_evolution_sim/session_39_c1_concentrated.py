"""Session 39: C1集中訓練 → 転移テスト

問い:
  「文脈依存行動が出ない」のは
  A) アーキテクチャ的に不可能（根本的限界）
  B) 経験が足りていないだけ（訓練環境の問題）

  Session 29-38 ではC1（食料近・捕食者近）の経験が
  全ステップの4〜15%しかなかった。
  これでは「捕食者がいる時は食事を控える」を
  Hebbianが学習するには不十分かもしれない。

実験設計:
  フェーズ1（集中訓練）:
    捕食者を食料セルに固定スポーン
    → 「食料に近づく = 捕食者に当たる」が
      ほぼ毎ステップ発生
    → C1経験を集中的に積む（目標: >50%）
    → 50世代進化

  フェーズ2（転移テスト）:
    通常環境（Session 38と同じ: pp=0.9, hp_decay=1）
    → エピソード中の文脈別行動を計測
    → C0-C1差がSession 38より大きくなるか？

  対照条件:
    フェーズ1なし（Session 38 hp_decay=1 の結果）
    → C0-C1差: mean=+2%

判断基準:
  集中訓練後にC0-C1差が有意に増加
  → 「経験が足りていただけ」= 仮説Bが正しい
  → Hebbianネットワークでも文脈依存学習は可能

  差が変わらない
  → 「アーキテクチャ的な限界」= 仮説Aが正しい
  → XOR的統合の困難さが本質

固定条件:
  N=25, grid=8, hp_decay=1（フェーズ1/2共通）
  pred_dist=1, food_dist=2

出力:
  images/session_39/results_s39_c1_rate.png     事前のC1発生率確認
  images/session_39/results_s39_transfer.png    転移テスト結果
  images/session_39/results_s39_multiseed.png   複数seed確認
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
    _s28_get_W,
)
from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import (
    WorldConfig, _s31_init_foods, _s31_init_pred,
)
from session_34_pursuit_x_s33 import (
    _S34_PRED_SPEED, _s34_pred_step,
)
from session_36_pred_dist_sweep import _s36_inp5
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate, _hebb,
    aggregate_context_actions,
)
from session_38_long_episode import (
    _S38_CFG, _S38_PURSUIT, _S38_N, _S38_PRED_DIST,
    _s38_run_ep, _run_context_log,
)

# ── Session 39 定数 ────────────────────────────────────────────────────────────

_S39_SEED      = _S28_SEED
_S39_N_GEN     = _S28_N_GEN     # 50世代（フェーズ1）
_S39_N         = _S38_N         # 25
_S39_CFG       = _S38_CFG       # grid=8, food_dist=2, pred_dist=1
_S39_HP_DECAY  = 1.0            # Session 38のベスト
_S39_PURSUIT   = _S38_PURSUIT   # 0.9
_S39_T_LONG    = 2000
_S39_N_SEEDS   = 5
_S39_SEEDS     = list(range(42, 42 + _S39_N_SEEDS))
_S39_N_TRIALS  = 20


# ── 集中訓練用の初期配置：捕食者を食料セルにスポーン ──────────────────────────

def _s39_init_pred_on_food(cfg: WorldConfig, rng,
                            food_positions, agent_row, agent_col):
    """捕食者を食料セルの上にスポーン。

    食料のいずれかをランダムに選んでその同一セルに配置。
    エージェントの初期位置と同じ場合は別の食料を試みる。

    Returns [row, col].
    """
    candidates = list(food_positions)
    rng.shuffle(candidates)
    for fr, fc in candidates:
        if not (fr == agent_row and fc == agent_col):
            return [fr, fc]
    # 全食料がエージェントと同じセルの場合（稀）はランダムフォールバック
    return _s31_init_pred(cfg, rng, agent_row, agent_col, min_dist=1)


# ── C1発生率を事前計測 ────────────────────────────────────────────────────────

def measure_c1_rates(cfg: WorldConfig, seed: int, n_ep: int = 20):
    """通常配置 vs 集中配置でのC1発生率を計測（ランダム行動）。

    Returns dict: {'normal': rate, 'concentrated': rate}
    """
    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    results = {}

    for mode in ('normal', 'concentrated'):
        rng = np.random.default_rng(seed + 39900 + (0 if mode == 'normal' else 1))
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

            if mode == 'normal':
                pred_pos = _s31_init_pred(cfg, ep_rng, row, col)
            else:
                pred_pos = _s39_init_pred_on_food(
                    cfg, ep_rng, food_positions, row, col)

            pred_resources = 1.0
            pred_dormant   = False

            for step in range(cfg.max_steps):
                if hp <= 0:
                    break
                if step % _S34_PRED_SPEED == 0:
                    pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                        cfg, pred_pos, [row, col],
                        _S39_PURSUIT, pred_resources, pred_dormant, ep_rng)
                if pred_pos[0] == row and pred_pos[1] == col:
                    hp -= _S28_PRED_DAMAGE
                hp -= _S39_HP_DECAY

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
                elif action == 4:
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

        rates = {c: ctx_counts[c] / total for c in range(4)} if total else {}
        results[mode] = rates
        c1 = rates.get(1, 0.0)
        print(f'  {mode:>13}: C0={rates.get(0,0):.0%}  C1={c1:.0%}  '
              f'C2={rates.get(2,0):.0%}  C3={rates.get(3,0):.0%}'
              f'  ({total}steps)')

    return results


# ── フェーズ1: 集中訓練エピソードランナー ─────────────────────────────────────

def _s39_run_ep_concentrated(cfg: WorldConfig, G, W, genome, rng,
                              hp_decay: float       = _S39_HP_DECAY,
                              pursuit_prob: float   = _S39_PURSUIT,
                              predator_speed: int   = _S34_PRED_SPEED,
                              record_activity: bool = False):
    """捕食者を食料セル上にスポーンするエピソード（集中訓練用）。

    Session 38のエピソードランナーとの差分:
      pred_pos の初期化だけ異なる。
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

    # ── ここだけSession 38と違う ──────────────────────────────────────────
    pred_pos = _s39_init_pred_on_food(
        cfg, rng, food_positions, row, col)
    # ─────────────────────────────────────────────────────────────────────

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


# ── フェーズ1: 進化（集中訓練） ───────────────────────────────────────────────

def _s39_evolve_concentrated(cfg: WorldConfig, n: int,
                              hp_decay: float  = _S39_HP_DECAY,
                              seed: int        = _S39_SEED,
                              n_gen: int       = _S39_N_GEN):
    """集中訓練環境で進化。"""
    rng = np.random.default_rng(seed + 39000)
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s39_run_ep_concentrated(
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

def plot_transfer(conc_result, normal_result, s38_baseline,
                  fname='images/session_39/results_s39_transfer.png'):
    """集中訓練 vs 通常訓練 vs Session 38ベースラインの比較。"""
    conditions = ['集中訓練後\n転移テスト', '通常訓練\n(Session 39)', 'Session 38\n(hp_decay=1)']
    c0_eats = [conc_result['c0_eat'],  normal_result['c0_eat'],  s38_baseline['c0_eat']]
    c1_eats = [conc_result['c1_eat'],  normal_result['c1_eat'],  s38_baseline['c1_eat']]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    c1_ns   = [conc_result['c1_steps'], normal_result['c1_steps'], s38_baseline['c1_steps']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 39: C1集中訓練 → 転移テスト\n'
        f'N={_S39_N}  grid={_S39_CFG.grid}x{_S39_CFG.grid}  '
        f'hp_decay={_S39_HP_DECAY}  pp={_S39_PURSUIT}',
        fontsize=13,
    )
    colors = ['tomato', 'steelblue', 'gray']

    ax = axes[0]
    x = np.arange(len(conditions))
    w = 0.35
    ax.bar(x - w/2, c0_eats, width=w, color='royalblue', alpha=0.85,
           label='C0(食料近・捕食者遠)', edgecolor='white')
    ax.bar(x + w/2, c1_eats, width=w, color='tomato',    alpha=0.85,
           label='C1(食料近・捕食者近)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('食事行動率')
    ax.set_title('C0 vs C1 の食事率\n(C0>C1 なら文脈依存あり)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (c0, c1) in enumerate(zip(c0_eats, c1_eats)):
        ax.text(i - w/2, c0 + 0.005, f'{c0:.0%}', ha='center', fontsize=9)
        ax.text(i + w/2, c1 + 0.005, f'{c1:.0%}', ha='center', fontsize=9)

    ax = axes[1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(conditions)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.02, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 38基準(+2%)')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1【核心】\n集中訓練後に増加すれば仮説B（経験不足）が正しい')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.012,
                f'{d:+.0%}', ha='center', fontsize=11, fontweight='bold')

    ax = axes[2]
    ax.bar(range(len(conditions)), c1_ns,
           color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('C1経験ステップ数')
    ax.set_title('C1（食料近・捕食者近）の経験量\n(集中訓練後に多いはず)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(c1_ns):
        ax.text(i, v + 3, f'{v}', ha='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_multiseed(multi_results,
                   fname='images/session_39/results_s39_multiseed.png'):
    seeds    = [r['seed']        for r in multi_results]
    c0_eats  = [r['c0_eat_rate'] for r in multi_results]
    c1_eats  = [r['c1_eat_rate'] for r in multi_results]
    diffs    = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f'Session 39: 複数seed確認（集中訓練後転移テスト）\n'
        f'N={_S39_N}  hp_decay={_S39_HP_DECAY}  T={_S39_T_LONG}',
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
    ax.axhline(0.02, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 38基準(+2%)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1\n(灰色=Session 38基準)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.01,
                f'{d:+.0%}', ha='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def _do_normal_evolve(cfg, n, seed):
    """通常訓練（Session 38と同じ設定）。"""
    from session_38_long_episode import _s38_evolve
    return _s38_evolve(cfg, n, hp_decay=_S39_HP_DECAY, seed=seed + 1)


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S39_CFG

    print('=== Session 39: C1集中訓練 → 転移テスト ===')
    print(f'N={_S39_N}  grid={cfg.grid}x{cfg.grid}  '
          f'pred_dist={cfg.pred_dist}  food_dist={cfg.food_dist}')
    print(f'hp_decay={_S39_HP_DECAY}  pp={_S39_PURSUIT}  n_gen={_S39_N_GEN}')
    print()

    # ── 事前確認: C1発生率 ─────────────────────────────────────────────────
    print('[事前確認] 通常配置 vs 集中配置のC1発生率（ランダム行動）')
    c1_rate_data = measure_c1_rates(cfg, seed=_S39_SEED)
    c1_normal      = c1_rate_data['normal'].get(1, 0.0)
    c1_concentrated = c1_rate_data['concentrated'].get(1, 0.0)
    print(f'  通常配置:   C1={c1_normal:.0%}')
    print(f'  集中配置:   C1={c1_concentrated:.0%}')
    print(f'  倍率: {c1_concentrated/c1_normal:.1f}x')
    print()

    # ── フェーズ1: 集中訓練（seed=42） ────────────────────────────────────
    print('[フェーズ1] 集中訓練（捕食者を食料セル上にスポーン）')
    best_conc, hist_conc = _s39_evolve_concentrated(
        cfg, _S39_N, seed=_S39_SEED)
    print(f'  → steps={hist_conc["gen_best_steps"][-1]:.1f}  '
          f'food={hist_conc["gen_food_count"][-1]:.2f}/ep  '
          f'hits={hist_conc["gen_pred_hits"][-1]:.2f}/ep')

    # ── フェーズ2: 転移テスト（通常環境で計測） ───────────────────────────
    print(f'\n[フェーズ2] 転移テスト（通常環境 T={_S39_T_LONG}）')
    rng_test = np.random.default_rng(_S39_SEED + 39200)
    G_test   = best_conc['G'].copy()
    W_test   = best_conc['W'].copy()
    log_test, act_test = _run_context_log(
        cfg, G_test, W_test, best_conc, rng_test,
        hp_decay=_S39_HP_DECAY, T=_S39_T_LONG)
    counts_t, fracs_t, totals_t = aggregate_context_actions(log_test)

    c0_conc = fracs_t[0, 4]
    c1_conc = fracs_t[1, 4]
    c1n_conc = totals_t[1]
    print(f'  C0食事率={c0_conc:.0%}  C1食事率={c1_conc:.0%}  '
          f'差={c0_conc-c1_conc:+.0%}  C1n={c1n_conc}')
    for c in range(4):
        if totals_t[c] > 0:
            dom = int(np.argmax(fracs_t[c]))
            print(f'  C{c}: {totals_t[c]}steps  '
                  f'主行動={_S28_ACTION_NAMES[dom]}({fracs_t[c,dom]:.0%})  '
                  f'食事={fracs_t[c,4]:.0%}')

    # ── 対照: 通常訓練（同条件、集中なし） ───────────────────────────────
    print(f'\n[対照] 通常訓練（集中なし）')
    best_norm, hist_norm = _do_normal_evolve(cfg, _S39_N, seed=_S39_SEED)
    rng_norm = np.random.default_rng(_S39_SEED + 39201)
    G_norm   = best_norm['G'].copy()
    W_norm   = best_norm['W'].copy()
    log_norm, _ = _run_context_log(
        cfg, G_norm, W_norm, best_norm, rng_norm,
        hp_decay=_S39_HP_DECAY, T=_S39_T_LONG)
    counts_n, fracs_n, totals_n = aggregate_context_actions(log_norm)
    c0_norm  = fracs_n[0, 4]
    c1_norm  = fracs_n[1, 4]
    c1n_norm = totals_n[1]
    print(f'  C0食事率={c0_norm:.0%}  C1食事率={c1_norm:.0%}  '
          f'差={c0_norm-c1_norm:+.0%}  C1n={c1n_norm}')

    # Session 38ベースライン（mean値）
    s38_baseline = {'c0_eat': 0.21, 'c1_eat': 0.19,
                    'c1_steps': 65}  # Session 38 hp_decay=1 の平均

    plot_transfer(
        conc_result  = {'c0_eat': c0_conc, 'c1_eat': c1_conc,
                        'c1_steps': c1n_conc},
        normal_result= {'c0_eat': c0_norm, 'c1_eat': c1_norm,
                        'c1_steps': c1n_norm},
        s38_baseline = s38_baseline,
    )

    # ── 複数seed確認 ─────────────────────────────────────────────────────
    print(f'\n[複数seed確認] seeds={_S39_SEEDS}')
    multi_results = []
    for seed in _S39_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, _ = _s39_evolve_concentrated(cfg, _S39_N, seed=seed)
        rng_s = np.random.default_rng(seed + 39200)
        G_s   = best_s['G'].copy()
        W_s   = best_s['W'].copy()
        log_s, _ = _run_context_log(
            cfg, G_s, W_s, best_s, rng_s,
            hp_decay=_S39_HP_DECAY, T=_S39_T_LONG)
        _, fracs_s, totals_s = aggregate_context_actions(log_s)
        c0 = fracs_s[0, 4]
        c1 = fracs_s[1, 4]
        print(f'    C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={totals_s[1]}')
        multi_results.append({
            'seed': seed, 'c0_eat_rate': c0,
            'c1_eat_rate': c1, 'c1_steps': totals_s[1],
        })

    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 39 Summary ===')
    print()
    print(f'C1発生率: 通常={c1_normal:.0%} → 集中={c1_concentrated:.0%}')
    print()
    print('seed=42 の結果:')
    print(f'  集中訓練後転移: C0={c0_conc:.0%}  C1={c1_conc:.0%}  '
          f'diff={c0_conc-c1_conc:+.0%}')
    print(f'  通常訓練:       C0={c0_norm:.0%}  C1={c1_norm:.0%}  '
          f'diff={c0_norm-c1_norm:+.0%}')
    print(f'  Session 38基準: C0=21%  C1=19%  diff=+2%')
    print()
    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    print(f'複数seed (n={len(_S39_SEEDS)}):')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S39_SEEDS)} seeds')
    print()
    print('--- 判断 ---')
    if np.mean(diffs) > 0.05 and n_pos >= 4:
        print('→ 仮説B支持: 十分な経験があれば文脈依存学習が可能')
        print('  Hebbianネットワークの限界ではなく訓練環境の問題だった')
    elif np.mean(diffs) <= 0.02:
        print('→ 仮説A支持: 経験を増やしても文脈依存が出ない')
        print('  アーキテクチャ的な限界（XOR的統合の困難さ）')
    else:
        print('→ 中間的な結果: さらなる検証が必要')
    print()
    print('Done.')
