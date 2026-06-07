"""Session 39b: pursuit_prob=0 + 食料上スポーンでC1を確実に増やす

Session 39の問題:
  pursuit_prob=0.9 では捕食者がエージェントを追跡して食料から離れる
  → C1（食料近・捕食者近）が5%にしか増えなかった
  → 実験の前提が成立していなかった

修正:
  フェーズ1（集中訓練）: pursuit_prob=0.0（ランダムウォーク）
    + 食料セル上にスポーン
    → 捕食者が食料付近に留まりやすい
    → エージェントが食料に近づくたびにC1が発生

  フェーズ2（転移テスト）: pursuit_prob=0.9（Session 38と同じ）
    → 通常の追跡型捕食者環境で文脈依存を計測

ただし訓練と転移の環境ギャップに注意:
  訓練: ランダムウォーク捕食者
  テスト: 追跡型捕食者
  → 「転移」の難しさ自体が観察対象になる

判断基準は Session 39 と同じ:
  転移後のC0-C1差がSession 38基準(+2%)を超えるか
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
    _S38_CFG, _S38_PURSUIT, _S38_N,
    _run_context_log,
)
from session_39_c1_concentrated import (
    _s39_init_pred_on_food,
)

# ── Session 39b 定数 ───────────────────────────────────────────────────────────

_S39B_SEED         = _S28_SEED
_S39B_N_GEN        = _S28_N_GEN
_S39B_N            = _S38_N          # 25
_S39B_CFG          = _S38_CFG        # grid=8, food_dist=2, pred_dist=1
_S39B_HP_DECAY     = 1.0
_S39B_PURSUIT_TRAIN = 0.0            # フェーズ1: ランダムウォーク
_S39B_PURSUIT_TEST  = _S38_PURSUIT   # フェーズ2: 0.9（通常環境）
_S39B_T_LONG       = 2000
_S39B_N_SEEDS      = 5
_S39B_SEEDS        = list(range(42, 42 + _S39B_N_SEEDS))


# ── C1発生率の事前計測 ────────────────────────────────────────────────────────

def measure_c1_rates(cfg: WorldConfig, seed: int, n_ep: int = 50):
    """3条件でC1発生率を計測（ランダム行動）。

    条件:
      normal:       通常スポーン + pursuit=0.9
      conc_pursuit: 食料上スポーン + pursuit=0.9（Session 39）
      conc_random:  食料上スポーン + pursuit=0.0（Session 39b）
    """
    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    conditions = {
        'normal':       (False, _S39B_PURSUIT_TEST),
        'conc_pursuit': (True,  _S39B_PURSUIT_TEST),
        'conc_random':  (True,  _S39B_PURSUIT_TRAIN),
    }
    results = {}

    for name, (on_food, pp) in conditions.items():
        rng = np.random.default_rng(seed + 39900 + hash(name) % 1000)
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

            if on_food:
                pred_pos = _s39_init_pred_on_food(
                    cfg, ep_rng, food_positions, row, col)
            else:
                pred_pos = _s31_init_pred(cfg, ep_rng, row, col)

            pred_resources = 1.0
            pred_dormant   = False

            for step in range(cfg.max_steps):
                if hp <= 0:
                    break
                if step % _S34_PRED_SPEED == 0:
                    pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                        cfg, pred_pos, [row, col],
                        pp, pred_resources, pred_dormant, ep_rng)
                if pred_pos[0] == row and pred_pos[1] == col:
                    hp -= _S28_PRED_DAMAGE
                hp -= _S39B_HP_DECAY

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
        results[name] = rates
        c1 = rates.get(1, 0.0)
        print(f'  {name:>14}: '
              f'C0={rates.get(0,0):.0%}  C1={c1:.0%}  '
              f'C2={rates.get(2,0):.0%}  C3={rates.get(3,0):.0%}  '
              f'({total}steps)')

    return results


# ── フェーズ1: 集中訓練エピソードランナー（pursuit=0） ────────────────────────

def _s39b_run_ep_concentrated(cfg: WorldConfig, G, W, genome, rng,
                               hp_decay: float     = _S39B_HP_DECAY,
                               predator_speed: int = _S34_PRED_SPEED,
                               record_activity: bool = False):
    """food上スポーン + pursuit=0のエピソード。

    Session 39との差分: pursuit_prob=0.0 固定。
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
    pred_pos       = _s39_init_pred_on_food(
        cfg, rng, food_positions, row, col)
    pred_resources = 1.0
    pred_dormant   = False

    steps = food = pred_hits = 0
    act_recs = [] if record_activity else None

    for step in range(cfg.max_steps):
        if hp <= 0:
            break

        # pursuit_prob=0.0: ランダムウォーク
        if step % predator_speed == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                0.0, pred_resources, pred_dormant, rng)

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


# ── 進化 ──────────────────────────────────────────────────────────────────────

def _s39b_evolve(cfg: WorldConfig, n: int,
                 hp_decay: float = _S39B_HP_DECAY,
                 seed: int       = _S39B_SEED,
                 n_gen: int      = _S39B_N_GEN):
    """食料上スポーン + pursuit=0 で進化。"""
    rng = np.random.default_rng(seed + 39100)
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits', 'gen_mean_active')}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s39b_run_ep_concentrated(
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

def plot_c1_rates(c1_rate_data,
                  fname='images/session_39b/results_s39b_c1_rates.png'):
    """3条件のC1発生率を可視化。"""
    names  = list(c1_rate_data.keys())
    labels = ['通常\n(pp=0.9)', '食料上\n(pp=0.9)', '食料上\n(pp=0.0)']
    c0s = [c1_rate_data[n].get(0, 0) for n in names]
    c1s = [c1_rate_data[n].get(1, 0) for n in names]
    c2s = [c1_rate_data[n].get(2, 0) for n in names]
    c3s = [c1_rate_data[n].get(3, 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Session 39b: 条件別C1発生率（ランダム行動）\n'
                 'C1=食料近・捕食者近（トレードオフ発生状況）', fontsize=13)

    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5*w, c0s, width=w, color='royalblue', alpha=0.85,
           label='C0(食料近・捕食者遠)', edgecolor='white')
    ax.bar(x - 0.5*w, c1s, width=w, color='tomato', alpha=0.85,
           label='C1(食料近・捕食者近)', edgecolor='white')
    ax.bar(x + 0.5*w, c2s, width=w, color='seagreen', alpha=0.85,
           label='C2(食料遠・捕食者近)', edgecolor='white')
    ax.bar(x + 1.5*w, c3s, width=w, color='gray', alpha=0.85,
           label='C3(食料遠・捕食者遠)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('発生率')
    ax.set_title('C1が「食料上pp=0.0」で増加するか確認')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(c1s):
        ax.text(i - 0.5*w, v + 0.005, f'{v:.0%}',
                ha='center', fontsize=10, color='tomato', fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_multiseed(multi_results,
                   fname='images/session_39b/results_s39b_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f'Session 39b: 複数seed確認\n'
        f'訓練: food上スポーン+pp=0  テスト: 通常pp=0.9\n'
        f'N={_S39B_N}  hp_decay={_S39B_HP_DECAY}  T={_S39B_T_LONG}',
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
    ax.set_title('C0 vs C1 の食事率')
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
    ax.set_title('食事率差 C0-C1\n集中訓練で改善するか？')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.012,
                f'{d:+.0%}', ha='center', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S39B_CFG

    print('=== Session 39b: pursuit=0 + food上スポーンでC1を確実に増やす ===')
    print(f'N={_S39B_N}  grid={cfg.grid}x{cfg.grid}  '
          f'hp_decay={_S39B_HP_DECAY}')
    print(f'訓練: food上スポーン + pp={_S39B_PURSUIT_TRAIN}（ランダムウォーク）')
    print(f'テスト: 通常環境 + pp={_S39B_PURSUIT_TEST}')
    print()

    # ── 事前確認: C1発生率 ─────────────────────────────────────────────────
    print('[事前確認] 3条件でのC1発生率（ランダム行動 50ep）')
    c1_rate_data = measure_c1_rates(cfg, seed=_S39B_SEED)
    plot_c1_rates(c1_rate_data)

    c1_conc = c1_rate_data['conc_random'].get(1, 0.0)
    c1_norm = c1_rate_data['normal'].get(1, 0.0)
    print(f'\n  通常:    C1={c1_norm:.0%}')
    print(f'  集中:    C1={c1_conc:.0%}  ({c1_conc/max(c1_norm,0.001):.1f}x)')

    if c1_conc < 0.15:
        print(f'  警告: C1={c1_conc:.0%} はまだ低い。')
        print(f'  ただし実験を継続する（データとして記録）')
    print()

    # ── 複数seed: フェーズ1→2 ─────────────────────────────────────────────
    print(f'[実験] seeds={_S39B_SEEDS}')
    multi_results = []

    for seed in _S39B_SEEDS:
        print(f'\n{"="*50}')
        print(f'seed={seed}')
        print(f'{"="*50}')

        # フェーズ1: 集中訓練（pursuit=0）
        print('  [フェーズ1] 集中訓練')
        best, hist = _s39b_evolve(cfg, _S39B_N, seed=seed)
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'food={hist["gen_food_count"][-1]:.2f}/ep  '
              f'hits={hist["gen_pred_hits"][-1]:.2f}/ep')

        # フェーズ2: 転移テスト（pursuit=0.9）
        print(f'  [フェーズ2] 転移テスト（pp={_S39B_PURSUIT_TEST}）')
        rng_test = np.random.default_rng(seed + 39200)
        G_test   = best['G'].copy()
        W_test   = best['W'].copy()
        log_test, _ = _run_context_log(
            cfg, G_test, W_test, best, rng_test,
            hp_decay=_S39B_HP_DECAY,
            pursuit_prob=_S39B_PURSUIT_TEST,
            T=_S39B_T_LONG)
        _, fracs_t, totals_t = aggregate_context_actions(log_test)

        c0 = fracs_t[0, 4]
        c1 = fracs_t[1, 4]
        n1 = totals_t[1]
        print(f'  C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={n1}')

        multi_results.append({
            'seed':        seed,
            'c0_eat_rate': c0,
            'c1_eat_rate': c1,
            'c1_steps':    n1,
        })

    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 39b Summary ===')
    print()
    print(f'C1発生率: 通常={c1_norm:.0%} → 集中(pp=0)={c1_conc:.0%}')
    print()
    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)
    print(f'転移テスト結果 (n={len(_S39B_SEEDS)} seeds):')
    print(f'  C0食事率: mean={np.mean(c0s):.0%}  std={np.std(c0s):.0%}')
    print(f'  C1食事率: mean={np.mean(c1s):.0%}  std={np.std(c1s):.0%}')
    print(f'  差C0-C1:  mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    print(f'  C0>C1: {n_pos}/{len(_S39B_SEEDS)} seeds')
    print(f'  Session 38基準: mean=+2%  C0>C1: 4/5')
    print()

    # 判断
    improved = np.mean(diffs) > 0.04 and n_pos >= 4
    same     = abs(np.mean(diffs)) <= 0.02
    print('--- 判断 ---')
    if improved:
        print('→ 仮説B支持: 十分な経験があれば文脈依存学習が可能')
        print('  訓練環境の設計の問題であってモデルの限界ではなかった')
    elif same:
        print('→ 仮説A支持: 経験を増やしても文脈依存が出ない')
        print('  アーキテクチャ的な限界（XOR的統合の困難さ）')
    else:
        print(f'→ 中間的な結果 (mean={np.mean(diffs):+.0%})')
        print('  C1集中訓練の効果はあるが転移が不完全な可能性')
    print()
    print('Done.')
