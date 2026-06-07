"""Session 42: フェーズ1を長く・ネットワークを大きく

Session 41の問題:
  フェーズ1が100世代でn_edges=20〜30にしか育たない
  Session 10-14の実績（n_edges=200〜360）には遠く及ばない
  → 「安定した」とは言えない状態でフェーズ2に移行していた

原因:
  1. 世代数が少ない（100世代）
  2. grid=8×8で活性化が分散 → Hebbianが収束しにくい

解決:
  世代数: 100 → 500世代（またはn_edges>200で早期終了）
  N: 25 と 40 を両方試す
  → N=40の方がより豊かな構造が形成される可能性

期待:
  フェーズ1で十分なn_edgesが確保されれば
  フェーズ2での文脈依存も安定するはず

実験:
  条件A: N=25, フェーズ1=500世代
  条件B: N=40, フェーズ1=500世代
  → 各条件でフェーズ2（Session 40環境）に転移
  → n_edges推移 + C0-C1差を計測

判断基準:
  フェーズ1でn_edges>200 が達成できるか
  フェーズ2でC0-C1差がSession 40(mean=+4%)を超えるか
  N=40がN=25より改善するか
"""

import os
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from session_28_predator import (
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_OUT_START, _S28_OUT_END,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_ACTION_NAMES,
    _s28_get_W,
)
from session_10_embodied_output import _N_PROP, _K, _INIT_W, _LR
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import WorldConfig, _s31_init_foods
from session_34_pursuit_x_s33 import _S34_PRED_SPEED, _s34_pred_step
from session_36_pred_dist_sweep import _s36_inp5
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate,
    aggregate_context_actions,
)
from session_40_unavoidable import (
    _S40_N, _S40_PURSUIT, _S40_CFG, _S40_SEEDS,
    _s40_init_foods, _s40_init_pred_on_food, _s40_inp5,
)
from session_41_curriculum import (
    _S41_PHASE1_HP_DECAY, _S41_PHASE2_HP_DECAY, _S41_PHASE2_PURSUIT,
    _S41_CFG_PHASE1, _S41_CFG_PHASE2,
    _hebb_s10_style,
    _s41_run_ep_phase1,
)

# ── 定数 ──────────────────────────────────────────────────────────────────────

_S42_SEED           = _S28_SEED
_S42_SEEDS          = list(range(42, 42 + 3))  # 3seedで両条件を比較
_S42_T_LONG         = 2000
_S42_T_COLLAPSE     = 5000
_S42_CHUNK          = 100

# フェーズ1
_S42_PHASE1_N_GEN   = 500
_S42_PHASE1_N_EP    = _S28_N_EP
_S42_STABLE_THRESH  = 200      # Session 10-14の実績に合わせる

# 試すノード数
_S42_NODE_SIZES     = [25, 40]


# ── フェーズ1: 進化（長期版） ──────────────────────────────────────────────────

def _s42_evolve_phase1(cfg: WorldConfig, n: int,
                       hp_decay: float = _S41_PHASE1_HP_DECAY,
                       seed: int       = _S42_SEED,
                       n_gen: int      = _S42_PHASE1_N_GEN,
                       stable_thresh:  int = _S42_STABLE_THRESH):
    """安全環境で長期進化。n_edges > stable_thresh になったら早期終了。"""
    rng = np.random.default_rng(seed + 42000 + n * 10)
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    best_genome  = None
    best_n_edges = 0
    best_steps   = 0

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food = 0, []
            for _ in range(_S42_PHASE1_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s41_run_ep_phase1(
                    cfg, g['G'], g['W'], g, ep_rng,
                    hp_decay=hp_decay)
                total += res['steps']
                ep_food.append(res['food'])
            fitnesses.append(total / _S42_PHASE1_N_EP)
            g['_ep_food']   = float(np.mean(ep_food))
            g['_n_edges']   = g['G'].number_of_edges()

        best_idx     = int(np.argmax(fitnesses))
        best_genome  = pop[best_idx]
        best_n_edges = best_genome['G'].number_of_edges()
        best_steps   = fitnesses[best_idx]

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f'  gen {gen+1:4d}: best={best_steps:7.1f}  '
                  f'food={best_genome["_ep_food"]:.2f}/ep  '
                  f'n_edges={best_n_edges}')

        if best_n_edges >= stable_thresh:
            print(f'  → 安定基準達成（n_edges={best_n_edges} >= {stable_thresh}）'
                  f' at gen {gen+1}')
            break

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_S28_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _S28_N_AGENTS:
            parent = survivors[int(rng.integers(0, _S28_N_SURV))]
            new_pop.append(_mutate_genome(parent, rng))
        pop = new_pop

    for g in pop:
        g.pop('_ep_food', None)
        g.pop('_n_edges', None)

    return best_genome, best_n_edges


# ── フェーズ2: 崩壊観察 ──────────────────────────────────────────────────────

def _s42_observe_phase2(cfg: WorldConfig, G, W, genome, rng,
                         hp_decay: float     = _S41_PHASE2_HP_DECAY,
                         pursuit_prob: float = _S41_PHASE2_PURSUIT,
                         T: int              = _S42_T_COLLAPSE,
                         chunk: int          = _S42_CHUNK):
    """Session 40環境でn_edgesとC0-C1差の推移を観察。"""
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

    ctx_map  = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    log      = []
    chunks   = []
    chunk_ctx = defaultdict(list)

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

        hp -= hp_decay
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        log.append((ctx_idx, action))
        chunk_ctx[ctx_idx].append(1 if action == 4 else 0)

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
            _hebb_s10_style(G, W, eff, rng)

        activity = eff.copy()

        if (step + 1) % chunk == 0:
            n_edges = G.number_of_edges()
            c0_eat  = float(np.mean(chunk_ctx[0])) if chunk_ctx[0] else float('nan')
            c1_eat  = float(np.mean(chunk_ctx[1])) if chunk_ctx[1] else float('nan')
            diff    = c0_eat - c1_eat if not (
                np.isnan(c0_eat) or np.isnan(c1_eat)) else float('nan')
            chunks.append({
                'step': step + 1, 'n_edges': n_edges,
                'c0_eat': c0_eat, 'c1_eat': c1_eat, 'diff': diff,
            })
            chunk_ctx = defaultdict(list)

    return chunks, log


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_results(results_by_condition,
                 fname='images/session_42/results_s42_main.png'):
    """N=25 vs N=40 の比較。"""
    n_conds = len(results_by_condition)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f'Session 42: フェーズ1長期化（{_S42_PHASE1_N_GEN}世代）+ N=25/40\n'
        f'Phase1: hp_decay={_S41_PHASE1_HP_DECAY}（安全）  '
        f'Phase2: hp_decay={_S41_PHASE2_HP_DECAY}（捕食者）  '
        f'安定閾値n_edges={_S42_STABLE_THRESH}',
        fontsize=12,
    )

    colors_n = {'N=25': 'steelblue', 'N=40': 'tomato'}

    # Panel 1: フェーズ1でのn_edges達成状況
    ax = axes[0][0]
    cond_names = list(results_by_condition.keys())
    p1_edges_by_cond = {
        name: [r['phase1_edges'] for r in results]
        for name, results in results_by_condition.items()
    }
    x = np.arange(len(_S42_SEEDS))
    w = 0.35
    for i, (name, edges) in enumerate(p1_edges_by_cond.items()):
        offset = (i - (n_conds-1)/2) * w
        ax.bar(x + offset, edges, width=w,
               color=list(colors_n.values())[i],
               alpha=0.85, label=name, edgecolor='white')
    ax.axhline(_S42_STABLE_THRESH, color='gray', linestyle='--',
               linewidth=2, label=f'安定閾値({_S42_STABLE_THRESH})')
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in _S42_SEEDS])
    ax.set_ylabel('n_edges（Phase1終了時）')
    ax.set_title('フェーズ1でのネットワーク成長\n閾値に届いたか？')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: フェーズ2後のn_edges
    ax = axes[0][1]
    p2_edges_by_cond = {
        name: [r['phase2_edges'] for r in results]
        for name, results in results_by_condition.items()
    }
    for i, (name, edges) in enumerate(p2_edges_by_cond.items()):
        offset = (i - (n_conds-1)/2) * w
        ax.bar(x + offset, edges, width=w,
               color=list(colors_n.values())[i],
               alpha=0.85, label=name, edgecolor='white')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in _S42_SEEDS])
    ax.set_ylabel('n_edges（Phase2後）')
    ax.set_title('フェーズ2後のネットワーク維持状況\n0に近いほど崩壊')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: C0-C1差
    ax = axes[0][2]
    diffs_by_cond = {
        name: [r['c0_eat_rate'] - r['c1_eat_rate'] for r in results]
        for name, results in results_by_condition.items()
    }
    for i, (name, diffs) in enumerate(diffs_by_cond.items()):
        offset = (i - (n_conds-1)/2) * w
        colors_bar = ['seagreen' if d > 0 else 'tomato' for d in diffs]
        for j, (d, c) in enumerate(zip(diffs, colors_bar)):
            ax.bar(x[j] + offset, d, width=w, color=c,
                   alpha=0.85, edgecolor='white',
                   label=name if j == 0 else '')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.04, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 40基準(+4%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in _S42_SEEDS])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('文脈依存の核心指標 C0-C1差')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: n_edgesの推移（N=25, seed=42）
    ax = axes[1][0]
    for name, results in results_by_condition.items():
        for r in results:
            if r['seed'] == 42:
                steps   = [c['step']    for c in r['chunks']]
                n_edges = [c['n_edges'] for c in r['chunks']]
                ax.plot(steps, n_edges, 'o-',
                        color=colors_n[name], linewidth=2,
                        markersize=3, label=f'{name} s42', alpha=0.8)
                ax.axhline(r['phase1_edges'], color=colors_n[name],
                           linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Step（Phase2）')
    ax.set_ylabel('n_edges')
    ax.set_title('n_edges推移（seed=42）\n破線=Phase1終了時')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: C0-C1差の推移（seed=42）
    ax = axes[1][1]
    for name, results in results_by_condition.items():
        for r in results:
            if r['seed'] == 42:
                steps = [c['step'] for c in r['chunks']]
                diffs = [c['diff'] for c in r['chunks']]
                valid = [(s, d) for s, d in zip(steps, diffs)
                         if not np.isnan(d)]
                if valid:
                    vs, vd = zip(*valid)
                    ax.plot(vs, vd, 'o-',
                            color=colors_n[name], linewidth=2,
                            markersize=3, label=f'{name} s42', alpha=0.8)
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.04, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 40基準')
    ax.set_xlabel('Step（Phase2）')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('C0-C1差の推移（seed=42）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: サマリーテキスト
    ax = axes[1][2]
    lines = ['条件別サマリー\n']
    for name, results in results_by_condition.items():
        p1s   = [r['phase1_edges'] for r in results]
        p2s   = [r['phase2_edges'] for r in results]
        diffs = [r['c0_eat_rate'] - r['c1_eat_rate'] for r in results]
        n_pos = sum(d > 0 for d in diffs)
        lines.append(f'{name}:')
        lines.append(f'  Phase1 n_edges: '
                     f'mean={np.mean(p1s):.0f}  max={max(p1s)}')
        lines.append(f'  Phase2 n_edges: '
                     f'mean={np.mean(p2s):.0f}  (崩壊=0)')
        lines.append(f'  C0-C1差: mean={np.mean(diffs):+.0%}  '
                     f'std={np.std(diffs):.0%}  C0>C1:{n_pos}/{len(diffs)}')
        lines.append('')
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


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 42: フェーズ1長期化 + N=25/40 ===')
    print(f'Phase1: {_S42_PHASE1_N_GEN}世代  安定閾値={_S42_STABLE_THRESH}')
    print(f'Phase2: hp_decay={_S41_PHASE2_HP_DECAY}  pp={_S41_PHASE2_PURSUIT}')
    print(f'N={_S42_NODE_SIZES}  seeds={_S42_SEEDS}')
    print()

    results_by_condition = {}

    for n in _S42_NODE_SIZES:
        cond_name = f'N={n}'
        print(f'\n{"#"*60}')
        print(f'# 条件: {cond_name}')
        print(f'{"#"*60}')

        cond_results = []

        for seed in _S42_SEEDS:
            print(f'\n{"="*50}')
            print(f'seed={seed}  N={n}')
            print(f'{"="*50}')

            # フェーズ1
            print('[フェーズ1] 安全環境で長期進化')
            best, n_edges_p1 = _s42_evolve_phase1(
                _S41_CFG_PHASE1, n, seed=seed)
            print(f'  Phase1完了: n_edges={n_edges_p1}')
            if n_edges_p1 < _S42_STABLE_THRESH:
                print(f'  ※ 安定閾値未達だが続行')

            # フェーズ2
            print(f'\n[フェーズ2] 捕食者環境に投入 (T={_S42_T_COLLAPSE})')
            rng_p2 = np.random.default_rng(seed + 42200 + n * 100)
            G_p2   = best['G'].copy()
            W_p2   = _s28_get_W(G_p2)

            chunks, log_p2 = _s42_observe_phase2(
                _S41_CFG_PHASE2, G_p2, W_p2, best, rng_p2,
                T=_S42_T_COLLAPSE)

            n_edges_p2 = G_p2.number_of_edges()
            print(f'  n_edges: {n_edges_p1} → {n_edges_p2}  '
                  f'({"崩壊" if n_edges_p2 < 5 else "維持"})')

            # 文脈別行動（T=2000分）
            log_short = log_p2[:_S42_T_LONG]
            _, fracs, totals = aggregate_context_actions(log_short)
            c0 = fracs[0, 4]
            c1 = fracs[1, 4]
            n1 = totals[1]
            print(f'  C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
                  f'差={c0-c1:+.0%}  C1n={n1}')
            for c in range(4):
                if totals[c] > 0:
                    dom = int(np.argmax(fracs[c]))
                    print(f'  C{c}: {totals[c]}steps  '
                          f'主行動={_S28_ACTION_NAMES[dom]}'
                          f'({fracs[c,dom]:.0%})  '
                          f'食事={fracs[c,4]:.0%}')

            cond_results.append({
                'seed':         seed,
                'n':            n,
                'phase1_edges': n_edges_p1,
                'phase2_edges': n_edges_p2,
                'c0_eat_rate':  c0,
                'c1_eat_rate':  c1,
                'c1_steps':     n1,
                'chunks':       chunks,
            })

        results_by_condition[cond_name] = cond_results

        # 条件別サマリー
        print(f'\n--- {cond_name} サマリー ---')
        p1s   = [r['phase1_edges'] for r in cond_results]
        p2s   = [r['phase2_edges'] for r in cond_results]
        diffs = [r['c0_eat_rate'] - r['c1_eat_rate'] for r in cond_results]
        n_pos = sum(d > 0 for d in diffs)
        print(f'  Phase1 n_edges: mean={np.mean(p1s):.0f}  '
              f'max={max(p1s)}  min={min(p1s)}')
        print(f'  Phase2 n_edges: mean={np.mean(p2s):.0f}')
        print(f'  C0-C1差: mean={np.mean(diffs):+.0%}  '
              f'std={np.std(diffs):.0%}  C0>C1: {n_pos}/{len(_S42_SEEDS)}')

    plot_results(results_by_condition)

    # 全体サマリー
    print('\n' + '=' * 60)
    print('=== Session 42 Summary ===')
    print()
    for name, results in results_by_condition.items():
        p1s   = [r['phase1_edges'] for r in results]
        p2s   = [r['phase2_edges'] for r in results]
        diffs = [r['c0_eat_rate'] - r['c1_eat_rate'] for r in results]
        n_pos = sum(d > 0 for d in diffs)
        n_col = sum(r['phase2_edges'] < 5 for r in results)
        print(f'{name}:')
        print(f'  Phase1 n_edges: mean={np.mean(p1s):.0f}  '
              f'max={max(p1s)}')
        print(f'  Phase2 崩壊: {n_col}/{len(results)}')
        print(f'  C0-C1差: mean={np.mean(diffs):+.0%}  '
              f'std={np.std(diffs):.0%}  '
              f'C0>C1: {n_pos}/{len(results)}')
        print()

    print('--- 判断基準 ---')
    print('Phase1でn_edges>200達成 → フェーズ1の問題が解決')
    print('Phase2でC0-C1差がSession 40(+4%)を超える → 文脈依存が安定')
    print('N=40がN=25より改善 → ネットワーク容量が効いている')
    print()
    print('Done.')
