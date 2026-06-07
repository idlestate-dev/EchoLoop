"""Session 43: depletion_rate=0 で睡眠様状態を無効化

仮説:
  TM資源モデル（depletion_rate>0）が
  文脈依存行動の必要条件である

  depletion_rate=0 にすると：
    resources = 常に1.0（枯渇しない）
    → ほぼ全エッジが強化され続ける
    → weightが1.0に飽和
    → 活動パターンが均一化
    → 文脈依存が消える

Session 42bとの対比：
  Session 42b: depletion_rate は進化で決定（0.1〜0.5）
    → N=25 seed=43: depl=0.321 → C0-C1差=+17%
    → N=25 seed=42: depl=0.123 → C0-C1差=-2%
    → depletion_rateが高いほど差が出る傾向

  Session 43: depletion_rate=0 に固定
    → 飽和が起きてC0-C1差=0になるか？

実験設計:
  Session 42bと全く同じ設定
  depletion_rate=0 に固定する点だけ異なる

  フェーズ1（安全環境）: 500世代
  フェーズ2（捕食者環境）: T=5000
  N=25のみ（飽和の確認が目的なのでN=40は不要）
  seeds=[42, 43, 44]

判断基準:
  C0-C1差がほぼ0 → 飽和仮説支持
    = TM資源モデルが文脈依存の必要条件
  C0-C1差が出る → 飽和しない
    = TM資源モデルは関係ない
"""

import os
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

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
from session_31_grid_sweep import WorldConfig
from session_34_pursuit_x_s33 import _S34_PRED_SPEED, _s34_pred_step
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate, aggregate_context_actions,
)
from session_40_unavoidable import (
    _S40_PURSUIT, _S40_CFG,
    _s40_init_foods, _s40_init_pred_on_food, _s40_inp5,
)
from session_41_curriculum import (
    _S41_PHASE1_HP_DECAY, _S41_PHASE2_HP_DECAY,
    _S41_CFG_PHASE1, _S41_CFG_PHASE2,
    _hebb_s10_style, _s41_run_ep_phase1,
)
from session_42_curriculum_long import (
    _S42_PHASE1_N_GEN, _S42_STABLE_THRESH,
)

# ── 定数 ──────────────────────────────────────────────────────────────────────

_S43_SEED        = _S28_SEED
_S43_SEEDS       = [42, 43, 44]
_S43_N           = 25
_S43_T_PHASE2    = 5000
_S43_T_CONTEXT   = 2000
_S43_CHUNK       = 100

# depletion_rateを固定（0=無効化、比較用に通常値も）
_S43_DEPL_FIXED  = 0.0   # 無効化

# Session 42bの結果（比較用）
_S43_S42B_RESULTS = {
    42: {'depl': 0.123, 'diff': -0.02},
    43: {'depl': 0.321, 'diff': +0.17},
    44: {'depl': 0.275, 'diff': +0.06},
}


# ── depletion_rate固定版のゲノム生成 ─────────────────────────────────────────

def _make_genome_fixed_depl(n: int, rng, depl: float) -> dict:
    """depletion_rateを固定したゲノム。"""
    import networkx as nx
    from session_10_embodied_output import _INIT_W
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=float(_INIT_W))
    W = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        W[i, j] = d['weight']
    return {
        'G':              G,
        'W':              W,
        'n':              n,
        'depletion_rate': depl,        # 固定値
        'edge_add_prob':  float(rng.uniform(0.0, 0.1)),
        'activity_ratio': float(rng.uniform(0.0, 0.6)),
        'metabolic_rate': 0.01,
    }


def _mutate_genome_fixed_depl(genome: dict, rng, depl: float) -> dict:
    """depletion_rateを固定したまま突然変異。"""
    from session_28_predator import _S28_MUT_STD, _S28_EDGE_CHNG
    from session_10_embodied_output import _INIT_W
    n   = genome['n']
    G   = genome['G'].copy()
    for i, j in list(G.edges()):
        w = float(G[i][j]['weight']) + rng.normal(0, _S28_MUT_STD)
        G[i][j]['weight'] = float(np.clip(w, 0.01, 1.0))
    existing = set(G.edges())
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) in existing:
                if rng.random() < _S28_EDGE_CHNG:
                    G.remove_edge(i, j)
            else:
                if rng.random() < _S28_EDGE_CHNG * 0.2:
                    G.add_edge(i, j, weight=float(_INIT_W))
    W = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        W[i, j] = d['weight']
    return {
        'G':              G,
        'W':              W,
        'n':              n,
        'depletion_rate': depl,        # 固定
        'edge_add_prob':  float(np.clip(
            genome['edge_add_prob'] + rng.normal(0, 0.01), 0.0, 0.1)),
        'activity_ratio': float(np.clip(
            genome['activity_ratio'] + rng.normal(0, 0.05), 0.0, 0.6)),
        'metabolic_rate': genome['metabolic_rate'],
    }


# ── Phase1エピソード（depletion_rate固定版） ──────────────────────────────────

def _s43_run_ep_phase1(cfg, G, W, genome, rng,
                       hp_decay=_S41_PHASE1_HP_DECAY):
    """depletion_rate固定版のPhase1エピソード。"""
    n              = genome['n']
    depletion_rate = genome['depletion_rate']  # 固定値（0.0）
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _make_tau_arr(n)
    resources = np.ones(n)
    activity  = np.zeros(n)

    center   = cfg.grid // 2
    row, col = center, center
    hp       = float(cfg.hp_start)

    from session_31_grid_sweep import _s31_init_foods
    food_positions = _s31_init_foods(cfg, rng, row, col)
    food_avail     = [True] * cfg.n_foods
    food_timer     = [0]   * cfg.n_foods

    steps = food = 0

    for step in range(cfg.max_steps):
        if hp <= 0:
            break

        from session_36_pred_dist_sweep import _s36_inp5
        inp5 = _s36_inp5(cfg, row, col, hp,
                         food_positions, food_avail,
                         pred_pos=[999, 999])

        for _ in range(_N_PROP):
            activity = _propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, n), 0.0, 1.0)

        # depletion_rate=0なら resources は変化しない
        resources = _s27_update_resources(
            resources, activity, tau_arr, depletion_rate)

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
            _hebb_s10_style(G, W, eff, rng)

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, _S28_T_CONSOL)
    return {'steps': steps, 'food': food}


# ── Phase1進化 ────────────────────────────────────────────────────────────────

def _s43_evolve_phase1(cfg, n, depl, seed,
                       n_gen=_S42_PHASE1_N_GEN):
    rng = np.random.default_rng(seed + 43000 + int(depl * 100))
    pop = [_make_genome_fixed_depl(n, rng, depl)
           for _ in range(_S28_N_AGENTS)]

    best_genome  = None
    best_n_edges = 0

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total = 0
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s43_run_ep_phase1(
                    cfg, g['G'], g['W'], g, ep_rng)
                total += res['steps']
            fitnesses.append(total / _S28_N_EP)

        best_idx     = int(np.argmax(fitnesses))
        best_genome  = pop[best_idx]
        best_n_edges = best_genome['G'].number_of_edges()

        if (gen + 1) % 100 == 0 or gen == 0:
            print(f'  gen {gen+1:4d}: best={fitnesses[best_idx]:7.1f}  '
                  f'n_edges={best_n_edges}')

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_S28_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _S28_N_AGENTS:
            parent = survivors[int(rng.integers(0, _S28_N_SURV))]
            new_pop.append(_mutate_genome_fixed_depl(parent, rng, depl))
        pop = new_pop

    return best_genome, best_n_edges


# ── Phase2エピソード（depletion_rate固定版） ──────────────────────────────────

def _s43_run_phase2_context_log(cfg, G, W, genome, rng,
                                 hp_decay=_S41_PHASE2_HP_DECAY,
                                 pursuit_prob=_S40_PURSUIT,
                                 T=_S43_T_PHASE2,
                                 chunk=_S43_CHUNK):
    """文脈別行動 + n_edges推移を記録（depletion_rate固定版）。"""
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

    ctx_map   = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    log       = []
    chunks    = []
    chunk_ctx = defaultdict(list)

    # 重み飽和の観察用
    weight_samples = []

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

        # depletion_rate=0なら resources は変化しない
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
            # 重みの分布を記録（飽和確認用）
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            mean_w  = float(np.mean(weights)) if weights else 0.0
            frac_sat = float(np.mean(
                [w > 0.95 for w in weights])) if weights else 0.0

            c0_eat = float(np.mean(chunk_ctx[0])) if chunk_ctx[0] else float('nan')
            c1_eat = float(np.mean(chunk_ctx[1])) if chunk_ctx[1] else float('nan')
            diff   = c0_eat - c1_eat if not (
                np.isnan(c0_eat) or np.isnan(c1_eat)) else float('nan')

            chunks.append({
                'step':     step + 1,
                'n_edges':  n_edges,
                'mean_w':   mean_w,
                'frac_sat': frac_sat,   # 飽和エッジの割合（w>0.95）
                'c0_eat':   c0_eat,
                'c1_eat':   c1_eat,
                'diff':     diff,
            })
            chunk_ctx = defaultdict(list)

    return chunks, log


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_comparison(results_depl0, results_s42b,
                    fname='images/session_43/results_s43_comparison.png'):
    """depletion_rate=0 vs Session 42b（通常）の比較。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Session 43: depletion_rate=0 vs 通常（Session 42b）\n'
        '「TM資源モデルが文脈依存の必要条件か」',
        fontsize=13,
    )

    seeds   = _S43_SEEDS
    colors  = ['steelblue', 'tomato', 'seagreen']

    # Panel 1: C0-C1差の比較（核心）
    ax = axes[0][0]
    x = np.arange(len(seeds))
    w = 0.35
    diffs_d0  = [r['diff']  for r in results_depl0]
    diffs_s42 = [_S43_S42B_RESULTS[s]['diff'] for s in seeds]
    ax.bar(x - w/2, diffs_d0,  width=w, color='tomato',    alpha=0.85,
           label='depl=0（無効化）', edgecolor='white')
    ax.bar(x + w/2, diffs_s42, width=w, color='steelblue', alpha=0.85,
           label='通常（Session 42b）', edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('C0-C1差【核心】\ndepl=0で消えれば「睡眠様状態が必要条件」')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (d0, ds) in enumerate(zip(diffs_d0, diffs_s42)):
        ax.text(i - w/2, d0 + 0.005 if d0 >= 0 else d0 - 0.015,
                f'{d0:+.0%}', ha='center', fontsize=10)
        ax.text(i + w/2, ds + 0.005 if ds >= 0 else ds - 0.015,
                f'{ds:+.0%}', ha='center', fontsize=10)

    # Panel 2: 重みの飽和（frac_sat）の推移
    ax = axes[0][1]
    for r, color in zip(results_depl0, colors):
        steps    = [c['step']     for c in r['chunks']]
        frac_sat = [c['frac_sat'] for c in r['chunks']]
        ax.plot(steps, frac_sat, 'o-', color=color, linewidth=2,
                markersize=4, label=f's{r["seed"]}', alpha=0.8)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5,
               label='完全飽和')
    ax.set_xlabel('Step（Phase2）')
    ax.set_ylabel('飽和エッジの割合（w>0.95）')
    ax.set_title('エッジ重みの飽和状況（depl=0）\n1.0に向かうほど飽和している')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel 3: 平均重みの推移
    ax = axes[0][2]
    for r, color in zip(results_depl0, colors):
        steps  = [c['step']  for c in r['chunks']]
        mean_w = [c['mean_w'] for c in r['chunks']]
        ax.plot(steps, mean_w, 'o-', color=color, linewidth=2,
                markersize=4, label=f's{r["seed"]}', alpha=0.8)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5,
               label='飽和値')
    ax.set_xlabel('Step（Phase2）')
    ax.set_ylabel('平均エッジ重み')
    ax.set_title('平均エッジ重みの推移（depl=0）\n1.0に向かって飽和するか？')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: n_edgesの推移
    ax = axes[1][0]
    for r, color in zip(results_depl0, colors):
        steps   = [c['step']    for c in r['chunks']]
        n_edges = [c['n_edges'] for c in r['chunks']]
        ax.plot(steps, n_edges, 'o-', color=color, linewidth=2,
                markersize=4, label=f's{r["seed"]}', alpha=0.8)
    ax.set_xlabel('Step（Phase2）')
    ax.set_ylabel('n_edges')
    ax.set_title('エッジ数の推移（depl=0）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: C0-C1差の推移
    ax = axes[1][1]
    for r, color in zip(results_depl0, colors):
        steps = [c['step'] for c in r['chunks']]
        diffs = [c['diff'] for c in r['chunks']]
        valid = [(s, d) for s, d in zip(steps, diffs) if not np.isnan(d)]
        if valid:
            vs, vd = zip(*valid)
            ax.plot(vs, vd, 'o-', color=color, linewidth=2,
                    markersize=4, label=f's{r["seed"]}', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Step（Phase2）')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('C0-C1差の推移（depl=0）\n飽和すると0に収束するはず')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: サマリー
    ax = axes[1][2]
    lines = [
        'depl=0 vs 通常（Session 42b）\n',
        f'{"seed":>5}  {"depl":>6}  {"diff_d0":>8}  {"diff_s42":>9}',
    ]
    for r in results_depl0:
        s    = r['seed']
        depl = _S43_S42B_RESULTS[s]['depl']
        ds42 = _S43_S42B_RESULTS[s]['diff']
        d0   = r['diff']
        lines.append(
            f'{s:>5}  {depl:>6.3f}  {d0:>+8.0%}  {ds42:>+9.0%}')

    lines.append('')
    diffs_d0_vals = [r['diff'] for r in results_depl0]
    lines.append(
        f'depl=0: mean={np.mean(diffs_d0_vals):+.0%}  '
        f'std={np.std(diffs_d0_vals):.0%}')
    lines.append(
        f'通常:   mean={np.mean(diffs_s42):+.0%}  '
        f'std={np.std(diffs_s42):.0%}')

    final_sat = [r['chunks'][-1]['frac_sat'] for r in results_depl0]
    lines.append(f'\n飽和率（Phase2終了時）: mean={np.mean(final_sat):.0%}')

    ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=10, family='monospace',
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
    print('=== Session 43: depletion_rate=0 で睡眠様状態を無効化 ===')
    print(f'N={_S43_N}  depl={_S43_DEPL_FIXED}  seeds={_S43_SEEDS}')
    print(f'Phase1: {_S42_PHASE1_N_GEN}世代  Phase2: T={_S43_T_PHASE2}')
    print()
    print('比較対象（Session 42b N=25）:')
    for s, v in _S43_S42B_RESULTS.items():
        print(f'  seed={s}: depl={v["depl"]:.3f} → C0-C1差={v["diff"]:+.0%}')
    print()

    results_depl0 = []

    for seed in _S43_SEEDS:
        print(f'\n{"="*50}')
        print(f'seed={seed}  depl={_S43_DEPL_FIXED}')
        print(f'{"="*50}')

        # Phase1
        print('[Phase1] 安全環境で進化（depl=0固定）')
        best, n_edges_p1 = _s43_evolve_phase1(
            _S41_CFG_PHASE1, _S43_N, _S43_DEPL_FIXED, seed=seed)
        print(f'  Phase1完了: n_edges={n_edges_p1}  depl={best["depletion_rate"]}')

        # エッジ重みの初期状態を確認
        weights_p1 = [d['weight'] for _, _, d in best['G'].edges(data=True)]
        mean_w_p1  = float(np.mean(weights_p1)) if weights_p1 else 0.0
        frac_sat_p1 = float(np.mean(
            [w > 0.95 for w in weights_p1])) if weights_p1 else 0.0
        print(f'  Phase1終了時の重み: mean={mean_w_p1:.3f}  '
              f'飽和率={frac_sat_p1:.0%}')

        # Phase2
        print(f'\n[Phase2] 捕食者環境 (T={_S43_T_PHASE2})')
        rng_p2 = np.random.default_rng(seed + 43200)
        G_p2   = best['G'].copy()
        W_p2   = _s28_get_W(G_p2)

        chunks, log_p2 = _s43_run_phase2_context_log(
            _S41_CFG_PHASE2, G_p2, W_p2, best, rng_p2,
            T=_S43_T_PHASE2)

        # 文脈別行動
        _, fracs, totals = aggregate_context_actions(
            log_p2[:_S43_T_CONTEXT])
        c0 = fracs[0, 4]
        c1 = fracs[1, 4]

        final = chunks[-1]
        print(f'  n_edges: {n_edges_p1} → {final["n_edges"]}')
        print(f'  重み飽和率: {final["frac_sat"]:.0%}  '
              f'平均重み: {final["mean_w"]:.3f}')
        print(f'  C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}')

        # Session 42bとの比較
        s42b = _S43_S42B_RESULTS[seed]
        print(f'  Session 42b比較: depl={s42b["depl"]:.3f} → '
              f'C0-C1差={s42b["diff"]:+.0%}')

        results_depl0.append({
            'seed':         seed,
            'phase1_edges': n_edges_p1,
            'c0':           c0,
            'c1':           c1,
            'diff':         c0 - c1,
            'chunks':       chunks,
        })

    plot_comparison(results_depl0, _S43_S42B_RESULTS)

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 43 Summary ===')
    print()
    diffs_d0  = [r['diff'] for r in results_depl0]
    diffs_s42 = [_S43_S42B_RESULTS[s]['diff'] for s in _S43_SEEDS]
    sat_rates  = [r['chunks'][-1]['frac_sat'] for r in results_depl0]
    n_pos_d0  = sum(d > 0 for d in diffs_d0)
    n_pos_s42 = sum(d > 0 for d in diffs_s42)

    print(f'depl=0:  C0-C1差 mean={np.mean(diffs_d0):+.0%}  '
          f'std={np.std(diffs_d0):.0%}  C0>C1: {n_pos_d0}/{len(_S43_SEEDS)}')
    print(f'通常:    C0-C1差 mean={np.mean(diffs_s42):+.0%}  '
          f'std={np.std(diffs_s42):.0%}  C0>C1: {n_pos_s42}/{len(_S43_SEEDS)}')
    print(f'飽和率（Phase2終了時）: mean={np.mean(sat_rates):.0%}')
    print()

    print('--- 判断 ---')
    if np.mean(diffs_d0) < 0.02 and np.mean(sat_rates) > 0.5:
        print('→ 仮説支持: depl=0でC0-C1差が消え、かつ飽和が確認された')
        print('  TM資源モデル（睡眠様状態）は文脈依存の必要条件')
    elif np.mean(diffs_d0) > 0.05:
        print('→ 仮説否定: depl=0でも文脈依存が出る')
        print('  TM資源モデルは文脈依存に必須ではない')
    else:
        print(f'→ 中間的な結果 (diff={np.mean(diffs_d0):+.0%}, '
              f'sat={np.mean(sat_rates):.0%})')
    print()
    print('Done.')
