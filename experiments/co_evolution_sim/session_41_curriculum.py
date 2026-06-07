"""Session 41: カリキュラム学習

問い:
  「睡眠様状態が確立したネットワークは
   捕食者環境に投入しても崩壊しないか？」

  Session 40bで判明したこと:
    Session 40の進化個体はn_edges=0（ネットワーク崩壊）
    行動がランダムに近い
    → 「文脈依存が出ない」ではなく
      「学習基盤自体が存在しない」

  仮説:
    A（袋小路）: どんな順序でも崩壊する
    B（橋渡し可能）: 安定したネットワークを
                    先に作ってから捕食者を導入すれば
                    崩壊しない

実験設計:
  フェーズ1（安定化）: Session 10相当
    捕食者なし、HP緩め（decay=1）
    → n_edges > 50 になるまで進化（最大100世代）
    → Session 14の基準（n_edges≈200, clustering≈0.8）を目標

  フェーズ2（転移）: Session 40の環境に一気に投入
    hp_decay=3、捕食者あり（pp=0.9、food固定）
    → ネットワークが崩壊するか？
    → 崩壊しなければC0-C1差は改善するか？

  対照条件:
    フェーズ1なしでSession 40をそのまま（Session 40の結果）

計測:
  転移直後（T=0〜2000）のn_edges推移
  → 崩壊の速度を見る
  文脈別行動（C0-C1差）
  → 文脈依存が出るか

判断基準:
  崩壊する（n_edges→0）: 仮説A支持 → 袋小路
  崩壊しない: 仮説B支持 → 閾値探索へ進む
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
from session_10_embodied_output import (
    _N_PROP, _K, _INIT_W, _LR,
)
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import WorldConfig, _s31_init_foods, _s31_init_pred
from session_34_pursuit_x_s33 import _S34_PRED_SPEED, _s34_pred_step
from session_36_pred_dist_sweep import _s36_inp5
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate,
    aggregate_context_actions,
)
from session_40_unavoidable import (
    _S40_N, _S40_PURSUIT, _S40_CFG,
    _S40_FOOD_POS, _S40_T_LONG, _S40_SEEDS,
    _s40_init_foods, _s40_init_pred_on_food, _s40_inp5,
    _s40_run_context_log,
)

# ── 定数 ──────────────────────────────────────────────────────────────────────

_S41_SEED         = _S28_SEED
_S41_N            = _S40_N         # 25
_S41_N_SEEDS      = 5
_S41_SEEDS        = list(range(42, 42 + _S41_N_SEEDS))
_S41_T_LONG       = 2000
_S41_T_COLLAPSE   = 5000           # 崩壊観察用の長さ
_S41_CHUNK        = 100            # n_edges推移の計測チャンク

# フェーズ1（安定化）
_S41_PHASE1_HP_DECAY  = 1
_S41_PHASE1_N_GEN     = 100        # 最大100世代（安定したら早期終了）
_S41_PHASE1_N_EP      = _S28_N_EP
_S41_STABLE_THRESHOLD = 50         # n_edges > 50 で「安定」と判断

# フェーズ2（転移）
_S41_PHASE2_HP_DECAY  = 3          # Session 40と同じ
_S41_PHASE2_PURSUIT   = _S40_PURSUIT  # 0.9

# フェーズ1用WorldConfig（捕食者なし・食料ランダム）
_S41_CFG_PHASE1 = WorldConfig(
    grid      = _S40_CFG.grid,
    max_steps = 1280,
    hp_start  = _S40_CFG.hp_start,
    n_foods   = 2,            # Session 10と同じ食料2つ
    food_dist = _S40_CFG.food_dist,
    pred_dist = _S40_CFG.pred_dist,
)

# フェーズ2用WorldConfig（Session 40と同じ）
_S41_CFG_PHASE2 = _S40_CFG


# ── Session 10相当のHebb（閾値0.5） ───────────────────────────────────────────

def _hebb_s10_style(G, W, activity, rng, edge_add_prob=0.01):
    """Session 10と同じHebbian（閾値=0.5）。

    Session 37-40の閾値0.1より厳しい基準で強化。
    「本当に活発なノードだけ」強化することで
    ネットワークの安定性が高まる。
    """
    n = W.shape[0]
    to_remove = []
    for i, j, d in list(G.edges(data=True)):
        w = d['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += _LR
        w -= 0.01
        if w < 0.01:
            to_remove.append((i, j))
            W[i, j] = 0.0
        else:
            w = min(w, 1.0)
            G[i][j]['weight'] = w
            W[i, j] = w
    G.remove_edges_from(to_remove)

    # ランダムエッジ追加
    existing = set(G.edges())
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) not in existing:
                if rng.random() < edge_add_prob:
                    G.add_edge(i, j, weight=float(_INIT_W))
                    W[i, j] = float(_INIT_W)


# ── フェーズ1: 安全環境エピソードランナー ─────────────────────────────────────

def _s41_run_ep_phase1(cfg: WorldConfig, G, W, genome, rng,
                       hp_decay: float       = _S41_PHASE1_HP_DECAY,
                       record_activity: bool = False):
    """捕食者なし・通常食料配置のエピソード（フェーズ1用）。

    Session 10相当:
      捕食者なし
      Hebbian閾値=0.5
      hp_decay=1
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

    steps = food = 0
    act_recs = [] if record_activity else None

    for step in range(cfg.max_steps):
        if hp <= 0:
            break

        inp5 = _s36_inp5(cfg, row, col, hp,
                         food_positions, food_avail,
                         pred_pos=[999, 999])  # 捕食者なし

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

        # Session 10スタイルのHebb（閾値=0.5）
        if (step + 1) % _K == 0:
            _hebb_s10_style(G, W, eff, rng)

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, _S28_T_CONSOL)

    return {'steps': steps, 'food': food, 'act_recs': act_recs}


# ── フェーズ1: 進化 ───────────────────────────────────────────────────────────

def _s41_evolve_phase1(cfg: WorldConfig, n: int,
                       hp_decay: float = _S41_PHASE1_HP_DECAY,
                       seed: int       = _S41_SEED,
                       n_gen: int      = _S41_PHASE1_N_GEN,
                       stable_thresh:  int = _S41_STABLE_THRESHOLD):
    """安全環境で進化。n_edges > stable_thresh になったら早期終了。"""
    rng = np.random.default_rng(seed + 41000)
    pop = [_make_genome(n, rng) for _ in range(_S28_N_AGENTS)]

    best_genome  = None
    best_n_edges = 0

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food = 0, []
            for _ in range(_S41_PHASE1_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s41_run_ep_phase1(
                    cfg, g['G'], g['W'], g, ep_rng,
                    hp_decay=hp_decay)
                total += res['steps']
                ep_food.append(res['food'])
            fitnesses.append(total / _S41_PHASE1_N_EP)
            g['_ep_food'] = float(np.mean(ep_food))

        best_idx   = int(np.argmax(fitnesses))
        best_genome = pop[best_idx]
        best_n_edges = best_genome['G'].number_of_edges()

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={best_genome["_ep_food"]:.2f}/ep  '
                  f'n_edges={best_n_edges}')

        # 安定判定
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

    return best_genome, best_n_edges


# ── フェーズ2: 崩壊観察ランナー ──────────────────────────────────────────────

def _s41_observe_collapse(cfg: WorldConfig, G, W, genome, rng,
                           hp_decay: float     = _S41_PHASE2_HP_DECAY,
                           pursuit_prob: float = _S41_PHASE2_PURSUIT,
                           T: int              = _S41_T_COLLAPSE,
                           chunk: int          = _S41_CHUNK):
    """Session 40環境でネットワークの崩壊を観察。

    チャンクごとにn_edgesを記録。
    同時に文脈別行動も記録。

    Returns:
      chunks: list of {'step', 'n_edges', 'c0_eat', 'c1_eat'}
      log:    文脈別行動ログ
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

    food_positions = _s40_init_foods()
    food_avail     = [True]
    food_timer     = [0]
    pred_pos       = _s40_init_pred_on_food()
    pred_resources = 1.0
    pred_dormant   = False

    ctx_map  = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    log      = []
    chunks   = []
    chunk_c0_eat = []
    chunk_c1_eat = []

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

        # 食事記録（チャンク集計用）
        if ctx_idx == 0:
            chunk_c0_eat.append(1 if action == 4 else 0)
        elif ctx_idx == 1:
            chunk_c1_eat.append(1 if action == 4 else 0)

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

        # Session 10スタイルのHebb（閾値=0.5）を継続
        if (step + 1) % _K == 0:
            _hebb_s10_style(G, W, eff, rng)

        activity = eff.copy()

        # チャンク集計
        if (step + 1) % chunk == 0:
            n_edges   = G.number_of_edges()
            c0_eat    = float(np.mean(chunk_c0_eat)) if chunk_c0_eat else float('nan')
            c1_eat    = float(np.mean(chunk_c1_eat)) if chunk_c1_eat else float('nan')
            chunks.append({
                'step':    step + 1,
                'n_edges': n_edges,
                'c0_eat':  c0_eat,
                'c1_eat':  c1_eat,
                'diff':    c0_eat - c1_eat if not (
                    np.isnan(c0_eat) or np.isnan(c1_eat)) else float('nan'),
            })
            chunk_c0_eat = []
            chunk_c1_eat = []

    return chunks, log


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_collapse(collapse_data_by_seed, phase1_edges,
                  fname='images/session_41/results_s41_collapse.png'):
    """n_edgesの推移（崩壊曲線）を可視化。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Session 41: カリキュラム学習 → 捕食者環境への転移\n'
        f'N={_S41_N}  Phase1: hp_decay={_S41_PHASE1_HP_DECAY}（安全）  '
        f'Phase2: hp_decay={_S41_PHASE2_HP_DECAY}（捕食者あり）',
        fontsize=12,
    )

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(collapse_data_by_seed)))

    ax = axes[0]
    for (seed, chunks), color in zip(collapse_data_by_seed.items(), colors):
        steps   = [c['step']    for c in chunks]
        n_edges = [c['n_edges'] for c in chunks]
        ax.plot(steps, n_edges, 'o-', color=color, linewidth=2,
                markersize=4, label=f's{seed}', alpha=0.8)
        ax.axhline(phase1_edges[seed], color=color, linestyle='--',
                   linewidth=1, alpha=0.5)

    ax.axhline(_S41_STABLE_THRESHOLD, color='gray', linestyle=':',
               linewidth=2, label=f'安定閾値({_S41_STABLE_THRESHOLD})')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Step（Phase2開始後）')
    ax.set_ylabel('n_edges')
    ax.set_title('ネットワークエッジ数の推移\n'
                 '破線=Phase1終了時の値  点線=安定閾値\n'
                 '0に向かって崩壊するか、維持されるか')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (seed, chunks), color in zip(collapse_data_by_seed.items(), colors):
        steps = [c['step'] for c in chunks]
        diffs = [c['diff'] for c in chunks]
        valid = [(s, d) for s, d in zip(steps, diffs) if not np.isnan(d)]
        if valid:
            vs, vd = zip(*valid)
            ax.plot(vs, vd, 'o-', color=color, linewidth=2,
                    markersize=4, label=f's{seed}', alpha=0.8)

    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.02, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 40基準(+2%)')
    ax.set_xlabel('Step（Phase2開始後）')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('文脈依存の推移\n'
                 'ネットワークが維持されれば差が出るか？')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_multiseed(multi_results,
                   fname='images/session_41/results_s41_multiseed.png'):
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    p1_edges = [r['phase1_edges'] for r in multi_results]
    p2_edges = [r['phase2_edges'] for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 41: 複数seed確認\n'
        f'Phase1（安全）→ Phase2（捕食者）',
        fontsize=13,
    )

    ax = axes[0]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, p1_edges, width=w, color='seagreen', alpha=0.85,
           label='Phase1終了時', edgecolor='white')
    ax.bar(x + w/2, p2_edges, width=w, color='tomato', alpha=0.85,
           label='Phase2後（T=2000）', edgecolor='white')
    ax.axhline(_S41_STABLE_THRESHOLD, color='gray', linestyle='--',
               label=f'安定閾値({_S41_STABLE_THRESHOLD})')
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('n_edges')
    ax.set_title('エッジ数の変化\n'
                 '転移後に崩壊するか維持されるか')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    x2 = np.arange(len(seeds))
    ax.bar(x2 - w/2, c0_eats, width=w, color='royalblue', alpha=0.85,
           label='C0(食料近・捕食者遠)', edgecolor='white')
    ax.bar(x2 + w/2, c1_eats, width=w, color='tomato', alpha=0.85,
           label='C1(食料近・捕食者近)', edgecolor='white')
    ax.set_xticks(x2)
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('食事行動率')
    ax.set_title('C0 vs C1 の食事率')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(seeds)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.02, color='gray',  linestyle='--', linewidth=1.5,
               label='Session 40基準(+2%)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率差 C0-C1')
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
    print('=== Session 41: カリキュラム学習 ===')
    print(f'N={_S41_N}')
    print(f'Phase1: 安全環境  hp_decay={_S41_PHASE1_HP_DECAY}  '
          f'安定閾値n_edges={_S41_STABLE_THRESHOLD}')
    print(f'Phase2: 捕食者環境  hp_decay={_S41_PHASE2_HP_DECAY}  '
          f'pp={_S41_PHASE2_PURSUIT}')
    print()

    collapse_data_by_seed = {}
    phase1_edges          = {}
    multi_results         = []

    for seed in _S41_SEEDS:
        print(f'\n{"="*55}')
        print(f'seed={seed}')
        print(f'{"="*55}')

        # ── フェーズ1: 安全環境で進化 ─────────────────────────────────
        print('\n[フェーズ1] 安全環境で進化')
        best, n_edges_p1 = _s41_evolve_phase1(
            _S41_CFG_PHASE1, _S41_N, seed=seed)
        phase1_edges[seed] = n_edges_p1
        print(f'  Phase1完了: n_edges={n_edges_p1}  '
              f'depl={best["depletion_rate"]:.3f}')

        if n_edges_p1 < _S41_STABLE_THRESHOLD:
            print(f'  警告: 安定閾値未達（{n_edges_p1} < {_S41_STABLE_THRESHOLD}）')
            print(f'  それでもPhase2に投入して観察する')

        # ── フェーズ2: 捕食者環境に一気に投入 ────────────────────────
        print(f'\n[フェーズ2] 捕食者環境に投入 (T={_S41_T_COLLAPSE})')
        rng_p2 = np.random.default_rng(seed + 41200)
        G_p2   = best['G'].copy()
        W_p2   = _s28_get_W(G_p2)

        chunks, log_p2 = _s41_observe_collapse(
            _S41_CFG_PHASE2, G_p2, W_p2, best, rng_p2,
            T=_S41_T_COLLAPSE)

        collapse_data_by_seed[seed] = chunks

        # 最終状態
        final_edges = G_p2.number_of_edges()
        print(f'  初期n_edges={n_edges_p1} → 最終n_edges={final_edges}')

        # 崩壊判定
        if final_edges < 5:
            print(f'  → 崩壊（n_edges={final_edges}）')
        elif final_edges >= _S41_STABLE_THRESHOLD:
            print(f'  → 維持（n_edges={final_edges}）')
        else:
            print(f'  → 部分的崩壊（n_edges={final_edges}）')

        # 文脈別行動（T=2000分）
        log_short = log_p2[:_S41_T_LONG]
        _, fracs, totals = aggregate_context_actions(log_short)
        c0 = fracs[0, 4]
        c1 = fracs[1, 4]
        n1 = totals[1]
        print(f'  C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
              f'差={c0-c1:+.0%}  C1n={n1}')

        multi_results.append({
            'seed':         seed,
            'phase1_edges': n_edges_p1,
            'phase2_edges': final_edges,
            'c0_eat_rate':  c0,
            'c1_eat_rate':  c1,
            'c1_steps':     n1,
        })

    plot_collapse(collapse_data_by_seed, phase1_edges)
    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 41 Summary ===')
    print()
    print(f'{"seed":>5}  {"p1_edges":>9}  {"p2_edges":>9}  '
          f'{"崩壊?":>6}  {"C0":>5}  {"C1":>5}  {"diff":>6}')
    for r in multi_results:
        collapsed = '崩壊' if r['phase2_edges'] < 5 else '維持'
        d = r['c0_eat_rate'] - r['c1_eat_rate']
        print(f'{r["seed"]:>5}  {r["phase1_edges"]:>9}  '
              f'{r["phase2_edges"]:>9}  {collapsed:>6}  '
              f'{r["c0_eat_rate"]:>5.0%}  {r["c1_eat_rate"]:>5.0%}  '
              f'{d:>+6.0%}')

    n_collapsed = sum(r['phase2_edges'] < 5 for r in multi_results)
    c0s   = [r['c0_eat_rate'] for r in multi_results]
    c1s   = [r['c1_eat_rate'] for r in multi_results]
    diffs = [c0 - c1 for c0, c1 in zip(c0s, c1s)]
    n_pos = sum(d > 0 for d in diffs)

    print()
    print(f'崩壊: {n_collapsed}/{len(_S41_SEEDS)} seeds')
    print(f'C0-C1差: mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}  '
          f'C0>C1: {n_pos}/{len(_S41_SEEDS)}')
    print()
    print('--- 判断 ---')
    if n_collapsed >= 4:
        print('→ 仮説A支持: カリキュラム学習でも崩壊する')
        print('  Hebbian + HP環境は根本的に両立しない')
        print('  = 袋小路')
    elif n_collapsed == 0:
        print('→ 仮説B支持: 安定したネットワークは崩壊しない')
        print('  閾値探索へ進む価値がある')
    else:
        print(f'→ 中間的な結果 ({n_collapsed}崩壊/{len(_S41_SEEDS)})')
        print('  崩壊条件の閾値が存在する可能性')
    print()
    print('Done.')
