"""Session 42b: Phase2でのネットワーク圧縮仮説の検証

仮説:
  「危険環境（Phase2）ではネットワークが圧縮され、
   重要なエッジだけが残る」

  N=25: 圧縮後に重要エッジが集中 → 文脈依存が強い
  N=40: 圧縮後もエッジが分散    → 文脈依存が弱い

検証:
  Session 42のchunksデータ（100stepごとのn_edges推移）を使う
  + Phase2を再実行してエッジの変化を詳しく記録

具体的に見ること:
  1. n_edgesの推移: Phase2で減少するか、安定するか、増加するか
  2. どのエッジが残るか: 入力→内部→出力の「重要経路」が残るか
  3. N=25 vs N=40: 圧縮のパターンが違うか
  4. 圧縮後のネットワーク構造と文脈依存の相関

注目ポイント:
  food_flag (node3) → 出力ノード(5-9) の経路が残るか
  pred_flag (node4) → 出力ノード(5-9) の経路が残るか
  → 「重要な入力への経路が選択的に保存される」なら
    圧縮仮説を支持
"""

import argparse
import csv
import json
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
    _S40_PURSUIT, _S40_CFG, _S40_SEEDS,
    _s40_init_foods, _s40_init_pred_on_food, _s40_inp5,
)
from session_41_curriculum import (
    _S41_PHASE1_HP_DECAY, _S41_PHASE2_HP_DECAY,
    _S41_CFG_PHASE1, _S41_CFG_PHASE2,
    _hebb_s10_style, _s41_run_ep_phase1,
)
from session_42_curriculum_long import (
    _S42_SEEDS, _S42_PHASE1_N_GEN, _S42_STABLE_THRESH,
    _s42_evolve_phase1,
)

# ── 定数 ──────────────────────────────────────────────────────────────────────

_S42B_SEED      = _S28_SEED
_S42B_SEEDS     = _S42_SEEDS       # [42, 43, 44]
_S42B_NODE_SIZES = [25, 40]
_S42B_T_PHASE2  = 5000
_S42B_CHUNK     = 100
_S42B_T_CONTEXT = 2000

# 入力ノードのインデックス
_INP_X         = 0
_INP_Y         = 1
_INP_HP        = 2
_INP_FOOD_FLAG = 3
_INP_PRED_FLAG = 4
_N_INP         = 5
_N_OUT         = 5


# ── エッジ変化の詳細記録 ──────────────────────────────────────────────────────

def _s42b_observe_with_edge_tracking(cfg: WorldConfig, G_init, genome, rng,
                                      hp_decay=_S41_PHASE2_HP_DECAY,
                                      pursuit_prob=_S40_PURSUIT,
                                      T=_S42B_T_PHASE2,
                                      chunk=_S42B_CHUNK):
    """Phase2でのエッジ変化を詳細に追跡。

    チャンクごとに記録:
      n_edges_total: 全エッジ数
      n_edges_inp_out: 入力→出力の経路に関わるエッジ
      n_edges_food_path: food_flag(node3)を含む経路
      n_edges_pred_path: pred_flag(node4)を含む経路
      c0_eat, c1_eat: 文脈別食事率
    """
    n  = genome['n']
    G  = G_init.copy()
    W  = _s28_get_W(G)

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

    ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
    chunks  = []
    log     = []
    chunk_ctx = defaultdict(list)

    # 初期エッジセットを記録
    initial_edges = set(G.edges())

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
            resources, activity, tau_arr, depletion_rate)

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
            # エッジ数の詳細分類
            edges = set(G.edges())
            n_total = len(edges)

            # food_flag(3)を起点または終点に持つエッジ
            n_food = sum(1 for (i, j) in edges
                        if i == _INP_FOOD_FLAG or j == _INP_FOOD_FLAG)

            # pred_flag(4)を起点または終点に持つエッジ
            n_pred = sum(1 for (i, j) in edges
                        if i == _INP_PRED_FLAG or j == _INP_PRED_FLAG)

            # 入力ノード(0-4)→出力ノード(5-9)への直接エッジ
            n_direct = sum(1 for (i, j) in edges
                          if i < _N_INP and _N_INP <= j < _N_INP + _N_OUT)

            # 入力→内部→出力の経路が存在するか
            try:
                n_inp_out_paths = sum(
                    1 for inp in range(_N_INP)
                    for out in range(_N_INP, _N_INP + _N_OUT)
                    if nx.has_path(G, inp, out))
            except Exception:
                n_inp_out_paths = 0

            # 新規エッジ（Phase1では存在しなかったもの）
            n_new = len(edges - initial_edges)
            # 消えたエッジ
            n_lost = len(initial_edges - edges)

            c0_eat = float(np.mean(chunk_ctx[0])) if chunk_ctx[0] else float('nan')
            c1_eat = float(np.mean(chunk_ctx[1])) if chunk_ctx[1] else float('nan')
            diff   = c0_eat - c1_eat if not (
                np.isnan(c0_eat) or np.isnan(c1_eat)) else float('nan')

            chunks.append({
                'step':          step + 1,
                'n_total':       n_total,
                'n_food_edges':  n_food,
                'n_pred_edges':  n_pred,
                'n_direct':      n_direct,
                'n_inp_out':     n_inp_out_paths,
                'n_new':         n_new,
                'n_lost':        n_lost,
                'c0_eat':        c0_eat,
                'c1_eat':        c1_eat,
                'diff':          diff,
            })
            chunk_ctx = defaultdict(list)

    return G, W, chunks, log


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_compression(results_all,
                     fname='images/session_42b/results_s42b_compression.png'):
    """N=25 vs N=40 のネットワーク圧縮パターンを比較。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Session 42b: Phase2でのネットワーク圧縮\n'
        'N=25 vs N=40  「危険環境で重要経路が選択的に保存されるか」',
        fontsize=13,
    )

    _palette = ['steelblue', 'tomato', 'seagreen', 'purple', 'orange', 'brown']
    _seed_order = {s: i for i, s in enumerate(sorted({r['seed'] for r in results_all}))}
    colors_seed = {s: _palette[i % len(_palette)] for s, i in _seed_order.items()}
    ls_by_n = {25: '-', 40: '--'}

    # Panel 1: n_edges総数の推移
    ax = axes[0][0]
    for r in results_all:
        steps  = [c['step']    for c in r['chunks']]
        totals = [c['n_total'] for c in r['chunks']]
        ax.plot(steps, totals,
                ls_by_n[r['n']], linewidth=2,
                color=colors_seed[r['seed']],
                label=f'N={r["n"]} s{r["seed"]}',
                alpha=0.8)
        ax.axhline(r['phase1_edges'], linestyle=':',
                   color=colors_seed[r['seed']],
                   alpha=0.4)
    ax.set_xlabel('Step')
    ax.set_ylabel('n_edges（総数）')
    ax.set_title('全エッジ数の推移\n点線=Phase1終了時')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: food_flag関連エッジの推移
    ax = axes[0][1]
    for r in results_all:
        steps  = [c['step']          for c in r['chunks']]
        n_food = [c['n_food_edges']  for c in r['chunks']]
        ax.plot(steps, n_food,
                ls_by_n[r['n']], linewidth=2,
                color=colors_seed[r['seed']],
                label=f'N={r["n"]} s{r["seed"]}',
                alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('food_flag関連エッジ数')
    ax.set_title('food_flag(node3)関連エッジの推移\n'
                 '「食料情報の経路」が維持されるか')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: pred_flag関連エッジの推移
    ax = axes[0][2]
    for r in results_all:
        steps  = [c['step']          for c in r['chunks']]
        n_pred = [c['n_pred_edges']  for c in r['chunks']]
        ax.plot(steps, n_pred,
                ls_by_n[r['n']], linewidth=2,
                color=colors_seed[r['seed']],
                label=f'N={r["n"]} s{r["seed"]}',
                alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('pred_flag関連エッジ数')
    ax.set_title('pred_flag(node4)関連エッジの推移\n'
                 '「捕食者情報の経路」が維持されるか')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: C0-C1差の推移
    ax = axes[1][0]
    for r in results_all:
        steps = [c['step'] for c in r['chunks']]
        diffs = [c['diff'] for c in r['chunks']]
        valid = [(s, d) for s, d in zip(steps, diffs) if not np.isnan(d)]
        if valid:
            vs, vd = zip(*valid)
            ax.plot(vs, vd,
                    ls_by_n[r['n']], linewidth=2,
                    color=colors_seed[r['seed']],
                    label=f'N={r["n"]} s{r["seed"]}',
                    alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('C0-C1差の推移\n文脈依存はどの時点で出るか')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 5: pred_flag経路の割合（pred_edges / total）
    ax = axes[1][1]
    for r in results_all:
        steps = [c['step'] for c in r['chunks']]
        ratio = [c['n_pred_edges'] / max(c['n_total'], 1)
                 for c in r['chunks']]
        ax.plot(steps, ratio,
                ls_by_n[r['n']], linewidth=2,
                color=colors_seed[r['seed']],
                label=f'N={r["n"]} s{r["seed"]}',
                alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('pred_edges / total_edges')
    ax.set_title('pred_flag経路の比率\n「圧縮後に捕食者経路が相対的に増えるか」')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: 終端状態のサマリー
    ax = axes[1][2]
    summary_lines = ['Phase2終了時のネットワーク構造\n']
    summary_lines.append(
        f'{"N":>4} {"seed":>5}  {"total":>6}  {"food":>5}  '
        f'{"pred":>5}  {"pred%":>6}  {"diff":>6}')
    for r in results_all:
        final = r['chunks'][-1]
        d     = r['c0'] - r['c1']
        pred_pct = final['n_pred_edges'] / max(final['n_total'], 1)
        marker = ' ←' if d > 0.05 else ''
        summary_lines.append(
            f'{r["n"]:>4} {r["seed"]:>5}  '
            f'{final["n_total"]:>6}  '
            f'{final["n_food_edges"]:>5}  '
            f'{final["n_pred_edges"]:>5}  '
            f'{pred_pct:>6.0%}  '
            f'{d:>+6.0%}{marker}')

    ax.text(0.02, 0.95, '\n'.join(summary_lines),
            transform=ax.transAxes, va='top', fontsize=8.5,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('終端状態サマリー（← = C0-C1差>5%）')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_compression_ratio(results_all,
                           fname='images/session_42b/results_s42b_ratio.png'):
    """N=25 vs N=40 の圧縮率と文脈依存の散布図。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 42b: ネットワーク圧縮率 vs 文脈依存\n'
        '「圧縮が進むほど文脈依存が強まるか」',
        fontsize=13,
    )

    colors_n = {25: 'steelblue', 40: 'tomato'}
    markers  = {25: 'o', 40: 's'}

    ax = axes[0]
    for r in results_all:
        # 圧縮率 = (Phase1 - Phase2終了) / Phase1
        p1 = r['phase1_edges']
        p2 = r['chunks'][-1]['n_total']
        compression = (p1 - p2) / max(p1, 1)
        diff        = r['c0'] - r['c1']
        ax.scatter(compression, diff,
                   color=colors_n[r['n']],
                   marker=markers[r['n']],
                   s=150, alpha=0.8, zorder=3,
                   label=f'N={r["n"]} s{r["seed"]}')
        ax.annotate(f's{r["seed"]}',
                    (compression, diff),
                    textcoords='offset points',
                    xytext=(5, 5), fontsize=8)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='gray',  linewidth=1, linestyle='--')
    ax.set_xlabel('圧縮率（(P1-P2)/P1）\n正=縮小、負=拡大')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('圧縮率 vs 文脈依存\n右上が「圧縮で文脈依存が強まる」')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r in results_all:
        final     = r['chunks'][-1]
        pred_pct  = final['n_pred_edges'] / max(final['n_total'], 1)
        diff      = r['c0'] - r['c1']
        ax.scatter(pred_pct, diff,
                   color=colors_n[r['n']],
                   marker=markers[r['n']],
                   s=150, alpha=0.8, zorder=3,
                   label=f'N={r["n"]} s{r["seed"]}')
        ax.annotate(f's{r["seed"]}',
                    (pred_pct, diff),
                    textcoords='offset points',
                    xytext=(5, 5), fontsize=8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('pred_flag経路の比率')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('pred_flag経路比率 vs 文脈依存\n'
                 '「捕食者経路の比率が高いほど文脈依存が強い」か')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── 結果保存 ──────────────────────────────────────────────────────────────────

def _s42b_save_results(results_all, out_dir='results/session_42b_result'):
    """サマリーCSV + チャンクCSV + JSONを保存。"""
    os.makedirs(out_dir, exist_ok=True)

    # summary.csv: 1行/run
    summary_path = os.path.join(out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n', 'seed', 'phase1_edges', 'final_edges',
            'compression_rate', 'n_food_edges', 'n_pred_edges',
            'pred_ratio', 'c0_eat', 'c1_eat', 'c0_minus_c1',
        ])
        for r in results_all:
            final = r['chunks'][-1]
            p1 = r['phase1_edges']
            p2 = final['n_total']
            compression = (p1 - p2) / max(p1, 1)
            pred_ratio = final['n_pred_edges'] / max(p2, 1)
            writer.writerow([
                r['n'], r['seed'], p1, p2,
                f'{compression:.4f}',
                final['n_food_edges'], final['n_pred_edges'],
                f'{pred_ratio:.4f}',
                f'{r["c0"]:.4f}', f'{r["c1"]:.4f}',
                f'{r["c0"] - r["c1"]:.4f}',
            ])
    print(f'Saved {summary_path}')

    # chunks_N{n}_s{seed}.csv: チャンク時系列
    for r in results_all:
        chunk_path = os.path.join(
            out_dir, f'chunks_N{r["n"]}_s{r["seed"]}.csv')
        with open(chunk_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=r['chunks'][0].keys())
            writer.writeheader()
            writer.writerows(r['chunks'])
        print(f'Saved {chunk_path}')

    # meta.json: 実行パラメータ
    meta = {
        'node_sizes': sorted({r['n'] for r in results_all}),
        'seeds':      sorted({r['seed'] for r in results_all}),
        'T_phase2':   _S42B_T_PHASE2,
        'T_context':  _S42B_T_CONTEXT,
        'chunk_size': _S42B_CHUNK,
    }
    meta_path = os.path.join(out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved {meta_path}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Session 42b: Phase2 ネットワーク圧縮仮説の検証')
    parser.add_argument(
        '--N', type=int, nargs='+', default=_S42B_NODE_SIZES,
        metavar='N', help='ノードサイズのリスト (default: 25 40)')
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=_S42B_SEEDS,
        metavar='S', help='シードのリスト (default: 42 43 44)')
    parser.add_argument(
        '--T', type=int, default=_S42B_T_PHASE2,
        help=f'Phase2ステップ数 (default: {_S42B_T_PHASE2})')
    parser.add_argument(
        '--out-dir', default='results/session_42b_result',
        help='結果出力ディレクトリ (default: results/session_42b_result)')
    args = parser.parse_args()

    node_sizes = args.N
    seeds      = args.seeds
    T_phase2   = args.T
    out_dir    = args.out_dir

    print('=== Session 42b: ネットワーク圧縮仮説の検証 ===')
    print(f'N={node_sizes}  seeds={seeds}')
    print(f'Phase2: hp_decay={_S41_PHASE2_HP_DECAY}  '
          f'pp={_S40_PURSUIT}  T={T_phase2}')
    print()

    results_all = []

    for n in node_sizes:
        print(f'\n{"#"*55}')
        print(f'# N={n}')
        print(f'{"#"*55}')

        for seed in seeds:
            print(f'\n  seed={seed}  N={n}')

            # Phase1
            print('  [Phase1] 安全環境で進化')
            best, n_edges_p1 = _s42_evolve_phase1(
                _S41_CFG_PHASE1, n, seed=seed)
            print(f'  Phase1完了: n_edges={n_edges_p1}')

            # Phase2（詳細追跡）
            print(f'  [Phase2] 捕食者環境 詳細追跡 (T={T_phase2})')
            rng_p2 = np.random.default_rng(seed + 42200 + n * 100)
            G_p2, W_p2, chunks, log_p2 = _s42b_observe_with_edge_tracking(
                _S41_CFG_PHASE2, best['G'], best, rng_p2,
                T=T_phase2)

            # 文脈別行動（T=2000分）
            _, fracs, totals = aggregate_context_actions(
                log_p2[:_S42B_T_CONTEXT])
            c0 = fracs[0, 4]
            c1 = fracs[1, 4]

            final = chunks[-1]
            print(f'  n_edges: {n_edges_p1} → {final["n_total"]}  '
                  f'food_edges={final["n_food_edges"]}  '
                  f'pred_edges={final["n_pred_edges"]}')
            print(f'  C0食事率={c0:.0%}  C1食事率={c1:.0%}  '
                  f'差={c0-c1:+.0%}')

            # 圧縮率
            compression = (n_edges_p1 - final['n_total']) / max(n_edges_p1, 1)
            pred_pct    = final['n_pred_edges'] / max(final['n_total'], 1)
            print(f'  圧縮率={compression:+.0%}  '
                  f'pred経路比率={pred_pct:.0%}')

            results_all.append({
                'seed':         seed,
                'n':            n,
                'phase1_edges': n_edges_p1,
                'c0':           c0,
                'c1':           c1,
                'chunks':       chunks,
            })

    plot_compression(results_all)
    plot_compression_ratio(results_all)
    _s42b_save_results(results_all, out_dir=out_dir)

    # サマリー
    print('\n' + '=' * 60)
    print('=== Session 42b Summary ===')
    print()

    for n in node_sizes:
        rs = [r for r in results_all if r['n'] == n]
        diffs       = [r['c0'] - r['c1'] for r in rs]
        compressions = [(r['phase1_edges'] - r['chunks'][-1]['n_total'])
                        / max(r['phase1_edges'], 1) for r in rs]
        pred_pcts   = [r['chunks'][-1]['n_pred_edges']
                       / max(r['chunks'][-1]['n_total'], 1) for r in rs]
        print(f'N={n}:')
        print(f'  圧縮率:        mean={np.mean(compressions):+.0%}  '
              f'(正=縮小、負=拡大)')
        print(f'  pred経路比率:  mean={np.mean(pred_pcts):.0%}')
        print(f'  C0-C1差:       mean={np.mean(diffs):+.0%}  '
              f'std={np.std(diffs):.0%}')
        print()

    # 圧縮率とC0-C1差の相関
    all_compressions = [(r['phase1_edges'] - r['chunks'][-1]['n_total'])
                        / max(r['phase1_edges'], 1) for r in results_all]
    all_diffs        = [r['c0'] - r['c1'] for r in results_all]
    all_pred_pcts    = [r['chunks'][-1]['n_pred_edges']
                        / max(r['chunks'][-1]['n_total'], 1)
                        for r in results_all]

    if len(all_compressions) > 2:
        r_comp, p_comp = scipy_stats.pearsonr(all_compressions, all_diffs)
        r_pred, p_pred = scipy_stats.pearsonr(all_pred_pcts, all_diffs)
        print(f'相関分析:')
        print(f'  圧縮率 × C0-C1差:        r={r_comp:.3f}  p={p_comp:.3f}')
        print(f'  pred経路比率 × C0-C1差:  r={r_pred:.3f}  p={p_pred:.3f}')
        print()

    print('--- 圧縮仮説の判断 ---')
    print('圧縮率とC0-C1差が正の相関 → 圧縮仮説支持')
    print('pred経路比率とC0-C1差が正の相関 → 捕食者経路の選択的保存が鍵')
    print()
    print('Done.')
