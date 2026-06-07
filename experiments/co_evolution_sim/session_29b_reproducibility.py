"""Session 29b: 再現性確認

Question: Session 29（seed=42）の文脈依存的な行動は
         複数シードで再現されるか？

方針:
  まず10シードの mean_cosine_dist の分布を見る。
  分布を確認してから再現基準を決める（先に閾値を決めない）。

Seeds: 42-51 (10 seeds)
固定条件: pursuit_prob=0.6, pred_depletion=0.05 (Session 29 best)

各シードで実行すること:
  1. 50世代進化（Session 29 Exp A と同じ条件）
  2. 進化後のベストゲノムで Exp B（4文脈 × 25サンプル）
  3. mean_cosine_dist, per-context 行動分布, p値を記録

出力:
  results_s29b_raw.csv        — シード別の生データ
  images/session_29b/results_s29b_distribution.png  — 分布の可視化
  images/session_29b/results_s29b_contexts.png      — 文脈別ヒートマップ
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_29_pursuit import (
    _s29_evolve,
    run_exp_b_context,
    _S29_PURSUIT_PROB_DEF,
    _S29_PRED_DEPLETION,
    _S29_PRED_SPEED,
    _S29_N_GEN,
    _S29_SEED,
)
from session_28_predator import (
    _S28_CONTEXTS,
    _S28_ACTION_NAMES,
)

# ── 定数 ───────────────────────────────────────────────────────────────────────

_SEEDS        = list(range(42, 52))
_PURSUIT_PROB = 0.6                  # Session 29 best
_CSV_PATH     = 'results_s29b_raw.csv'
_DIST_PNG     = 'images/session_29b/results_s29b_distribution.png'
_CTX_PNG      = 'images/session_29b/results_s29b_contexts.png'

_CTX_LABELS   = [c['label'].replace('\n', ' ') for c in _S28_CONTEXTS]
_N_CTX        = len(_S28_CONTEXTS)
_N_ACT        = len(_S28_ACTION_NAMES)


# ── Per-seed 実行 ──────────────────────────────────────────────────────────────

def run_one_seed(seed):
    """1シード分の進化 + Exp B を実行して結果を返す。

    Returns dict:
      seed, mean_cosine_dist,
      action_dists (4×5 array),
      mean_outputs (4×5 array),
      cos_matrix (4×4 array),
      p_values (6要素 list, upper-triangle順),
      min_p,
      dominant_actions (4要素 list of str),
    """
    print(f'\n{"="*60}')
    print(f'Seed {seed}')
    print(f'{"="*60}')

    # 進化
    best, hist = _s29_evolve(
        pursuit_prob=_PURSUIT_PROB,
        predator_speed=_S29_PRED_SPEED,
        pred_depletion=_S29_PRED_DEPLETION,
        seed=seed,
        n_gen=_S29_N_GEN,
    )
    print(f'  進化完了: best_steps={hist["gen_best_steps"][-1]:.1f}  '
          f'depl={best["depletion_rate"]:.3f}')

    # Exp B: 4文脈計測
    exp_b = run_exp_b_context(best, _PURSUIT_PROB, seed=seed)

    mcd     = exp_b['mean_cosine_dist']
    cos_mat = exp_b['cosine_matrix']
    results = exp_b['context_results']
    p_vals  = exp_b['p_values']

    action_dists  = np.array([r['action_dist']  for r in results])  # 4×5
    mean_outputs  = np.array([r['mean_output']   for r in results])  # 4×5

    pairs    = [(i, j) for i in range(_N_CTX) for j in range(i + 1, _N_CTX)]
    p_list   = [p_vals[pair] for pair in pairs]
    min_p    = float(min(p_list))
    dom_acts = [_S28_ACTION_NAMES[int(np.argmax(r['action_dist']))] for r in results]

    print(f'  mean_cosine_dist = {mcd:.4f}  min_p = {min_p:.4f}')
    for lbl, dom in zip(_CTX_LABELS, dom_acts):
        print(f'    [{lbl}] → {dom}')

    return {
        'seed':             seed,
        'mean_cosine_dist': mcd,
        'action_dists':     action_dists,
        'mean_outputs':     mean_outputs,
        'cos_matrix':       cos_mat,
        'p_values':         p_list,
        'min_p':            min_p,
        'dominant_actions': dom_acts,
        'best_steps':       hist['gen_best_steps'][-1],
        'depl':             best['depletion_rate'],
    }


# ── CSV 保存 ───────────────────────────────────────────────────────────────────

def save_csv(records, path=_CSV_PATH):
    """シード別の生データをCSVに保存。"""
    pairs = [(i, j) for i in range(_N_CTX) for j in range(i + 1, _N_CTX)]
    pair_labels = [f'cos_{i}{j}' for i, j in pairs]
    p_labels    = [f'p_{i}{j}'   for i, j in pairs]
    act_labels  = [f'act_ctx{c}_act{a}'
                   for c in range(_N_CTX) for a in range(_N_ACT)]
    dom_labels  = [f'dom_ctx{c}' for c in range(_N_CTX)]

    fieldnames = (
        ['seed', 'mean_cosine_dist', 'min_p', 'best_steps', 'depl']
        + pair_labels + p_labels + act_labels + dom_labels
    )

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                'seed':             r['seed'],
                'mean_cosine_dist': f'{r["mean_cosine_dist"]:.6f}',
                'min_p':            f'{r["min_p"]:.6f}',
                'best_steps':       f'{r["best_steps"]:.1f}',
                'depl':             f'{r["depl"]:.4f}',
            }
            for k, (i, j) in zip(pair_labels, pairs):
                row[k] = f'{r["cos_matrix"][i, j]:.6f}'
            for k, pv in zip(p_labels, r['p_values']):
                row[k] = f'{pv:.6f}'
            for ci in range(_N_CTX):
                for ai in range(_N_ACT):
                    row[f'act_ctx{ci}_act{ai}'] = f'{r["action_dists"][ci, ai]:.4f}'
            for ci in range(_N_CTX):
                row[f'dom_ctx{ci}'] = r['dominant_actions'][ci]
            writer.writerow(row)

    print(f'\nSaved {path}')


# ── 可視化1: mcd 分布 ──────────────────────────────────────────────────────────

def plot_distribution(records, path=_DIST_PNG):
    """mean_cosine_dist の分布と per-pair cosine distance を可視化。"""
    mcds      = [r['mean_cosine_dist'] for r in records]
    min_ps    = [r['min_p']            for r in records]
    seeds     = [r['seed']             for r in records]
    pairs     = [(i, j) for i in range(_N_CTX) for j in range(i + 1, _N_CTX)]
    pair_lbls = [f'C{i}↔C{j}' for i, j in pairs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 29b: 再現性確認\n'
        f'pursuit_prob={_PURSUIT_PROB}  seeds {min(seeds)}–{max(seeds)}  '
        f'n={len(seeds)}',
        fontsize=13,
    )

    # Panel 1: mcd per seed（棒グラフ）
    ax = axes[0]
    colors = ['tomato' if m > np.median(mcds) else 'steelblue' for m in mcds]
    ax.bar(range(len(seeds)), mcds, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(np.mean(mcds),   color='black',  linestyle='--', linewidth=1.5,
               label=f'mean={np.mean(mcds):.4f}')
    ax.axhline(np.median(mcds), color='gray',   linestyle=':',  linewidth=1.5,
               label=f'median={np.median(mcds):.4f}')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds], fontsize=9)
    ax.set_ylabel('mean cosine dist')
    ax.set_title('シード別 mean_cosine_dist\n(赤=中央値以上)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (v, s) in enumerate(zip(mcds, seeds)):
        ax.text(i, v + 0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=7)

    # Panel 2: boxplot + strip
    ax = axes[1]
    bp = ax.boxplot(mcds, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='navy', linewidth=2))
    rng_jitter = np.random.default_rng(0)
    jitter = rng_jitter.uniform(-0.1, 0.1, len(mcds))
    ax.scatter(np.ones(len(mcds)) + jitter, mcds,
               color='tomato', zorder=5, s=60, alpha=0.85)
    # seed=42 を赤丸で強調
    idx42 = seeds.index(42) if 42 in seeds else None
    if idx42 is not None:
        ax.scatter([1 + jitter[idx42]], [mcds[idx42]],
                   color='red', zorder=6, s=120, marker='o',
                   label='seed=42 (original)')
    ax.set_ylabel('mean cosine dist')
    ax.set_title('分布（Boxplot + Strip）\n赤丸=seed=42（元実験）')
    ax.set_xticks([1])
    ax.set_xticklabels(['全10シード'])
    if idx42 is not None:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    # 統計サマリーをテキストで
    ax.text(0.97, 0.97,
            f'mean  = {np.mean(mcds):.4f}\n'
            f'std   = {np.std(mcds):.4f}\n'
            f'min   = {np.min(mcds):.4f}\n'
            f'max   = {np.max(mcds):.4f}\n'
            f'p<0.05: {sum(p < 0.05 for p in min_ps)}/{len(min_ps)} seeds',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Panel 3: per-pair cosine distance（6ペア × 10シード）
    ax = axes[2]
    pair_data = np.array([[r['cos_matrix'][i, j] for i, j in pairs]
                          for r in records])  # 10×6
    colors_p = plt.cm.tab10(np.linspace(0, 0.6, len(pairs)))
    for pi, (lbl, col) in enumerate(zip(pair_lbls, colors_p)):
        ax.plot(range(len(seeds)), pair_data[:, pi],
                marker='o', markersize=5, color=col,
                linewidth=1.5, label=lbl, alpha=0.85)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds], fontsize=9)
    ax.set_ylabel('cosine dist')
    ax.set_title('文脈ペア別 cosine distance\n(どのペアが差を生んでいるか)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ── 可視化2: 文脈別ヒートマップ ────────────────────────────────────────────────

def plot_contexts(records, path=_CTX_PNG):
    """文脈別の行動分布をシード×文脈のヒートマップで可視化。"""
    seeds  = [r['seed'] for r in records]
    n_seed = len(seeds)

    # dominant action per (seed, context)
    dom_matrix = np.array([
        [int(np.argmax(r['action_dists'][ci])) for ci in range(_N_CTX)]
        for r in records
    ])  # n_seed × 4

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Session 29b: 文脈別行動パターン\n'
        f'pursuit_prob={_PURSUIT_PROB}  seeds {min(seeds)}–{max(seeds)}',
        fontsize=13,
    )

    # Panel 1: 主行動ヒートマップ（整数値 → カラーマップ）
    ax = axes[0]
    cmap = plt.cm.get_cmap('tab10', _N_ACT)
    im = ax.imshow(dom_matrix, cmap=cmap, vmin=0, vmax=_N_ACT - 1, aspect='auto')
    ax.set_xticks(range(_N_CTX))
    ax.set_xticklabels(_CTX_LABELS, fontsize=9)
    ax.set_yticks(range(n_seed))
    ax.set_yticklabels([f'seed={s}' for s in seeds], fontsize=9)
    ax.set_title('主行動（文脈 × シード）\n色=行動の種類')
    for si in range(n_seed):
        for ci in range(_N_CTX):
            ai = dom_matrix[si, ci]
            ax.text(ci, si, _S28_ACTION_NAMES[ai],
                    ha='center', va='center', fontsize=8,
                    color='white' if ai in (1, 2, 3) else 'black')
    cbar = plt.colorbar(im, ax=ax, ticks=range(_N_ACT), shrink=0.8)
    cbar.set_ticklabels(_S28_ACTION_NAMES)

    # Panel 2: mean_cosine_dist と主行動の一致率
    ax = axes[1]
    # 期待パターン: [食料近捕食者遠→食事, 食料近捕食者近→非食事,
    #               食料遠捕食者近→非食事, 食料遠捕食者遠→非食事]
    # 「食料近捕食者遠で食事(=4)」かつ「食料遠捕食者近で非食事(!=4)」が出ているか
    expected = [
        ('食料近・捕食者遠 → 食事', lambda r: int(np.argmax(r['action_dists'][0])) == 4),
        ('食料近・捕食者近 → 非食事', lambda r: int(np.argmax(r['action_dists'][1])) != 4),
        ('食料遠・捕食者近 → 非食事', lambda r: int(np.argmax(r['action_dists'][2])) != 4),
        ('食料遠・捕食者遠 → 非食事', lambda r: int(np.argmax(r['action_dists'][3])) != 4),
    ]
    exp_labels = [e[0] for e in expected]
    exp_counts = [sum(1 for r in records if e[1](r)) for e in expected]
    colors_e   = ['seagreen' if c >= 8 else 'gold' if c >= 5 else 'tomato'
                  for c in exp_counts]

    ax.barh(range(len(expected)), exp_counts, color=colors_e, alpha=0.85,
            edgecolor='white')
    ax.axvline(8, color='black', linestyle='--', linewidth=1.5,
               label='再現基準候補 (8/10)')
    ax.set_yticks(range(len(expected)))
    ax.set_yticklabels(exp_labels, fontsize=9)
    ax.set_xlim(0, n_seed + 0.5)
    ax.set_xlabel('該当シード数 / 10')
    ax.set_title('期待される行動パターンの出現率\n(緑=8+/10, 黄=5-7/10, 赤=<5/10)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    for i, c in enumerate(exp_counts):
        ax.text(c + 0.1, i, f'{c}/10', va='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 29b: 再現性確認 ===')
    print(f'Seeds: {_SEEDS}')
    print(f'pursuit_prob={_PURSUIT_PROB}  pred_depletion={_S29_PRED_DEPLETION}')
    print(f'n_gen={_S29_N_GEN}  ※先に分布を見て基準を決める')
    print()

    records = []
    for seed in _SEEDS:
        rec = run_one_seed(seed)
        records.append(rec)

    # CSV保存
    save_csv(records)

    # 可視化
    plot_distribution(records)
    plot_contexts(records)

    # サマリー出力
    mcds   = [r['mean_cosine_dist'] for r in records]
    min_ps = [r['min_p']            for r in records]

    print('\n' + '=' * 60)
    print('=== Session 29b Summary ===')
    print()
    print(f'mean_cosine_dist across 10 seeds:')
    print(f'  mean   = {np.mean(mcds):.4f}')
    print(f'  std    = {np.std(mcds):.4f}')
    print(f'  min    = {np.min(mcds):.4f}')
    print(f'  max    = {np.max(mcds):.4f}')
    print(f'  median = {np.median(mcds):.4f}')
    print()
    print(f'統計的有意性 (min_p < 0.05): {sum(p < 0.05 for p in min_ps)}/10 seeds')
    print()
    print('シード別結果:')
    for r in records:
        doms = ' / '.join(r['dominant_actions'])
        print(f'  seed={r["seed"]}: mcd={r["mean_cosine_dist"]:.4f}  '
              f'min_p={r["min_p"]:.4f}  [{doms}]')

    print()
    print('--- 観察してから判断すること ---')
    print('1. mcd の分布は集中しているか、ばらついているか？')
    print('2. seed=42 は外れ値か、典型的な値か？')
    print('3. 「食事/非食事」の切り替えは何シードで出ているか？')
    print('4. この結果は「文脈依存的な行動が生まれた」と主張できるか？')
    print()
    print('Done.')
