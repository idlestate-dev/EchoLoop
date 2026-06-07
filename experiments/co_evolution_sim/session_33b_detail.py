"""Session 33b: decay=5 の詳細分析

Session 33でmcd=0.1201が得られたdecay=5の条件を詳細に調べる。

確認すること:
  1. 4文脈の主行動の中身
     → 「食料近・捕食者遠 → 食事」「食料近・捕食者近 → 逃げる」が出ているか
  2. 出力ノードの活動値の大きさ
     → 0.02〜0.08（ほぼフラット）か、それとも有意な差があるか
  3. mcd=0.1201を生み出しているペアはどれか
     → 全6ペアの内訳を確認
  4. 10サンプル → 100サンプルに増やして安定性を確認
     → seed=42の結果が外れ値でないか
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_28_predator import (
    _S28_N, _S28_SEED,
    _S28_CONTEXTS, _S28_ACTION_NAMES,
    _s28_get_W, _s28_cosine_dist, _s28_measure_context,
)
from session_31_grid_sweep import WorldConfig, _s31_measure_context_dep
from session_33_eat_range import (
    _S33_GRID, _S33_FOOD_BIAS, _S33_CFG,
    _s33_evolve,
)

_S33B_HP_DECAY   = 5
_S33B_SEED       = _S28_SEED
_S33B_N_CONTEXT  = 100   # 25 → 100 で安定性確認
_S33B_CONTEXT_T  = 100
_S33B_N_SEEDS    = 5     # 複数seedで外れ値チェック
_S33B_SEEDS      = list(range(42, 42 + _S33B_N_SEEDS))


def _measure_context_detailed(genome, seed, n_context=_S33B_N_CONTEXT):
    """4文脈の詳細計測。出力値の絶対値・ペア別mcdも返す。"""
    G_copy = genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 33200)

    results = []
    for ctx in _S28_CONTEXTS:
        samples = []
        for _ in range(n_context):
            r = _s28_measure_context(
                G_copy, W_copy, genome, ctx, rng, T=_S33B_CONTEXT_T)
            samples.append(r['mean_output'])
        mean_out     = np.mean(samples, axis=0)
        std_out      = np.std(samples, axis=0)
        action_count = np.zeros(5)
        for s in samples:
            action_count[int(np.argmax(s))] += 1
        results.append({
            'mean_output':    mean_out,
            'std_output':     std_out,
            'action_dist':    action_count / n_context,
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

    # ペア別のmcdと p値
    pair_details = {}
    for (i, j) in pairs:
        si = results[i]['output_samples'].max(axis=1)
        sj = results[j]['output_samples'].max(axis=1)
        _, p = scipy_stats.ttest_ind(si, sj)
        lbl_i = _S28_CONTEXTS[i]['label'].replace('\n', ' ')
        lbl_j = _S28_CONTEXTS[j]['label'].replace('\n', ' ')
        pair_details[(i, j)] = {
            'cos_dist': cos_mat[i, j],
            'p_value':  float(p),
            'label':    f'{lbl_i} vs {lbl_j}',
        }

    return {
        'context_results': results,
        'cosine_matrix':   cos_mat,
        'mean_cosine_dist': mcd,
        'pair_details':    pair_details,
    }


def plot_detailed(exp_b, seed,
                  fname='images/session_33b/results_s33b_detail.png'):
    """詳細な文脈依存行動の可視化。"""
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    pairs    = exp_b['pair_details']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f'Session 33b: decay=5 詳細分析  (seed={seed}, n={_S33B_N_CONTEXT}/ctx)\n'
        f'grid={_S33_GRID}x{_S33_GRID}  food_dist={_S33_CFG.food_dist}  '
        f'food_bias={_S33_FOOD_BIAS}  '
        f'mcd={exp_b["mean_cosine_dist"]:.4f}',
        fontsize=12,
    )

    # Panel 1: 出力ノード平均活動（絶対値を確認）
    ax = axes[0][0]
    out_mat = np.array([r['mean_output'] for r in results])
    std_mat = np.array([r['std_output']  for r in results])
    vmax = max(out_mat.max(), 0.01)
    im = ax.imshow(out_mat, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels(
        [f'node{i+5}\n({a})' for i, a in enumerate(_S28_ACTION_NAMES)],
        fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=9)
    ax.set_title(f'出力ノード平均活動\n(vmax={vmax:.3f})')
    for i in range(n_ctx):
        for j in range(5):
            ax.text(j, i,
                    f'{out_mat[i,j]:.3f}\n±{std_mat[i,j]:.3f}',
                    ha='center', va='center', fontsize=6.5,
                    color='white' if out_mat[i,j] > vmax * 0.6 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: cosine距離行列（ペア別の値を確認）
    ax = axes[0][1]
    vmax2 = max(cos_mat.max(), 0.01)
    im2 = ax.imshow(cos_mat, cmap='Blues', vmin=0, vmax=vmax2, aspect='auto')
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=8)
    ax.set_title('Cosine距離行列\n(*=p<0.05)')
    for i in range(n_ctx):
        for j in range(n_ctx):
            pair = (min(i, j), max(i, j))
            if i == j:
                label = '0.000'
            else:
                pd   = pairs[pair]
                sig  = '*' if pd['p_value'] < 0.05 else ''
                label = f'{cos_mat[i,j]:.3f}{sig}'
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='white' if cos_mat[i,j] > vmax2 * 0.6 else 'black')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 3: 行動分布（積み上げ棒）
    ax = axes[0][2]
    act_mat  = np.array([r['action_dist'] for r in results])
    colors_a = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    bottoms  = np.zeros(n_ctx)
    for ai, (aname, col) in enumerate(zip(_S28_ACTION_NAMES, colors_a)):
        bars = ax.bar(range(n_ctx), act_mat[:, ai], bottom=bottoms,
                      color=col, alpha=0.85, label=aname, edgecolor='white')
        # 割合が大きい場合だけラベル表示
        for bi, (b, v) in enumerate(zip(bottoms, act_mat[:, ai])):
            if v > 0.1:
                ax.text(bi, b + v/2, f'{v:.0%}',
                        ha='center', va='center', fontsize=8, color='white',
                        fontweight='bold')
        bottoms += act_mat[:, ai]
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=9)
    ax.set_ylabel('Action ratio')
    ax.set_title('文脈別行動分布\n（食事と逃げが文脈で切り替わるか）')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.15)

    # Panel 4: ペア別cosine距離の棒グラフ（どのペアが差を生んでいるか）
    ax = axes[1][0]
    pair_keys  = sorted(pairs.keys())
    pair_vals  = [pairs[k]['cos_dist'] for k in pair_keys]
    pair_ps    = [pairs[k]['p_value']  for k in pair_keys]
    pair_lbls  = [pairs[k]['label']    for k in pair_keys]
    bar_colors = ['seagreen' if p < 0.05 else 'lightgray'
                  for p in pair_ps]
    ax.barh(range(len(pair_keys)), pair_vals, color=bar_colors, alpha=0.85,
            edgecolor='white')
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_lbls, fontsize=7.5)
    ax.set_xlabel('Cosine distance')
    ax.set_title('ペア別cosine距離\n(緑=p<0.05, 灰=非有意)')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    for i, (v, p) in enumerate(zip(pair_vals, pair_ps)):
        ax.text(v + 0.001, i, f'{v:.4f} (p={p:.3f})',
                va='center', fontsize=7.5)

    # Panel 5: 出力値の分布（boxplot）
    ax = axes[1][1]
    # 食事ノード（node9, action=4）の活動値を文脈別に比較
    eat_node_idx = 4  # 食事アクションのインデックス
    eat_samples  = [r['output_samples'][:, eat_node_idx] for r in results]
    bp = ax.boxplot(eat_samples, patch_artist=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color='black', linewidth=2))
    colors_ctx = ['royalblue', 'tomato', 'seagreen', 'darkorange']
    for patch, col in zip(bp['boxes'], colors_ctx):
        patch.set_facecolor(col)
    ax.set_xticks(range(1, n_ctx + 1))
    ax.set_xticklabels(ctx_lbls, fontsize=8)
    ax.set_ylabel('Output value (食事ノード)')
    ax.set_title('食事ノード活動値の分布\n(文脈間で差があるか)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: 期待される文脈依存パターンのチェック
    ax = axes[1][2]
    # 期待: 食料近・捕食者遠 → 食事率高い
    #       食料近・捕食者近 → 食事率低い（逃げる）
    ctx_names   = [c['label'].replace('\n', ' ') for c in _S28_CONTEXTS]
    eat_rates   = [r['action_dist'][4] for r in results]  # 食事の割合
    escape_rates = [r['action_dist'][0] + r['action_dist'][1] +
                    r['action_dist'][2] + r['action_dist'][3]
                    for r in results]  # 移動系の割合

    x     = np.arange(n_ctx)
    width = 0.35
    ax.bar(x - width/2, eat_rates,    width, color='purple',    alpha=0.85,
           label='食事', edgecolor='white')
    ax.bar(x + width/2, escape_rates, width, color='steelblue', alpha=0.85,
           label='移動（逃げ）', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(ctx_names, fontsize=8)
    ax.set_ylabel('Action ratio')
    ax.set_title('食事 vs 移動の比率\n期待: C0=食事↑, C1=食事↓(捕食者近)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (e, s) in enumerate(zip(eat_rates, escape_rates)):
        ax.text(i - width/2, e + 0.01, f'{e:.0%}', ha='center', fontsize=8)
        ax.text(i + width/2, s + 0.01, f'{s:.0%}', ha='center', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_multi_seed(multi_results,
                    fname='images/session_33b/results_s33b_multiseed.png'):
    """複数seedのmcdと主行動を可視化。外れ値チェック。"""
    seeds = [r['seed'] for r in multi_results]
    mcds  = [r['mcd']  for r in multi_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Session 33b: decay=5 複数seed確認 (n={len(seeds)} seeds)\n'
        f'mcd=0.1201 (seed=42) は外れ値か？',
        fontsize=13,
    )

    ax = axes[0]
    colors = ['tomato' if s == 42 else 'steelblue' for s in seeds]
    ax.bar(range(len(seeds)), mcds, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(np.mean(mcds), color='black', linestyle='--', linewidth=1.5,
               label=f'mean={np.mean(mcds):.4f}')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds], fontsize=9)
    ax.set_ylabel('mean cosine dist')
    ax.set_title('seed別 mcd\n(赤=seed42, 元実験)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (s, m) in enumerate(zip(seeds, mcds)):
        ax.text(i, m + 0.002, f'{m:.4f}', ha='center', fontsize=8)

    ax = axes[1]
    # 各seedの主行動パターンをテキストで表示
    lines = [f'{"seed":>6}  {"mcd":>8}  C0(食近捕遠)  C1(食近捕近)  '
             f'C2(食遠捕近)  C3(食遠捕遠)']
    for r in multi_results:
        doms = r['dominant_actions']
        lines.append(
            f'{r["seed"]:>6}  {r["mcd"]:>8.4f}  '
            f'{doms[0]:>12}  {doms[1]:>12}  '
            f'{doms[2]:>12}  {doms[3]:>12}')
    ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.axis('off')
    ax.set_title('seed別の主行動パターン\n期待: C0→食事, C1→移動')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S33_CFG

    print('=== Session 33b: decay=5 詳細分析 ===')
    print(f'hp_decay={_S33B_HP_DECAY}  grid={_S33_GRID}x{_S33_GRID}  '
          f'food_dist={cfg.food_dist}  food_bias={_S33_FOOD_BIAS}')
    print(f'n_context={_S33B_N_CONTEXT} (Session 33の4倍)')
    print()

    # ── seed=42 で再進化 → 詳細計測 ─────────────────────────────────────────
    print('[Step 1] seed=42 で再進化（decay=5）')
    best42, hist42 = _s33_evolve(cfg, hp_decay=_S33B_HP_DECAY, seed=_S33B_SEED)
    print(f'  steps={hist42["gen_best_steps"][-1]:.1f}  '
          f'food={hist42["gen_food_count"][-1]:.2f}/ep')

    print(f'\n[Step 2] 詳細文脈計測（n={_S33B_N_CONTEXT}/文脈）')
    exp_b42 = _measure_context_detailed(best42, seed=_S33B_SEED)
    mcd42   = exp_b42['mean_cosine_dist']
    print(f'  mcd={mcd42:.4f}  (Session 33: 0.1201)')
    print()

    # 文脈別の主行動と詳細を出力
    ctx_lbls = [c['label'].replace('\n', ' ') for c in _S28_CONTEXTS]
    print('  文脈別の主行動:')
    for lbl, res in zip(ctx_lbls, exp_b42['context_results']):
        dom_i = int(np.argmax(res['action_dist']))
        dom   = _S28_ACTION_NAMES[dom_i]
        rate  = res['action_dist'][dom_i]
        out   = res['mean_output']
        print(f'    [{lbl}]: {dom}({rate:.0%})  '
              f'out={out.round(3)}')

    print('\n  ペア別cosine距離:')
    for (i, j), pd in sorted(exp_b42['pair_details'].items(),
                              key=lambda x: -x[1]['cos_dist']):
        sig = '* p<0.05' if pd['p_value'] < 0.05 else '  n.s.'
        print(f'    {pd["label"]}: {pd["cos_dist"]:.4f}  {sig}')

    plot_detailed(exp_b42, seed=_S33B_SEED)

    # ── 複数seedで外れ値チェック ─────────────────────────────────────────────
    print(f'\n[Step 3] 複数seed確認 (seeds={_S33B_SEEDS})')
    multi_results = []
    for seed in _S33B_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, hist_s = _s33_evolve(cfg, hp_decay=_S33B_HP_DECAY, seed=seed)
        exp_s = _measure_context_detailed(best_s, seed=seed, n_context=25)
        mcd_s = exp_s['mean_cosine_dist']
        doms  = [_S28_ACTION_NAMES[int(np.argmax(r['action_dist']))]
                 for r in exp_s['context_results']]
        print(f'    mcd={mcd_s:.4f}  [{" / ".join(doms)}]')
        multi_results.append({
            'seed':             seed,
            'mcd':              mcd_s,
            'dominant_actions': doms,
            'ctx_data':         exp_s,
        })

    plot_multi_seed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 33b Summary ===')
    print()
    mcds = [r['mcd'] for r in multi_results]
    print(f'mcd across {len(_S33B_SEEDS)} seeds:')
    print(f'  mean={np.mean(mcds):.4f}  std={np.std(mcds):.4f}  '
          f'min={np.min(mcds):.4f}  max={np.max(mcds):.4f}')
    print()
    print('期待パターン確認（C0=食事↑, C1=捕食者近で食事↓）:')
    for r in multi_results:
        doms = r['dominant_actions']
        c0_eat = doms[0] == '食事'
        c1_esc = doms[1] != '食事'
        ok = '✓' if (c0_eat and c1_esc) else '✗'
        print(f'  seed={r["seed"]}: {ok}  '
              f'C0={doms[0]}  C1={doms[1]}  '
              f'C2={doms[2]}  C3={doms[3]}  '
              f'mcd={r["mcd"]:.4f}')

    print()
    print('--- 観察してから判断すること ---')
    print('1. mcd=0.1201はseed=42の外れ値か、複数seedで再現するか？')
    print('2. 出力値の絶対値は有意な差があるか（0.02〜0.08のフラットではないか）？')
    print('3. 「食料近・捕食者遠→食事」「食料近・捕食者近→逃げ」が出ているか？')
    print('4. mcdを生んでいるペアはどれか？')
    print()
    print('Done.')
