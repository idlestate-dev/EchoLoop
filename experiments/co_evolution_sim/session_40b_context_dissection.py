"""Session 40b: seed=43 の詳細分析

Session 40の発見:
  seed=43だけ C0-C1差=+20% という明確な文脈依存行動が出た
  他のseedは +2〜+4% 程度

問い:
  1. seed=43のゲノムは何が構造的に違うか？
     → ネットワーク構造（エッジ数、クラスタリング、経路長）
     → depletion_rate、edge_add_prob などのゲノムパラメータ

  2. seed=43の結果は再現するか？
     → 同じseed=43で再度進化させて同じ結果が出るか
     → seed=43環境で異なる初期値でも出るか

  3. 「捕食者がいる時は北へ逃げる」という行動は
     どのステップで学習されたか？
     → 進化の途中経過を記録して、何世代目で文脈依存が出たか

  4. seed=43の行動をより詳細に分析
     → 捕食者の休眠サイクルと食事タイミングの相関
     → pred_flagが0になった直後に食事が増えるか
"""

import os
from collections import defaultdict

import numpy as np
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
from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_27_tm_resources import _s27_update_resources
from session_31_grid_sweep import WorldConfig
from session_34_pursuit_x_s33 import (
    _S34_PRED_SPEED, _s34_pred_step,
)
from session_37_node_sweep import (
    _make_tau_arr, _make_genome, _mutate_genome,
    _propagate, _hebb,
    aggregate_context_actions,
)
from session_40_unavoidable import (
    _S40_N, _S40_HP_DECAY, _S40_PURSUIT, _S40_CFG,
    _S40_FOOD_POS, _S40_T_LONG, _S40_SEEDS,
    _s40_init_foods, _s40_init_pred_on_food, _s40_inp5,
    _s40_run_ep, _s40_run_context_log, _s40_evolve,
)

# ── 定数 ──────────────────────────────────────────────────────────────────────

_S40B_SEED_FOCUS = 43          # 詳細分析対象
_S40B_N_REPEAT   = 10          # 再現確認の繰り返し数
_S40B_T_DETAIL   = 5000        # 詳細観察の長さ
_S40B_CHUNK      = 50          # pred_flag→食事タイミング分析のウィンドウ


# ── 分析1: ゲノムパラメータの比較 ────────────────────────────────────────────

def analyze_genome_params(genomes_by_seed: dict):
    """全seedのゲノムパラメータを比較。

    seed=43が他のseedと何が違うかを確認。
    """
    print('\n[分析1] ゲノムパラメータ比較')
    print(f'  {"seed":>5}  {"depl":>6}  {"edge_add":>9}  '
          f'{"act_ratio":>10}  {"n_edges":>8}  {"density":>8}')
    for seed, g in sorted(genomes_by_seed.items()):
        n_edges  = g['G'].number_of_edges()
        n_nodes  = g['n']
        density  = n_edges / (n_nodes * (n_nodes - 1))
        marker   = ' ←' if seed == _S40B_SEED_FOCUS else ''
        print(f'  {seed:>5}  {g["depletion_rate"]:>6.3f}  '
              f'{g["edge_add_prob"]:>9.3f}  '
              f'{g["activity_ratio"]:>10.3f}  '
              f'{n_edges:>8d}  {density:>8.3f}{marker}')


# ── 分析2: ネットワーク構造の比較 ────────────────────────────────────────────

def analyze_network_structure(genomes_by_seed: dict):
    """入力→出力の経路長・クラスタリングを比較。"""
    import networkx as nx

    print('\n[分析2] ネットワーク構造比較')
    print(f'  {"seed":>5}  {"avg_path":>9}  {"clustering":>11}  '
          f'{"inp_out_conn":>13}')

    for seed, g in sorted(genomes_by_seed.items()):
        G = g['G']

        # 入力ノード(0-4)→出力ノード(5-9)への接続性
        inp_nodes = list(range(5))
        out_nodes = list(range(5, 10))
        conn_count = 0
        path_lengths = []
        for inp in inp_nodes:
            for out in out_nodes:
                try:
                    length = nx.shortest_path_length(G, inp, out)
                    path_lengths.append(length)
                    conn_count += 1
                except nx.NetworkXNoPath:
                    pass

        avg_path    = float(np.mean(path_lengths)) if path_lengths else float('inf')
        clustering  = float(nx.average_clustering(G))
        marker      = ' ←' if seed == _S40B_SEED_FOCUS else ''
        print(f'  {seed:>5}  {avg_path:>9.2f}  {clustering:>11.3f}  '
              f'{conn_count:>13d}/{len(inp_nodes)*len(out_nodes)}{marker}')


# ── 分析3: 再現性確認（seed=43を複数回進化） ──────────────────────────────────

def analyze_reproducibility(cfg: WorldConfig, seed: int,
                             n_repeat: int = _S40B_N_REPEAT):
    """同じseedで複数回進化させて、C0-C1差の分布を確認。"""
    print(f'\n[分析3] seed={seed} の再現性確認（{n_repeat}回）')
    results = []
    for trial in range(n_repeat):
        # seedをずらして独立した進化を実現
        trial_seed = seed + trial * 1000
        best, _ = _s40_evolve(cfg, _S40_N, seed=trial_seed)
        rng_ep = np.random.default_rng(trial_seed + 40200)
        G_ep   = best['G'].copy()
        W_ep   = best['W'].copy()
        log_ep, _ = _s40_run_context_log(
            cfg, G_ep, W_ep, best, rng_ep, T=_S40_T_LONG)
        _, fracs, totals = aggregate_context_actions(log_ep)
        c0 = fracs[0, 4]
        c1 = fracs[1, 4]
        diff = c0 - c1
        marker = ' *** 大きな差' if diff > 0.10 else ''
        print(f'  trial={trial:2d} (seed={trial_seed}): '
              f'C0={c0:.0%}  C1={c1:.0%}  diff={diff:+.0%}{marker}')
        results.append({'trial': trial, 'seed': trial_seed,
                        'c0': c0, 'c1': c1, 'diff': diff})

    diffs = [r['diff'] for r in results]
    n_pos = sum(d > 0 for d in diffs)
    n_large = sum(d > 0.10 for d in diffs)
    print(f'\n  要約: mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}  '
          f'C0>C1: {n_pos}/{n_repeat}  diff>10%: {n_large}/{n_repeat}')
    return results


# ── 分析4: 捕食者休眠サイクルと食事タイミングの相関 ─────────────────────────

def analyze_timing(cfg: WorldConfig, genome: dict, seed: int,
                   T: int = _S40B_T_DETAIL):
    """pred_flagが0→1に変わる前後の食事行動を分析。

    捕食者が休眠（pred_flag=0）→ 活動（pred_flag=1）の遷移の前後で
    食事行動がどう変化するかを確認。
    """
    print(f'\n[分析4] 捕食者休眠サイクルと食事タイミング（T={T}）')

    n              = genome['n']
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    rng       = np.random.default_rng(seed + 40400)
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

    G = genome['G'].copy()
    W = genome['W'].copy()

    # 記録
    pred_flag_log  = []
    action_log     = []
    food_flag_log  = []

    for step in range(T):
        if hp <= 0:
            row, col   = center, center
            hp         = float(cfg.hp_start)
            food_positions = _s40_init_foods()
            food_avail = [True]
            food_timer = [0]
            pred_pos   = _s40_init_pred_on_food()
            pred_resources = 1.0
            pred_dormant   = False
            resources  = np.ones(n)
            activity   = np.zeros(n)

        if step % _S34_PRED_SPEED == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                _S40_PURSUIT, pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE

        inp5      = _s40_inp5(cfg, row, col, hp,
                              food_positions, food_avail, pred_pos)
        food_flag = int(inp5[3])
        pred_flag = int(inp5[4])

        for _ in range(_N_PROP):
            activity = _propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, n), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, _make_tau_arr(n), depletion_rate)

        hp -= _S40_HP_DECAY
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))

        pred_flag_log.append(pred_flag)
        food_flag_log.append(food_flag)
        action_log.append(action)

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
            _hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    pred_flag_arr  = np.array(pred_flag_log)
    food_flag_arr  = np.array(food_flag_log)
    action_arr     = np.array(action_log)
    eat_arr        = (action_arr == 4).astype(int)

    # pred_flag変化点の前後での食事率
    transitions_1to0 = []  # 捕食者が去る（危険解除）
    transitions_0to1 = []  # 捕食者が来る（危険発生）

    window = 10
    for t in range(1, T):
        if pred_flag_arr[t-1] == 1 and pred_flag_arr[t] == 0:
            # pred_flag: 1→0（捕食者が去った）
            after = eat_arr[t:min(t+window, T)]
            transitions_1to0.append(float(np.mean(after)) if len(after) > 0 else 0)
        elif pred_flag_arr[t-1] == 0 and pred_flag_arr[t] == 1:
            # pred_flag: 0→1（捕食者が来た）
            after = eat_arr[t:min(t+window, T)]
            transitions_0to1.append(float(np.mean(after)) if len(after) > 0 else 0)

    print(f'  pred_flag 1→0（捕食者去る）後{window}stepsの食事率:')
    if transitions_1to0:
        print(f'    mean={np.mean(transitions_1to0):.0%}  '
              f'n={len(transitions_1to0)}回')
    print(f'  pred_flag 0→1（捕食者来る）後{window}stepsの食事率:')
    if transitions_0to1:
        print(f'    mean={np.mean(transitions_0to1):.0%}  '
              f'n={len(transitions_0to1)}回')

    if transitions_1to0 and transitions_0to1:
        diff = np.mean(transitions_1to0) - np.mean(transitions_0to1)
        print(f'  差（去る後 - 来る後）: {diff:+.0%}')
        if diff > 0.05:
            print(f'  → 捕食者が去った後に食事が増える = 文脈依存あり')
        else:
            print(f'  → 捕食者の有無で食事タイミングは変わらない')

    return {
        'pred_flag':       pred_flag_arr,
        'food_flag':       food_flag_arr,
        'eat':             eat_arr,
        'transitions_1to0': transitions_1to0,
        'transitions_0to1': transitions_0to1,
    }


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_analysis(timing_data, seed,
                  fname='images/session_40b/results_s40b_timing.png'):
    """食事タイミングと捕食者フラグの関係を可視化。"""
    pred = timing_data['pred_flag']
    food = timing_data['food_flag']
    eat  = timing_data['eat']
    t1to0 = timing_data['transitions_1to0']
    t0to1 = timing_data['transitions_0to1']

    # 最初の500stepsを可視化
    T_show = min(500, len(pred))

    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(
        f'Session 40b: seed={seed} の詳細分析\n'
        f'捕食者休眠サイクルと食事タイミングの相関',
        fontsize=13,
    )

    t = np.arange(T_show)

    ax = axes[0]
    ax.fill_between(t, pred[:T_show], alpha=0.4, color='tomato',
                    label='pred_flag')
    ax.fill_between(t, food[:T_show], alpha=0.4, color='seagreen',
                    label='food_flag')
    ax.set_ylabel('Flag')
    ax.set_title('捕食者フラグ(赤)と食料フラグ(緑)の時系列')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.3)

    ax = axes[1]
    ax.fill_between(t, eat[:T_show], alpha=0.6, color='steelblue',
                    label='食事行動')
    ax.set_ylabel('食事(1=食べた)')
    ax.set_title('食事行動の時系列\n捕食者が去った直後に食べているか？')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.3)

    ax = axes[2]
    if t1to0 and t0to1:
        categories = ['捕食者が去った後\n(pred: 1→0)', '捕食者が来た後\n(pred: 0→1)']
        means = [np.mean(t1to0), np.mean(t0to1)]
        sems  = [scipy_stats.sem(t1to0), scipy_stats.sem(t0to1)]
        colors = ['seagreen', 'tomato']
        bars = ax.bar(categories, means, color=colors, alpha=0.85,
                      edgecolor='white', yerr=sems, capsize=5)
        ax.set_ylabel('食事率（次の10steps）')
        ax.set_title(f'捕食者遷移後の食事率\n'
                     f'差={np.mean(t1to0)-np.mean(t0to1):+.0%}  '
                     f'n_1to0={len(t1to0)}  n_0to1={len(t0to1)}')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.0%}', ha='center', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, '遷移データ不足', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
    ax.set_ylim(0, max(max(means)*1.3, 0.3) if t1to0 and t0to1 else 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_reproducibility(repro_results,
                         fname='images/session_40b/results_s40b_repro.png'):
    diffs  = [r['diff']  for r in repro_results]
    trials = [r['trial'] for r in repro_results]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f'Session 40b: seed={_S40B_SEED_FOCUS} の再現性確認\n'
        f'同環境で{_S40B_N_REPEAT}回進化させた時のC0-C1差の分布',
        fontsize=13,
    )

    colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(trials, diffs, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0,    color='black', linewidth=1.5)
    ax.axhline(0.10, color='gray',  linestyle='--', linewidth=1.5,
               label='顕著な文脈依存の閾値(+10%)')
    ax.axhline(0.02, color='lightgray', linestyle='--', linewidth=1,
               label='Session 38基準(+2%)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title(f'mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}  '
                 f'C0>C1: {sum(d>0 for d in diffs)}/{len(diffs)}  '
                 f'diff>10%: {sum(d>0.10 for d in diffs)}/{len(diffs)}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.015,
                f'{d:+.0%}', ha='center', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S40_CFG

    print('=== Session 40b: seed=43 の詳細分析 ===')
    print(f'N={_S40_N}  hp_decay={_S40_HP_DECAY}  pp={_S40_PURSUIT}')
    print()

    # ── 全seedのゲノムを再進化して取得 ────────────────────────────────────
    print('[全seedのゲノム取得]')
    genomes_by_seed = {}
    for seed in _S40_SEEDS:
        print(f'  seed={seed} 進化中...')
        best, _ = _s40_evolve(cfg, _S40_N, seed=seed)
        genomes_by_seed[seed] = best

    # ── 分析1: ゲノムパラメータ比較 ───────────────────────────────────────
    analyze_genome_params(genomes_by_seed)

    # ── 分析2: ネットワーク構造比較 ───────────────────────────────────────
    analyze_network_structure(genomes_by_seed)

    # ── 分析3: seed=43の再現性確認 ────────────────────────────────────────
    repro_results = analyze_reproducibility(
        cfg, seed=_S40B_SEED_FOCUS, n_repeat=_S40B_N_REPEAT)
    plot_reproducibility(repro_results)

    # ── 分析4: seed=43の食事タイミング分析 ───────────────────────────────
    best43 = genomes_by_seed[_S40B_SEED_FOCUS]
    timing_data = analyze_timing(cfg, best43, seed=_S40B_SEED_FOCUS)
    plot_analysis(timing_data, seed=_S40B_SEED_FOCUS)

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 40b Summary ===')
    print()
    repro_diffs = [r['diff'] for r in repro_results]
    n_large = sum(d > 0.10 for d in repro_diffs)
    t1to0 = timing_data['transitions_1to0']
    t0to1 = timing_data['transitions_0to1']

    print(f'再現性: {_S40B_N_REPEAT}回中 diff>10% が {n_large}回')
    if n_large >= 3:
        print('→ seed=43の文脈依存行動はある程度再現する')
    else:
        print('→ seed=43の文脈依存行動は偶発的（再現性低い）')

    if t1to0 and t0to1:
        diff_timing = np.mean(t1to0) - np.mean(t0to1)
        print(f'\n捕食者遷移後の食事率差: {diff_timing:+.0%}')
        if diff_timing > 0.05:
            print('→ 捕食者が去った直後に食事が増える')
            print('  = 「捕食者の休眠を待って食べる」行動が確認された')
        else:
            print('→ 捕食者の遷移と食事タイミングに相関なし')

    print()
    print('--- 全体的な結論 ---')
    if n_large >= 3 and (t1to0 and t0to1 and
                          np.mean(t1to0) - np.mean(t0to1) > 0.05):
        print('仮説B支持: Hebbianネットワークでも')
        print('  回避不可能なトレードオフ環境では')
        print('  文脈依存行動が出現する（再現性あり）')
    elif n_large >= 1:
        print('部分的支持: 文脈依存行動は出現するが再現性が低い')
        print('  → 進化の確率的な性質による')
        print('  → 世代数を増やせば頻度が上がる可能性')
    else:
        print('仮説A支持: 偶発的な外れ値であり')
        print('  アーキテクチャ的な限界が示唆される')
    print()
    print('Done.')
