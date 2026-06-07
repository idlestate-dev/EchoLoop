"""Session 34b: エピソード中の実際の行動を文脈別に計測

Session 34の問題:
  静的計測（固定inp5を注入）では mcd=0.0013 と低かった。
  しかし進化個体は固定戦略を+11〜12steps上回っている。
  → 静的計測と実際の行動が乖離している可能性。

この実験でやること:
  エピソードを実際に走らせて、各ステップの「文脈」と「行動」を記録する。
  文脈を以下の4つに分類してステップを仕分ける:
    C0: food_flag=1 & pred_flag=0  （食料近・捕食者遠）
    C1: food_flag=1 & pred_flag=1  （食料近・捕食者近）← トレードオフ
    C2: food_flag=0 & pred_flag=1  （食料遠・捕食者近）
    C3: food_flag=0 & pred_flag=0  （食料遠・捕食者遠）

  各文脈で実際にどの行動を取っているかを集計する。
  → 「文脈を読んだ行動」が出ているかを直接確認できる。

固定条件:
  pp=0.9（Session 34でmcd最大・固定戦略との差+11.4steps）
  hp_decay=5, grid=8x8, food_dist=2

実験:
  A: seed=42 のベストゲノムで長エピソード（T=2000）を実行
     → 文脈別の行動分布を集計
  B: ランダムゲノム（対照）との比較
  C: 複数seed（42〜46）での再現確認
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from collections import defaultdict

from session_28_predator import (
    _S28_N, _S28_OUT_START, _S28_OUT_END,
    _S28_HP_MAX, _S28_FOOD_VALUE, _S28_FOOD_RESPAWN,
    _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_SEED,
    _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACTION_NAMES,
    _s28_get_W, _s28_propagate, _s28_make_tau_arr,
    _s28_hebb, _s28_make_genome,
)
from session_27_tm_resources import _s27_update_resources
from session_10_embodied_output import _N_PROP, _K
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_31_grid_sweep import (
    WorldConfig, _s31_init_foods, _s31_init_pred, _s31_inp5,
)
from session_34_pursuit_x_s33 import (
    _S34_HP_DECAY, _S34_CFG, _S34_PRED_SPEED,
    _S34_PRED_DEPLETION, _S34_PRED_RECOVERY,
    _S34_PRED_DORMANT_LO, _S34_PRED_DORMANT_HI,
    _s34_pred_step, _s34_evolve,
)

# ── 定数 ──────────────────────────────────────────────────────────────────────

_S34B_SEED        = _S28_SEED
_S34B_PURSUIT     = 0.9          # Session 34 best
_S34B_T_LONG      = 2000         # 長エピソード
_S34B_N_SEEDS     = 5
_S34B_SEEDS       = list(range(42, 42 + _S34B_N_SEEDS))

_CTX_LABELS = [
    'food=1 pred=0\n(食料近・捕食者遠)',
    'food=1 pred=1\n(食料近・捕食者近)',
    'food=0 pred=1\n(食料遠・捕食者近)',
    'food=0 pred=0\n(食料遠・捕食者遠)',
]


# ── エピソード中の文脈別行動を記録するランナー ────────────────────────────────

def _run_ep_with_context_log(cfg: WorldConfig, G, W, genome, rng,
                              pursuit_prob: float,
                              hp_decay: float    = _S34_HP_DECAY,
                              predator_speed: int = _S34_PRED_SPEED,
                              T: int              = _S34B_T_LONG,
                              reset_on_death: bool = True):
    """エピソードを走らせながら各ステップの（文脈, 行動）を記録。

    reset_on_death=True の場合、HPが0になってもリセットして観察を続ける
    （長時間観察のため）。

    Returns:
      context_action_log: list of (ctx_idx, action_idx) per step
      stats: dict（steps, food, pred_hits）
    """
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
    pred_resources = 1.0
    pred_dormant   = False

    context_action_log = []
    total_food = total_pred_hits = 0

    for step in range(T):
        # 死亡時リセット
        if hp <= 0:
            if not reset_on_death:
                break
            row, col = center, center
            hp       = float(cfg.hp_start)
            food_positions = _s31_init_foods(cfg, rng, row, col)
            food_avail     = [True] * cfg.n_foods
            food_timer     = [0]   * cfg.n_foods
            pred_pos       = _s31_init_pred(cfg, rng, row, col)
            pred_resources = 1.0
            pred_dormant   = False
            resources      = np.ones(_S28_N)
            activity       = np.zeros(_S28_N)

        if step % predator_speed == 0:
            pred_pos, pred_resources, pred_dormant = _s34_pred_step(
                cfg, pred_pos, [row, col],
                pursuit_prob, pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            total_pred_hits += 1

        inp5 = _s31_inp5(cfg, row, col, hp, food_positions, food_avail, pred_pos)

        # 文脈を分類
        food_flag = int(inp5[3])
        pred_flag = int(inp5[4])
        ctx_idx   = food_flag * 2 + (1 - pred_flag)
        # C0: food=1,pred=0 → food_flag=1, pred_flag=0 → 1*2+(1-0)=3 … 要修正
        # 正しいマッピング:
        #   food=1,pred=0: idx=0
        #   food=1,pred=1: idx=1
        #   food=0,pred=1: idx=2
        #   food=0,pred=0: idx=3
        ctx_idx = (1 - food_flag) * 2 + pred_flag  # 一旦計算
        # 上記は: food=1,pred=0→0*2+0=0, food=1,pred=1→0*2+1=1,
        #         food=0,pred=1→1*2+1=3, food=0,pred=0→1*2+0=2
        # 期待するマッピングに合わせる:
        ctx_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
        ctx_idx = ctx_map[(food_flag, pred_flag)]

        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(
                eff + rng.normal(0, _S28_ACT_NOISE, _S28_N), 0.0, 1.0)

        resources = _s27_update_resources(
            resources, activity, _s28_make_tau_arr(), depletion_rate)

        hp -= hp_decay
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        context_action_log.append((ctx_idx, action))

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
                    total_food += 1
                    break

        for fi in range(cfg.n_foods):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        if (step + 1) % _K == 0:
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    return context_action_log, {
        'steps': T, 'food': total_food, 'pred_hits': total_pred_hits}


def aggregate_context_actions(log):
    """文脈別の行動分布を集計。

    Returns:
      counts:   4×5 array (ctx × action の出現数)
      fracs:    4×5 array (ctx × action の割合)
      ctx_totals: 各文脈のステップ数
    """
    counts = np.zeros((4, 5), dtype=int)
    for ctx_idx, action in log:
        counts[ctx_idx, action] += 1
    ctx_totals = counts.sum(axis=1)
    fracs = np.zeros((4, 5))
    for c in range(4):
        if ctx_totals[c] > 0:
            fracs[c] = counts[c] / ctx_totals[c]
    return counts, fracs, ctx_totals


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_in_episode(evolved_result, random_result, seed,
                    fname='images/session_34b/results_s34b_in_episode.png'):
    """エピソード中の文脈別行動分布を可視化。進化 vs ランダムゲノムの比較。"""
    e_fracs = evolved_result['fracs']
    r_fracs = random_result['fracs']
    e_totals = evolved_result['ctx_totals']
    r_totals = random_result['ctx_totals']
    n_ctx = 4

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f'Session 34b: エピソード中の文脈別行動（実測）\n'
        f'pp={_S34B_PURSUIT}  hp_decay={_S34_HP_DECAY}  '
        f'grid={_S34_CFG.grid}x{_S34_CFG.grid}  '
        f'T={_S34B_T_LONG}  seed={seed}',
        fontsize=12,
    )
    colors_a = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']

    def _stacked_bar(ax, fracs, totals, title):
        bottoms = np.zeros(n_ctx)
        for ai, (aname, col) in enumerate(zip(_S28_ACTION_NAMES, colors_a)):
            bars = ax.bar(range(n_ctx), fracs[:, ai], bottom=bottoms,
                          color=col, alpha=0.85, label=aname, edgecolor='white')
            for bi, (b, v) in enumerate(zip(bottoms, fracs[:, ai])):
                if v > 0.08:
                    ax.text(bi, b + v / 2, f'{v:.0%}',
                            ha='center', va='center', fontsize=8,
                            color='white', fontweight='bold')
            bottoms += fracs[:, ai]
        ax.set_xticks(range(n_ctx))
        ax.set_xticklabels(
            [f'C{i}\n{_CTX_LABELS[i]}\n(n={totals[i]})'
             for i in range(n_ctx)], fontsize=7.5)
        ax.set_ylabel('Action ratio')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, 1.15)

    # Panel 1: 進化個体の行動分布
    _stacked_bar(axes[0][0], e_fracs, e_totals,
                 '進化個体の文脈別行動\n期待: C0=食事↑, C1=食事↓')

    # Panel 2: ランダムゲノムの行動分布（対照）
    _stacked_bar(axes[0][1], r_fracs, r_totals,
                 'ランダムゲノム（対照）\n文脈依存がなければ均等なはず')

    # Panel 3: 食事行動の比率を文脈別に比較
    ax = axes[0][2]
    e_eat = e_fracs[:, 4]
    r_eat = r_fracs[:, 4]
    x = np.arange(n_ctx)
    w = 0.35
    ax.bar(x - w/2, e_eat, width=w, color='purple', alpha=0.85,
           label='進化個体', edgecolor='white')
    ax.bar(x + w/2, r_eat, width=w, color='lightgray', alpha=0.85,
           label='ランダム', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(n_ctx)], fontsize=9)
    ax.set_ylabel('食事行動の割合')
    ax.set_title('食事行動率の文脈別比較\n期待: C0>C1（食料近でも捕食者近なら食事が減る）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (e, r) in enumerate(zip(e_eat, r_eat)):
        ax.text(i - w/2, e + 0.01, f'{e:.0%}', ha='center', fontsize=9)
        ax.text(i + w/2, r + 0.01, f'{r:.0%}', ha='center', fontsize=9)

    # Panel 4: 文脈別ステップ数（どの文脈を何回経験したか）
    ax = axes[1][0]
    ax.bar(range(n_ctx), e_totals, color=['royalblue', 'tomato', 'seagreen', 'darkorange'],
           alpha=0.85, edgecolor='white')
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels([f'C{i}' for i in range(n_ctx)], fontsize=9)
    ax.set_ylabel('Steps')
    ax.set_title('文脈別の経験ステップ数\n(C1=トレードオフ発生頻度)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(e_totals):
        ax.text(i, v + 5, f'{v}\n({v/sum(e_totals):.1%})',
                ha='center', fontsize=9)

    # Panel 5: 移動方向の文脈別比較（北/南/西/東の合計）
    ax = axes[1][1]
    e_escape = e_fracs[:, :4].sum(axis=1)  # 移動系合計
    r_escape = r_fracs[:, :4].sum(axis=1)
    ax.bar(x - w/2, e_escape, width=w, color='steelblue', alpha=0.85,
           label='進化個体', edgecolor='white')
    ax.bar(x + w/2, r_escape, width=w, color='lightgray', alpha=0.85,
           label='ランダム', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(n_ctx)], fontsize=9)
    ax.set_ylabel('移動行動の割合')
    ax.set_title('移動行動率（逃げ/探索）の文脈別比較\n期待: C1,C2で移動↑')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: キー指標のサマリー
    ax = axes[1][2]
    # C0 vs C1 の食事率の差が「文脈依存」の核心
    c0_eat = e_fracs[0, 4]
    c1_eat = e_fracs[1, 4]
    c1_steps = e_totals[1]
    lines = [
        f'=== 文脈依存の核心指標 ===\n',
        f'C0 食事率（食料近・捕食者遠）: {c0_eat:.1%}',
        f'C1 食事率（食料近・捕食者近）: {c1_eat:.1%}',
        f'差 C0-C1: {c0_eat - c1_eat:+.1%}',
        f'',
        f'C1経験ステップ数: {c1_steps}',
        f'  ({c1_steps/sum(e_totals):.1%} of total)',
        f'',
        f'=== 統計 ===',
        f'食事率 C0 vs C1 のカイ二乗検定:',
    ]
    # C0 vs C1 の食事行動のカイ二乗検定
    if e_totals[0] > 0 and e_totals[1] > 0:
        obs = np.array([
            [e_fracs[0, 4] * e_totals[0],
             (1 - e_fracs[0, 4]) * e_totals[0]],
            [e_fracs[1, 4] * e_totals[1],
             (1 - e_fracs[1, 4]) * e_totals[1]],
        ])
        try:
            chi2, p, _, _ = scipy_stats.chi2_contingency(obs)
            lines.append(f'  chi2={chi2:.2f}  p={p:.4f}')
            lines.append(f'  {"有意差あり *" if p < 0.05 else "有意差なし"}')
        except Exception:
            lines.append('  計算不可')
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.axis('off')
    ax.set_title('文脈依存の核心指標')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_multiseed(multi_results,
                   fname='images/session_34b/results_s34b_multiseed.png'):
    """複数seedの結果を比較。"""
    seeds   = [r['seed']        for r in multi_results]
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    c1_ns   = [r['c1_steps']    for r in multi_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f'Session 34b: 複数seed確認\n'
        f'pp={_S34B_PURSUIT}  hp_decay={_S34_HP_DECAY}  '
        f'T={_S34B_T_LONG}  seeds={seeds}',
        fontsize=13,
    )

    # Panel 1: C0 vs C1 の食事率
    ax = axes[0]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, c0_eats, width=w, color='royalblue', alpha=0.85,
           label='C0(食料近・捕食者遠)', edgecolor='white')
    ax.bar(x + w/2, c1_eats, width=w, color='tomato', alpha=0.85,
           label='C1(食料近・捕食者近)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('食事行動率')
    ax.set_title('C0 vs C1 の食事率\n(C0>C1 なら文脈依存あり)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (c0, c1) in enumerate(zip(c0_eats, c1_eats)):
        ax.text(i - w/2, c0 + 0.005, f'{c0:.0%}', ha='center', fontsize=8)
        ax.text(i + w/2, c1 + 0.005, f'{c1:.0%}', ha='center', fontsize=8)

    # Panel 2: C0-C1 の差
    ax = axes[1]
    bar_colors = ['seagreen' if d > 0 else 'tomato' for d in diffs]
    ax.bar(range(len(seeds)), diffs, color=bar_colors, alpha=0.85,
           edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('C0_eat - C1_eat')
    ax.set_title('食事率の差 C0-C1\n(正=捕食者がいると食事を減らす)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.005 if d >= 0 else d - 0.01,
                f'{d:+.0%}', ha='center', fontsize=9)

    # Panel 3: C1経験ステップ数
    ax = axes[2]
    ax.bar(range(len(seeds)), c1_ns, color='darkorange', alpha=0.85,
           edgecolor='white')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{s}' for s in seeds])
    ax.set_ylabel('Steps')
    ax.set_title('C1（トレードオフ）の経験ステップ数\n(多いほど文脈依存を学習する機会がある)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(c1_ns):
        ax.text(i, v + 5, f'{v}', ha='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── メイン ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = _S34_CFG

    print('=== Session 34b: エピソード中の文脈別行動計測 ===')
    print(f'pp={_S34B_PURSUIT}  hp_decay={_S34_HP_DECAY}  '
          f'grid={cfg.grid}x{cfg.grid}  T={_S34B_T_LONG}')
    print()

    # ── seed=42 で詳細計測 ────────────────────────────────────────────────────
    print('[Step 1] seed=42 で進化')
    best42, hist42 = _s34_evolve(
        cfg, pursuit_prob=_S34B_PURSUIT, seed=_S34B_SEED)
    print(f'  steps={hist42["gen_best_steps"][-1]:.1f}  '
          f'food={hist42["gen_food_count"][-1]:.2f}/ep')

    print(f'\n[Step 2] 進化個体でT={_S34B_T_LONG}ステップ実行（文脈別行動を記録）')
    rng42 = np.random.default_rng(_S34B_SEED + 34200)
    G42   = best42['G'].copy()
    W42   = _s28_get_W(G42)
    log42, stats42 = _run_ep_with_context_log(
        cfg, G42, W42, best42, rng42,
        pursuit_prob=_S34B_PURSUIT, T=_S34B_T_LONG)
    counts42, fracs42, totals42 = aggregate_context_actions(log42)

    print(f'  総ステップ: {len(log42)}  food={stats42["food"]}  '
          f'hits={stats42["pred_hits"]}')
    print()
    print('  文脈別の経験ステップ数と行動分布:')
    for c, lbl in enumerate(_CTX_LABELS):
        lbl_short = lbl.replace('\n', ' ')
        if totals42[c] > 0:
            dom_a = int(np.argmax(fracs42[c]))
            print(f'  C{c} [{lbl_short}]: {totals42[c]}steps  '
                  f'主行動={_S28_ACTION_NAMES[dom_a]}({fracs42[c,dom_a]:.0%})  '
                  f'食事={fracs42[c,4]:.0%}')
        else:
            print(f'  C{c} [{lbl_short}]: 0steps (経験なし)')

    # ランダムゲノム（対照）
    print('\n[Step 3] ランダムゲノム（対照）でT={_S34B_T_LONG}ステップ実行')
    rng_rand = np.random.default_rng(_S34B_SEED + 34299)
    rand_genome = _s28_make_genome(rng_rand)
    G_rand = rand_genome['G'].copy()
    W_rand = _s28_get_W(G_rand)
    log_rand, stats_rand = _run_ep_with_context_log(
        cfg, G_rand, W_rand, rand_genome, rng_rand,
        pursuit_prob=_S34B_PURSUIT, T=_S34B_T_LONG)
    counts_rand, fracs_rand, totals_rand = aggregate_context_actions(log_rand)

    plot_in_episode(
        {'fracs': fracs42, 'ctx_totals': totals42},
        {'fracs': fracs_rand, 'ctx_totals': totals_rand},
        seed=_S34B_SEED)

    # ── 複数seedで確認 ────────────────────────────────────────────────────────
    print(f'\n[Step 4] 複数seed確認 (seeds={_S34B_SEEDS})')
    multi_results = []
    for seed in _S34B_SEEDS:
        print(f'\n  seed={seed}:')
        best_s, hist_s = _s34_evolve(
            cfg, pursuit_prob=_S34B_PURSUIT, seed=seed)
        rng_s = np.random.default_rng(seed + 34200)
        G_s   = best_s['G'].copy()
        W_s   = _s28_get_W(G_s)
        log_s, _ = _run_ep_with_context_log(
            cfg, G_s, W_s, best_s, rng_s,
            pursuit_prob=_S34B_PURSUIT, T=_S34B_T_LONG)
        counts_s, fracs_s, totals_s = aggregate_context_actions(log_s)
        c0_eat = fracs_s[0, 4]
        c1_eat = fracs_s[1, 4]
        c1_n   = totals_s[1]
        print(f'    C0食事率={c0_eat:.0%}  C1食事率={c1_eat:.0%}  '
              f'差={c0_eat-c1_eat:+.0%}  C1n={c1_n}')
        multi_results.append({
            'seed':        seed,
            'c0_eat_rate': c0_eat,
            'c1_eat_rate': c1_eat,
            'c1_steps':    c1_n,
            'fracs':       fracs_s,
            'totals':      totals_s,
        })

    plot_multiseed(multi_results)

    # ── サマリー ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 34b Summary ===')
    print()
    c0_eats = [r['c0_eat_rate'] for r in multi_results]
    c1_eats = [r['c1_eat_rate'] for r in multi_results]
    diffs   = [c0 - c1 for c0, c1 in zip(c0_eats, c1_eats)]
    print(f'C0食事率（食料近・捕食者遠）: '
          f'mean={np.mean(c0_eats):.0%}  std={np.std(c0_eats):.0%}')
    print(f'C1食事率（食料近・捕食者近）: '
          f'mean={np.mean(c1_eats):.0%}  std={np.std(c1_eats):.0%}')
    print(f'差 C0-C1: '
          f'mean={np.mean(diffs):+.0%}  std={np.std(diffs):.0%}')
    n_positive = sum(d > 0 for d in diffs)
    print(f'C0>C1（捕食者がいると食事減少）: {n_positive}/{len(diffs)} seeds')
    print()
    print('--- 判断基準 ---')
    print('C0>C1 が多数のseedで成立 → 文脈依存的な行動が実際のエピソードで出ている')
    print('C0≈C1 → 文脈を読んでいない（食料フラグにだけ反応）')
    print()
    print('Done.')
