"""Session 26: 疲労による飽和の解消と文脈分離

Session 25bで確定したこと:
  文脈分離が起きない根本原因は飽和
  全ノードが常に0.76〜0.98で活動
  → モードAとBで同じパターン → 文脈を区別できない

疲労メカニズム:
  各ノードが疲労度（fatigue, 0-1）を持つ
  effective_activity = activity * (1 - fatigue)
  fatigue += activity * fatigue_rate
  fatigue = clip(fatigue - recovery_rate, 0, 1)

疲労パラメータはノードタイプ別に進化させる:
  fr_s: 感覚器ノード（node0-3）の疲労速度
  fr_o: 出力ノード（node4-8）の疲労速度
  fr_i: 内部ノード（node9-19）の疲労速度
  rec:  全ノード共通の回復速度

副産物: fr_s/fr_i > 1 なら「感覚器が先に疲れる」→ 睡眠様状態が創発するか

実験:
  A 疲労パラメータの進化収束（fr_s/fr_o/fr_i/rec の推移、活動ノード数）
  B 飽和の解消確認（疲労あり vs なしの活動分布比較）
  C 文脈分離の計測（acc_A/acc_B/cosine_dist、統計的有意性）
  D 睡眠様状態の観察（T=2000連続実行、層別活動と疲労度の時系列）

出力:
  images/session_26/results_s26_genome_convergence.png
  images/session_26/results_s26_saturation.png
  images/session_26/results_s26_context.png
  images/session_26/results_s26_sleep_pattern.png
"""

import os
import numpy as np
import _jp_font  # noqa: F401 — sets Japanese font in matplotlib rcParams
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _N, _K, _N_PROP,
)
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_18_ratio_evolution import (
    _s18_hebb,
    _ACTIVITY_NOISE,
    _N_GEN, _N_SURV,
    _EP_INIT_MAX, _AR_INIT_MAX,
    _EP_MUT_STD, _AR_MUT_STD,
)
from session_19_context_reboot import _CGRID, _CHP_MAX, _CRESPAWN, _CSTEPS
from session_20_penalty_context import _s20_inp4, _PENALTY
from session_25_hunger_learning import (
    _N_AGENTS, _N_EP, _ACT_THRESHOLD,
    _INP_START, _INP_END, _OUT_START, _OUT_END, _INT_START, _INT_END,
    _S25_HP_START, _S25_FOOD_VALUE,
)

_S26_SEED   = 42
_S26_N_GEN  = 50
_S26_MR     = 0.01
_S26_HUNGER_THR = 96
_S26_HUNGER_PEN = 1.0
_S26_T_CONSOL   = 0    # 疲労で代替するため睡眠フェーズなし

# Fatigue genome bounds
_S26_FR_LO       = 0.0
_S26_FR_HI       = 0.2
_S26_REC_LO      = 0.0
_S26_REC_HI      = 0.2
_S26_FR_MUT_STD  = 0.01
_S26_REC_MUT_STD = 0.01

# Sleep observation
_S26_SLEEP_T     = 2000
_S26_SLEEP_CHUNK = 100


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s26_make_genome(rng):
    G = _s10_build_graph(rng)
    return {
        'G':                G,
        'W':                _s10_get_W(G),
        'fr_s':             float(rng.uniform(_S26_FR_LO, _S26_FR_HI)),
        'fr_o':             float(rng.uniform(_S26_FR_LO, _S26_FR_HI)),
        'fr_i':             float(rng.uniform(_S26_FR_LO, _S26_FR_HI)),
        'rec':              float(rng.uniform(_S26_REC_LO, _S26_REC_HI)),
        'edge_add_prob':    float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio':   float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate':   _S26_MR,
        'hunger_threshold': _S26_HUNGER_THR,
        'hunger_penalty':   _S26_HUNGER_PEN,
    }


def _s26_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    return {
        'G':                G_new,
        'W':                _s10_get_W(G_new),
        'fr_s':             float(np.clip(genome['fr_s'] + rng.normal(0, _S26_FR_MUT_STD),
                                          _S26_FR_LO, _S26_FR_HI)),
        'fr_o':             float(np.clip(genome['fr_o'] + rng.normal(0, _S26_FR_MUT_STD),
                                          _S26_FR_LO, _S26_FR_HI)),
        'fr_i':             float(np.clip(genome['fr_i'] + rng.normal(0, _S26_FR_MUT_STD),
                                          _S26_FR_LO, _S26_FR_HI)),
        'rec':              float(np.clip(genome['rec']  + rng.normal(0, _S26_REC_MUT_STD),
                                          _S26_REC_LO, _S26_REC_HI)),
        'edge_add_prob':    float(np.clip(genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD),
                                          0.0, _EP_INIT_MAX)),
        'activity_ratio':   float(np.clip(genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD),
                                          0.0, _AR_INIT_MAX)),
        'metabolic_rate':   genome['metabolic_rate'],
        'hunger_threshold': genome['hunger_threshold'],
        'hunger_penalty':   genome['hunger_penalty'],
    }


# ── Episode runner ─────────────────────────────────────────────────────────────

def _s26_run_ep(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                hunger_threshold, hunger_penalty,
                fr_s, fr_o, fr_i, rec,
                rng, mode=None,
                activity_noise=_ACTIVITY_NOISE,
                T_consolidation=_S26_T_CONSOL,
                record_activity=False,
                record_fatigue=False):
    """PenaltyContextGridWorld + 飢餓 + 疲労メカニズム。

    effective_activity = activity * (1 - fatigue) がすべての下流処理に使われる。
    疲労は raw activity で蓄積され、rec ずつ回復する。

    Returns (steps, food, mode, penalties, act_records_or_None, fat_records_or_None).
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos    = (0, 0) if mode == 'A' else (4, 4)
    penalty_pos = (4, 4) if mode == 'A' else (0, 0)
    food_avail  = True
    food_timer  = 0
    fr, fc = food_pos
    pr, pc = penalty_pos

    fatigue_rates              = np.zeros(_N)
    fatigue_rates[_INP_START:_INP_END] = fr_s
    fatigue_rates[_OUT_START:_OUT_END] = fr_o
    fatigue_rates[_INT_START:_INT_END] = fr_i

    fatigue  = np.zeros(_N)
    activity = np.zeros(_N)

    row, col  = 2, 2
    hp        = float(_S25_HP_START)
    hunger    = 0
    steps     = 0
    food      = 0
    penalties = 0
    act_recs  = [] if record_activity else None
    fat_recs  = [] if record_fatigue  else None

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        inp4 = _s20_inp4(row, col, hp, food_avail)

        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        # 疲労によって実効活動を減衰させる
        eff = np.clip(activity * (1.0 - fatigue), 0.0, 1.0)

        if activity_noise > 0.0:
            eff = np.clip(eff + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        # 疲労更新: raw activity で蓄積、rec で回復
        fatigue = np.clip(fatigue + activity * fatigue_rates - rec, 0.0, 1.0)

        if record_activity:
            act_recs.append(eff.copy())
        if record_fatigue:
            fat_recs.append(fatigue.copy())

        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_OUT_START:_OUT_END]))

        food_eaten = False
        if action in (0, 1, 2, 3):
            if   action == 0: row = max(0, row - 1)
            elif action == 1: row = min(_CGRID - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(_CGRID - 1, col + 1)
            if row == pr and col == pc:
                hp -= _PENALTY
                penalties += 1
        elif action == 4:
            if row == fr and col == fc and food_avail:
                hp = min(_CHP_MAX, hp + _S25_FOOD_VALUE)
                food_avail = False
                food_timer = 0
                food_eaten = True
                food += 1

        hunger += 1
        if food_eaten:
            hunger = 0
        elif hunger > hunger_threshold:
            hp -= hunger_penalty

        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        # 次ステップへ effective activity を引き継ぐ
        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food, mode, penalties, act_recs, fat_recs


# ── Evolution ─────────────────────────────────────────────────────────────────

def _s26_evolve(seed=_S26_SEED):
    """疲労パラメータを含むゲノムを PenaltyContextGridWorld + 飢餓で進化させる。

    Returns (best_genome, history_dict).
    history keys: gen_best_steps, gen_food_count, gen_mean_active,
                  gen_fr_s, gen_fr_o, gen_fr_i, gen_rec, gen_fr_ratio.
    """
    rng = np.random.default_rng(seed + 26000)
    pop = [_s26_make_genome(rng) for _ in range(_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_mean_active',
        'gen_fr_s', 'gen_fr_o', 'gen_fr_i', 'gen_rec', 'gen_fr_ratio',
    )}

    for gen in range(_S26_N_GEN):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_active = 0, [], []
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, _, _, recs, _ = _s26_run_ep(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'], g['metabolic_rate'],
                    g['hunger_threshold'], g['hunger_penalty'],
                    g['fr_s'], g['fr_o'], g['fr_i'], g['rec'],
                    ep_rng, record_activity=True)
                total += s
                ep_food.append(f)
                if recs:
                    arr = np.array(recs)
                    ep_active.append(float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD)))
            fitnesses.append(total / _N_EP)
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0
            g['_ep_food']   = float(np.mean(ep_food))

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]
        fr_ratio = bg['fr_s'] / bg['fr_i'] if bg['fr_i'] > 1e-9 else float('nan')

        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_fr_s'].append(bg['fr_s'])
        hist['gen_fr_o'].append(bg['fr_o'])
        hist['gen_fr_i'].append(bg['fr_i'])
        hist['gen_rec'].append(bg['rec'])
        hist['gen_fr_ratio'].append(fr_ratio)

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s26_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'fr_s={bg["fr_s"]:.3f}  fr_i={bg["fr_i"]:.3f}  '
                  f'ratio={fr_ratio:.2f}  rec={bg["rec"]:.3f}  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        g.pop('_ep_active', None)
        g.pop('_ep_food',   None)

    return pop[0], hist


# ── Experiment A: genome convergence ──────────────────────────────────────────

def run_exp_a_genome_convergence(seed=_S26_SEED):
    """疲労パラメータの進化収束を記録する。

    Returns (best_genome, history_dict).
    """
    print(f'\n  [Exp A] 疲労パラメータの進化 ({_S26_N_GEN} 世代, {_N_AGENTS} 個体, {_N_EP} エピソード)')
    best, hist = _s26_evolve(seed)
    print(f'  → fr_s={best["fr_s"]:.4f}  fr_o={best["fr_o"]:.4f}  '
          f'fr_i={best["fr_i"]:.4f}  rec={best["rec"]:.4f}')
    fr_ratio = best['fr_s'] / best['fr_i'] if best['fr_i'] > 1e-9 else float('nan')
    print(f'     fr_s/fr_i比={fr_ratio:.3f}  '
          f'("{"感覚器が先に疲れる" if fr_ratio > 1.0 else "内部が先に疲れる"}")')
    print(f'     food/ep={hist["gen_food_count"][-1]:.2f}  '
          f'active={hist["gen_mean_active"][-1]:.1f}  '
          f'steps={hist["gen_best_steps"][-1]:.1f}')
    return best, hist


def plot_exp_a_genome_convergence(best, hist,
                                   fname='images/session_26/results_s26_genome_convergence.png'):
    xs = np.arange(1, len(hist['gen_best_steps']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Session 26 Exp A: 疲労パラメータの進化収束\n'
        f'PenaltyContextGridWorld + 飢餓 (mr={_S26_MR}, '
        f'hunger_thr={_S26_HUNGER_THR}, {_S26_N_GEN}世代)',
        fontsize=12,
    )

    # Panel 1: fr_s, fr_o, fr_i 推移
    ax = axes[0][0]
    ax.plot(xs, hist['gen_fr_s'], color='steelblue', linewidth=2, label='fr_s (感覚器)')
    ax.plot(xs, hist['gen_fr_o'], color='tomato',    linewidth=2, label='fr_o (出力)')
    ax.plot(xs, hist['gen_fr_i'], color='green',     linewidth=2, label='fr_i (内部)')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fatigue rate')
    ax.set_title('疲労速度パラメータの収束')
    ax.set_ylim(-0.01, _S26_FR_HI + 0.01)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: rec 推移
    ax = axes[0][1]
    ax.plot(xs, hist['gen_rec'], color='purple', linewidth=2, label='rec (回復速度)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Recovery rate')
    ax.set_title('回復速度パラメータの収束')
    ax.set_ylim(-0.01, _S26_REC_HI + 0.01)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: fr_s / fr_i 比率（重要な観察指標）
    ax = axes[0][2]
    ratio_vals = hist['gen_fr_ratio']
    finite_mask = np.isfinite(ratio_vals)
    if np.any(finite_mask):
        ax.plot(xs[finite_mask], np.array(ratio_vals)[finite_mask],
                color='darkorange', linewidth=2, label='fr_s / fr_i')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
               label='比率=1.0 (同等疲労)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('fr_s / fr_i')
    ax.set_title('fr_s/fr_i 比率\n(>1: 感覚器が先に疲れる)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    final_ratio = hist['gen_fr_ratio'][-1]
    label = '感覚器優先疲労' if (np.isfinite(final_ratio) and final_ratio > 1.0) else '内部優先疲労'
    ax.text(0.97, 0.97, f'最終値: {final_ratio:.2f}\n({label})',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Panel 4: 活動ノード数の推移
    ax = axes[1][0]
    ax.plot(xs, hist['gen_mean_active'], color='navy', linewidth=2)
    ax.axhline(_N, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'最大{_N}ノード')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Active nodes (mean)')
    ax.set_title('活動ノード数の推移\n(飽和解消で減少するか)')
    ax.set_ylim(0, _N + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: food/ep の推移
    ax = axes[1][1]
    ax.plot(xs, hist['gen_food_count'], color='forestgreen', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Food / episode')
    ax.set_title('食料獲得数の推移\n(学習が進むほど増加)')
    ax.grid(True, alpha=0.3)

    # Panel 6: 最終ゲノムのサマリー
    ax = axes[1][2]
    fr_ratio_final = best['fr_s'] / best['fr_i'] if best['fr_i'] > 1e-9 else float('nan')
    summary = (
        f'最終ベスト個体\n\n'
        f'fr_s  = {best["fr_s"]:.4f}\n'
        f'fr_o  = {best["fr_o"]:.4f}\n'
        f'fr_i  = {best["fr_i"]:.4f}\n'
        f'rec   = {best["rec"]:.4f}\n\n'
        f'fr_s/fr_i = {fr_ratio_final:.3f}\n'
        f'{"→ 感覚器優先疲労" if np.isfinite(fr_ratio_final) and fr_ratio_final > 1.0 else "→ 内部優先疲労"}\n\n'
        f'食料/ep = {hist["gen_food_count"][-1]:.2f}\n'
        f'活動ノード = {hist["gen_mean_active"][-1]:.1f}\n'
        f'steps   = {hist["gen_best_steps"][-1]:.1f}'
    )
    ax.text(0.5, 0.5, summary, transform=ax.transAxes,
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax.set_title('最終ゲノムサマリー')
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment B: saturation check ────────────────────────────────────────────

def _s26_collect_activity_stats(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                                  hunger_threshold, hunger_penalty,
                                  fr_s, fr_o, fr_i, rec,
                                  seed, n_episodes=20, with_fatigue=True):
    """複数エピソードにわたるノード活動統計を収集する。

    with_fatigue=False のとき、疲労速度をゼロにして比較ベースラインを取得する。
    Returns dict: node_means, node_stds, layer_means (sensory/output/internal), all_activities.
    """
    rng = np.random.default_rng(seed)
    all_acts = []

    _fr_s = fr_s if with_fatigue else 0.0
    _fr_o = fr_o if with_fatigue else 0.0
    _fr_i = fr_i if with_fatigue else 0.0
    _rec  = rec  if with_fatigue else 0.0

    for _ in range(n_episodes):
        G_copy = G.copy()
        W_copy = _s10_get_W(G_copy)
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        _, _, _, _, recs, _ = _s26_run_ep(
            G_copy, W_copy,
            edge_add_prob, activity_ratio, metabolic_rate,
            hunger_threshold, hunger_penalty,
            _fr_s, _fr_o, _fr_i, _rec,
            ep_rng, record_activity=True)
        if recs:
            all_acts.append(np.array(recs))

    if not all_acts:
        empty = np.zeros(_N)
        return {'node_means': empty, 'node_stds': empty,
                'layer_means': {}, 'all_activities': []}

    combined = np.concatenate(all_acts, axis=0)
    node_means = np.mean(combined, axis=0)
    node_stds  = np.std(combined,  axis=0)

    return {
        'node_means': node_means,
        'node_stds':  node_stds,
        'layer_means': {
            'sensory':  float(np.mean(node_means[_INP_START:_INP_END])),
            'output':   float(np.mean(node_means[_OUT_START:_OUT_END])),
            'internal': float(np.mean(node_means[_INT_START:_INT_END])),
        },
        'all_activities': all_acts,
        'combined': combined,
    }


def _s26_collect_mode_patterns(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                                 hunger_threshold, hunger_penalty,
                                 fr_s, fr_o, fr_i, rec,
                                 seed, n_per_mode=15, with_fatigue=True):
    """モード A/B 別の出力ノード活動パターンを収集。cosine_distance も計算。"""
    rng  = np.random.default_rng(seed)
    _fr_s = fr_s if with_fatigue else 0.0
    _fr_o = fr_o if with_fatigue else 0.0
    _fr_i = fr_i if with_fatigue else 0.0
    _rec  = rec  if with_fatigue else 0.0

    out_A, out_B = [], []

    for ei in range(n_per_mode * 2):
        mode   = 'A' if ei < n_per_mode else 'B'
        G_copy = G.copy()
        W_copy = _s10_get_W(G_copy)
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        _, _, _, _, recs, _ = _s26_run_ep(
            G_copy, W_copy,
            edge_add_prob, activity_ratio, metabolic_rate,
            hunger_threshold, hunger_penalty,
            _fr_s, _fr_o, _fr_i, _rec,
            ep_rng, mode=mode, record_activity=True)
        if recs:
            arr = np.array(recs)
            mean_out = arr[:, _OUT_START:_OUT_END].mean(axis=0)
            (out_A if mode == 'A' else out_B).append(mean_out)

    mA = np.mean(out_A, axis=0) if out_A else np.zeros(_OUT_END - _OUT_START)
    mB = np.mean(out_B, axis=0) if out_B else np.zeros(_OUT_END - _OUT_START)
    nA, nB = np.linalg.norm(mA), np.linalg.norm(mB)
    cos_dist = (1.0 - float(np.dot(mA, mB) / (nA * nB))
                if nA > 1e-9 and nB > 1e-9 else 0.0)

    return {'mean_A': mA, 'mean_B': mB, 'cosine_dist': cos_dist}


def run_exp_b_saturation(best, seed=_S26_SEED, n_episodes=20):
    """疲労あり vs なし の活動分布を比較して飽和解消を確認。

    Returns dict: with_fatigue, without_fatigue (各 _s26_collect_activity_stats の戻り値).
    """
    print(f'\n  [Exp B] 飽和解消確認 (n_episodes={n_episodes})')
    G = best['G'].copy()
    W = _s10_get_W(G)

    res_with = _s26_collect_activity_stats(
        G, W,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['fr_s'], best['fr_o'], best['fr_i'], best['rec'],
        seed + 26100, n_episodes=n_episodes, with_fatigue=True)

    res_without = _s26_collect_activity_stats(
        G, W,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['fr_s'], best['fr_o'], best['fr_i'], best['rec'],
        seed + 26101, n_episodes=n_episodes, with_fatigue=False)

    pat_with    = _s26_collect_mode_patterns(
        G, W,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['fr_s'], best['fr_o'], best['fr_i'], best['rec'],
        seed + 26102, n_per_mode=10, with_fatigue=True)

    pat_without = _s26_collect_mode_patterns(
        G, W,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['fr_s'], best['fr_o'], best['fr_i'], best['rec'],
        seed + 26103, n_per_mode=10, with_fatigue=False)

    print(f'    疲労あり:  全ノード平均活動={np.mean(res_with["node_means"]):.3f}  '
          f'分散={np.mean(res_with["node_stds"]):.3f}  '
          f'cos_dist={pat_with["cosine_dist"]:.4f}')
    print(f'    疲労なし:  全ノード平均活動={np.mean(res_without["node_means"]):.3f}  '
          f'分散={np.mean(res_without["node_stds"]):.3f}  '
          f'cos_dist={pat_without["cosine_dist"]:.4f}')

    return {
        'with_fatigue':    {**res_with,    'mode_patterns': pat_with},
        'without_fatigue': {**res_without, 'mode_patterns': pat_without},
    }


def plot_exp_b_saturation(exp_b,
                           fname='images/session_26/results_s26_saturation.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Session 26 Exp B: 疲労による飽和の解消\n'
        '疲労あり vs 疲労なし の活動分布とモードA/B パターン比較',
        fontsize=12,
    )

    node_labels = ([f'inp{i}' for i in range(_INP_END - _INP_START)] +
                   [f'out{i}' for i in range(_OUT_END - _OUT_START)] +
                   [f'int{i}' for i in range(_INT_END - _INT_START)])

    for row_idx, (key, title_suffix, color) in enumerate([
        ('without_fatigue', '疲労なし（飽和状態）', '#888888'),
        ('with_fatigue',    '疲労あり（S26）',      'steelblue'),
    ]):
        data = exp_b[key]
        nm   = data['node_means']
        ns   = data['node_stds']

        # Panel 1: 全ノードの平均活動
        ax = axes[row_idx][0]
        colors_node = (['#aec7e8'] * (_INP_END - _INP_START) +
                       ['#ffbb78'] * (_OUT_END - _OUT_START) +
                       ['#98df8a'] * (_INT_END - _INT_START))
        ax.bar(range(_N), nm, color=colors_node, alpha=0.85, edgecolor='white')
        ax.errorbar(range(_N), nm, yerr=ns, fmt='none', color='black',
                    capsize=3, linewidth=0.8, alpha=0.6)
        ax.axhline(0.76, color='red',  linestyle='--', linewidth=1.2, alpha=0.7,
                   label='飽和下限 (0.76)')
        ax.axhline(0.98, color='darkred', linestyle='--', linewidth=1.2, alpha=0.7,
                   label='飽和上限 (0.98)')
        ax.set_xticks(range(0, _N, 2))
        ax.set_xticklabels([node_labels[i] for i in range(0, _N, 2)], fontsize=8, rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Mean activity')
        ax.set_title(f'{title_suffix}\n全ノード平均活動  μ={np.mean(nm):.3f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 2: 活動値の分布（ヒストグラム）
        ax = axes[row_idx][1]
        if 'combined' in data and len(data['combined']) > 0:
            flat = data['combined'].ravel()
            ax.hist(flat, bins=40, color=color, alpha=0.75, edgecolor='white',
                    density=True)
            ax.axvline(0.76, color='red',     linestyle='--', linewidth=1.5,
                       label='飽和下限 0.76')
            ax.axvline(0.98, color='darkred', linestyle='--', linewidth=1.5,
                       label='飽和上限 0.98')
            ax.set_xlabel('Activity value')
            ax.set_ylabel('Density')
            ax.set_title(f'{title_suffix}\n活動値の分布')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Panel 3: モードA vs B 出力ノード活動
        ax = axes[row_idx][2]
        pat = data['mode_patterns']
        x   = np.arange(_OUT_END - _OUT_START)
        ax.bar(x - 0.2, pat['mean_A'], 0.35, color='steelblue', alpha=0.8,
               label='Mode A (食料左上)')
        ax.bar(x + 0.2, pat['mean_B'], 0.35, color='tomato',    alpha=0.8,
               label='Mode B (食料右下)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'out{i}' for i in range(_OUT_END - _OUT_START)])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Mean output activity')
        ax.set_title(f'{title_suffix}\n出力ノード A vs B  cos_dist={pat["cosine_dist"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment C: context separation ──────────────────────────────────────────

def _s26_eval_acc(best, seed, n_per_mode=25):
    """モードA・B それぞれ n_per_mode エピソード評価 → acc_A, acc_B, cos_dist, p_value."""
    rng    = np.random.default_rng(seed)
    G_base = best['G'].copy()
    W_base = _s10_get_W(G_base)

    food_A, food_B = [], []
    out_A, out_B   = [], []

    for ei in range(n_per_mode * 2):
        mode   = 'A' if ei < n_per_mode else 'B'
        G_copy = G_base.copy()
        W_copy = _s10_get_W(G_copy)
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        _, f, _, _, recs, _ = _s26_run_ep(
            G_copy, W_copy,
            best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
            best['hunger_threshold'], best['hunger_penalty'],
            best['fr_s'], best['fr_o'], best['fr_i'], best['rec'],
            ep_rng, mode=mode, record_activity=True)
        if mode == 'A':
            food_A.append(f)
            if recs:
                out_A.append(np.array(recs)[:, _OUT_START:_OUT_END].mean(axis=0))
        else:
            food_B.append(f)
            if recs:
                out_B.append(np.array(recs)[:, _OUT_START:_OUT_END].mean(axis=0))

    acc_A = float(np.mean(np.array(food_A) >= 1))
    acc_B = float(np.mean(np.array(food_B) >= 1))

    mA = np.mean(out_A, axis=0) if out_A else np.zeros(_OUT_END - _OUT_START)
    mB = np.mean(out_B, axis=0) if out_B else np.zeros(_OUT_END - _OUT_START)
    nA, nB = np.linalg.norm(mA), np.linalg.norm(mB)
    cos_dist = (1.0 - float(np.dot(mA, mB) / (nA * nB))
                if nA > 1e-9 and nB > 1e-9 else 0.0)

    # one-sided binomial test: acc_B > 0.30 (Session 25 のベースライン)
    k_B = sum(f >= 1 for f in food_B)
    n_B = len(food_B)
    p_val = scipy_stats.binomtest(k_B, n_B, p=0.30, alternative='greater').pvalue

    z     = scipy_stats.norm.ppf(0.975)
    p_hat = k_B / n_B if n_B > 0 else 0.0
    denom  = 1 + z**2 / n_B
    center = (p_hat + z**2 / (2 * n_B)) / denom
    margin = z * (p_hat * (1 - p_hat) / n_B + z**2 / (4 * n_B**2))**0.5 / denom
    ci_lo  = max(0.0, center - margin)
    ci_hi  = min(1.0, center + margin)

    return {
        'acc_A': acc_A, 'acc_B': acc_B,
        'cosine_dist': cos_dist,
        'k_B': k_B, 'n_B': n_B,
        'p_value': p_val,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'food_A': food_A, 'food_B': food_B,
    }


def run_exp_c_context(best, seed=_S26_SEED, n_per_mode=25):
    """文脈分離の計測 (acc_A, acc_B, cosine_dist, p値)。

    Returns result dict.
    """
    print(f'\n  [Exp C] 文脈分離計測 (n_per_mode={n_per_mode})')
    res = _s26_eval_acc(best, seed + 26200, n_per_mode=n_per_mode)
    sig = '有意' if res['p_value'] < 0.05 else '非有意'
    print(f'  → acc_A={res["acc_A"]:.3f}  acc_B={res["acc_B"]:.3f}  '
          f'cos_dist={res["cosine_dist"]:.4f}')
    print(f'     k_B={res["k_B"]}/{res["n_B"]}  '
          f'95%CI=[{res["ci_lo"]:.3f},{res["ci_hi"]:.3f}]  '
          f'p={res["p_value"]:.4f} ({sig})')
    both_separated = res['acc_A'] >= 0.6 and res['acc_B'] >= 0.6
    print(f'     文脈分離判定: {"✓ 達成 (両モード ≥ 0.6)" if both_separated else "✗ 未達成"}')
    return res


# セッション推移の参照値（Session 25b までの近似値。可視化用）
_S26_HISTORY = {
    'sessions': ['S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25\n(mr=0.01)', 'S25b', 'S26'],
    'acc_A':    [0.50,  0.60,  0.70,  0.75,  0.80,  0.85,  0.88,             0.88,   None],
    'acc_B':    [0.00,  0.00,  0.05,  0.05,  0.05,  0.08,  0.31,             0.31,   None],
    'cos_dist': [0.05,  0.05,  0.08,  0.10,  0.08,  0.10,  0.15,             0.15,   None],
}


def plot_exp_c_context(exp_c,
                        fname='images/session_26/results_s26_context.png'):
    sessions   = _S26_HISTORY['sessions']
    hist_acc_A = list(_S26_HISTORY['acc_A'])
    hist_acc_B = list(_S26_HISTORY['acc_B'])
    hist_cos   = list(_S26_HISTORY['cos_dist'])

    # Session 26 の実測値を代入
    hist_acc_A[-1] = exp_c['acc_A']
    hist_acc_B[-1] = exp_c['acc_B']
    hist_cos[-1]   = exp_c['cosine_dist']

    xs      = np.arange(len(sessions))
    s26_idx = len(sessions) - 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Session 26 Exp C: 文脈分離の計測\n'
        f'Session 19〜26 の推移（S19-S25bは参照値）  '
        f'acc_A={exp_c["acc_A"]:.3f}  acc_B={exp_c["acc_B"]:.3f}',
        fontsize=12,
    )

    for ai, (vals, title, threshold, ylabel, color) in enumerate([
        (hist_acc_A, 'Mode A Accuracy',    0.6,  'acc_A',    'steelblue'),
        (hist_acc_B, 'Mode B Accuracy',    0.6,  'acc_B',    'tomato'),
        (hist_cos,   'Cosine Distance\n(A vs B 出力パターン)', None, 'cosine dist', 'purple'),
    ]):
        ax    = axes[ai]
        bars  = ax.bar(xs[:-1], vals[:-1], color='lightgray', alpha=0.7,
                       edgecolor='gray', linewidth=0.8)
        ax.bar([xs[s26_idx]], [vals[s26_idx]], color=color, alpha=0.9,
               edgecolor='white', linewidth=1.0, label='Session 26 (実測)')

        for xi, v in enumerate(vals):
            if v is not None:
                ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold' if xi == s26_idx else 'normal')

        if threshold is not None:
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5,
                       label=f'文脈分離閾値 {threshold}')
        ax.set_xticks(xs)
        ax.set_xticklabels(sessions, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ylim_top = max([v for v in vals if v is not None] + [threshold or 0]) * 1.35 + 0.05
        ax.set_ylim(0, ylim_top)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # acc_B パネルに統計情報を追記
    ax = axes[1]
    sig = '有意 (p < 0.05)' if exp_c['p_value'] < 0.05 else '非有意 (p ≥ 0.05)'
    ax.errorbar([xs[s26_idx]], [exp_c['acc_B']],
                yerr=[[exp_c['acc_B'] - exp_c['ci_lo']],
                       [exp_c['ci_hi'] - exp_c['acc_B']]],
                fmt='none', color='black', capsize=6, linewidth=2)
    ax.text(0.97, 0.97,
            f'p={exp_c["p_value"]:.4f}\n{sig}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color='darkred' if exp_c['p_value'] < 0.05 else 'gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment D: sleep pattern observation ───────────────────────────────────

def _s26_run_long(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                   hunger_threshold, hunger_penalty,
                   fr_s, fr_o, fr_i, rec,
                   rng, T=_S26_SLEEP_T,
                   activity_noise=_ACTIVITY_NOISE,
                   T_consolidation=_S26_T_CONSOL):
    """T=2000 ステップの連続実行。食料・ペナルティ位置は動的に変わらない（文脈Aで固定）。

    100ステップごとの層別平均活動と平均疲労度を返す。
    Returns dict: chunks (list of dicts per 100-step window).
    """
    food_pos    = (0, 0)  # Mode A で観察
    penalty_pos = (4, 4)
    food_avail  = True
    food_timer  = 0
    fr, fc = food_pos
    pr, pc = penalty_pos

    fatigue_rates              = np.zeros(_N)
    fatigue_rates[_INP_START:_INP_END] = fr_s
    fatigue_rates[_OUT_START:_OUT_END] = fr_o
    fatigue_rates[_INT_START:_INT_END] = fr_i

    fatigue  = np.zeros(_N)
    activity = np.zeros(_N)

    row, col = 2, 2
    hp       = float(_S25_HP_START)
    hunger   = 0
    food     = 0

    chunks      = []
    chunk_acts  = []
    chunk_fats  = []

    for step in range(T):
        if hp <= 0:
            hp = float(_S25_HP_START)  # 死亡時にHP回復（観察継続のため）

        inp4 = _s20_inp4(row, col, hp, food_avail)

        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        eff = np.clip(activity * (1.0 - fatigue), 0.0, 1.0)

        if activity_noise > 0.0:
            eff = np.clip(eff + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        fatigue = np.clip(fatigue + activity * fatigue_rates - rec, 0.0, 1.0)

        chunk_acts.append(eff.copy())
        chunk_fats.append(fatigue.copy())

        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_OUT_START:_OUT_END]))
        food_eaten = False
        if action in (0, 1, 2, 3):
            if   action == 0: row = max(0, row - 1)
            elif action == 1: row = min(_CGRID - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(_CGRID - 1, col + 1)
            if row == pr and col == pc:
                hp -= _PENALTY
        elif action == 4:
            if row == fr and col == fc and food_avail:
                hp = min(_CHP_MAX, hp + _S25_FOOD_VALUE)
                food_avail = False
                food_timer = 0
                food_eaten = True
                food += 1

        hunger += 1
        if food_eaten:
            hunger = 0
        elif hunger > hunger_threshold:
            hp -= hunger_penalty

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

        # 100ステップごとにチャンク統計を記録
        if (step + 1) % _S26_SLEEP_CHUNK == 0:
            arr_a = np.array(chunk_acts)
            arr_f = np.array(chunk_fats)
            chunks.append({
                't_end':            step + 1,
                'sensory_activity': float(np.mean(arr_a[:, _INP_START:_INP_END])),
                'output_activity':  float(np.mean(arr_a[:, _OUT_START:_OUT_END])),
                'internal_activity':float(np.mean(arr_a[:, _INT_START:_INT_END])),
                'fatigue_mean':     float(np.mean(arr_f)),
                'fatigue_sensory':  float(np.mean(arr_f[:, _INP_START:_INP_END])),
                'fatigue_internal': float(np.mean(arr_f[:, _INT_START:_INT_END])),
            })
            chunk_acts = []
            chunk_fats = []

    return {'chunks': chunks, 'food_total': food}


def run_exp_d_sleep_pattern(best, seed=_S26_SEED):
    """T=2000の連続実行で睡眠様状態を観察する。

    Returns dict: chunks (list), food_total.
    """
    print(f'\n  [Exp D] 睡眠様状態の観察 (T={_S26_SLEEP_T}ステップ連続実行)')
    G_copy = best['G'].copy()
    W_copy = _s10_get_W(G_copy)
    rng    = np.random.default_rng(seed + 26300)
    res    = _s26_run_long(
        G_copy, W_copy,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['fr_s'], best['fr_o'], best['fr_i'], best['rec'],
        rng)
    n_chunks = len(res['chunks'])
    if n_chunks > 0:
        first  = res['chunks'][0]
        last   = res['chunks'][-1]
        print(f'     チャンク数={n_chunks}  food_total={res["food_total"]}')
        print(f'     最初の100step: sens={first["sensory_activity"]:.3f}  '
              f'out={first["output_activity"]:.3f}  int={first["internal_activity"]:.3f}  '
              f'fatigue={first["fatigue_mean"]:.3f}')
        print(f'     最後の100step: sens={last["sensory_activity"]:.3f}  '
              f'out={last["output_activity"]:.3f}  int={last["internal_activity"]:.3f}  '
              f'fatigue={last["fatigue_mean"]:.3f}')
    return res


def plot_exp_d_sleep_pattern(exp_d,
                               fname='images/session_26/results_s26_sleep_pattern.png'):
    chunks = exp_d['chunks']
    if not chunks:
        print('Warning: no chunks recorded for Exp D')
        return

    t_vals  = [c['t_end']             for c in chunks]
    sens    = [c['sensory_activity']  for c in chunks]
    out     = [c['output_activity']   for c in chunks]
    intern  = [c['internal_activity'] for c in chunks]
    fat_all = [c['fatigue_mean']      for c in chunks]
    fat_sen = [c['fatigue_sensory']   for c in chunks]
    fat_int = [c['fatigue_internal']  for c in chunks]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'Session 26 Exp D: 睡眠様状態の観察 (T={_S26_SLEEP_T}ステップ連続実行)\n'
        '疲労が蓄積し「感覚器が先に疲れる」2段階パターンが出るか？',
        fontsize=12,
    )

    # Panel 1: 層別活動の時系列
    ax = axes[0][0]
    ax.plot(t_vals, sens,   color='steelblue', linewidth=2, marker='o', markersize=4,
            label='感覚器 (node0-3)')
    ax.plot(t_vals, out,    color='tomato',    linewidth=2, marker='s', markersize=4,
            label='出力   (node4-8)')
    ax.plot(t_vals, intern, color='green',     linewidth=2, marker='^', markersize=4,
            label='内部   (node9-19)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean activity')
    ax.set_title('層別平均活動の時系列\n(100stepごとの平均)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: 疲労度の時系列
    ax = axes[0][1]
    ax.plot(t_vals, fat_all, color='purple',  linewidth=2, marker='o', markersize=4,
            label='全ノード平均疲労')
    ax.plot(t_vals, fat_sen, color='steelblue', linewidth=2, linestyle='--', marker='s',
            markersize=4, label='感覚器疲労')
    ax.plot(t_vals, fat_int, color='green',   linewidth=2, linestyle='--', marker='^',
            markersize=4, label='内部ノード疲労')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean fatigue')
    ax.set_title('疲労度の時系列\n(感覚器が先に疲れるか)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: 活動と疲労の相関（感覚器）
    ax = axes[1][0]
    ax.scatter(fat_sen, sens, color='steelblue', alpha=0.7, s=60, zorder=3)
    ax.set_xlabel('Sensory fatigue')
    ax.set_ylabel('Sensory activity')
    ax.set_title('感覚器: 疲労 vs 活動\n(負の相関が「感覚器疲弊」の証拠)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    if len(fat_sen) > 2:
        r, p = scipy_stats.pearsonr(fat_sen, sens)
        ax.text(0.95, 0.95, f'r={r:.3f}\np={p:.4f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.grid(True, alpha=0.3)

    # Panel 4: 2段階サマリー
    ax = axes[1][1]
    mid_idx   = len(chunks) // 2
    half1_avg_sens = float(np.mean(sens[:mid_idx]))     if mid_idx > 0 else 0.0
    half2_avg_sens = float(np.mean(sens[mid_idx:]))     if mid_idx < len(sens) else 0.0
    half1_avg_int  = float(np.mean(intern[:mid_idx]))   if mid_idx > 0 else 0.0
    half2_avg_int  = float(np.mean(intern[mid_idx:]))   if mid_idx < len(intern) else 0.0

    sens_drop = half1_avg_sens - half2_avg_sens
    int_drop  = half1_avg_int  - half2_avg_int
    two_stage = sens_drop > int_drop  # 感覚器が先に下がっているか

    summary = (
        f'2段階パターン分析\n\n'
        f'前半 (〜{t_vals[mid_idx-1] if mid_idx > 0 else "?"}step)\n'
        f'  感覚器活動: {half1_avg_sens:.3f}\n'
        f'  内部活動:   {half1_avg_int:.3f}\n\n'
        f'後半 ({t_vals[mid_idx] if mid_idx < len(t_vals) else "?"}step〜)\n'
        f'  感覚器活動: {half2_avg_sens:.3f}\n'
        f'  内部活動:   {half2_avg_int:.3f}\n\n'
        f'感覚器の低下: Δ={sens_drop:.3f}\n'
        f'内部の低下:   Δ={int_drop:.3f}\n\n'
        f'{"✓ 感覚器優先低下 → 睡眠様状態の萌芽" if two_stage else "✗ 2段階パターンは不明確"}'
    )
    color = 'lightgreen' if two_stage else 'lightyellow'
    ax.text(0.5, 0.5, summary, transform=ax.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.85))
    ax.set_title('2段階睡眠パターン判定')
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 26: 疲労による飽和の解消と文脈分離 ===')
    print('疲労メカニズム: effective_activity = activity * (1 - fatigue)')
    print('疲労パラメータ fr_s/fr_o/fr_i/rec を進化させて飽和を解消する')
    print()

    # Experiment A: 疲労パラメータの進化
    print('[Exp A] 疲労パラメータの進化収束')
    best, hist = run_exp_a_genome_convergence(seed=_S26_SEED)
    plot_exp_a_genome_convergence(best, hist)

    # Experiment B: 飽和の解消確認
    print('\n[Exp B] 飽和の解消確認 (疲労あり vs なし)')
    exp_b = run_exp_b_saturation(best, seed=_S26_SEED, n_episodes=20)
    plot_exp_b_saturation(exp_b)

    # Experiment C: 文脈分離の計測
    print('\n[Exp C] 文脈分離の計測')
    exp_c = run_exp_c_context(best, seed=_S26_SEED, n_per_mode=25)
    plot_exp_c_context(exp_c)

    # Experiment D: 睡眠様状態の観察
    print('\n[Exp D] 睡眠様状態の観察')
    exp_d = run_exp_d_sleep_pattern(best, seed=_S26_SEED)
    plot_exp_d_sleep_pattern(exp_d)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('=== Session 26 Summary ===')
    print()

    fr_ratio = best['fr_s'] / best['fr_i'] if best['fr_i'] > 1e-9 else float('nan')
    print('[A] 疲労パラメータの収束')
    print(f'  fr_s={best["fr_s"]:.4f}  fr_o={best["fr_o"]:.4f}  '
          f'fr_i={best["fr_i"]:.4f}  rec={best["rec"]:.4f}')
    print(f'  fr_s/fr_i = {fr_ratio:.3f}  '
          f'→ {"感覚器優先疲労（睡眠様状態の前提が整った）" if not np.isnan(fr_ratio) and fr_ratio > 1.0 else "内部優先疲労"}')

    print('\n[B] 飽和の解消')
    wf  = exp_b['with_fatigue']
    wof = exp_b['without_fatigue']
    print(f'  疲労なし: 全ノード平均活動 = {np.mean(wof["node_means"]):.3f}  '
          f'std = {np.mean(wof["node_stds"]):.3f}  '
          f'cos_dist = {wof["mode_patterns"]["cosine_dist"]:.4f}')
    print(f'  疲労あり: 全ノード平均活動 = {np.mean(wf["node_means"]):.3f}  '
          f'std = {np.mean(wf["node_stds"]):.3f}  '
          f'cos_dist = {wf["mode_patterns"]["cosine_dist"]:.4f}')
    saturation_resolved = np.mean(wf['node_means']) < np.mean(wof['node_means']) - 0.05
    print(f'  → {"✓ 飽和が解消されている" if saturation_resolved else "△ 飽和の解消は限定的"}')

    print('\n[C] 文脈分離')
    sig = exp_c['p_value'] < 0.05
    both = exp_c['acc_A'] >= 0.6 and exp_c['acc_B'] >= 0.6
    print(f'  acc_A={exp_c["acc_A"]:.3f}  acc_B={exp_c["acc_B"]:.3f}  '
          f'cos_dist={exp_c["cosine_dist"]:.4f}')
    print(f'  p={exp_c["p_value"]:.4f}  95%CI=[{exp_c["ci_lo"]:.3f},{exp_c["ci_hi"]:.3f}]')
    print(f'  → {"✓ 文脈分離達成（両モード ≥ 0.6）" if both else "✗ 文脈分離未達成"}')
    print(f'     統計的有意性: {"✓ 有意 (p < 0.05)" if sig else "✗ 非有意"}')

    print('\n[D] 睡眠様状態')
    if exp_d['chunks']:
        mid   = len(exp_d['chunks']) // 2
        sens1 = np.mean([c['sensory_activity']   for c in exp_d['chunks'][:mid]])
        sens2 = np.mean([c['sensory_activity']   for c in exp_d['chunks'][mid:]])
        int1  = np.mean([c['internal_activity']  for c in exp_d['chunks'][:mid]])
        int2  = np.mean([c['internal_activity']  for c in exp_d['chunks'][mid:]])
        two_stage = (sens1 - sens2) > (int1 - int2)
        print(f'  感覚器活動: 前半={sens1:.3f} → 後半={sens2:.3f}  Δ={sens1-sens2:.3f}')
        print(f'  内部活動:   前半={int1:.3f} → 後半={int2:.3f}  Δ={int1-int2:.3f}')
        print(f'  → {"✓ 感覚器優先低下 → 睡眠様状態の萌芽" if two_stage else "✗ 2段階パターン不明確"}')

    print('\nDone.')
