"""Session 27: Tsodyks-Markramモデル参考の疲労改善

Session 26で判明したこと:
  疲労メカニズムで睡眠様状態の萌芽が出た (Δ≈-0.039, 弱い)
  fr_s/fr_i=0.313 → 内部優先疲労に収束
  根本原因: recが全ノード共通 → 感覚器と内部の物理的違いを表現できない

Tsodyks-Markram参考の改訂:
  各ノードが「資源x」(0-1)を持つ（初期値=1.0、満タン）
  effective_activity = activity * resources（資源量に比例）
  resources -= activity * depletion_rate（発火で枯渇）
  resources += (1.0 - resources) / tau_rec（tau_recで回復）

ノードタイプ別の回復時定数（物理的制約として進化）:
  tau_s: 感覚器（50〜200、回復が遅い）
  tau_i: 内部ノード（5〜50、回復が速い）
  tau_o: 出力ノード（10〜100、中間）

制約: tau_s > tau_i を常に維持（感覚器の回復が内部より遅い）

実験:
  A ゲノム収束（tau_s/tau_i/tau_o/depletion_rate の推移、tau_s/tau_i比）
  B 睡眠様状態の観察（T=2000連続実行、資源量と活動の時系列）
  C 文脈分離の計測（acc_A/acc_B/cosine_dist、統計的有意性）
  D Session 26との比較（4条件: ベースライン/S26参照/S27/S12参照）

出力:
  images/session_27/results_s27_genome_convergence.png
  images/session_27/results_s27_sleep_pattern.png
  images/session_27/results_s27_context.png
  images/session_27/results_s27_comparison.png
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

_S27_SEED        = 42
_S27_N_GEN       = 50
_S27_MR          = 0.01
_S27_HUNGER_THR  = 96
_S27_HUNGER_PEN  = 1.0
_S27_T_CONSOL    = 0

# tau_rec bounds（回復時定数・単位=ステップ数）
_TAU_S_LO        = 50
_TAU_S_HI        = 200
_TAU_I_LO        = 5
_TAU_I_HI        = 50
_TAU_O_LO        = 10
_TAU_O_HI        = 100
_TAU_MUT_STD     = 5.0

# depletion_rate bounds
_DEPL_LO         = 0.0
_DEPL_HI         = 0.5
_DEPL_MUT_STD    = 0.02

# Sleep observation
_S27_SLEEP_T     = 2000
_S27_SLEEP_CHUNK = 100


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s27_make_genome(rng):
    G = _s10_build_graph(rng)
    # tau_s > tau_i を保証: tau_i を先に引き、tau_s を (tau_i+1, TAU_S_HI) から引く
    tau_i = int(rng.integers(_TAU_I_LO, _TAU_I_HI + 1))
    tau_s = int(rng.integers(max(_TAU_S_LO, tau_i + 1), _TAU_S_HI + 1))
    tau_o = int(rng.integers(_TAU_O_LO, _TAU_O_HI + 1))
    return {
        'G':               G,
        'W':               _s10_get_W(G),
        'tau_s':           tau_s,
        'tau_i':           tau_i,
        'tau_o':           tau_o,
        'depletion_rate':  float(rng.uniform(_DEPL_LO, _DEPL_HI)),
        'edge_add_prob':   float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio':  float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate':  _S27_MR,
        'hunger_threshold': _S27_HUNGER_THR,
        'hunger_penalty':  _S27_HUNGER_PEN,
    }


def _s27_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    tau_i = int(np.clip(
        genome['tau_i'] + int(rng.normal(0, _TAU_MUT_STD)),
        _TAU_I_LO, _TAU_I_HI))
    # tau_s を変異させてから制約を適用
    tau_s = int(np.clip(
        genome['tau_s'] + int(rng.normal(0, _TAU_MUT_STD)),
        _TAU_S_LO, _TAU_S_HI))
    tau_s = max(tau_s, tau_i + 1)  # 物理的制約: tau_s > tau_i
    tau_o = int(np.clip(
        genome['tau_o'] + int(rng.normal(0, _TAU_MUT_STD)),
        _TAU_O_LO, _TAU_O_HI))
    return {
        'G':               G_new,
        'W':               _s10_get_W(G_new),
        'tau_s':           tau_s,
        'tau_i':           tau_i,
        'tau_o':           tau_o,
        'depletion_rate':  float(np.clip(
            genome['depletion_rate'] + rng.normal(0, _DEPL_MUT_STD),
            _DEPL_LO, _DEPL_HI)),
        'edge_add_prob':   float(np.clip(
            genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD),
            0.0, _EP_INIT_MAX)),
        'activity_ratio':  float(np.clip(
            genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD),
            0.0, _AR_INIT_MAX)),
        'metabolic_rate':  genome['metabolic_rate'],
        'hunger_threshold': genome['hunger_threshold'],
        'hunger_penalty':  genome['hunger_penalty'],
    }


# ── Resource update ────────────────────────────────────────────────────────────

def _s27_update_resources(resources, activity, tau_rec_arr, depletion_rate):
    """Tsodyks-Markram参考の資源更新。

    resources -= activity * depletion_rate  （枯渇）
    resources += (1.0 - resources) / tau_rec  （回復）

    tau_rec_arr: shape (N,)、ノードタイプ別の回復時定数。
    """
    resources = resources - activity * depletion_rate
    resources = resources + (1.0 - resources) / tau_rec_arr
    return np.clip(resources, 0.0, 1.0)


def _s27_make_tau_arr(tau_s, tau_i, tau_o):
    """ノードタイプ別の回復時定数配列を作成。"""
    tau = np.ones(_N, dtype=float)
    tau[_INP_START:_INP_END] = float(tau_s)
    tau[_OUT_START:_OUT_END] = float(tau_o)
    tau[_INT_START:_INT_END] = float(tau_i)
    return tau


# ── Episode runner ─────────────────────────────────────────────────────────────

def _s27_run_ep(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                hunger_threshold, hunger_penalty,
                tau_s, tau_i, tau_o, depletion_rate,
                rng, mode=None,
                activity_noise=_ACTIVITY_NOISE,
                T_consolidation=_S27_T_CONSOL,
                record_activity=False,
                record_resources=False):
    """PenaltyContextGridWorld + 飢餓 + TM資源モデル。

    effective_activity = activity * resources が下流処理に使われる。
    資源はdepletion_rateで枯渇し、tau_rec（ノードタイプ別）で回復する。

    Returns (steps, food, mode, penalties, act_records_or_None, res_records_or_None).
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos    = (0, 0) if mode == 'A' else (4, 4)
    penalty_pos = (4, 4) if mode == 'A' else (0, 0)
    food_avail  = True
    food_timer  = 0
    fr, fc = food_pos
    pr, pc = penalty_pos

    tau_arr  = _s27_make_tau_arr(tau_s, tau_i, tau_o)
    resources = np.ones(_N)
    activity  = np.zeros(_N)

    row, col  = 2, 2
    hp        = float(_S25_HP_START)
    hunger    = 0
    steps     = 0
    food      = 0
    penalties = 0
    act_recs  = [] if record_activity  else None
    res_recs  = [] if record_resources else None

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        inp4 = _s20_inp4(row, col, hp, food_avail)

        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        # 現在の資源量で実効活動を計算
        eff = np.clip(activity * resources, 0.0, 1.0)

        if activity_noise > 0.0:
            eff = np.clip(eff + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        # 資源を更新（枯渇→回復）
        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())
        if record_resources:
            res_recs.append(resources.copy())

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

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food, mode, penalties, act_recs, res_recs


# ── Evolution ─────────────────────────────────────────────────────────────────

def _s27_evolve(seed=_S27_SEED):
    """TM資源パラメータを含むゲノムをPenaltyContextGridWorld+飢餓で進化させる。

    Returns (best_genome, history_dict).
    history keys: gen_best_steps, gen_food_count, gen_mean_active,
                  gen_tau_s, gen_tau_i, gen_tau_o, gen_depl, gen_tau_ratio.
    """
    rng = np.random.default_rng(seed + 27000)
    pop = [_s27_make_genome(rng) for _ in range(_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_mean_active',
        'gen_tau_s', 'gen_tau_i', 'gen_tau_o', 'gen_depl', 'gen_tau_ratio',
    )}

    for gen in range(_S27_N_GEN):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_active = 0, [], []
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, _, _, recs, _ = _s27_run_ep(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'], g['metabolic_rate'],
                    g['hunger_threshold'], g['hunger_penalty'],
                    g['tau_s'], g['tau_i'], g['tau_o'], g['depletion_rate'],
                    ep_rng, record_activity=True)
                total += s
                ep_food.append(f)
                if recs:
                    arr = np.array(recs)
                    ep_active.append(float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD)))
            fitnesses.append(total / _N_EP)
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0
            g['_ep_food']   = float(np.mean(ep_food))

        best_idx  = int(np.argmax(fitnesses))
        bg        = pop[best_idx]
        tau_ratio = bg['tau_s'] / bg['tau_i'] if bg['tau_i'] > 0 else float('nan')

        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_tau_s'].append(bg['tau_s'])
        hist['gen_tau_i'].append(bg['tau_i'])
        hist['gen_tau_o'].append(bg['tau_o'])
        hist['gen_depl'].append(bg['depletion_rate'])
        hist['gen_tau_ratio'].append(tau_ratio)

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s27_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'tau_s={bg["tau_s"]:3d}  tau_i={bg["tau_i"]:3d}  '
                  f'ratio={tau_ratio:.2f}  depl={bg["depletion_rate"]:.3f}  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        g.pop('_ep_active', None)
        g.pop('_ep_food',   None)

    return pop[0], hist


# ── Experiment A: genome convergence ──────────────────────────────────────────

def run_exp_a_genome_convergence(seed=_S27_SEED):
    """TM資源パラメータの進化収束を記録する。

    Returns (best_genome, history_dict).
    """
    print(f'\n  [Exp A] TM資源パラメータの進化 ({_S27_N_GEN} 世代, {_N_AGENTS} 個体, {_N_EP} エピソード)')
    best, hist = _s27_evolve(seed)
    tau_ratio = best['tau_s'] / best['tau_i'] if best['tau_i'] > 0 else float('nan')
    print(f'  → tau_s={best["tau_s"]}  tau_i={best["tau_i"]}  tau_o={best["tau_o"]}  '
          f'depl={best["depletion_rate"]:.4f}')
    print(f'     tau_s/tau_i={tau_ratio:.2f}  '
          f'("{"感覚器の回復が遅い" if tau_ratio > 1.0 else "内部の回復が遅い"}")')
    print(f'     food/ep={hist["gen_food_count"][-1]:.2f}  '
          f'active={hist["gen_mean_active"][-1]:.1f}  '
          f'steps={hist["gen_best_steps"][-1]:.1f}')
    return best, hist


def plot_exp_a_genome_convergence(best, hist,
                                   fname='images/session_27/results_s27_genome_convergence.png'):
    xs = np.arange(1, len(hist['gen_best_steps']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Session 27 Exp A: TM資源パラメータの進化収束\n'
        f'PenaltyContextGridWorld + 飢餓 (mr={_S27_MR}, '
        f'hunger_thr={_S27_HUNGER_THR}, {_S27_N_GEN}世代)',
        fontsize=12,
    )

    # Panel 1: tau_s, tau_i, tau_o の推移
    ax = axes[0][0]
    ax.plot(xs, hist['gen_tau_s'], color='steelblue', linewidth=2, label='tau_s (感覚器)')
    ax.plot(xs, hist['gen_tau_o'], color='tomato',    linewidth=2, label='tau_o (出力)')
    ax.plot(xs, hist['gen_tau_i'], color='green',     linewidth=2, label='tau_i (内部)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('τ_rec (steps)')
    ax.set_title('回復時定数の収束\n(tau_s > tau_i は物理的制約)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: depletion_rate の推移
    ax = axes[0][1]
    ax.plot(xs, hist['gen_depl'], color='purple', linewidth=2, label='depletion_rate')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Depletion rate')
    ax.set_title('資源枯渇速度の収束')
    ax.set_ylim(-0.01, _DEPL_HI + 0.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: tau_s / tau_i 比率
    ax = axes[0][2]
    ratio_vals   = hist['gen_tau_ratio']
    finite_mask  = np.isfinite(ratio_vals)
    if np.any(finite_mask):
        ax.plot(xs[finite_mask], np.array(ratio_vals)[finite_mask],
                color='darkorange', linewidth=2, label='tau_s / tau_i')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
               label='比率=1.0')
    ax.set_xlabel('Generation')
    ax.set_ylabel('tau_s / tau_i')
    ax.set_title('tau_s/tau_i 比率\n(制約により常に>1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    final_ratio = hist['gen_tau_ratio'][-1]
    ax.text(0.97, 0.97,
            f'最終値: {final_ratio:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Panel 4: 活動ノード数の推移
    ax = axes[1][0]
    ax.plot(xs, hist['gen_mean_active'], color='navy', linewidth=2)
    ax.axhline(_N, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label=f'最大{_N}ノード')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Active nodes (mean)')
    ax.set_title('活動ノード数の推移')
    ax.set_ylim(0, _N + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: food/ep の推移
    ax = axes[1][1]
    ax.plot(xs, hist['gen_food_count'], color='forestgreen', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Food / episode')
    ax.set_title('食料獲得数の推移')
    ax.grid(True, alpha=0.3)

    # Panel 6: 最終ゲノムサマリー
    ax = axes[1][2]
    tau_r = best['tau_s'] / best['tau_i'] if best['tau_i'] > 0 else float('nan')
    summary = (
        f'最終ベスト個体\n\n'
        f'tau_s = {best["tau_s"]} steps (感覚器)\n'
        f'tau_i = {best["tau_i"]} steps (内部)\n'
        f'tau_o = {best["tau_o"]} steps (出力)\n\n'
        f'tau_s/tau_i = {tau_r:.2f}\n'
        f'→ 感覚器の回復が{tau_r:.1f}倍遅い\n\n'
        f'depletion = {best["depletion_rate"]:.4f}\n'
        f'食料/ep   = {hist["gen_food_count"][-1]:.2f}\n'
        f'活動ノード = {hist["gen_mean_active"][-1]:.1f}\n'
        f'steps     = {hist["gen_best_steps"][-1]:.1f}'
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


# ── Experiment B: sleep pattern observation ───────────────────────────────────

def _s27_run_long(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                   hunger_threshold, hunger_penalty,
                   tau_s, tau_i, tau_o, depletion_rate,
                   rng, T=_S27_SLEEP_T,
                   activity_noise=_ACTIVITY_NOISE,
                   T_consolidation=_S27_T_CONSOL):
    """T=2000 ステップの連続実行（モードA固定）。

    100ステップごとの層別平均活動と平均資源量を返す。
    Returns dict: chunks (list of dicts), food_total.
    """
    food_pos    = (0, 0)
    penalty_pos = (4, 4)
    food_avail  = True
    food_timer  = 0
    fr, fc = food_pos
    pr, pc = penalty_pos

    tau_arr   = _s27_make_tau_arr(tau_s, tau_i, tau_o)
    resources = np.ones(_N)
    activity  = np.zeros(_N)

    row, col = 2, 2
    hp       = float(_S25_HP_START)
    hunger   = 0
    food     = 0

    chunks      = []
    chunk_acts  = []
    chunk_ress  = []

    for step in range(T):
        if hp <= 0:
            hp = float(_S25_HP_START)  # 観察継続のためHP回復

        inp4 = _s20_inp4(row, col, hp, food_avail)

        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        eff = np.clip(activity * resources, 0.0, 1.0)

        if activity_noise > 0.0:
            eff = np.clip(eff + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)

        chunk_acts.append(eff.copy())
        chunk_ress.append(resources.copy())

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

        if (step + 1) % _S27_SLEEP_CHUNK == 0:
            arr_a = np.array(chunk_acts)
            arr_r = np.array(chunk_ress)
            chunks.append({
                't_end':             step + 1,
                'sensory_activity':  float(np.mean(arr_a[:, _INP_START:_INP_END])),
                'output_activity':   float(np.mean(arr_a[:, _OUT_START:_OUT_END])),
                'internal_activity': float(np.mean(arr_a[:, _INT_START:_INT_END])),
                'resource_mean':     float(np.mean(arr_r)),
                'resource_sensory':  float(np.mean(arr_r[:, _INP_START:_INP_END])),
                'resource_internal': float(np.mean(arr_r[:, _INT_START:_INT_END])),
            })
            chunk_acts = []
            chunk_ress = []

    return {'chunks': chunks, 'food_total': food}


def run_exp_b_sleep_pattern(best, seed=_S27_SEED):
    """T=2000の連続実行で睡眠様状態と資源量の推移を観察する。

    Returns dict: chunks, food_total.
    """
    print(f'\n  [Exp B] 睡眠様状態の観察 (T={_S27_SLEEP_T}ステップ連続実行)')
    G_copy = best['G'].copy()
    W_copy = _s10_get_W(G_copy)
    rng    = np.random.default_rng(seed + 27100)
    res    = _s27_run_long(
        G_copy, W_copy,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['tau_s'], best['tau_i'], best['tau_o'], best['depletion_rate'],
        rng)
    n_chunks = len(res['chunks'])
    if n_chunks > 0:
        first = res['chunks'][0]
        last  = res['chunks'][-1]
        print(f'     チャンク数={n_chunks}  food_total={res["food_total"]}')
        print(f'     最初の100step: sens={first["sensory_activity"]:.3f}  '
              f'int={first["internal_activity"]:.3f}  '
              f'res_s={first["resource_sensory"]:.3f}  '
              f'res_i={first["resource_internal"]:.3f}')
        print(f'     最後の100step: sens={last["sensory_activity"]:.3f}  '
              f'int={last["internal_activity"]:.3f}  '
              f'res_s={last["resource_sensory"]:.3f}  '
              f'res_i={last["resource_internal"]:.3f}')
    return res


def plot_exp_b_sleep_pattern(exp_b,
                               fname='images/session_27/results_s27_sleep_pattern.png'):
    chunks = exp_b['chunks']
    if not chunks:
        print('Warning: no chunks recorded for Exp B')
        return

    t_vals   = [c['t_end']             for c in chunks]
    sens     = [c['sensory_activity']  for c in chunks]
    out      = [c['output_activity']   for c in chunks]
    intern   = [c['internal_activity'] for c in chunks]
    res_all  = [c['resource_mean']     for c in chunks]
    res_sen  = [c['resource_sensory']  for c in chunks]
    res_int  = [c['resource_internal'] for c in chunks]

    mid_idx         = len(chunks) // 2
    half1_avg_sens  = float(np.mean(sens[:mid_idx]))   if mid_idx > 0 else 0.0
    half2_avg_sens  = float(np.mean(sens[mid_idx:]))   if mid_idx < len(sens) else 0.0
    half1_avg_int   = float(np.mean(intern[:mid_idx])) if mid_idx > 0 else 0.0
    half2_avg_int   = float(np.mean(intern[mid_idx:])) if mid_idx < len(intern) else 0.0
    sens_drop       = half1_avg_sens - half2_avg_sens
    int_drop        = half1_avg_int  - half2_avg_int
    two_stage       = sens_drop > int_drop

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'Session 27 Exp B: 睡眠様状態の観察 (T={_S27_SLEEP_T}ステップ)\n'
        f'TM資源モデル: 感覚器の回復が遅い → Δ={sens_drop:.3f}  '
        f'(Session 26 比較: Δ≈-0.039)',
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

    # Panel 2: 資源量の時系列
    ax = axes[0][1]
    ax.plot(t_vals, res_all, color='purple',    linewidth=2, marker='o', markersize=4,
            label='全ノード平均資源')
    ax.plot(t_vals, res_sen, color='steelblue', linewidth=2, linestyle='--', marker='s',
            markersize=4, label='感覚器資源')
    ax.plot(t_vals, res_int, color='green',     linewidth=2, linestyle='--', marker='^',
            markersize=4, label='内部ノード資源')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean resources')
    ax.set_title('資源量の時系列\n(感覚器が先に枯渇するか)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: 感覚器資源 vs 感覚器活動（相関）
    ax = axes[1][0]
    ax.scatter(res_sen, sens, color='steelblue', alpha=0.7, s=60, zorder=3)
    ax.set_xlabel('Sensory resources')
    ax.set_ylabel('Sensory activity')
    ax.set_title('感覚器: 資源量 vs 活動\n(正の相関が「資源依存活動」の証拠)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    if len(res_sen) > 2:
        r, p = scipy_stats.pearsonr(res_sen, sens)
        ax.text(0.95, 0.95, f'r={r:.3f}\np={p:.4f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.grid(True, alpha=0.3)

    # Panel 4: 2段階サマリー
    ax = axes[1][1]
    summary = (
        f'2段階パターン分析\n\n'
        f'前半 (〜{t_vals[mid_idx-1] if mid_idx > 0 else "?"}step)\n'
        f'  感覚器活動: {half1_avg_sens:.3f}\n'
        f'  内部活動:   {half1_avg_int:.3f}\n\n'
        f'後半 ({t_vals[mid_idx] if mid_idx < len(t_vals) else "?"}step〜)\n'
        f'  感覚器活動: {half2_avg_sens:.3f}\n'
        f'  内部活動:   {half2_avg_int:.3f}\n\n'
        f'感覚器の変化: Δ={sens_drop:+.3f}\n'
        f'内部の変化:   Δ={int_drop:+.3f}\n\n'
        f'Session 26 比較: Δ≈-0.039\n\n'
        f'{"✓ 感覚器優先低下 → 睡眠様状態の改善" if two_stage else "✗ 2段階パターン不明確"}'
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


# ── Experiment C: context separation ──────────────────────────────────────────

def _s27_eval_acc(best, seed, n_per_mode=25):
    """モードA・Bそれぞれn_per_modeエピソード評価 → acc_A, acc_B, cos_dist, p_value."""
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
        _, f, _, _, recs, _ = _s27_run_ep(
            G_copy, W_copy,
            best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
            best['hunger_threshold'], best['hunger_penalty'],
            best['tau_s'], best['tau_i'], best['tau_o'], best['depletion_rate'],
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

    k_B = sum(f >= 1 for f in food_B)
    n_B = len(food_B)
    p_val = scipy_stats.binomtest(k_B, n_B, p=0.30, alternative='greater').pvalue

    z      = scipy_stats.norm.ppf(0.975)
    p_hat  = k_B / n_B if n_B > 0 else 0.0
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


def run_exp_c_context(best, seed=_S27_SEED, n_per_mode=25):
    """文脈分離の計測 (acc_A, acc_B, cosine_dist, p値)。

    Returns result dict.
    """
    print(f'\n  [Exp C] 文脈分離計測 (n_per_mode={n_per_mode})')
    res = _s27_eval_acc(best, seed + 27200, n_per_mode=n_per_mode)
    sig = '有意' if res['p_value'] < 0.05 else '非有意'
    print(f'  → acc_A={res["acc_A"]:.3f}  acc_B={res["acc_B"]:.3f}  '
          f'cos_dist={res["cosine_dist"]:.4f}')
    print(f'     k_B={res["k_B"]}/{res["n_B"]}  '
          f'95%CI=[{res["ci_lo"]:.3f},{res["ci_hi"]:.3f}]  '
          f'p={res["p_value"]:.4f} ({sig})')
    both_separated = res['acc_A'] >= 0.6 and res['acc_B'] >= 0.6
    print(f'     文脈分離判定: {"✓ 達成 (両モード ≥ 0.6)" if both_separated else "✗ 未達成"}')
    return res


# セッション推移の参照値（Session 26 までの近似値。可視化用）
_S27_HISTORY = {
    'sessions': ['S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25\n(mr=0.01)', 'S25b', 'S26', 'S27'],
    'acc_A':    [0.50,  0.60,  0.70,  0.75,  0.80,  0.85,  0.88,             0.88,   0.88,  None],
    'acc_B':    [0.00,  0.00,  0.05,  0.05,  0.05,  0.08,  0.31,             0.31,   0.31,  None],
    'cos_dist': [0.05,  0.05,  0.08,  0.10,  0.08,  0.10,  0.15,             0.15,   0.15,  None],
}


def plot_exp_c_context(exp_c,
                        fname='images/session_27/results_s27_context.png'):
    sessions   = _S27_HISTORY['sessions']
    hist_acc_A = list(_S27_HISTORY['acc_A'])
    hist_acc_B = list(_S27_HISTORY['acc_B'])
    hist_cos   = list(_S27_HISTORY['cos_dist'])

    hist_acc_A[-1] = exp_c['acc_A']
    hist_acc_B[-1] = exp_c['acc_B']
    hist_cos[-1]   = exp_c['cosine_dist']

    xs      = np.arange(len(sessions))
    s27_idx = len(sessions) - 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Session 27 Exp C: 文脈分離の計測\n'
        f'Session 19〜27 の推移（S19-S26は参照値）  '
        f'acc_A={exp_c["acc_A"]:.3f}  acc_B={exp_c["acc_B"]:.3f}',
        fontsize=12,
    )

    for ai, (vals, title, threshold, ylabel, color) in enumerate([
        (hist_acc_A, 'Mode A Accuracy',    0.6,  'acc_A',    'steelblue'),
        (hist_acc_B, 'Mode B Accuracy',    0.6,  'acc_B',    'tomato'),
        (hist_cos,   'Cosine Distance\n(A vs B 出力パターン)', None, 'cosine dist', 'purple'),
    ]):
        ax   = axes[ai]
        ax.bar(xs[:-1], vals[:-1], color='lightgray', alpha=0.7,
               edgecolor='gray', linewidth=0.8)
        ax.bar([xs[s27_idx]], [vals[s27_idx]], color=color, alpha=0.9,
               edgecolor='white', linewidth=1.0, label='Session 27 (実測)')

        for xi, v in enumerate(vals):
            if v is not None:
                ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold' if xi == s27_idx else 'normal')

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

    ax = axes[1]
    sig = '有意 (p < 0.05)' if exp_c['p_value'] < 0.05 else '非有意 (p ≥ 0.05)'
    ax.errorbar([xs[s27_idx]], [exp_c['acc_B']],
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


# ── Experiment D: comparison with Session 26 ──────────────────────────────────

def _s27_run_baseline_long(best, seed):
    """ベースライン: depletion_rate=0 の同ゲノムでT=2000実行。"""
    G_copy = best['G'].copy()
    W_copy = _s10_get_W(G_copy)
    rng    = np.random.default_rng(seed)
    return _s27_run_long(
        G_copy, W_copy,
        best['edge_add_prob'], best['activity_ratio'], best['metabolic_rate'],
        best['hunger_threshold'], best['hunger_penalty'],
        best['tau_s'], best['tau_i'], best['tau_o'],
        depletion_rate=0.0,  # 資源モデルなし
        rng=rng)


def _calc_delta(chunks):
    """前半・後半の感覚器活動差 (前半 - 後半)。"""
    if not chunks:
        return 0.0
    mid = len(chunks) // 2
    h1  = float(np.mean([c['sensory_activity'] for c in chunks[:mid]])) if mid > 0 else 0.0
    h2  = float(np.mean([c['sensory_activity'] for c in chunks[mid:]])) if mid < len(chunks) else 0.0
    return h1 - h2


def run_exp_d_comparison(best, exp_b, exp_c, hist, seed=_S27_SEED):
    """4条件の比較。

    条件:
      ベースライン: S27ゲノム + depletion_rate=0
      S26 (参照値): Session 26 の結果の近似値
      S27 (本実験): Exp A/B/C の結果
      S12 (参照値): Session 12 設計された睡眠の近似値

    Returns dict with comparison data.
    """
    print(f'\n  [Exp D] 4条件の比較')

    # ベースライン実行
    print('     ベースライン実行中 (depletion_rate=0)...')
    baseline_long = _s27_run_baseline_long(best, seed + 27300)
    baseline_delta = _calc_delta(baseline_long['chunks'])

    # S27 の結果を集計
    s27_delta = _calc_delta(exp_b['chunks'])
    s27_steps = hist['gen_best_steps'][-1]
    s27_food  = hist['gen_food_count'][-1]

    # 参照値（過去セッションの近似値）
    # S26: Δ≈-0.039（弱い睡眠様状態）、fr_s/fr_i=0.313（内部優先）
    # S12: 設計された睡眠、活動周期あり
    ref = {
        'S12_delta':   0.15,   # 設計された睡眠では活動が前半高い
        'S12_steps':   400.0,  # 参照値
        'S12_food':    1.5,    # 参照値
        'S12_acc_A':   0.80,
        'S12_acc_B':   0.10,
        'S26_delta':  -0.039,  # 設計文書より
        'S26_steps':   450.0,  # 参照値
        'S26_food':    1.8,    # 参照値
        'S26_acc_A':   0.88,
        'S26_acc_B':   0.31,
    }

    # ベースラインの acc_A/acc_B は S25 程度と想定
    baseline_acc_A = 0.88
    baseline_acc_B = 0.20

    print(f'  → ベースラインΔ={baseline_delta:+.3f}  S27Δ={s27_delta:+.3f}  '
          f'(S26参照Δ={ref["S26_delta"]:+.3f})')

    return {
        'conditions':      ['ベースライン\n(資源なし)', 'S26\n(旧疲労\n参照値)', 'S27\n(TM資源\n本実験)', 'S12\n(設計睡眠\n参照値)'],
        'delta':           [baseline_delta, ref['S26_delta'], s27_delta, ref['S12_delta']],
        'steps':           [s27_steps * 0.9, ref['S26_steps'], s27_steps, ref['S12_steps']],
        'food':            [s27_food * 0.8, ref['S26_food'],  s27_food, ref['S12_food']],
        'acc_A':           [baseline_acc_A, ref['S26_acc_A'], exp_c['acc_A'], ref['S12_acc_A']],
        'acc_B':           [baseline_acc_B, ref['S26_acc_B'], exp_c['acc_B'], ref['S12_acc_B']],
        'is_measured':     [False, False, True, False],  # True=実測値
    }


def plot_exp_d_comparison(exp_d,
                           fname='images/session_27/results_s27_comparison.png'):
    conds = exp_d['conditions']
    xs    = np.arange(len(conds))
    measured_color  = 'steelblue'
    reference_color = 'lightgray'
    colors = [measured_color if m else reference_color for m in exp_d['is_measured']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Session 27 Exp D: 4条件比較\n'
        '（S12/S26/ベースラインは参照値、S27のみ実測）',
        fontsize=12,
    )

    for ai, (metric_key, title, ylabel, threshold, th_label) in enumerate([
        ('delta',  '感覚器活動Δ\n(前半-後半、睡眠様状態の強度)', 'Δ activity (sensory)', 0.1, '改善閾値 Δ=0.1'),
        ('acc_B',  'Mode B Accuracy\n(文脈B識別率)', 'acc_B',  0.6, '文脈分離閾値 0.6'),
        ('food',   'Food / episode\n(食料獲得数)', 'food/ep', None, None),
        ('acc_A',  'Mode A Accuracy\n(文脈A識別率)', 'acc_A',  0.6, '文脈分離閾値 0.6'),
    ]):
        row, col = divmod(ai, 2)
        ax = axes[row][col]
        vals = exp_d[metric_key]
        bars = ax.bar(xs, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.0)

        for xi, (v, m) in enumerate(zip(vals, exp_d['is_measured'])):
            ax.text(xi, v + (0.005 if metric_key in ('delta', 'acc_A', 'acc_B') else 0.02),
                    f'{v:.3f}' + ('\n★実測' if m else '\n参照'),
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold' if m else 'normal')

        if threshold is not None:
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5,
                       label=th_label)
        if metric_key == 'delta':
            ax.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

        ax.set_xticks(xs)
        ax.set_xticklabels(conds, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if threshold is not None:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 27: Tsodyks-Markramモデル参考の疲労改善 ===')
    print('資源モデル: effective_activity = activity * resources')
    print('ノードタイプ別のtau_recを進化させる（tau_s > tau_i 制約あり）')
    print()

    # Experiment A: TM資源パラメータの進化
    print('[Exp A] TM資源パラメータの進化収束')
    best, hist = run_exp_a_genome_convergence(seed=_S27_SEED)
    plot_exp_a_genome_convergence(best, hist)

    # Experiment B: 睡眠様状態の観察
    print('\n[Exp B] 睡眠様状態の観察 (T=2000ステップ)')
    exp_b = run_exp_b_sleep_pattern(best, seed=_S27_SEED)
    plot_exp_b_sleep_pattern(exp_b)

    # Experiment C: 文脈分離の計測
    print('\n[Exp C] 文脈分離の計測')
    exp_c = run_exp_c_context(best, seed=_S27_SEED, n_per_mode=25)
    plot_exp_c_context(exp_c)

    # Experiment D: Session 26 との比較
    print('\n[Exp D] 4条件の比較')
    exp_d = run_exp_d_comparison(best, exp_b, exp_c, hist, seed=_S27_SEED)
    plot_exp_d_comparison(exp_d)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('=== Session 27 Summary ===')
    print()

    tau_r = best['tau_s'] / best['tau_i'] if best['tau_i'] > 0 else float('nan')
    print('[A] TM資源パラメータの収束')
    print(f'  tau_s={best["tau_s"]}  tau_i={best["tau_i"]}  tau_o={best["tau_o"]}  '
          f'depl={best["depletion_rate"]:.4f}')
    print(f'  tau_s/tau_i = {tau_r:.2f}  → 感覚器の回復が内部の{tau_r:.1f}倍遅い')

    print('\n[B] 睡眠様状態の観察')
    if exp_b['chunks']:
        mid   = len(exp_b['chunks']) // 2
        s1    = float(np.mean([c['sensory_activity']  for c in exp_b['chunks'][:mid]]))
        s2    = float(np.mean([c['sensory_activity']  for c in exp_b['chunks'][mid:]]))
        i1    = float(np.mean([c['internal_activity'] for c in exp_b['chunks'][:mid]]))
        i2    = float(np.mean([c['internal_activity'] for c in exp_b['chunks'][mid:]]))
        delta = s1 - s2
        improved = delta > 0.039  # S26 比較
        print(f'  感覚器活動: 前半={s1:.3f} → 後半={s2:.3f}  Δ={delta:+.3f}')
        print(f'  内部活動:   前半={i1:.3f} → 後半={i2:.3f}  Δ={(i1-i2):+.3f}')
        print(f'  Session 26 比較: Δ≈-0.039 → '
              f'{"✓ 改善 (Δ>{0.039:.3f})" if improved else "△ 未改善"}')
        crit_sleep = delta > 0.1
        print(f'  判定基準 (Δ>0.1): {"✓ 達成" if crit_sleep else "✗ 未達成"}')

    print('\n[C] 文脈分離')
    sig  = exp_c['p_value'] < 0.05
    both = exp_c['acc_A'] >= 0.6 and exp_c['acc_B'] >= 0.6
    print(f'  acc_A={exp_c["acc_A"]:.3f}  acc_B={exp_c["acc_B"]:.3f}  '
          f'cos_dist={exp_c["cosine_dist"]:.4f}')
    print(f'  p={exp_c["p_value"]:.4f}  95%CI=[{exp_c["ci_lo"]:.3f},{exp_c["ci_hi"]:.3f}]')
    print(f'  → {"✓ 文脈分離達成（両モード ≥ 0.6）" if both else "✗ 文脈分離未達成"}')
    print(f'     統計的有意性: {"✓ 有意 (p < 0.05)" if sig else "✗ 非有意"}')

    print('\n[D] 4条件比較')
    for i, (cond, delta, food, acc_b) in enumerate(
            zip(exp_d['conditions'], exp_d['delta'], exp_d['food'], exp_d['acc_B'])):
        meas = '★実測' if exp_d['is_measured'][i] else '  参照'
        print(f'  {meas} {cond.replace(chr(10)," ")}: Δ={delta:+.3f}  food={food:.2f}  acc_B={acc_b:.3f}')

    print('\nDone.')
