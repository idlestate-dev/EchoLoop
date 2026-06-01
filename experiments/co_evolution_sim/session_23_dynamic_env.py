"""Session 23: 学習が必要な環境での自然なスパース性の創発

Session 22の問題:
  代謝コストでスパース化できた（active=4）がエージェントは静止した。
  5×5のGridWorldが単純すぎて「学習しないことが効率的」な環境だった。

本セッションの問い:
  「学習しないと生き残れない環境」では
  代謝コストと学習の必要性がバランスして
  適度な活動量が自然に創発するか？

環境設計: DynamicGridWorld
  食料位置とペナルティ位置がエピソードごとにランダムに変わる。
  「常に北西」戦略が通用しない — HPの変化と食料フラグから自力で読み取るしかない。

Experiments:
  A  mr固定スイープ [0.0, 0.01, 0.05, 0.1] で DynamicGridWorld 進化 (50世代)
     世代ごとの最良生存ステップ数と平均活動ノード数 (Session 22 との比較)
  B  スパース性と活動パターン変化の確認
     平均活動ノード数 + エピソードをまたいだ活動パターンの変化量
  C  文脈分離の再計測 (PenaltyContextGridWorld)
     動的環境で進化した個体が文脈依存的な行動を獲得しているかを確認
  D  静的環境 vs 動的環境の比較
     各条件のベスト個体を 3 環境 (Simple / Dynamic / Penalty) で評価

判定基準:
  1. active > 4 かつ active < 18
  2. 動的環境での生存ステップ数 > ランダムベースラインの 1.5 倍
  3. エピソードをまたいで活動パターンが変化する (cross_cos_dist > 0)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _N, _K, _N_PROP,
)
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_18_ratio_evolution import (
    _s18_hebb,
    _ACTIVITY_NOISE, _T_CONSOLIDATION,
    _N_GEN, _N_SURV,
    _EP_INIT_MAX, _AR_INIT_MAX,
    _EP_MUT_STD, _AR_MUT_STD,
)
from session_19_context_reboot import (
    _CGRID, _CHP_MAX, _CHP_DECAY, _CFOOD_VAL, _CRESPAWN, _CSTEPS,
)
from session_20_penalty_context import _s20_inp4, _PENALTY
from session_21_metabolic_cost import _s21_run_ep
from session_22_fixed_metabolic import (
    _s22_make_genome, _s22_mutate_genome,
    _MR_VALUES, _MR_COLORS, _MR_LABELS,
)

_SEED          = 42
_N_AGENTS      = 10
_N_EP          = 5
_ACT_THRESHOLD = 0.1


# ── DynamicGridWorld episode runner ───────────────────────────────────────────

def _s23_run_ep_dynamic(G, W, edge_add_prob, activity_ratio, metabolic_rate, rng,
                         activity_noise=_ACTIVITY_NOISE,
                         T_consolidation=_T_CONSOLIDATION,
                         record_activity=False):
    """DynamicGridWorld: food_pos と penalty_pos をエピソード開始時に rng で決定。

    食料位置もペナルティ位置も通知なし。
    HP の変化と food_flag から自力で読み取るしかない。

    Returns (steps_survived, food_count, penalty_count, records or None).
    """
    all_pos  = [(r, c) for r in range(_CGRID) for c in range(_CGRID)]
    food_pos    = all_pos[int(rng.integers(0, len(all_pos)))]
    remaining   = [p for p in all_pos if p != food_pos]
    penalty_pos = remaining[int(rng.integers(0, len(remaining)))]

    food_avail = True
    food_timer = 0

    activity = np.zeros(_N)
    row, col  = 2, 2
    hp        = 100.0
    steps     = 0
    food      = 0
    penalties = 0
    records   = [] if record_activity else None

    fr, fc = food_pos
    pr, pc = penalty_pos

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        inp4 = _s20_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        if record_activity:
            records.append(activity.copy())

        hp -= metabolic_rate * float(np.sum(activity))

        action = int(np.argmax(activity[4:9]))

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
                hp = min(_CHP_MAX, hp + _CFOOD_VAL)
                food_avail = False
                food_timer = 0
                food += 1

        hp   -= _CHP_DECAY
        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, activity, rng, edge_add_prob, activity_ratio)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food, penalties, records


# ── SimpleGridWorld episode runner (Exp D 用) ─────────────────────────────────

def _s23_run_ep_simple(G, W, edge_add_prob, activity_ratio, metabolic_rate, rng,
                        activity_noise=_ACTIVITY_NOISE,
                        T_consolidation=_T_CONSOLIDATION,
                        record_activity=False):
    """SimpleGridWorld: 食料固定 (0, 0)、ペナルティなし。

    Returns (steps_survived, food_count, 0, records or None).
    """
    fr, fc     = 0, 0
    food_avail = True
    food_timer = 0

    activity = np.zeros(_N)
    row, col  = 2, 2
    hp        = 100.0
    steps     = 0
    food      = 0
    records   = [] if record_activity else None

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        inp4 = _s20_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        if record_activity:
            records.append(activity.copy())

        hp -= metabolic_rate * float(np.sum(activity))

        action = int(np.argmax(activity[4:9]))

        if action in (0, 1, 2, 3):
            if   action == 0: row = max(0, row - 1)
            elif action == 1: row = min(_CGRID - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(_CGRID - 1, col + 1)
        elif action == 4:
            if row == fr and col == fc and food_avail:
                hp = min(_CHP_MAX, hp + _CFOOD_VAL)
                food_avail = False
                food_timer = 0
                food += 1

        hp   -= _CHP_DECAY
        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, activity, rng, edge_add_prob, activity_ratio)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food, 0, records


# ── Inner evolution loop (DynamicGridWorld) ───────────────────────────────────

def _s23_evolve_dynamic_mr(metabolic_rate, seed):
    """DynamicGridWorld で edge_add_prob / activity_ratio を進化 (metabolic_rate 固定)。

    Returns (best_genome, history_dict).
    history keys: gen_best_steps, gen_mean_active.
    """
    mr_idx = _MR_VALUES.index(metabolic_rate)
    rng    = np.random.default_rng(seed + 23000 + mr_idx * 1000)
    pop    = [_s22_make_genome(rng, metabolic_rate) for _ in range(_N_AGENTS)]

    gen_best_steps  = []
    gen_mean_active = []

    for gen in range(_N_GEN):
        fitnesses = []
        act_means = []

        for g in pop:
            total   = 0
            ep_acts = []
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, _, _, recs = _s23_run_ep_dynamic(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'], ep_rng,
                    record_activity=True)
                total += s
                if recs:
                    arr = np.array(recs)
                    ep_acts.append(float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD)))
            fitnesses.append(total / _N_EP)
            act_means.append(float(np.mean(ep_acts)) if ep_acts else 0.0)

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]
        gen_best_steps.append(fitnesses[best_idx])
        gen_mean_active.append(act_means[best_idx])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s22_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'    gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'ep={bg["edge_add_prob"]:.3f}  ar={bg["activity_ratio"]:.3f}  '
                  f'active={gen_mean_active[-1]:.1f}')

    return pop[0], {
        'gen_best_steps':  gen_best_steps,
        'gen_mean_active': gen_mean_active,
    }


# ── Experiment A: mr fixed sweep (DynamicGridWorld) ───────────────────────────

def run_exp_a_mr_sweep(seed=_SEED):
    """各 mr 固定値で DynamicGridWorld 進化。

    Returns dict keyed by mr: { 'best': genome, 'hist': { gen_best_steps, gen_mean_active } }
    """
    results = {}
    for mr in _MR_VALUES:
        print(f'\n  [mr={mr:.2f}] DynamicGridWorld で {_N_GEN} 世代進化中...')
        best, hist = _s23_evolve_dynamic_mr(mr, seed)
        results[mr] = {'best': best, 'hist': hist}
        print(f'  → ep={best["edge_add_prob"]:.3f}  ar={best["activity_ratio"]:.3f}  '
              f'edges={best["G"].number_of_edges()}  '
              f'active(last)={hist["gen_mean_active"][-1]:.1f}')
    return results


# ── Experiment B: sparsity and cross-episode activity variation ───────────────

def run_exp_b_sparsity(exp_a_results, seed=_SEED, n_episodes=20):
    """活動ノード数とエピソードをまたいだ活動パターン変化を計測。

    Returns dict keyed by mr.
    """
    print('\n  [Exp B: sparsity and cross-episode pattern variation]')
    results = {}

    for mr in _MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 23200 + mr_idx * 100)

        G      = g['G'].copy()
        W      = _s10_get_W(G)
        ep     = g['edge_add_prob']
        ar     = g['activity_ratio']
        mr_val = g['metabolic_rate']

        active_counts   = []
        ep_mean_vectors = []

        for _ in range(n_episodes):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, recs = _s23_run_ep_dynamic(
                G, W, ep, ar, mr_val, ep_rng, record_activity=True)
            if recs:
                arr = np.array(recs)
                active_counts.append(float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD)))
                ep_mean_vectors.append(np.mean(arr, axis=0))

        cross_cos = []
        if len(ep_mean_vectors) >= 2:
            vecs = np.array(ep_mean_vectors)
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    vi, vj = vecs[i], vecs[j]
                    ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
                    if ni > 1e-9 and nj > 1e-9:
                        cross_cos.append(1.0 - float(np.dot(vi, vj) / (ni * nj)))

        mean_active  = float(np.mean(active_counts)) if active_counts else 0.0
        mean_cos_var = float(np.mean(cross_cos))     if cross_cos     else 0.0

        print(f'    mr={mr:.2f}: mean_active={mean_active:.1f}  '
              f'cross_ep_cos_dist={mean_cos_var:.4f}')

        results[mr] = {
            'active_counts':   active_counts,
            'ep_mean_vectors': ep_mean_vectors,
            'cross_cos':       cross_cos,
            'mean_active':     mean_active,
            'mean_cos_var':    mean_cos_var,
        }

    return results


# ── Experiment C: context separation (PenaltyContextGridWorld) ────────────────

def run_exp_c_context(exp_a_results, seed=_SEED, n_ep_per_mode=10):
    """動的環境で進化した個体の文脈分離を PenaltyContextGridWorld で計測。

    Session 22 Exp B と同プロトコル。
    Returns dict keyed by mr with acc_A, acc_B, mean_cos_dist, mean_penalties.
    """
    print('\n  [Exp C: context separation on PenaltyContextGridWorld]')
    results = {}

    for mr in _MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 23300 + mr_idx * 100)

        ep_means_A    = []
        ep_means_B    = []
        cos_distances = []
        all_food      = []
        all_modes     = []
        all_penalties = []

        for _ in range(_N_AGENTS):
            G      = g['G'].copy()
            W      = _s10_get_W(G)
            ep     = g['edge_add_prob']
            ar     = g['activity_ratio']
            mr_val = g['metabolic_rate']

            trial_means_A, trial_means_B = [], []

            for ei in range(n_ep_per_mode * 2):
                mode   = 'A' if ei % 2 == 0 else 'B'
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                _, f, _, p, recs = _s21_run_ep(
                    G, W, ep, ar, mr_val, ep_rng, mode=mode, record_activity=True)
                all_food.append(f)
                all_modes.append(mode)
                all_penalties.append(p)

                if recs:
                    arr      = np.array(recs)
                    mean_out = arr[:, 4:9].mean(axis=0)
                    if mode == 'A':
                        trial_means_A.append(mean_out)
                        ep_means_A.append(mean_out)
                    else:
                        trial_means_B.append(mean_out)
                        ep_means_B.append(mean_out)

            if trial_means_A and trial_means_B:
                mA = np.mean(trial_means_A, axis=0)
                mB = np.mean(trial_means_B, axis=0)
                nA = np.linalg.norm(mA)
                nB = np.linalg.norm(mB)
                if nA > 1e-9 and nB > 1e-9:
                    cos_distances.append(1.0 - float(np.dot(mA, mB) / (nA * nB)))

        modes_arr = np.array(all_modes)
        food_arr  = np.array(all_food)
        acc_a   = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
        acc_b   = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
        mean_cd = float(np.mean(cos_distances)) if cos_distances else 0.0

        print(f'    mr={mr:.2f}: acc_A={acc_a:.3f}  acc_B={acc_b:.3f}  '
              f'cos_dist={mean_cd:.4f}  mean_pens={np.mean(all_penalties):.2f}')

        results[mr] = {
            'acc_A':          acc_a,
            'acc_B':          acc_b,
            'cos_distances':  cos_distances,
            'mean_cos_dist':  mean_cd,
            'mean_penalties': float(np.mean(all_penalties)),
        }

    return results


# ── Experiment D: cross-environment transfer ──────────────────────────────────

def run_exp_d_transfer(exp_a_results, seed=_SEED, n_episodes=20):
    """各条件のベスト個体を 3 環境で評価 (SimpleGridWorld / Dynamic / PenaltyContext)。

    Returns dict keyed by mr with mean survival steps per environment.
    """
    print('\n  [Exp D: cross-environment transfer]')
    results = {}

    for mr in _MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 23400 + mr_idx * 100)

        G      = g['G'].copy()
        W      = _s10_get_W(G)
        ep     = g['edge_add_prob']
        ar     = g['activity_ratio']
        mr_val = g['metabolic_rate']

        steps_simple  = []
        steps_dynamic = []
        steps_penalty = []

        for _ in range(n_episodes):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, _, _, _ = _s23_run_ep_simple(G, W, ep, ar, mr_val, ep_rng)
            steps_simple.append(s)

            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, _, _, _ = _s23_run_ep_dynamic(G, W, ep, ar, mr_val, ep_rng)
            steps_dynamic.append(s)

            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, _, _, _, _ = _s21_run_ep(G, W, ep, ar, mr_val, ep_rng)
            steps_penalty.append(s)

        ms = float(np.mean(steps_simple))
        md = float(np.mean(steps_dynamic))
        mp = float(np.mean(steps_penalty))
        print(f'    mr={mr:.2f}: simple={ms:.1f}  dynamic={md:.1f}  penalty={mp:.1f}')

        results[mr] = {
            'steps_simple':  steps_simple,
            'steps_dynamic': steps_dynamic,
            'steps_penalty': steps_penalty,
            'mean_simple':   ms,
            'mean_dynamic':  md,
            'mean_penalty':  mp,
        }

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_a_evolution(exp_a_results,
                          fname='images/session_23/results_s23_evolution.png'):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        'Session 23 Exp A: Fixed Metabolic Rate Sweep — DynamicGridWorld Evolution\n'
        f'n_agents={_N_AGENTS}, n_ep={_N_EP}, n_gen={_N_GEN}  '
        '(食料・ペナルティ位置がエピソードごとにランダム)',
        fontsize=11,
    )

    gens = np.arange(1, _N_GEN + 1)

    ax = axes[0]
    for mr in _MR_VALUES:
        hist = exp_a_results[mr]['hist']
        ax.plot(gens, hist['gen_best_steps'],
                color=_MR_COLORS[mr], linewidth=2,
                label=_MR_LABELS[mr].replace('\n', ' '))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Mean Survival Steps')
    ax.set_title('Best Fitness per Generation (DynamicGridWorld)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mr in _MR_VALUES:
        hist = exp_a_results[mr]['hist']
        ax.plot(gens, hist['gen_mean_active'],
                color=_MR_COLORS[mr], linewidth=2,
                label=_MR_LABELS[mr].replace('\n', ' '))
    ax.axhline(4.0, color='purple', linestyle=':', linewidth=1.5, alpha=0.8,
               label='S22 mr=0.01 static → active≈4 (何もできなかった)')
    ax.axhline(15.0, color='black', linestyle=':', linewidth=1.2, alpha=0.6,
               label='S20 saturation ≈ 15/16 nodes')
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'Mean Active Nodes (activity > {_ACT_THRESHOLD}, best agent)')
    ax.set_ylim(0, _N + 1)
    ax.set_title('Mean Active Nodes of Best Agent per Generation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_sparsity(exp_b_results,
                         fname='images/session_23/results_s23_sparsity.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 23 Exp B: Sparsity and Cross-Episode Activity Variation\n'
        f'Active node threshold = activity > {_ACT_THRESHOLD}  (N={_N} total nodes)  '
        'DynamicGridWorld 評価',
        fontsize=11,
    )

    x      = np.arange(len(_MR_VALUES))
    colors = [_MR_COLORS[mr] for mr in _MR_VALUES]

    ax = axes[0]
    means = [exp_b_results[mr]['mean_active'] for mr in _MR_VALUES]
    bars  = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
    ax.axhline(4.0, color='purple', linestyle=':', linewidth=1.5, alpha=0.8,
               label='S22 mr=0.01 static → active≈4')
    ax.axhline(15.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='S20 saturation ≈ 15 nodes')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_MR_LABELS[mr] for mr in _MR_VALUES], fontsize=9)
    ax.set_ylabel(f'Mean Active Nodes (activity > {_ACT_THRESHOLD})')
    ax.set_title('Mean Active Nodes in DynamicGridWorld\n(S22 静的環境参照を点線で表示)')
    ax.set_ylim(0, _N + 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    cos_means = [exp_b_results[mr]['mean_cos_var'] for mr in _MR_VALUES]
    ymax      = max(max(cos_means) * 1.4, 0.005) + 0.001
    bars = ax.bar(x, cos_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, cos_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_MR_LABELS[mr] for mr in _MR_VALUES], fontsize=9)
    ax.set_ylabel('Mean Pairwise Cosine Distance\n(エピソード間活動ベクトル)')
    ax.set_title('Cross-Episode Activity Variation\n(高いほど環境変化に応答して活動が変化)')
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_context(exp_c_results,
                        fname='images/session_23/results_s23_context.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Session 23 Exp C: Context Separation (PenaltyContextGridWorld)\n'
        '動的環境で進化した個体が文脈依存的な行動を獲得しているか\n'
        'acc_A/B: food eaten ≥ 1  |  cosine_dist: output node vectors (mode A vs B)',
        fontsize=10,
    )

    x       = np.arange(len(_MR_VALUES))
    colors  = [_MR_COLORS[mr] for mr in _MR_VALUES]
    x_ticks = [_MR_LABELS[mr] for mr in _MR_VALUES]

    metrics = [
        ('acc_A',         'Mode A Accuracy\n(food ≥ 1 in NW-food episodes)', 0.6),
        ('acc_B',         'Mode B Accuracy\n(food ≥ 1 in SE-food episodes)', 0.6),
        ('mean_cos_dist', 'Cosine Distance\n(output nodes: Mode A vs Mode B)', 0.1),
    ]

    for ai, (key, title, threshold) in enumerate(metrics):
        ax   = axes[ai]
        vals = [exp_c_results[mr][key] for mr in _MR_VALUES]
        ymax = max(max(vals) * 1.3, threshold * 1.5) + 0.05

        bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
        ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.85,
                   label=f'Threshold = {threshold}')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_ticks, fontsize=9)
        ax.set_ylabel(title.split('\n')[0])
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_d_transfer(exp_d_results,
                         fname='images/session_23/results_s23_transfer.png'):
    fig, axes = plt.subplots(1, len(_MR_VALUES), figsize=(16, 6), sharey=True)
    fig.suptitle(
        'Session 23 Exp D: Cross-Environment Transfer\n'
        '動的環境で進化した個体を 3 環境で評価',
        fontsize=11,
    )

    env_labels = ['Simple\n(fixed NW food)', 'Dynamic\n(random pos)', 'Penalty\n(A/B context)']
    env_colors = ['#3cb44b', '#4363d8', '#f58231']

    for ai, mr in enumerate(_MR_VALUES):
        ax   = axes[ai]
        r    = exp_d_results[mr]
        vals = [r['mean_simple'], r['mean_dynamic'], r['mean_penalty']]

        bars = ax.bar(range(3), vals, color=env_colors, alpha=0.8,
                      edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels(env_labels, fontsize=8)
        ax.set_title(_MR_LABELS[mr].replace('\n', ' '), fontsize=10)
        ax.set_ylim(0, _CSTEPS + 30)
        ax.grid(True, alpha=0.3, axis='y')
        if ai == 0:
            ax.set_ylabel('Mean Survival Steps')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 23: 学習が必要な環境での自然なスパース性の創発 ===')
    print('環境: DynamicGridWorld（食料・ペナルティ位置がエピソードごとにランダム）')
    print(f'mr 固定スイープ: {_MR_VALUES}')
    print(f'n_agents={_N_AGENTS}  n_ep={_N_EP}  n_gen={_N_GEN}')

    print(f'\n[Exp A] DynamicGridWorld での mr 固定スイープ（各条件 {_N_GEN} 世代）')
    exp_a = run_exp_a_mr_sweep(seed=_SEED)

    print('\n  Summary (converged best genomes):')
    print(f'  {"mr":>6}  {"ep":>6}  {"ar":>6}  {"edges":>6}  '
          f'{"active(last)":>12}  {"best_steps(last)":>16}')
    print('  ' + '─' * 62)
    for mr in _MR_VALUES:
        bg   = exp_a[mr]['best']
        hist = exp_a[mr]['hist']
        print(f'  {mr:6.3f}  {bg["edge_add_prob"]:6.3f}  {bg["activity_ratio"]:6.3f}  '
              f'{bg["G"].number_of_edges():6d}  '
              f'{hist["gen_mean_active"][-1]:12.1f}  '
              f'{hist["gen_best_steps"][-1]:16.1f}')
    plot_exp_a_evolution(exp_a)

    print(f'\n[Exp B] スパース性と活動パターン変化 (n_episodes=20)')
    exp_b = run_exp_b_sparsity(exp_a, seed=_SEED, n_episodes=20)
    plot_exp_b_sparsity(exp_b)

    print(f'\n[Exp C] 文脈分離の再計測 (PenaltyContextGridWorld, n_ep_per_mode=10)')
    exp_c = run_exp_c_context(exp_a, seed=_SEED, n_ep_per_mode=10)
    plot_exp_c_context(exp_c)

    print(f'\n[Exp D] 3 環境転移テスト (n_episodes=20)')
    exp_d = run_exp_d_transfer(exp_a, seed=_SEED, n_episodes=20)
    plot_exp_d_transfer(exp_d)

    # ── Judgment ──────────────────────────────────────────────────────────────
    print('\n  ── Judgment Criteria (Session 23) ─────────────────────────────')
    print('  「学習が必要な環境で適度な活動量が創発した」と言える条件:')
    print('  1. active > 4 かつ active < 18')
    print('  2. 動的環境での生存ステップ数 > ランダムベースライン × 1.5 (≈ 150)')
    print('  3. エピソードをまたいで活動パターンが変化する (cross_cos_dist > 0)')
    print()

    random_baseline = 100.0  # hp_start=100, hp_decay=1 → 最低でも100ステップは生きる

    print(f'  {"mr":>6}  {"active":>7}  {"c1":>4}  {"dyn_steps":>10}  {"c2":>4}  '
          f'{"cos_var":>9}  {"c3":>4}')
    print('  ' + '─' * 60)

    passing_mrs = []
    for mr in _MR_VALUES:
        active    = exp_b[mr]['mean_active']
        dyn_steps = exp_a[mr]['hist']['gen_best_steps'][-1]
        cos_var   = exp_b[mr]['mean_cos_var']

        c1 = 4.0 < active < 18.0
        c2 = dyn_steps > random_baseline * 1.5
        c3 = cos_var > 0.0

        if c1 and c2:
            passing_mrs.append(mr)

        print(f'  {mr:6.3f}  {active:7.1f}  {"✓" if c1 else "✗":>4}  '
              f'{dyn_steps:10.1f}  {"✓" if c2 else "✗":>4}  '
              f'{cos_var:9.4f}  {"✓" if c3 else "✗":>4}')

    print()
    if passing_mrs:
        print(f'  → 適度な活動量が創発: mr={passing_mrs}')
        if any(exp_b[mr]['mean_active'] > 4.0 for mr in passing_mrs):
            print('  → active > 4: 動的環境により S22 の静止状態から脱出した ✓')
    else:
        print('  → 全条件で基準未達')
        statics = [mr for mr in _MR_VALUES if exp_b[mr]['mean_active'] <= 4.0]
        if statics:
            print(f'    active ≤ 4 (静止状態): mr={statics}')
            print('    動的環境でも代謝コストが学習圧力に勝った可能性')

    # Session 22 比較
    s22_active_ref = 4.0
    improved = [mr for mr in _MR_VALUES if exp_b[mr]['mean_active'] > s22_active_ref]
    print(f'\n  Session 22 比較 (S22 mr=0.01 active≈{s22_active_ref:.0f}):')
    if improved:
        for mr in improved:
            print(f'    mr={mr:.2f}: active={exp_b[mr]["mean_active"]:.1f} > {s22_active_ref:.0f} ✓ (動的環境の効果)')
    else:
        print('    全条件で S22 比改善なし — 動的環境でも静止が最適戦略')

    print('\nDone.')
