"""Session 21: 代謝コストによる飽和の自然な制御

問題（Session 20）:
  活動パターンが常に0.9〜1.0に飽和 → モードA/B区別できない → cos_dist≈0

設計:
  活動することに「代謝コスト」を設ける
  hp -= metabolic_rate * sum(activity)  毎ステップ
  metabolic_rateをゲノムとして進化
  → 飽和した個体は死にやすくなり、スパースな活動が有利になる
  → 生物的根拠: ニューロン発火にはATPが必要（スパースコーディングの進化的根拠）

Experiments:
  A  代謝コストの進化: 50世代でのゲノム（metabolic_rate/ep/ar）収束と活動量推移
  B  文脈分離の再計測: Session 20と同プロトコル（cosine_distance, acc_A/B, penalties）
  C  スパース性の確認: S20（飽和）vs S21（代謝コストあり）活動ノード数分布
  D  代謝コストと活動量の関係: metabolic_rateと平均活動ノード数の散布図

判定基準（Session 20と同じ）:
  acc_A > 0.6 かつ acc_B > 0.6
  cosine_dist > 0.1
  後半のペナルティ踏み込み < 前半
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
from session_20_penalty_context import (
    _s20_inp4,
    _PENALTY, _N_EP_EVAL,
)

_SEED      = 42
_N_AGENTS  = 10
_N_EP      = 5

_MR_INIT_MAX = 0.1    # initial range for metabolic_rate
_MR_MUT_STD  = 0.005  # mutation σ for metabolic_rate


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s21_make_genome(rng):
    G = _s10_build_graph(rng)
    return {
        'G':              G,
        'W':              _s10_get_W(G),
        'edge_add_prob':  float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio': float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate': float(rng.uniform(0.0, _MR_INIT_MAX)),
    }


def _s21_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    ep = float(np.clip(
        genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD), 0.0, _EP_INIT_MAX))
    ar = float(np.clip(
        genome['activity_ratio'] + rng.normal(0, _AR_INIT_MAX * 0.05), 0.0, _AR_INIT_MAX))
    mr = float(np.clip(
        genome['metabolic_rate'] + rng.normal(0, _MR_MUT_STD), 0.0, _MR_INIT_MAX))
    return {
        'G':              G_new,
        'W':              _s10_get_W(G_new),
        'edge_add_prob':  ep,
        'activity_ratio': ar,
        'metabolic_rate': mr,
    }


# ── Episode runner with metabolic cost ────────────────────────────────────────

def _s21_run_ep(G, W, edge_add_prob, activity_ratio, metabolic_rate, rng,
                mode=None, activity_noise=_ACTIVITY_NOISE,
                T_consolidation=_T_CONSOLIDATION,
                record_activity=False):
    """PenaltyContextGridWorld + 代謝コスト episode.

    毎ステップ: hp -= metabolic_rate * sum(activity)
    metabolic_rate=0.0 のとき Session 20 と同じ挙動。

    Returns (steps_survived, food_count, mode, penalty_count, records or None).
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos    = (0, 0) if mode == 'A' else (4, 4)
    penalty_pos = (4, 4) if mode == 'A' else (0, 0)
    food_avail  = True
    food_timer  = 0

    activity  = np.zeros(_N)
    row, col  = 2, 2
    hp        = 100.0
    steps     = 0
    food      = 0
    penalties = 0
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

        # 代謝コスト（毎ステップ）
        hp -= metabolic_rate * float(np.sum(activity))

        action = int(np.argmax(activity[4:9]))

        fr, fc = food_pos
        pr, pc = penalty_pos

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
    return steps, food, mode, penalties, records


# ── Experiment A: 代謝コストの進化 ────────────────────────────────────────────

def run_exp_a_genome_convergence(seed=_SEED):
    """50世代の進化でゲノムパラメータと活動量がどう変化するかを追跡。

    Returns (best_genome, history_dict).
    history_dict keys:
      gen_best_mr, gen_best_ep, gen_best_ar — best agent's genome per generation
      gen_best_fit — best fitness per generation
      gen_mean_activity — mean activity of best agent averaged over episodes per gen
    """
    rng = np.random.default_rng(seed + 21000)
    pop = [_s21_make_genome(rng) for _ in range(_N_AGENTS)]

    gen_best_mr, gen_best_ep, gen_best_ar = [], [], []
    gen_best_fit     = []
    gen_mean_activity = []

    for gen in range(_N_GEN):
        fitnesses     = []
        mean_acts     = []

        for g in pop:
            total     = 0
            ep_acts   = []
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, _, _, _, recs = _s21_run_ep(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'], ep_rng,
                    record_activity=True)
                total += s
                if recs:
                    ep_acts.append(float(np.mean([np.mean(a) for a in recs])))
            fitnesses.append(total / _N_EP)
            mean_acts.append(float(np.mean(ep_acts)) if ep_acts else 0.0)

        best_idx = int(np.argmax(fitnesses))
        best     = fitnesses[best_idx]
        bg       = pop[best_idx]

        gen_best_fit.append(best)
        gen_best_mr.append(bg['metabolic_rate'])
        gen_best_ep.append(bg['edge_add_prob'])
        gen_best_ar.append(bg['activity_ratio'])
        gen_mean_activity.append(mean_acts[best_idx])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s21_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={best:7.1f}  '
                  f'mr={bg["metabolic_rate"]:.4f}  '
                  f'ep={bg["edge_add_prob"]:.3f}  '
                  f'ar={bg["activity_ratio"]:.3f}  '
                  f'mean_act={gen_mean_activity[-1]:.3f}')

    return pop[0], {
        'gen_best_fit':      gen_best_fit,
        'gen_best_mr':       gen_best_mr,
        'gen_best_ep':       gen_best_ep,
        'gen_best_ar':       gen_best_ar,
        'gen_mean_activity': gen_mean_activity,
    }


# ── Experiment B: 文脈分離の再計測 ────────────────────────────────────────────

def run_exp_b_context_separation(best_genome, seed=_SEED, n_ep_per_mode=10):
    """Session 20 Exp B と同プロトコルで文脈分離を計測。

    Returns { mean_A, mean_B, ep_means_A, ep_means_B, cos_distances,
              acc_A, acc_B, mean_penalties }.
    """
    print('\n  [Exp B: context separation]')
    rng           = np.random.default_rng(seed + 21200)
    ep_means_A    = []
    ep_means_B    = []
    cos_distances = []
    all_food      = []
    all_modes     = []
    all_penalties = []

    for trial in range(_N_AGENTS):
        G  = best_genome['G'].copy()
        W  = _s10_get_W(G)
        ep = best_genome['edge_add_prob']
        ar = best_genome['activity_ratio']
        mr = best_genome['metabolic_rate']

        trial_means_A, trial_means_B = [], []

        for ei in range(n_ep_per_mode * 2):
            mode   = 'A' if ei % 2 == 0 else 'B'
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, f, _, p, recs = _s21_run_ep(
                G, W, ep, ar, mr, ep_rng, mode=mode, record_activity=True)
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

    mean_A = np.mean(ep_means_A, axis=0) if ep_means_A else np.zeros(5)
    mean_B = np.mean(ep_means_B, axis=0) if ep_means_B else np.zeros(5)

    modes_arr = np.array(all_modes)
    food_arr  = np.array(all_food)
    acc_a = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
    acc_b = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0

    print(f'    n_trials={len(cos_distances)}  '
          f'mean_cos_dist={np.mean(cos_distances) if cos_distances else 0.0:.4f}  '
          f'acc_A={acc_a:.3f}  acc_B={acc_b:.3f}  '
          f'mean_pens={np.mean(all_penalties):.2f}')
    print(f'    mean_A={np.round(mean_A, 3)}')
    print(f'    mean_B={np.round(mean_B, 3)}')

    return {
        'mean_A':        mean_A,
        'mean_B':        mean_B,
        'ep_means_A':    ep_means_A,
        'ep_means_B':    ep_means_B,
        'cos_distances': cos_distances,
        'acc_A':         acc_a,
        'acc_B':         acc_b,
        'mean_penalties': float(np.mean(all_penalties)),
    }


# ── Experiment C: スパース性の確認 ────────────────────────────────────────────

def run_exp_c_sparsity(best_genome, seed=_SEED, n_episodes=20, threshold=0.1):
    """活動ノード数分布を計測してS20（飽和）と比較。

    threshold: activity > threshold のノードを「活動中」とみなす閾値。
    Returns { active_counts_A, active_counts_B, mean_activity_A, mean_activity_B }.
    """
    print('\n  [Exp C: sparsity analysis]')
    rng             = np.random.default_rng(seed + 21300)
    active_counts_A = []
    active_counts_B = []
    mean_acts_A     = []
    mean_acts_B     = []

    G  = best_genome['G'].copy()
    W  = _s10_get_W(G)
    ep = best_genome['edge_add_prob']
    ar = best_genome['activity_ratio']
    mr = best_genome['metabolic_rate']

    for _ in range(n_episodes):
        for mode in ('A', 'B'):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, _, recs = _s21_run_ep(
                G, W, ep, ar, mr, ep_rng, mode=mode, record_activity=True)
            if recs:
                arr   = np.array(recs)
                counts = np.mean(arr > threshold, axis=0)
                n_active = float(np.sum(np.mean(arr, axis=0) > threshold))
                mean_a   = float(np.mean(arr))
                if mode == 'A':
                    active_counts_A.append(n_active)
                    mean_acts_A.append(mean_a)
                else:
                    active_counts_B.append(n_active)
                    mean_acts_B.append(mean_a)

    s20_saturated = 15.0  # S20: ~15/16 nodes saturated
    print(f'    S21 mean active nodes: '
          f'A={np.mean(active_counts_A):.2f}  B={np.mean(active_counts_B):.2f}  '
          f'(S20 saturated ≈ {s20_saturated:.0f}/16)')
    print(f'    mean activity: A={np.mean(mean_acts_A):.3f}  B={np.mean(mean_acts_B):.3f}')

    return {
        'active_counts_A': active_counts_A,
        'active_counts_B': active_counts_B,
        'mean_acts_A':     mean_acts_A,
        'mean_acts_B':     mean_acts_B,
        'threshold':       threshold,
    }


# ── Experiment D: 代謝コストと活動量の関係 ─────────────────────────────────────

def run_exp_d_metabolic_scatter(seed=_SEED, n_genome=30, n_ep=3, threshold=0.1):
    """metabolic_rateと平均活動ノード数の散布図データを生成。

    n_genomeの多様なゲノムを各n_ep評価し散布図を作る。
    Returns list of (metabolic_rate, mean_active_nodes).
    """
    print('\n  [Exp D: metabolic_rate vs active nodes scatter]')
    rng     = np.random.default_rng(seed + 21400)
    results = []

    # 多様なmetabolic_rateを均等にサンプリング
    mr_values = np.linspace(0.0, _MR_INIT_MAX, n_genome)

    for mr_val in mr_values:
        g  = _s21_make_genome(rng)
        g['metabolic_rate'] = float(mr_val)
        G  = g['G']
        W  = _s10_get_W(G)
        ep = g['edge_add_prob']
        ar = g['activity_ratio']

        act_list = []
        for _ in range(n_ep):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, _, recs = _s21_run_ep(
                G, W, ep, ar, mr_val, ep_rng, record_activity=True)
            if recs:
                arr = np.array(recs)
                n_active = float(np.sum(np.mean(arr, axis=0) > threshold))
                act_list.append(n_active)

        if act_list:
            results.append((float(mr_val), float(np.mean(act_list))))

    results.sort(key=lambda x: x[0])
    if results:
        mrs, acts = zip(*results)
        print(f'    mr range: [{min(mrs):.4f}, {max(mrs):.4f}]  '
              f'active_nodes range: [{min(acts):.1f}, {max(acts):.1f}]')
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_a_genome_convergence(
        history, fname='images/session_21/results_s21_genome_convergence.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Session 21 Exp A: Genome Convergence with Metabolic Cost\n'
        f'n_agents={_N_AGENTS}, n_ep={_N_EP}, n_gen={_N_GEN}',
        fontsize=12,
    )

    gens = np.arange(1, _N_GEN + 1)

    axes[0, 0].plot(gens, history['gen_best_fit'], color='#e6194b', linewidth=2)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Best Mean Survival Steps')
    axes[0, 0].set_title('Fitness (Best Mean Steps per Episode)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gens, history['gen_best_mr'], color='#f58231', linewidth=2)
    axes[0, 1].axhline(np.mean(history['gen_best_mr'][-10:]),
                       color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
                       label=f'Last-10 mean={np.mean(history["gen_best_mr"][-10:]):.4f}')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('metabolic_rate')
    axes[0, 1].set_title('metabolic_rate Evolution')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(gens, history['gen_best_ep'], color='steelblue',
                    linewidth=2, label='edge_add_prob')
    axes[1, 0].plot(gens, history['gen_best_ar'], color='#3cb44b',
                    linewidth=2, label='activity_ratio', linestyle='--')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('edge_add_prob & activity_ratio Evolution')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gens, history['gen_mean_activity'], color='#800000', linewidth=2)
    axes[1, 1].axhline(0.9, color='red', linestyle=':', linewidth=1.2, alpha=0.7,
                       label='S20 saturation level (~0.9)')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Mean Activity (best agent, all nodes)')
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title('Mean Activity Level (best agent per gen)')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_context_separation(
        data, fname='images/session_21/results_s21_context_separation.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Session 21 Exp B: Context Separation (Metabolic Cost Agent)\n'
        'Output nodes (n4-n8): Mode A (NW food / SE penalty) vs Mode B (SE food / NW penalty)',
        fontsize=11,
    )

    action_names = ['North\n(n4)', 'South\n(n5)', 'West\n(n6)',
                    'East\n(n7)', 'Eat\n(n8)']
    hm_data = np.array([data['mean_A'], data['mean_B']])
    vmax    = max(float(hm_data.max()), 0.01)
    im      = axes[0].imshow(hm_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(action_names, fontsize=9)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(
        ['Mode A\n(NW food\nSE penalty)', 'Mode B\n(SE food\nNW penalty)'], fontsize=9)
    axes[0].set_title('Mean Output Node Activity per Mode')
    for (r, c), val in np.ndenumerate(hm_data):
        axes[0].text(c, r, f'{val:.3f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=axes[0])

    cos_d   = data['cos_distances']
    rng_jit = np.random.default_rng(0)
    if cos_d:
        axes[1].boxplot([cos_d], patch_artist=True,
                        boxprops=dict(facecolor='#f58231', alpha=0.7))
        jitter = rng_jit.uniform(-0.15, 0.15, len(cos_d))
        axes[1].scatter(1 + jitter, cos_d, color='darkorange', alpha=0.7, s=35, zorder=3)
        axes[1].axhline(0.1, color='#e6194b', linestyle='--', linewidth=1.3, alpha=0.8,
                        label='Threshold (0.1)')
        axes[1].axhline(0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
        mean_cd = float(np.mean(cos_d))
        std_cd  = float(np.std(cos_d))
        axes[1].set_ylabel('Cosine Distance  (Mode A vs B output nodes)')
        axes[1].set_xticks([1])
        axes[1].set_xticklabels(['metabolic_evolved'])
        axes[1].set_title(
            f'Cosine Distance Distribution\n'
            f'n_trials={len(cos_d)}  mean={mean_cd:.4f}±{std_cd:.4f}  '
            f'acc_A={data["acc_A"]:.3f}  acc_B={data["acc_B"]:.3f}  '
            f'{"✓ > 0.1" if mean_cd > 0.1 else "✗ ≤ 0.1"}')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_sparsity(
        data, fname='images/session_21/results_s21_sparsity.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Session 21 Exp C: Sparsity Comparison\n'
        f'Active node threshold = activity > {data["threshold"]}  (N={_N} total nodes)',
        fontsize=11,
    )

    s20_saturated = 15.0  # Session 20 reference

    all_counts = data['active_counts_A'] + data['active_counts_B']
    ax = axes[0]
    ax.hist(data['active_counts_A'], bins=range(0, _N + 2), alpha=0.6,
            color='#e6194b', label='Mode A', density=False)
    ax.hist(data['active_counts_B'], bins=range(0, _N + 2), alpha=0.6,
            color='steelblue', label='Mode B', density=False)
    ax.axvline(s20_saturated, color='black', linestyle='--', linewidth=1.5,
               label=f'S20 saturated ≈ {s20_saturated:.0f}')
    ax.axvline(np.mean(all_counts), color='orange', linestyle='-', linewidth=2,
               label=f'S21 mean = {np.mean(all_counts):.1f}')
    ax.set_xlabel(f'Active Nodes (activity > {data["threshold"]})')
    ax.set_ylabel('Episode Count')
    ax.set_title('Active Node Count Distribution\nS21 (metabolic cost) vs S20 (saturated)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, _N + 0.5)

    ax = axes[1]
    means_all = data['mean_acts_A'] + data['mean_acts_B']
    ax.hist(data['mean_acts_A'], bins=20, alpha=0.6, color='#e6194b',
            label='Mode A', density=True)
    ax.hist(data['mean_acts_B'], bins=20, alpha=0.6, color='steelblue',
            label='Mode B', density=True)
    ax.axvline(0.9, color='black', linestyle='--', linewidth=1.5,
               label='S20 saturation level (~0.9)')
    ax.axvline(np.mean(means_all), color='orange', linestyle='-', linewidth=2,
               label=f'S21 mean = {np.mean(means_all):.3f}')
    ax.set_xlabel('Mean Activity (all nodes, all steps)')
    ax.set_ylabel('Density')
    ax.set_title('Mean Activity Distribution\nS21 vs S20 saturation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_d_metabolic_scatter(
        scatter_data, fname='images/session_21/results_s21_metabolic_scatter.png'):
    if not scatter_data:
        print('No data for Exp D scatter plot.')
        return

    mrs, acts = zip(*scatter_data)
    mrs  = np.array(mrs)
    acts = np.array(acts)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mrs, acts, color='#f58231', alpha=0.8, s=60, edgecolors='darkorange',
               linewidths=0.8, zorder=3)

    if len(mrs) >= 2:
        coeffs = np.polyfit(mrs, acts, 1)
        trend  = np.polyval(coeffs, mrs)
        ax.plot(mrs, trend, 'k--', linewidth=1.5,
                label=f'Trend (slope={coeffs[0]:.1f})')

    ax.axhline(15.0, color='red', linestyle=':', linewidth=1.2, alpha=0.7,
               label='S20 saturation ≈ 15/16 nodes')
    ax.set_xlabel('metabolic_rate (genome value)', fontsize=11)
    ax.set_ylabel(f'Mean Active Nodes (activity > 0.1)', fontsize=11)
    ax.set_title(
        'Session 21 Exp D: Metabolic Rate vs Active Node Count\n'
        'Negative slope = metabolic cost drives sparse coding',
        fontsize=11,
    )
    ax.set_xlim(-0.002, _MR_INIT_MAX + 0.005)
    ax.set_ylim(0, _N + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 21: 代謝コストによる飽和の自然な制御 ===')
    print(f'metabolic cost: hp -= metabolic_rate * sum(activity) per step')
    print(f'metabolic_rate evolves in [0.0, {_MR_INIT_MAX}]  mutation σ={_MR_MUT_STD}')

    print('\n[Exp A] 代謝コストの進化 (ゲノム収束と活動量推移)')
    best_genome, history = run_exp_a_genome_convergence(seed=_SEED)
    mr_final = best_genome['metabolic_rate']
    ep_final = best_genome['edge_add_prob']
    ar_final = best_genome['activity_ratio']
    print(f'\n  Best genome: mr={mr_final:.4f}  ep={ep_final:.3f}  ar={ar_final:.3f}  '
          f'edges={best_genome["G"].number_of_edges()}')
    print(f'  Last-10 mean: mr={np.mean(history["gen_best_mr"][-10:]):.4f}  '
          f'ep={np.mean(history["gen_best_ep"][-10:]):.3f}  '
          f'ar={np.mean(history["gen_best_ar"][-10:]):.3f}')
    plot_exp_a_genome_convergence(history)

    print(f'\n[Exp B] 文脈分離の再計測 (n_agents={_N_AGENTS}, n_ep_per_mode=10)')
    exp_b = run_exp_b_context_separation(best_genome, seed=_SEED, n_ep_per_mode=10)
    plot_exp_b_context_separation(exp_b)

    print(f'\n[Exp C] スパース性の確認 (n_episodes=20)')
    exp_c = run_exp_c_sparsity(best_genome, seed=_SEED, n_episodes=20, threshold=0.1)
    plot_exp_c_sparsity(exp_c)

    print(f'\n[Exp D] 代謝コストと活動量の関係 (n_genome=30, n_ep=3)')
    exp_d = run_exp_d_metabolic_scatter(seed=_SEED, n_genome=30, n_ep=3)
    plot_exp_d_metabolic_scatter(exp_d)

    # ── Judgment Criteria ─────────────────────────────────────────────────────
    print('\n  ── Judgment Criteria (Session 21) ─────────────────────────────')
    cos_mean  = float(np.mean(exp_b['cos_distances'])) if exp_b['cos_distances'] else 0.0
    acc_a_fin = exp_b['acc_A']
    acc_b_fin = exp_b['acc_B']

    all_acts_s21 = exp_c['active_counts_A'] + exp_c['active_counts_B']
    s20_ref      = 15.0
    sparse_ok    = float(np.mean(all_acts_s21)) < s20_ref

    print(f'  acc_A > 0.6      : {acc_a_fin:.3f}  {"✓" if acc_a_fin > 0.6 else "✗"}')
    print(f'  acc_B > 0.6      : {acc_b_fin:.3f}  {"✓" if acc_b_fin > 0.6 else "✗"}')
    print(f'  cos_dist > 0.1   : {cos_mean:.4f}  {"✓" if cos_mean > 0.1 else "✗"}')
    print(f'  active < S20({s20_ref:.0f}): {np.mean(all_acts_s21):.2f}  '
          f'{"✓" if sparse_ok else "✗"}  (代謝コストによるスパース化)')

    # S20との活動量比較
    all_acts_mean = float(np.mean(exp_c['mean_acts_A'] + exp_c['mean_acts_B']))
    print(f'\n  Activity level: S21={all_acts_mean:.3f}  S20≈0.9  '
          f'{"↓ reduced" if all_acts_mean < 0.9 else "≈ still saturated"}')

    n_pass = sum([
        acc_a_fin > 0.6,
        acc_b_fin > 0.6,
        cos_mean > 0.1,
        sparse_ok,
    ])
    print(f'\n  → {n_pass}/4 criteria met')
    if sparse_ok:
        print('  代謝コストが自然にスパース性を生み出した ✓')
    else:
        print('  代謝コストだけでは飽和を抑制できなかった — 追加実験が必要')

    print('\nDone.')
