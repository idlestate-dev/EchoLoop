"""Session 22: 最小限の神の手による飽和制御

Session 21の知見:
  代謝コスト（mr）を進化に委ねるとmr≈0に収束し飽和が解決しない。
  mr=0.1固定では active_nodes≈3.7（スパース）だった。

本セッションの問い:
  metabolic_rateを固定（神の手）したとき
  文脈依存的な行動が創発するか？
  どの程度のmrが必要か？

Experiments:
  A  mr固定値スイープ [0.0, 0.01, 0.05, 0.1] で進化 (50世代)
     世代ごとの最良生存ステップ数と平均活動ノード数を追跡
  B  各条件のベスト個体の文脈分離計測
     acc_A, acc_B, cosine_distance, mean_penalties
  C  活動量とスパース性の計測
     mr値と活動量の関係 + モードA vs Bの活動パターン差異

判定基準:
  acc_A > 0.6 かつ acc_B > 0.6
  cosine_dist > 0.1
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_mutate,
    _N, _K, _N_PROP,
)
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
from session_21_metabolic_cost import _s21_run_ep

_SEED          = 42
_N_AGENTS      = 10
_N_EP          = 5
_ACT_THRESHOLD = 0.1

_MR_VALUES = [0.0, 0.01, 0.05, 0.1]
_MR_COLORS = {0.0: 'gray', 0.01: 'steelblue', 0.05: '#f58231', 0.1: '#e6194b'}
_MR_LABELS = {
    0.0:  'mr=0.00\n(no cost)',
    0.01: 'mr=0.01\n(light)',
    0.05: 'mr=0.05\n(medium)',
    0.1:  'mr=0.10\n(strong)',
}


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s22_make_genome(rng, metabolic_rate):
    G = _s10_build_graph(rng)
    return {
        'G':              G,
        'W':              _s10_get_W(G),
        'edge_add_prob':  float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio': float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate': float(metabolic_rate),
    }


def _s22_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    ep = float(np.clip(
        genome['edge_add_prob'] + rng.normal(0, _EP_MUT_STD), 0.0, _EP_INIT_MAX))
    ar = float(np.clip(
        genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD), 0.0, _AR_INIT_MAX))
    return {
        'G':              G_new,
        'W':              _s10_get_W(G_new),
        'edge_add_prob':  ep,
        'activity_ratio': ar,
        'metabolic_rate': genome['metabolic_rate'],  # fixed — not mutated
    }


# ── Inner evolution loop ───────────────────────────────────────────────────────

def _s22_evolve_fixed_mr(metabolic_rate, seed):
    """Evolve edge_add_prob / activity_ratio with fixed metabolic_rate.

    Returns (best_genome, history_dict).
    history keys: gen_best_steps, gen_mean_active.
    """
    mr_idx = _MR_VALUES.index(metabolic_rate)
    rng    = np.random.default_rng(seed + 22000 + mr_idx * 1000)
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
                s, _, _, _, recs = _s21_run_ep(
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


# ── Experiment A: mr fixed sweep ───────────────────────────────────────────────

def run_exp_a_mr_sweep(seed=_SEED):
    """Evolve under each fixed mr value; return best genomes and training histories.

    Returns dict keyed by mr value:
      { mr: { 'best': genome, 'hist': { gen_best_steps, gen_mean_active } } }
    """
    results = {}
    for mr in _MR_VALUES:
        print(f'\n  [mr={mr:.2f}] Evolving for {_N_GEN} generations...')
        best, hist = _s22_evolve_fixed_mr(mr, seed)
        results[mr] = {'best': best, 'hist': hist}
        print(f'  → ep={best["edge_add_prob"]:.3f}  ar={best["activity_ratio"]:.3f}  '
              f'edges={best["G"].number_of_edges()}  '
              f'active(last)={hist["gen_mean_active"][-1]:.1f}')
    return results


# ── Experiment B: context separation ──────────────────────────────────────────

def run_exp_b_context(exp_a_results, seed=_SEED, n_ep_per_mode=10):
    """Measure context separation for each condition's best genome.

    n_agents trials × n_ep_per_mode episodes per mode per trial.
    Returns dict keyed by mr value with acc_A, acc_B, cos_distances, mean_penalties.
    """
    print('\n  [Exp B: context separation per mr condition]')
    results = {}

    for mr in _MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 22200 + mr_idx * 100)

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
        acc_a  = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
        acc_b  = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
        mean_cd = float(np.mean(cos_distances)) if cos_distances else 0.0

        print(f'    mr={mr:.2f}: acc_A={acc_a:.3f}  acc_B={acc_b:.3f}  '
              f'cos_dist={mean_cd:.4f}  mean_pens={np.mean(all_penalties):.2f}')

        results[mr] = {
            'acc_A':          acc_a,
            'acc_B':          acc_b,
            'cos_distances':  cos_distances,
            'mean_cos_dist':  mean_cd,
            'mean_penalties': float(np.mean(all_penalties)),
            'mean_A':         np.mean(ep_means_A, axis=0) if ep_means_A else np.zeros(5),
            'mean_B':         np.mean(ep_means_B, axis=0) if ep_means_B else np.zeros(5),
        }

    return results


# ── Experiment C: activity and sparsity ───────────────────────────────────────

def run_exp_c_sparsity(exp_a_results, seed=_SEED, n_episodes=20):
    """Measure mean active nodes and mode A/B activity patterns.

    n_episodes of mode A and n_episodes of mode B per condition.
    Returns dict keyed by mr value.
    """
    print('\n  [Exp C: sparsity and mode activity patterns]')
    results = {}

    for mr in _MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 22300 + mr_idx * 100)

        G      = g['G'].copy()
        W      = _s10_get_W(G)
        ep     = g['edge_add_prob']
        ar     = g['activity_ratio']
        mr_val = g['metabolic_rate']

        active_A,    active_B    = [], []
        mean_acts_A, mean_acts_B = [], []
        node_means_A, node_means_B = [], []

        for _ in range(n_episodes):
            for mode in ('A', 'B'):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                _, _, _, _, recs = _s21_run_ep(
                    G, W, ep, ar, mr_val, ep_rng, mode=mode, record_activity=True)
                if recs:
                    arr      = np.array(recs)
                    n_active = float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD))
                    mean_act = float(np.mean(arr))
                    nm       = np.mean(arr, axis=0)
                    if mode == 'A':
                        active_A.append(n_active)
                        mean_acts_A.append(mean_act)
                        node_means_A.append(nm)
                    else:
                        active_B.append(n_active)
                        mean_acts_B.append(mean_act)
                        node_means_B.append(nm)

        mn_A   = np.mean(node_means_A, axis=0) if node_means_A else np.zeros(_N)
        mn_B   = np.mean(node_means_B, axis=0) if node_means_B else np.zeros(_N)
        diff_AB = float(np.mean(np.abs(mn_A - mn_B)))

        print(f'    mr={mr:.2f}: active A={np.mean(active_A):.1f}  '
              f'B={np.mean(active_B):.1f}  '
              f'mean_act={np.mean(mean_acts_A + mean_acts_B):.3f}  '
              f'|A-B|={diff_AB:.4f}')

        results[mr] = {
            'active_A':    active_A,
            'active_B':    active_B,
            'mean_acts_A': mean_acts_A,
            'mean_acts_B': mean_acts_B,
            'node_mean_A': mn_A,
            'node_mean_B': mn_B,
            'diff_AB':     diff_AB,
        }

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_a_evolution(exp_a_results,
                          fname='images/session_22/results_s22_evolution.png'):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        'Session 22 Exp A: Fixed Metabolic Rate Sweep — Evolution Progress\n'
        f'n_agents={_N_AGENTS}, n_ep={_N_EP}, n_gen={_N_GEN}',
        fontsize=12,
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
    ax.set_title('Best Fitness per Generation (PenaltyContextGridWorld)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mr in _MR_VALUES:
        hist = exp_a_results[mr]['hist']
        ax.plot(gens, hist['gen_mean_active'],
                color=_MR_COLORS[mr], linewidth=2,
                label=_MR_LABELS[mr].replace('\n', ' '))
    ax.axhline(15.0, color='black', linestyle=':', linewidth=1.2, alpha=0.7,
               label='S20 saturation ≈ 15/16 nodes')
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'Mean Active Nodes (activity > {_ACT_THRESHOLD}, best agent)')
    ax.set_ylim(0, _N + 1)
    ax.set_title('Mean Active Nodes of Best Agent per Generation')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_context(exp_b_results,
                        fname='images/session_22/results_s22_context.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Session 22 Exp B: Context Separation per Fixed metabolic_rate\n'
        'acc_A/B: food eaten ≥ 1 in mode  |  cosine_dist: output node vectors (mode A vs B)',
        fontsize=11,
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
        vals = [exp_b_results[mr][key] for mr in _MR_VALUES]
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


def plot_exp_c_sparsity(exp_c_results,
                         fname='images/session_22/results_s22_sparsity.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 22 Exp C: Activity and Sparsity per Fixed metabolic_rate\n'
        f'Active node threshold = activity > {_ACT_THRESHOLD}  (N={_N} total nodes)',
        fontsize=11,
    )

    mrs    = np.array(_MR_VALUES)
    mean_A = [np.mean(exp_c_results[mr]['active_A']) for mr in _MR_VALUES]
    mean_B = [np.mean(exp_c_results[mr]['active_B']) for mr in _MR_VALUES]

    ax = axes[0]
    ax.plot(mrs, mean_A, 'o-', color='#e6194b', linewidth=2, markersize=9,
            label='Mode A (NW food)')
    ax.plot(mrs, mean_B, 's--', color='steelblue', linewidth=2, markersize=9,
            label='Mode B (SE food)')
    ax.axhline(15.0, color='black', linestyle=':', linewidth=1.2, alpha=0.7,
               label='S20 saturation ≈ 15/16 nodes')
    for mr, mA, mB in zip(_MR_VALUES, mean_A, mean_B):
        ax.text(mr, mA + 0.4, f'{mA:.1f}', ha='center', fontsize=9, color='#e6194b')
        ax.text(mr, mB - 0.8, f'{mB:.1f}', ha='center', fontsize=9, color='steelblue')
    ax.set_xlabel('Fixed metabolic_rate', fontsize=11)
    ax.set_ylabel(f'Mean Active Nodes (activity > {_ACT_THRESHOLD})', fontsize=11)
    ax.set_title('mr Value vs Active Node Count\n(Mode A = NW food, Mode B = SE food)')
    ax.set_ylim(0, _N + 1)
    ax.set_xlim(-0.005, 0.11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x      = np.arange(len(_MR_VALUES))
    colors = [_MR_COLORS[mr] for mr in _MR_VALUES]
    diffs  = [exp_c_results[mr]['diff_AB'] for mr in _MR_VALUES]
    ymax   = max(diffs) * 1.4 + 0.002

    bars = ax.bar(x, diffs, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_MR_LABELS[mr] for mr in _MR_VALUES], fontsize=9)
    ax.set_ylabel('Mean |Activity(A) − Activity(B)| per Node')
    ax.set_title('Mode A vs Mode B\nMean Node-wise Activity Difference')
    ax.set_ylim(0, ymax)
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

    print('=== Session 22: 最小限の神の手による飽和制御 ===')
    print(f'mr固定スイープ: {_MR_VALUES}')
    print(f'進化パラメータ: edge_add_prob + activity_ratio のみ（mr固定）')
    print(f'n_agents={_N_AGENTS}  n_ep={_N_EP}  n_gen={_N_GEN}')

    print(f'\n[Exp A] mr固定値スイープ（各条件 {_N_GEN}世代進化）')
    exp_a = run_exp_a_mr_sweep(seed=_SEED)

    print('\n  Summary (converged best genomes):')
    print(f'  {"mr":>6}  {"ep":>6}  {"ar":>6}  {"edges":>6}  {"active(last)":>12}  {"best_steps(last)":>16}')
    print('  ' + '─' * 60)
    for mr in _MR_VALUES:
        bg   = exp_a[mr]['best']
        hist = exp_a[mr]['hist']
        print(f'  {mr:6.3f}  {bg["edge_add_prob"]:6.3f}  {bg["activity_ratio"]:6.3f}  '
              f'{bg["G"].number_of_edges():6d}  '
              f'{hist["gen_mean_active"][-1]:12.1f}  '
              f'{hist["gen_best_steps"][-1]:16.1f}')
    plot_exp_a_evolution(exp_a)

    print(f'\n[Exp B] 文脈分離の計測 (n_agents={_N_AGENTS}, n_ep_per_mode=10)')
    exp_b = run_exp_b_context(exp_a, seed=_SEED, n_ep_per_mode=10)
    plot_exp_b_context(exp_b)

    print(f'\n[Exp C] スパース性と活動パターン (n_episodes=20)')
    exp_c = run_exp_c_sparsity(exp_a, seed=_SEED, n_episodes=20)
    plot_exp_c_sparsity(exp_c)

    # ── Judgment ──────────────────────────────────────────────────────────────
    print('\n  ── Judgment Criteria (Session 22) ─────────────────────────────')
    print(f'  {"mr":>6}  {"acc_A":>6}  {"acc_B":>6}  {"cos_dist":>8}  {"result"}')
    print('  ' + '─' * 55)

    any_pass = False
    passing_mrs = []
    for mr in _MR_VALUES:
        r      = exp_b[mr]
        acc_a  = r['acc_A']
        acc_b  = r['acc_B']
        cos_d  = r['mean_cos_dist']
        ok_a   = acc_a > 0.6
        ok_b   = acc_b > 0.6
        ok_cos = cos_d > 0.1
        both   = ok_a and ok_b
        if both:
            any_pass = True
            passing_mrs.append(mr)
        markers = f'{"✓A" if ok_a else "✗A"}  {"✓B" if ok_b else "✗B"}  {"✓cos" if ok_cos else "✗cos"}'
        print(f'  {mr:6.3f}  {acc_a:6.3f}  {acc_b:6.3f}  {cos_d:8.4f}  {markers}')

    print()
    if passing_mrs:
        min_mr = passing_mrs[0]
        if min_mr == 0.0:
            print('  → mr=0.0でacc_B>0.6: 代謝コスト不要・飽和は関係なかった')
        elif min_mr <= 0.05:
            print(f'  → mr={min_mr:.2f}で文脈分離が創発: 最小限の神の手として正当化できる')
        else:
            print(f'  → mr=0.10でのみ文脈分離が創発: 強い代謝コストへの依存が大きい')
    else:
        print('  → 全条件でacc_B≤0.6: 飽和以外に別の根本問題がある → Session 23へ')

    print('\nDone.')
