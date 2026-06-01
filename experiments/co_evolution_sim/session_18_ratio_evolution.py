"""Session 18: ランダム性と経験依存性の混合比率の進化

Session 17 findings:
  - Activity-dependent edge addition → fixation (diversity loss)
  - Fully random edge addition → preserves diversity
  → Each plays a different role

Biological analogy: synapse formation is neither fully random nor fully
activity-dependent — both occur simultaneously.

Session 18 question:
  Does the optimal mixing ratio (activity_ratio) between randomness and
  activity-dependence converge through world interaction?
  Does the optimal ratio vary with episode length (lifespan)?

Design: activity_ratio is not fixed by the designer — each agent holds a
random value, and selection pressure carves out the optimal ratio.

Experiments:
  A  Evolution of mixing ratio across 3 episode lengths [500, 1000, 2000]
     n_gen=50, n_agents=10, n_ep=5, n_surv=3
     Genome: edge_add_prob ∈ [0.0, 0.2], activity_ratio ∈ [0.0, 1.0]
     Track: best agent's genome values per generation + fitness curves
  B  Long evaluation: best genomes from Exp A on 2000-step × 5 ep
     6 conditions: evolved (ms=500/1000/2000) + S17-random + S17-activity + random-baseline
  C  Scatter: converged activity_ratio vs episode length
     x=episode length, y=mean of last-10-gen best activity_ratio, err=std
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _s10_world_step, _s10_inp4,
    _N, _K, _INIT_W, _LR, _HP_DECAY, _RESPAWN, _N_PROP,
)
from session_12_sleep_consolidation import _s12_consolidation_phase, _BEST_AN
from session_16_noise_evolution import _s16_run_random_ep

_SEED            = 42
_ACTIVITY_NOISE  = _BEST_AN  # 0.05, fixed from S12
_T_CONSOLIDATION = 200       # fixed from S12

_EXP_A_EP_LENGTHS = [500, 1000, 2000]
_N_GEN    = 50
_N_AGENTS = 10
_N_EP     = 5
_N_SURV   = 3

_EP_INIT_MAX = 0.2   # initial range for edge_add_prob
_AR_INIT_MAX = 1.0   # initial range for activity_ratio

_EP_MUT_STD = 0.02   # mutation σ for edge_add_prob
_AR_MUT_STD = 0.05   # mutation σ for activity_ratio

_EXP_B_MAX_STEPS = 2000
_EXP_B_N_EP      = 5

_CONV_WINDOW = 10    # last N generations for convergence stats (Exp C)

# S17 reference fixed values
_S17_REF_EP = 0.1


# ── Core Hebbian function ─────────────────────────────────────────────────────

def _s18_hebb(G, W, activity, rng, edge_add_prob, activity_ratio):
    """Hebbian update + mixed edge addition.

    activity_ratio: probability a new edge is placed activity-dependently.
      0.0 = fully random exploration, 1.0 = fully activity-dependent.
    Per call: O(edges) for weight updates + one O(1) edge-addition attempt.
    """
    to_remove = []
    for i, j, data in list(G.edges(data=True)):
        w = data['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += _LR
        w -= 0.01
        if w < 0.01:
            to_remove.append((i, j))
            W[i, j] = 0.0
        else:
            w = min(w, 1.0)
            G[i][j]['weight'] = w
            W[i, j] = w
    G.remove_edges_from(to_remove)

    if edge_add_prob <= 0.0:
        return

    if rng.random() < edge_add_prob:
        if rng.random() < activity_ratio:
            # activity-dependent: sample nodes proportional to current activity
            act_pos = np.clip(activity, 0, None)
            if act_pos.sum() > 1e-9:
                probs = act_pos / act_pos.sum()
                i = int(rng.choice(_N, p=probs))
                j = int(rng.choice(_N, p=probs))
            else:
                i = int(rng.integers(0, _N))
                j = int(rng.integers(0, _N))
        else:
            # fully random exploration
            i = int(rng.integers(0, _N))
            j = int(rng.integers(0, _N))
        if i != j and not G.has_edge(i, j):
            G.add_edge(i, j, weight=_INIT_W)
            W[i, j] = _INIT_W


# ── Episode runner ────────────────────────────────────────────────────────────

def _s18_run_ep(G, W, edge_add_prob, activity_ratio, rng, max_steps,
                activity_noise=_ACTIVITY_NOISE, T_consolidation=_T_CONSOLIDATION):
    """Awake + sleep (consolidation) phase. Modifies G/W in place.

    Returns (steps_survived, food_count).
    """
    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    food       = 0

    for step in range(max_steps):
        if hp <= 0:
            break

        inp4 = _s10_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        action = int(np.argmax(activity[4:9]))
        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            food += 1

        hp    -= _HP_DECAY
        steps  = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, activity, rng, edge_add_prob, activity_ratio)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food


# ── Genome helpers ────────────────────────────────────────────────────────────

def _s18_make_genome(rng):
    G = _s10_build_graph(rng)
    return {
        'G':              G,
        'W':              _s10_get_W(G),
        'edge_add_prob':  float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio': float(rng.uniform(0.0, _AR_INIT_MAX)),
    }


def _s18_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    ep = float(np.clip(
        genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD), 0.0, _EP_INIT_MAX))
    ar = float(np.clip(
        genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD), 0.0, _AR_INIT_MAX))
    return {
        'G':              G_new,
        'W':              _s10_get_W(G_new),
        'edge_add_prob':  ep,
        'activity_ratio': ar,
    }


# ── Experiment A: Evolution ───────────────────────────────────────────────────

def run_exp_a_genome_evolution(seed=_SEED):
    """Evolve (edge_add_prob, activity_ratio) independently for each episode length.

    Returns dict keyed by max_steps:
      {
        'gen_means':    list[float],   mean fitness per generation
        'gen_bests':    list[float],   best fitness per generation
        'best_ep_hist': list[float],   best agent's edge_add_prob per generation
        'best_ar_hist': list[float],   best agent's activity_ratio per generation
        'final_best':   genome dict,   best genome at end of evolution
      }
    """
    results = {}

    for max_steps in _EXP_A_EP_LENGTHS:
        print(f'\n  [Exp A: max_steps={max_steps}  n_gen={_N_GEN}]')
        rng = np.random.default_rng(seed + max_steps)

        pop = [_s18_make_genome(rng) for _ in range(_N_AGENTS)]
        gen_means, gen_bests = [], []
        best_ep_hist, best_ar_hist = [], []

        for gen in range(_N_GEN):
            fitnesses = []
            for g in pop:
                total = 0
                for _ in range(_N_EP):
                    ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                    s, _   = _s18_run_ep(
                        g['G'], g['W'],
                        g['edge_add_prob'], g['activity_ratio'],
                        ep_rng, max_steps,
                    )
                    total += s
                fitnesses.append(total / _N_EP)

            gen_means.append(float(np.mean(fitnesses)))
            gen_bests.append(float(np.max(fitnesses)))

            idx_sorted = np.argsort(fitnesses)[::-1]
            best_g     = pop[idx_sorted[0]]
            best_ep_hist.append(best_g['edge_add_prob'])
            best_ar_hist.append(best_g['activity_ratio'])

            survivors = [pop[i] for i in idx_sorted[:_N_SURV]]
            new_pop   = list(survivors)
            while len(new_pop) < _N_AGENTS:
                parent = survivors[int(rng.integers(0, _N_SURV))]
                new_pop.append(_s18_mutate_genome(parent, rng))
            pop = new_pop

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f'    gen {gen+1:3d}: mean={gen_means[-1]:7.1f}  '
                      f'best={gen_bests[-1]:7.1f}  '
                      f'ep={best_ep_hist[-1]:.3f}  ar={best_ar_hist[-1]:.3f}')

        results[max_steps] = {
            'gen_means':    gen_means,
            'gen_bests':    gen_bests,
            'best_ep_hist': best_ep_hist,
            'best_ar_hist': best_ar_hist,
            'final_best':   pop[0],
        }
        final = pop[0]
        print(f'  → final: ep={final["edge_add_prob"]:.3f}  '
              f'ar={final["activity_ratio"]:.3f}  '
              f'edges={final["G"].number_of_edges()}')

    return results


# ── Experiment B: Long evaluation ─────────────────────────────────────────────

def run_exp_b_long_eval(exp_a_results, seed=_SEED):
    """Evaluate best genomes from Exp A on long episodes + S17 reference conditions.

    Returns dict keyed by condition name:
      {'steps': ndarray (n_ep,), 'food': ndarray (n_ep,), 'ep': float, 'ar': float}
    """
    rng     = np.random.default_rng(seed + 8888)
    results = {}

    for max_steps in _EXP_A_EP_LENGTHS:
        key    = f'evolved_ms{max_steps}'
        genome = exp_a_results[max_steps]['final_best']
        ep     = genome['edge_add_prob']
        ar     = genome['activity_ratio']
        print(f'\n  [Exp B: {key}  ep={ep:.3f}  ar={ar:.3f}]')

        G = genome['G'].copy()
        W = _s10_get_W(G)
        steps_l, food_l = [], []
        for _ in range(_EXP_B_N_EP):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, f   = _s18_run_ep(G, W, ep, ar, ep_rng, _EXP_B_MAX_STEPS)
            steps_l.append(s)
            food_l.append(f)
        results[key] = {
            'steps': np.array(steps_l),
            'food':  np.array(food_l),
            'ep':    ep,
            'ar':    ar,
        }
        print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    # S17 reference conditions with fresh topology (fixed ratio, no topology evolution)
    for ref_name, ref_ar in [('s17_random', 0.0), ('s17_activity', 1.0)]:
        topo_rng = np.random.default_rng(seed + 9000 + int(ref_ar * 1000))
        print(f'\n  [Exp B: {ref_name}  ep={_S17_REF_EP:.1f}  ar={ref_ar:.1f}]')

        G = _s10_build_graph(topo_rng)
        W = _s10_get_W(G)
        steps_l, food_l = [], []
        for _ in range(_EXP_B_N_EP):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, f   = _s18_run_ep(G, W, _S17_REF_EP, ref_ar, ep_rng, _EXP_B_MAX_STEPS)
            steps_l.append(s)
            food_l.append(f)
        results[ref_name] = {
            'steps': np.array(steps_l),
            'food':  np.array(food_l),
            'ep':    _S17_REF_EP,
            'ar':    ref_ar,
        }
        print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    print('\n  [Exp B: random_baseline]')
    steps_l, food_l = [], []
    for _ in range(_EXP_B_N_EP):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        s, f   = _s16_run_random_ep(ep_rng, _EXP_B_MAX_STEPS)
        steps_l.append(s)
        food_l.append(f)
    results['random_baseline'] = {
        'steps': np.array(steps_l),
        'food':  np.array(food_l),
        'ep':    0.0,
        'ar':    0.0,
    }
    print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

_EP_COLORS = {500: '#e6194b', 1000: '#f58231', 2000: '#3cb44b'}

_COND_COLORS_B = {
    'evolved_ms500':   '#e6194b',
    'evolved_ms1000':  '#f58231',
    'evolved_ms2000':  '#3cb44b',
    's17_random':      'steelblue',
    's17_activity':    'mediumseagreen',
    'random_baseline': 'gray',
}
_COND_LABELS_B = {
    'evolved_ms500':   'Evolved\n(ms=500)',
    'evolved_ms1000':  'Evolved\n(ms=1000)',
    'evolved_ms2000':  'Evolved\n(ms=2000)',
    's17_random':      'S17 random\n(ar=0.0)',
    's17_activity':    'S17 activity\n(ar=1.0)',
    'random_baseline': 'Random\nbaseline',
}
_COND_ORDER_B = (
    ['evolved_ms500', 'evolved_ms1000', 'evolved_ms2000',
     's17_random', 's17_activity', 'random_baseline']
)


def plot_exp_a_genome_convergence(
        data, fname='images/session_18/results_s18_genome_convergence.png'):
    """2-row × 3-col: (edge_add_prob / activity_ratio) × 3 episode lengths."""
    xs = np.arange(1, _N_GEN + 1)

    fig, axes = plt.subplots(2, len(_EXP_A_EP_LENGTHS), figsize=(15, 9))
    fig.suptitle(
        f'Session 18 Exp A: Genome Convergence  '
        f'(n_gen={_N_GEN}, n_agents={_N_AGENTS}, n_ep={_N_EP})\n'
        'Evolving edge_add_prob + activity_ratio across 3 episode lengths  '
        f'(activity_noise={_ACTIVITY_NOISE}, seed={_SEED})',
        fontsize=10,
    )

    for col_idx, max_steps in enumerate(_EXP_A_EP_LENGTHS):
        d     = data[max_steps]
        color = _EP_COLORS[max_steps]

        # Row 0: edge_add_prob
        ax = axes[0][col_idx]
        ax.plot(xs, d['best_ep_hist'], color=color, linewidth=2)
        ax.fill_between(xs, d['best_ep_hist'], alpha=0.15, color=color)
        ax.axhline(_S17_REF_EP, color='gray', linestyle='--',
                   linewidth=1.2, alpha=0.7, label='S17 ref (ep=0.1)')
        conv_ep = d['best_ep_hist'][-_CONV_WINDOW:]
        ax.set_title(
            f'ms={max_steps}  |  edge_add_prob\n'
            f'final={d["best_ep_hist"][-1]:.3f}  '
            f'conv={np.mean(conv_ep):.3f}±{np.std(conv_ep):.3f}',
            fontsize=8,
        )
        ax.set_xlabel('Generation')
        ax.set_ylabel('edge_add_prob')
        ax.set_ylim(-0.02, _EP_INIT_MAX + 0.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1: activity_ratio
        ax = axes[1][col_idx]
        ax.plot(xs, d['best_ar_hist'], color=color, linewidth=2)
        ax.fill_between(xs, d['best_ar_hist'], alpha=0.15, color=color)
        ax.axhline(0.0, color='steelblue', linestyle='--',
                   linewidth=1.2, alpha=0.7, label='S17 random (ar=0.0)')
        ax.axhline(1.0, color='mediumseagreen', linestyle='--',
                   linewidth=1.2, alpha=0.7, label='S17 activity (ar=1.0)')
        conv_ar = d['best_ar_hist'][-_CONV_WINDOW:]
        ax.set_title(
            f'ms={max_steps}  |  activity_ratio\n'
            f'final={d["best_ar_hist"][-1]:.3f}  '
            f'conv={np.mean(conv_ar):.3f}±{np.std(conv_ar):.3f}',
            fontsize=8,
        )
        ax.set_xlabel('Generation')
        ax.set_ylabel('activity_ratio')
        ax.set_ylim(-0.05, 1.05)
        if col_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_long_eval(
        data, fname='images/session_18/results_s18_long_eval.png'):
    """2-panel: survival steps box plot + early vs late food bar."""
    n_ep = _EXP_B_N_EP
    half = max(1, n_ep // 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f'Session 18 Exp B: Long Episode Evaluation  '
        f'({_EXP_B_MAX_STEPS}-step × {n_ep} ep)\n'
        '6 conditions: evolved (ms=500/1000/2000) + S17 random/activity + random baseline',
        fontsize=10,
    )

    # Panel 1: survival steps box plot
    ax       = axes[0]
    box_data = [data[k]['steps'].tolist() for k in _COND_ORDER_B]
    bp       = ax.boxplot(box_data, patch_artist=True, widths=0.5)
    for patch, key in zip(bp['boxes'], _COND_ORDER_B):
        patch.set_facecolor(_COND_COLORS_B[key])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(_COND_ORDER_B) + 1))
    ax.set_xticklabels([_COND_LABELS_B[k] for k in _COND_ORDER_B], fontsize=8)
    ax.set_ylabel('Survival Steps')
    ax.set_title(f'Survival Steps per Episode  (max_steps={_EXP_B_MAX_STEPS})')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: early vs late food bar
    ax    = axes[1]
    x_pos = np.arange(len(_COND_ORDER_B))
    for gi, key in enumerate(_COND_ORDER_B):
        food  = data[key]['food']
        early = float(np.mean(food[:half]))
        late  = float(np.mean(food[half:]))
        color = _COND_COLORS_B[key]
        ax.bar(x_pos[gi] - 0.18, early, width=0.34,
               color=color, alpha=0.4, edgecolor='gray', linewidth=0.7)
        ax.bar(x_pos[gi] + 0.18, late,  width=0.34,
               color=color, alpha=1.0, edgecolor='gray', linewidth=0.7)

    from matplotlib.patches import Patch
    handles = [
        Patch(color='gray', alpha=0.4, label=f'ep 1–{half} (early)'),
        Patch(color='gray', alpha=1.0, label=f'ep {half+1}–{n_ep} (late)'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_COND_LABELS_B[k] for k in _COND_ORDER_B], fontsize=8)
    ax.set_ylabel('Mean Food Count per Episode')
    ax.set_title('Food Gain: Early vs Late Episodes')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_lifespan_ratio(
        data, fname='images/session_18/results_s18_lifespan_ratio.png'):
    """Scatter: episode length vs converged activity_ratio (Exp C)."""
    ep_lengths = _EXP_A_EP_LENGTHS
    means, stds = [], []
    for ms in ep_lengths:
        conv = data[ms]['best_ar_hist'][-_CONV_WINDOW:]
        means.append(float(np.mean(conv)))
        stds.append(float(np.std(conv)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        'Session 18 Exp C: Lifespan vs Converged Activity Ratio\n'
        f'Convergence window = last {_CONV_WINDOW} generations  '
        f'(n_gen={_N_GEN}, seed={_SEED})',
        fontsize=10,
    )

    ax.errorbar(
        ep_lengths, means, yerr=stds,
        fmt='o-', linewidth=2, markersize=8,
        color='darkviolet', ecolor='violet', elinewidth=2, capsize=6,
        label=f'Converged activity_ratio\n(mean ± std, last {_CONV_WINDOW} gen)',
        zorder=3,
    )
    for ms, m, s in zip(ep_lengths, means, stds):
        ax.annotate(f'ar={m:.2f}±{s:.2f}',
                    xy=(ms, m), xytext=(12, 10),
                    textcoords='offset points', fontsize=8)

    ax.axhline(0.0, color='steelblue',      linestyle='--',
               linewidth=1.5, alpha=0.7, label='Fully random (ar=0.0)')
    ax.axhline(1.0, color='mediumseagreen', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Fully activity-dep (ar=1.0)')
    ax.axhline(0.5, color='gray',           linestyle=':',
               linewidth=1.0, alpha=0.5, label='50/50 mix (ar=0.5)')

    ax.set_xscale('log')
    ax.set_xticks(ep_lengths)
    ax.set_xticklabels([str(ms) for ms in ep_lengths])
    ax.set_xlabel('Episode Length (max_steps)', fontsize=11)
    ax.set_ylabel('Converged activity_ratio', fontsize=11)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 18: ランダム性と経験依存性の混合比率の進化 ===')

    print(f'\n[Exp A] Genome evolution  '
          f'(n_gen={_N_GEN}, n_agents={_N_AGENTS}, n_ep={_N_EP}, '
          f'ep_lengths={_EXP_A_EP_LENGTHS})')
    exp_a = run_exp_a_genome_evolution(seed=_SEED)

    print(f'\n  Summary: converged genome (last {_CONV_WINDOW} gen)')
    print(f'  {"ms":>6}  {"ep_mean":>8}  {"ep_std":>7}  '
          f'{"ar_mean":>8}  {"ar_std":>7}  {"edges":>6}')
    print('  ' + '─' * 54)
    for ms in _EXP_A_EP_LENGTHS:
        d      = exp_a[ms]
        ep_arr = d['best_ep_hist'][-_CONV_WINDOW:]
        ar_arr = d['best_ar_hist'][-_CONV_WINDOW:]
        edges  = d['final_best']['G'].number_of_edges()
        print(f'  {ms:>6}  {np.mean(ep_arr):8.3f}  {np.std(ep_arr):7.3f}  '
              f'{np.mean(ar_arr):8.3f}  {np.std(ar_arr):7.3f}  {edges:6d}')

    plot_exp_a_genome_convergence(exp_a)
    plot_exp_c_lifespan_ratio(exp_a)

    print(f'\n[Exp B] Long evaluation  '
          f'({_EXP_B_MAX_STEPS}-step × {_EXP_B_N_EP} ep)')
    exp_b = run_exp_b_long_eval(exp_a, seed=_SEED)
    plot_exp_b_long_eval(exp_b)

    print('\n  Summary: Exp B results')
    half = max(1, _EXP_B_N_EP // 2)
    print(f'  {"condition":>20}  {"ep":>5}  {"ar":>5}  '
          f'{"mean_steps":>10}  {"mean_food":>9}  {"early":>7}  {"late":>7}')
    print('  ' + '─' * 72)
    for key in _COND_ORDER_B:
        d     = exp_b[key]
        food  = d['food']
        early = float(np.mean(food[:half]))
        late  = float(np.mean(food[half:]))
        print(f'  {key:>20}  {d["ep"]:5.3f}  {d["ar"]:5.3f}  '
              f'{np.mean(d["steps"]):10.1f}  {np.mean(food):9.2f}  '
              f'{early:7.2f}  {late:7.2f}')

    print('\nDone.')
