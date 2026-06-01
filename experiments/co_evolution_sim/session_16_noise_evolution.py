"""Session 16: ノイズ強度の進化

Question: エピソード長（寿命）に応じた最適なノイズ強度があるのか？

Session 15 finding: edge_add_prob=0.1 (fixed) causes topology inflation in
  long episodes (max_steps=2000) — edges increase +26 to +52 without damage.
Hypothesis: longer-lived agents benefit from weaker noise; optimal noise
  declines as episode length grows.

Genome: {edge_add_prob ∈ [0.0, 0.2], activity_noise ∈ [0.0, 0.1]}
  T_consolidation = 200 (fixed, Session 12 best)
  E/I thresholds not evolved (too many free variables)

Experiments:
  A  Genome evolution × 3 episode lengths [500, 1000, 2000]
     Track: best agent's (ep, an) per generation + mean/best survival steps
  B  Long evaluation: 5 conditions × (2000 steps × 5 ep)
     Compare evolved genomes vs Session 12 fixed params vs random
  C  Noise-lifespan scatter: converged edge_add_prob vs episode length
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _s10_world_step, _s10_inp4,
    _N, _K, _HP_MAX, _HP_DECAY,
    _FOOD_VAL, _RESPAWN, _N_PROP, _MAX_STEPS,
)
from session_11_noise_escape import _s11_hebb, _s11_entropy
from session_12_sleep_consolidation import _s12_consolidation_phase, _BEST_EP, _BEST_AN

_SEED           = 42
_T_CONSOLIDATION = 200    # fixed from Session 12

_EXP_A_EP_LENGTHS = [500, 1000, 2000]
_N_GEN    = 50
_N_AGENTS = 10
_N_EP     = 5
_N_SURV   = 3

_S12_EP = _BEST_EP   # 0.10
_S12_AN = _BEST_AN   # 0.05

_EXP_B_MAX_STEPS = 2000
_EXP_B_N_EP      = 5

_EP_INIT_MAX = 0.2   # initial genome range for edge_add_prob
_AN_INIT_MAX = 0.1   # initial genome range for activity_noise

_EP_MUT_STD = 0.02   # mutation σ for edge_add_prob
_AN_MUT_STD = 0.01   # mutation σ for activity_noise


# ── Genome helpers ────────────────────────────────────────────────────────────

def _s16_make_genome(rng):
    """Create a random genome: fresh topology + random noise params."""
    G = _s10_build_graph(rng)
    return {
        'G':              G,
        'W':              _s10_get_W(G),
        'edge_add_prob':  float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_noise': float(rng.uniform(0.0, _AN_INIT_MAX)),
    }


def _s16_mutate_genome(genome, rng):
    """Return a mutated offspring: topology via _s10_mutate + gaussian noise on params."""
    G_new = _s10_mutate(genome['G'], rng)
    ep = float(np.clip(
        genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD), 0.0, _EP_INIT_MAX))
    an = float(np.clip(
        genome['activity_noise'] + rng.normal(0, _AN_MUT_STD), 0.0, _AN_INIT_MAX))
    return {'G': G_new, 'W': _s10_get_W(G_new), 'edge_add_prob': ep, 'activity_noise': an}


# ── Episode runner ────────────────────────────────────────────────────────────

def _s16_run_ep(G, W, edge_add_prob, activity_noise, rng, max_steps,
                T_consolidation=_T_CONSOLIDATION):
    """Awake phase + sleep (consolidation). Modifies G/W in place.

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
            _s11_hebb(G, W, activity, rng, edge_add_prob)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food


# ── Experiment A: Genome evolution per episode length ─────────────────────────

def run_exp_a_genome_evolution(seed=_SEED, n_gen=_N_GEN, n_agents=_N_AGENTS,
                               n_ep=_N_EP, n_surv=_N_SURV):
    """Evolve (edge_add_prob, activity_noise) independently for each episode length.

    Returns dict keyed by max_steps:
      {
        'gen_means':   list[float],   mean fitness per generation
        'gen_bests':   list[float],   best fitness per generation
        'best_ep_hist': list[float],  best agent's edge_add_prob per generation
        'best_an_hist': list[float],  best agent's activity_noise per generation
        'final_best':  genome dict,   best genome at end of evolution
      }
    """
    results = {}

    for max_steps in _EXP_A_EP_LENGTHS:
        print(f'\n  [Exp A: max_steps={max_steps}  n_gen={n_gen}]')
        rng = np.random.default_rng(seed + max_steps)

        pop = [_s16_make_genome(rng) for _ in range(n_agents)]
        gen_means, gen_bests = [], []
        best_ep_hist, best_an_hist = [], []

        for gen in range(n_gen):
            fitnesses = []
            for g in pop:
                total = 0
                for _ in range(n_ep):
                    ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                    s, _ = _s16_run_ep(g['G'], g['W'],
                                       g['edge_add_prob'], g['activity_noise'],
                                       ep_rng, max_steps)
                    total += s
                fitnesses.append(total / n_ep)

            gen_means.append(float(np.mean(fitnesses)))
            gen_bests.append(float(np.max(fitnesses)))

            idx_sorted = np.argsort(fitnesses)[::-1]
            best_g     = pop[idx_sorted[0]]
            best_ep_hist.append(best_g['edge_add_prob'])
            best_an_hist.append(best_g['activity_noise'])

            survivors = [pop[i] for i in idx_sorted[:n_surv]]
            new_pop   = list(survivors)
            while len(new_pop) < n_agents:
                parent = survivors[int(rng.integers(0, n_surv))]
                new_pop.append(_s16_mutate_genome(parent, rng))
            pop = new_pop

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f'    gen {gen+1:3d}: mean={gen_means[-1]:7.1f}  '
                      f'best={gen_bests[-1]:7.1f}  '
                      f'best_ep={best_ep_hist[-1]:.4f}  '
                      f'best_an={best_an_hist[-1]:.4f}')

        # Identify final best (first survivor from last generation)
        results[max_steps] = {
            'gen_means':    gen_means,
            'gen_bests':    gen_bests,
            'best_ep_hist': best_ep_hist,
            'best_an_hist': best_an_hist,
            'final_best':   pop[0],  # top survivor from last generation
        }
        print(f'  → Converged: ep={pop[0]["edge_add_prob"]:.4f}  '
              f'an={pop[0]["activity_noise"]:.4f}')

    return results


# ── Experiment B: Long evaluation of evolved genomes ─────────────────────────

def _s16_run_random_ep(rng, max_steps):
    """Random-action baseline. Returns (steps, food)."""
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    food       = 0

    from session_10_embodied_output import _FOOD_POS, _GRID
    for step in range(max_steps):
        if hp <= 0:
            break

        action = int(rng.integers(0, 5))

        ate = -1
        if   action == 0: row = max(0, row - 1)
        elif action == 1: row = min(_GRID - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(_GRID - 1, col + 1)
        elif action == 4:
            for idx, (fr, fc) in enumerate(_FOOD_POS):
                if row == fr and col == fc and food_avail[idx]:
                    hp  = min(_HP_MAX, hp + _FOOD_VAL)
                    ate = idx
                    break

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

    return steps, food


def run_exp_b_long_eval(exp_a_results, seed=_SEED):
    """Evaluate evolved genomes (Exp A) plus S12 fixed baseline and random.

    Each condition: fresh topology copy × _EXP_B_N_EP episodes × _EXP_B_MAX_STEPS.

    Returns dict:
      {
        '500':    {'steps': np.ndarray (n_ep,), 'food': np.ndarray (n_ep,)},
        '1000':   ...
        '2000':   ...
        's12':    ...
        'random': ...
      }
    """
    rng     = np.random.default_rng(seed + 9999)
    results = {}

    # Evolved genomes from Exp A
    for max_steps in _EXP_A_EP_LENGTHS:
        label = str(max_steps)
        print(f'\n  [Exp B: evolved-{label}]')
        genome = exp_a_results[max_steps]['final_best']
        G = genome['G'].copy()
        W = _s10_get_W(G)
        ep = genome['edge_add_prob']
        an = genome['activity_noise']

        steps_l, food_l = [], []
        for _ in range(_EXP_B_N_EP):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, f = _s16_run_ep(G, W, ep, an, ep_rng, _EXP_B_MAX_STEPS)
            steps_l.append(s); food_l.append(f)
        results[label] = {
            'steps': np.array(steps_l),
            'food':  np.array(food_l),
            'ep': ep, 'an': an,
        }
        print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}  '
              f'ep={ep:.4f}  an={an:.4f}')

    # Session 12 best (fixed params, fresh topology)
    print('\n  [Exp B: S12-fixed]')
    G  = _s10_build_graph(np.random.default_rng(seed))
    W  = _s10_get_W(G)
    steps_l, food_l = [], []
    for _ in range(_EXP_B_N_EP):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        s, f = _s16_run_ep(G, W, _S12_EP, _S12_AN, ep_rng, _EXP_B_MAX_STEPS)
        steps_l.append(s); food_l.append(f)
    results['s12'] = {
        'steps': np.array(steps_l),
        'food':  np.array(food_l),
        'ep': _S12_EP, 'an': _S12_AN,
    }
    print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    # Random baseline
    print('\n  [Exp B: random]')
    steps_l, food_l = [], []
    for _ in range(_EXP_B_N_EP):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        s, f = _s16_run_random_ep(ep_rng, _EXP_B_MAX_STEPS)
        steps_l.append(s); food_l.append(f)
    results['random'] = {
        'steps': np.array(steps_l),
        'food':  np.array(food_l),
        'ep': None, 'an': None,
    }
    print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_exp_a_genome_convergence(
        data, fname='images/session_16/results_s16_genome_convergence.png'):
    """2-row × 3-col grid: genome param convergence per episode length.

    Row 0: edge_add_prob trajectory.  Row 1: activity_noise trajectory.
    Columns: episode lengths [500, 1000, 2000].
    S12 best value shown as horizontal dashed line.
    """
    ep_lengths = _EXP_A_EP_LENGTHS
    n_gen_actual = len(data[ep_lengths[0]]['best_ep_hist'])
    xs           = np.arange(1, n_gen_actual + 1)
    colors       = {'best': 'steelblue', 'mean': 'slategray'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
    fig.suptitle(
        'Session 16 Exp A: Genome Convergence — noise params evolving per episode length\n'
        f'n_gen={_N_GEN}, n_agents={_N_AGENTS}, n_ep={_N_EP}, n_surv={_N_SURV}  |  seed={_SEED}',
        fontsize=10,
    )

    param_rows = [
        ('best_ep_hist', 'edge_add_prob', _S12_EP, f'S12 best ({_S12_EP:.2f})'),
        ('best_an_hist', 'activity_noise', _S12_AN, f'S12 best ({_S12_AN:.2f})'),
    ]

    for row_idx, (hist_key, ylabel, s12_val, s12_label) in enumerate(param_rows):
        for col_idx, max_steps in enumerate(ep_lengths):
            ax   = axes[row_idx][col_idx]
            d    = data[max_steps]
            hist = d[hist_key]

            ax.plot(xs, hist, color=colors['best'], linewidth=1.8,
                    label='best agent', zorder=3)

            # Smoothed trend (5-gen rolling mean)
            if len(hist) >= 5:
                smooth = np.convolve(hist, np.ones(5) / 5, mode='valid')
                ax.plot(xs[4:], smooth, color=colors['best'],
                        linewidth=3.5, alpha=0.3, zorder=2)

            # S12 reference line
            ax.axhline(s12_val, color='crimson', linestyle='--',
                       linewidth=1.2, alpha=0.6,
                       label=s12_label if col_idx == 0 else None)

            final_val = hist[-1]
            ax.axhline(final_val, color='navy', linestyle=':',
                       linewidth=1.0, alpha=0.5,
                       label=f'final={final_val:.4f}' if col_idx == 0 else None)

            if row_idx == 0:
                ax.set_title(f'max_steps = {max_steps}', fontsize=10)
            ax.set_xlabel('Generation')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.legend(fontsize=8)

            ax.annotate(f'{final_val:.4f}',
                        xy=(xs[-1], final_val),
                        xytext=(xs[-1] - len(xs) * 0.1, final_val),
                        fontsize=8, color='navy',
                        arrowprops=dict(arrowstyle='->', color='navy', lw=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_long_eval(
        data, fname='images/session_16/results_s16_long_eval.png'):
    """2-panel figure for Experiment B.

    Panel 1 (left):  Survival steps per episode for each condition (line plot).
    Panel 2 (right): First 2 vs last 2 episodes' mean food (grouped bars).
    """
    cond_order  = ['500', '1000', '2000', 's12', 'random']
    cond_labels = {
        '500':    'Evolved\n(500-step)',
        '1000':   'Evolved\n(1000-step)',
        '2000':   'Evolved\n(2000-step)',
        's12':    f'S12 fixed\n(ep={_S12_EP}, an={_S12_AN})',
        'random': 'Random\nbaseline',
    }
    colors = {
        '500':    'steelblue',
        '1000':   'mediumseagreen',
        '2000':   'darkorchid',
        's12':    'darkorange',
        'random': 'gray',
    }

    n_ep = _EXP_B_N_EP
    xs   = np.arange(1, n_ep + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Session 16 Exp B: Long Episode Evaluation — {_EXP_B_MAX_STEPS}-step × {n_ep} ep\n'
        f'5 conditions: genome evolved at [500,1000,2000] steps vs S12-fixed vs Random',
        fontsize=10,
    )

    # ── Panel 1: survival steps per episode ───────────────────────────────────
    ax = axes[0]
    for key in cond_order:
        d     = data[key]
        steps = d['steps']
        ep_val = d['ep']
        label = cond_labels[key]
        if ep_val is not None:
            label += f'\n  ep={ep_val:.3f}'
        ax.plot(xs, steps, color=colors[key], label=label,
                linewidth=2, marker='o', markersize=5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Survival Steps')
    ax.set_title(f'Survival Steps per Episode\n(max_steps={_EXP_B_MAX_STEPS})')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)

    # ── Panel 2: first vs last episodes food ──────────────────────────────────
    ax     = axes[1]
    x_pos  = np.arange(len(cond_order))
    # first 2 episodes vs last 2 episodes
    half_f = 2
    half_l = n_ep - 2

    for gi, key in enumerate(cond_order):
        food  = data[key]['food']   # (n_ep,)
        first = float(np.mean(food[:half_f]))
        last  = float(np.mean(food[half_l:]))
        color = colors[key]
        ax.bar(x_pos[gi] - 0.18, first, width=0.34,
               color=color, alpha=0.45, edgecolor='gray', linewidth=0.7)
        ax.bar(x_pos[gi] + 0.18, last,  width=0.34,
               color=color, alpha=1.0,  edgecolor='gray', linewidth=0.7)

    from matplotlib.patches import Patch
    handles = [
        Patch(color='gray', alpha=0.45, label=f'ep 1–{half_f} (early)'),
        Patch(color='gray', alpha=1.0,  label=f'ep {half_l+1}–{n_ep} (late)'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([cond_labels[k] for k in cond_order], fontsize=7.5)
    ax.set_ylabel('Mean Food Count per Episode')
    ax.set_title(f'Food Gain: Early vs Late Episodes\n(degradation check)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_lifespan_noise(
        data, fname='images/session_16/results_s16_lifespan_noise.png'):
    """Scatter: converged edge_add_prob vs episode length.

    x-axis: episode length.
    y-axis: converged edge_add_prob (mean of last 10 generations' best values).
    Error bars: std over the last 10 generations.
    A dotted trend line connects the three points.
    """
    ep_lengths = _EXP_A_EP_LENGTHS
    ep_means, ep_stds = [], []
    an_means, an_stds = [], []

    for max_steps in ep_lengths:
        ep_hist = np.array(data[max_steps]['best_ep_hist'][-10:])
        an_hist = np.array(data[max_steps]['best_an_hist'][-10:])
        ep_means.append(float(np.mean(ep_hist)))
        ep_stds.append(float(np.std(ep_hist)))
        an_means.append(float(np.mean(an_hist)))
        an_stds.append(float(np.std(an_hist)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Session 16 Exp C: Noise-Lifespan Relationship\n'
        'Converged genome values (mean ± std of last 10 generations\' best agent)',
        fontsize=10,
    )

    param_rows = [
        (axes[0], ep_means, ep_stds, _S12_EP, 'edge_add_prob', 'steelblue'),
        (axes[1], an_means, an_stds, _S12_AN, 'activity_noise', 'mediumseagreen'),
    ]

    for ax, means, stds, s12_val, ylabel, color in param_rows:
        ax.errorbar(ep_lengths, means, yerr=stds,
                    fmt='o-', color=color, linewidth=2.2,
                    markersize=8, capsize=5, capthick=1.5, elinewidth=1.5,
                    label='converged (mean ± std, last 10 gen)')

        # Trend annotation
        for x, m, s in zip(ep_lengths, means, stds):
            ax.annotate(f'{m:.4f}', xy=(x, m), xytext=(x, m + s + 0.003),
                        ha='center', fontsize=9, color=color)

        ax.axhline(s12_val, color='crimson', linestyle='--',
                   linewidth=1.3, alpha=0.7,
                   label=f'S12 fixed = {s12_val}')

        ax.set_xlabel('Episode Length (max_steps)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Converged {ylabel} vs Episode Length')
        ax.set_xticks(ep_lengths)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 16: ノイズ強度の進化 ===')

    print(f'\n[Exp A] Genome evolution × 3 episode lengths '
          f'(n_gen={_N_GEN}, n_agents={_N_AGENTS}, n_ep={_N_EP})')
    exp_a = run_exp_a_genome_evolution(seed=_SEED)

    print('\n  Summary: converged genome values')
    print(f'  {"max_steps":>10}  {"edge_add_prob":>13}  {"activity_noise":>14}')
    print('  ' + '─' * 42)
    for ms in _EXP_A_EP_LENGTHS:
        ep_hist = exp_a[ms]['best_ep_hist'][-10:]
        an_hist = exp_a[ms]['best_an_hist'][-10:]
        print(f'  {ms:10d}  {np.mean(ep_hist):13.4f}  {np.mean(an_hist):14.4f}  '
              f'(last 10 gen mean)')
    print(f'  {"S12 fixed":>10}  {_S12_EP:13.4f}  {_S12_AN:14.4f}')

    plot_exp_a_genome_convergence(exp_a)

    print('\n[Exp C] Noise-lifespan scatter')
    plot_exp_c_lifespan_noise(exp_a)

    print('\n[Exp B] Long evaluation (2000-step × 5 ep) for all conditions')
    exp_b = run_exp_b_long_eval(exp_a, seed=_SEED)

    print('\n  Summary: Exp B results')
    print(f'  {"condition":>14}  {"mean_steps":>10}  {"mean_food":>9}  '
          f'{"early_food":>10}  {"late_food":>9}')
    print('  ' + '─' * 56)
    cond_order = ['500', '1000', '2000', 's12', 'random']
    for key in cond_order:
        d         = exp_b[key]
        food      = d['food']
        early     = float(np.mean(food[:2]))
        late      = float(np.mean(food[-2:]))
        print(f'  {key:>14}  {np.mean(d["steps"]):10.1f}  '
              f'{np.mean(food):9.2f}  {early:10.2f}  {late:9.2f}')
    plot_exp_b_long_eval(exp_b)

    print('\nDone.')
