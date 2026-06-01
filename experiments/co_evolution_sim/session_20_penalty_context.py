"""Session 20: 文脈を読まないと死ぬ環境

PenaltyContextGridWorld:
  モードA: 食料=北西(0,0)、ペナルティゾーン=南東(4,4)  HP-50
  モードB: 食料=南東(4,4)、ペナルティゾーン=北西(0,0)  HP-50
  モード通知なし — HPの急激な変化から推測するしかない

「常に北西」や「常に南東」のような文脈盲目な戦略は
ペナルティゾーンに突っ込んで死ぬ。生き残るには文脈を読むしかない。

Experiments:
  A  PenaltyContextGridWorldでの進化（50世代）＋ 4条件比較
     (penalty_evolved / context_evolved-S19 / simple_evolved-S18 / random)
  B  ベスト個体の文脈分離
     出力ノード活動ヒートマップ + cosine_distance分布
  C  ペナルティ回避学習
     20エピソード連続実行：ペナルティ踏み込み推移・モードA/B正答率推移

判定基準:
  acc_A > 0.6 かつ acc_B > 0.6
  cosine_dist > 0.1
  後半のペナルティ踏み込み < 前半
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate,
    _N, _K, _N_PROP,
)
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_18_ratio_evolution import (
    _s18_hebb, _s18_make_genome, _s18_mutate_genome, _s18_run_ep,
    _ACTIVITY_NOISE, _T_CONSOLIDATION,
    _N_GEN, _N_SURV,
)
from session_19_context_reboot import (
    _CGRID, _CHP_MAX, _CHP_DECAY, _CFOOD_VAL, _CRESPAWN, _CSTEPS,
    _s19_get_evolved_ms1000_genome,
)

_SEED      = 42
_N_AGENTS  = 10
_N_EP      = 5    # per agent per generation (evolution)
_N_EP_EVAL = 20   # for evaluation experiments
_PENALTY   = 50   # HP damage on penalty zone entry


# ── PenaltyContextGridWorld episode runners ────────────────────────────────────

def _s20_inp4(row, col, hp, food_avail):
    return np.array([
        col / (_CGRID - 1),
        row / (_CGRID - 1),
        hp  / _CHP_MAX,
        1.0 if food_avail else 0.0,
    ])


def _s20_run_penalty_ep(G, W, edge_add_prob, activity_ratio, rng,
                         mode=None, activity_noise=_ACTIVITY_NOISE,
                         T_consolidation=_T_CONSOLIDATION,
                         record_activity=False):
    """PenaltyContextGridWorld episode (new-arch Hebb + sleep).

    Mode A: food at NW(0,0), penalty zone at SE(4,4).
    Mode B: food at SE(4,4), penalty zone at NW(0,0).
    Penalty triggers on zone entry: HP -= _PENALTY.
    No mode signal — agent must infer from HP changes.

    Modifies G and W in place (lifetime Hebb learning).
    Returns (steps_survived, food_count, mode, penalty_count, records or None).
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos    = (0, 0) if mode == 'A' else (4, 4)   # (row, col)
    penalty_pos = (4, 4) if mode == 'A' else (0, 0)
    food_avail  = True
    food_timer  = 0

    activity  = np.zeros(_N)
    row, col  = 2, 2
    hp        = 100
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

        action = int(np.argmax(activity[4:9]))

        fr, fc = food_pos
        pr, pc = penalty_pos

        if action in (0, 1, 2, 3):
            if   action == 0: row = max(0, row - 1)
            elif action == 1: row = min(_CGRID - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(_CGRID - 1, col + 1)

            # Penalty triggers on entry to the zone
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


def _s20_run_penalty_ep_random(rng, mode=None):
    """Random agent in PenaltyContextGridWorld.

    Returns (steps, food, mode, penalty_count).
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos    = (0, 0) if mode == 'A' else (4, 4)
    penalty_pos = (4, 4) if mode == 'A' else (0, 0)
    food_avail  = True
    food_timer  = 0

    row, col  = 2, 2
    hp        = 100
    steps     = 0
    food      = 0
    penalties = 0

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        action = int(rng.integers(0, 5))
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

    return steps, food, mode, penalties


# ── Evolution helpers ─────────────────────────────────────────────────────────

def _s20_evolve_penalty_world(seed=_SEED):
    """Evolve genomes with fitness = survival in PenaltyContextGridWorld.

    Returns (best_genome, fitness_history) where fitness_history is best
    mean-steps-per-episode at each generation.
    """
    rng      = np.random.default_rng(seed + 20000)
    pop      = [_s18_make_genome(rng) for _ in range(_N_AGENTS)]
    fit_hist = []

    for gen in range(_N_GEN):
        fitnesses = []
        for g in pop:
            total = 0
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, _, _, _, _ = _s20_run_penalty_ep(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'], ep_rng)
                total += s
            fitnesses.append(total / _N_EP)

        best = max(fitnesses)
        fit_hist.append(best)

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s18_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            bg = pop[0]
            print(f'    gen {gen+1:3d}: best={best:7.1f}  '
                  f'ep={bg["edge_add_prob"]:.3f}  ar={bg["activity_ratio"]:.3f}')

    return pop[0], fit_hist


def _s20_get_s18_genome(seed=_SEED):
    """Re-derive S18 best genome (SimpleGridWorld, ms=500).

    Returns best genome dict.
    """
    max_steps = 500
    rng       = np.random.default_rng(seed + 18500)
    pop       = [_s18_make_genome(rng) for _ in range(_N_AGENTS)]

    for gen in range(_N_GEN):
        fitnesses = []
        for g in pop:
            total = 0
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, _   = _s18_run_ep(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'],
                    ep_rng, max_steps)
                total += s
            fitnesses.append(total / _N_EP)

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s18_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 25 == 0:
            print(f'    [S18] gen {gen+1:3d}: best={max(fitnesses):7.1f}')

    return pop[0]


# ── Experiment A: Evolution + 4-condition comparison ─────────────────────────

def run_exp_a_evolution(seed=_SEED):
    """Evolve in PenaltyContextGridWorld; evaluate 4 conditions in penalty world.

    Returns (evolved_genome, fit_hist, cond_results).
    cond_results: dict keyed by condition with steps/food/modes/penalties arrays.
    """
    # 1. Evolve in PenaltyContextGridWorld
    print('\n  [Evolving in PenaltyContextGridWorld (50 gen)]')
    evolved, fit_hist = _s20_evolve_penalty_world(seed=seed)
    print(f'  → ep={evolved["edge_add_prob"]:.3f}  '
          f'ar={evolved["activity_ratio"]:.3f}  '
          f'edges={evolved["G"].number_of_edges()}')

    # 2. Re-derive S18 genome (SimpleGridWorld)
    print('\n  [Re-deriving S18 genome (SimpleGridWorld, seed+18500)]')
    s18_genome = _s20_get_s18_genome(seed=seed)
    print(f'  → S18: ep={s18_genome["edge_add_prob"]:.3f}  '
          f'ar={s18_genome["activity_ratio"]:.3f}')

    # 3. Re-derive S19 genome (ContextGridWorld ms=1000)
    print('\n  [Re-deriving S19 genome (ContextGridWorld ms=1000)]')
    s19_genome = _s19_get_evolved_ms1000_genome(seed=seed)
    print(f'  → S19: ep={s19_genome["edge_add_prob"]:.3f}  '
          f'ar={s19_genome["activity_ratio"]:.3f}')

    # 4. Evaluate all 4 conditions in PenaltyContextGridWorld
    print('\n  [Evaluating 4 conditions in PenaltyContextGridWorld]')
    named_genomes = [
        ('penalty_evolved', evolved),
        ('context_evolved', s19_genome),
        ('simple_evolved',  s18_genome),
    ]

    rng = np.random.default_rng(seed + 20100)
    cond_results = {}

    for cond, genome in named_genomes:
        print(f'\n    [{cond}]')
        all_steps, all_food, all_modes, all_pens = [], [], [], []

        for _ in range(_N_AGENTS):
            G  = genome['G'].copy()
            W  = _s10_get_W(G)
            ep = genome['edge_add_prob']
            ar = genome['activity_ratio']

            for _ in range(_N_EP_EVAL):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, m, p, _ = _s20_run_penalty_ep(G, W, ep, ar, ep_rng)
                all_steps.append(s)
                all_food.append(f)
                all_modes.append(m)
                all_pens.append(p)

        modes_arr = np.array(all_modes)
        food_arr  = np.array(all_food)
        acc_a = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
        acc_b = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
        print(f'      mean_steps={np.mean(all_steps):.1f}  acc_A={acc_a:.3f}  '
              f'acc_B={acc_b:.3f}  mean_penalties={np.mean(all_pens):.2f}')

        cond_results[cond] = {
            'steps':     np.array(all_steps),
            'food':      np.array(all_food),
            'modes':     all_modes,
            'penalties': np.array(all_pens),
        }

    # Random baseline
    print('\n    [random]')
    all_steps, all_food, all_modes, all_pens = [], [], [], []
    for _ in range(_N_AGENTS * _N_EP_EVAL):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        s, f, m, p = _s20_run_penalty_ep_random(ep_rng)
        all_steps.append(s)
        all_food.append(f)
        all_modes.append(m)
        all_pens.append(p)

    modes_arr = np.array(all_modes)
    food_arr  = np.array(all_food)
    acc_a = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
    acc_b = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
    print(f'      mean_steps={np.mean(all_steps):.1f}  acc_A={acc_a:.3f}  '
          f'acc_B={acc_b:.3f}  mean_penalties={np.mean(all_pens):.2f}')

    cond_results['random'] = {
        'steps':     np.array(all_steps),
        'food':      np.array(all_food),
        'modes':     all_modes,
        'penalties': np.array(all_pens),
    }

    return evolved, fit_hist, cond_results


# ── Experiment B: Context separation ─────────────────────────────────────────

def run_exp_b_context_separation(evolved_genome, seed=_SEED, n_ep_per_mode=10):
    """Record output-node activity per mode in PenaltyContextGridWorld.

    n_agents trials; n_ep_per_mode episodes per mode per trial.
    Returns { mean_A, mean_B, ep_means_A, ep_means_B, cos_distances }.
    """
    print('\n  [Exp B: context separation]')
    rng           = np.random.default_rng(seed + 20200)
    ep_means_A    = []
    ep_means_B    = []
    cos_distances = []

    for trial in range(_N_AGENTS):
        G  = evolved_genome['G'].copy()
        W  = _s10_get_W(G)
        ep = evolved_genome['edge_add_prob']
        ar = evolved_genome['activity_ratio']

        trial_means_A, trial_means_B = [], []

        for ei in range(n_ep_per_mode * 2):
            mode   = 'A' if ei % 2 == 0 else 'B'
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, _, recs = _s20_run_penalty_ep(
                G, W, ep, ar, ep_rng, mode=mode, record_activity=True)

            if recs:
                arr      = np.array(recs)
                mean_out = arr[:, 4:9].mean(axis=0)   # output nodes only
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

    print(f'    n_trials={len(cos_distances)}  '
          f'mean_cos_dist={np.mean(cos_distances):.4f} ± {np.std(cos_distances):.4f}')
    print(f'    mean_A={np.round(mean_A, 3)}  (N=North, S=South, W=West, E=East, Eat)')
    print(f'    mean_B={np.round(mean_B, 3)}')

    return {
        'mean_A':        mean_A,
        'mean_B':        mean_B,
        'ep_means_A':    ep_means_A,
        'ep_means_B':    ep_means_B,
        'cos_distances': cos_distances,
    }


# ── Experiment C: Penalty avoidance learning ──────────────────────────────────

def run_exp_c_penalty_avoidance(evolved_genome, seed=_SEED, n_episodes=20):
    """Run best genome sequentially for n_episodes; track penalty hits & accuracy.

    Same G/W used throughout — Hebb learning accumulates across episodes.
    Returns per-episode dict.
    """
    print('\n  [Exp C: penalty avoidance over 20 episodes]')
    rng = np.random.default_rng(seed + 20400)

    G  = evolved_genome['G'].copy()
    W  = _s10_get_W(G)
    ep = evolved_genome['edge_add_prob']
    ar = evolved_genome['activity_ratio']

    ep_penalties = []
    ep_food      = []
    ep_modes     = []
    ep_steps     = []

    for ei in range(n_episodes):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        s, f, m, p, _ = _s20_run_penalty_ep(G, W, ep, ar, ep_rng)
        ep_penalties.append(p)
        ep_food.append(f)
        ep_modes.append(m)
        ep_steps.append(s)
        print(f'    ep {ei+1:2d} [{m}]: steps={s:4d}  food={f}  penalties={p}')

    modes_arr = np.array(ep_modes)
    food_arr  = np.array(ep_food)

    # Rolling mode A/B accuracy (cumulative over episodes seen so far)
    acc_a_series = []
    acc_b_series = []
    for ei in range(1, n_episodes + 1):
        m_sub = modes_arr[:ei]
        f_sub = food_arr[:ei]
        acc_a = float(np.mean(f_sub[m_sub == 'A'] >= 1)) if np.any(m_sub == 'A') else float('nan')
        acc_b = float(np.mean(f_sub[m_sub == 'B'] >= 1)) if np.any(m_sub == 'B') else float('nan')
        acc_a_series.append(acc_a)
        acc_b_series.append(acc_b)

    pen_early = float(np.mean(ep_penalties[:5]))
    pen_late  = float(np.mean(ep_penalties[-5:]))
    final_a   = next((v for v in reversed(acc_a_series) if not np.isnan(v)), 0.0)
    final_b   = next((v for v in reversed(acc_b_series) if not np.isnan(v)), 0.0)
    print(f'  Penalties: early={pen_early:.2f} → late={pen_late:.2f}  '
          f'{"↓ decreasing" if pen_late < pen_early else "→ flat/increasing"}')
    print(f'  Final rolling acc: A={final_a:.3f}  B={final_b:.3f}')

    return {
        'penalties': ep_penalties,
        'food':      ep_food,
        'modes':     ep_modes,
        'steps':     ep_steps,
        'acc_A':     acc_a_series,
        'acc_B':     acc_b_series,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

_COND_ORDER  = ['penalty_evolved', 'context_evolved', 'simple_evolved', 'random']
_COND_COLORS = {
    'penalty_evolved': '#e6194b',
    'context_evolved': '#f58231',
    'simple_evolved':  'steelblue',
    'random':          'gray',
}
_COND_LABELS = {
    'penalty_evolved': 'Penalty\nevolved\n(S20)',
    'context_evolved': 'Context\nevolved\n(S19)',
    'simple_evolved':  'Simple\nevolved\n(S18)',
    'random':          'Random\nbaseline',
}


def plot_exp_a_evolution(fit_hist, cond_results,
                          fname='images/session_20/results_s20_evolution.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 20 Exp A: PenaltyContextGridWorld — Evolution + 4-Condition Comparison\n'
        f'n_agents={_N_AGENTS}, n_episodes={_N_EP_EVAL} (eval), max_steps={_CSTEPS}',
        fontsize=11,
    )

    # Left: evolution fitness curve + reference lines
    ax   = axes[0]
    gens = np.arange(1, len(fit_hist) + 1)
    ax.plot(gens, fit_hist, color='#e6194b', linewidth=2,
            label='penalty_evolved (training fitness)')

    for cond in ['context_evolved', 'simple_evolved', 'random']:
        ref   = float(np.mean(cond_results[cond]['steps']))
        label = _COND_LABELS[cond].replace('\n', ' ') + f' (mean={ref:.0f})'
        ax.axhline(ref, color=_COND_COLORS[cond], linestyle='--',
                   linewidth=1.5, alpha=0.85, label=label)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Mean Survival Steps  (PenaltyContextGridWorld)')
    ax.set_title('Evolution Fitness Curve\nvs. Transfer Baselines')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Right: box plot of survival steps for all 4 conditions
    ax = axes[1]
    bp = ax.boxplot(
        [cond_results[k]['steps'].tolist() for k in _COND_ORDER],
        patch_artist=True, widths=0.5,
    )
    for patch, key in zip(bp['boxes'], _COND_ORDER):
        patch.set_facecolor(_COND_COLORS[key])
        patch.set_alpha(0.75)

    # Overlay acc_A / acc_B as text
    for gi, key in enumerate(_COND_ORDER, 1):
        d         = cond_results[key]
        modes_arr = np.array(d['modes'])
        food_arr  = np.array(d['food'])
        acc_a = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
        acc_b = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
        ax.text(gi, 10, f'A:{acc_a:.2f}\nB:{acc_b:.2f}',
                ha='center', va='bottom', fontsize=8, color=_COND_COLORS[key])

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([_COND_LABELS[k] for k in _COND_ORDER], fontsize=9)
    ax.set_ylabel('Survival Steps per Episode')
    ax.set_title(f'4-Condition Survival in PenaltyContextGridWorld\n'
                 f'(text = acc_A / acc_B,  max={_CSTEPS})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_context_separation(
        data, fname='images/session_20/results_s20_context_separation.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Session 20 Exp B: Context Separation in PenaltyContextGridWorld\n'
        'Output nodes (node4-8) mean activity: Mode A (NW food / SE penalty)'
        ' vs Mode B (SE food / NW penalty)',
        fontsize=11,
    )

    action_names = ['North\n(n4)', 'South\n(n5)', 'West\n(n6)',
                    'East\n(n7)', 'Eat\n(n8)']

    ax      = axes[0]
    hm_data = np.array([data['mean_A'], data['mean_B']])
    vmax    = max(float(hm_data.max()), 0.01)
    im      = ax.imshow(hm_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(action_names, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Mode A\n(NW food\nSE penalty)', 'Mode B\n(SE food\nNW penalty)'],
                       fontsize=9)
    ax.set_title('Mean Output Node Activity per Mode')
    for (r, c), val in np.ndenumerate(hm_data):
        ax.text(c, r, f'{val:.3f}', ha='center', va='center', fontsize=9, color='black')
    plt.colorbar(im, ax=ax)

    ax      = axes[1]
    cos_d   = data['cos_distances']
    rng_jit = np.random.default_rng(0)
    if cos_d:
        ax.boxplot([cos_d], patch_artist=True,
                   boxprops=dict(facecolor='#e6194b', alpha=0.7))
        jitter = rng_jit.uniform(-0.15, 0.15, len(cos_d))
        ax.scatter(1 + jitter, cos_d, color='darkred', alpha=0.7, s=35, zorder=3)
        ax.axhline(0.1, color='#e6194b', linestyle='--', linewidth=1.3, alpha=0.8,
                   label='Threshold (cos_dist = 0.1)')
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6,
                   label='No separation (cos_dist = 0)')
        mean_cd = float(np.mean(cos_d))
        std_cd  = float(np.std(cos_d))
        ax.set_ylabel('Cosine Distance  (Mode A vs B output nodes)')
        ax.set_xticks([1])
        ax.set_xticklabels(['penalty_evolved'])
        ax.set_title(
            f'Cosine Distance Distribution\n'
            f'n_trials={len(cos_d)}  mean={mean_cd:.4f} ± {std_cd:.4f}  '
            f'{"✓ > 0.1" if mean_cd > 0.1 else "✗ ≤ 0.1"}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_penalty_avoidance(
        data, fname='images/session_20/results_s20_penalty_avoidance.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 20 Exp C: Penalty Avoidance Learning — 20 Sequential Episodes\n'
        '(same agent, Hebb learning accumulates across episodes)',
        fontsize=11,
    )

    n_ep   = len(data['penalties'])
    ep_idx = np.arange(1, n_ep + 1)
    modes  = np.array(data['modes'])
    pens   = np.array(data['penalties'], dtype=float)

    # Left: per-episode penalty count with trend
    ax     = axes[0]
    colors = ['#e6194b' if m == 'A' else 'steelblue' for m in modes]
    ax.bar(ep_idx, pens, color=colors, alpha=0.85, edgecolor='white')

    coeffs = np.polyfit(ep_idx, pens, 1)
    trend  = np.polyval(coeffs, ep_idx)
    slope  = float(coeffs[0])
    ax.plot(ep_idx, trend, 'k--', linewidth=1.5,
            label=f'Trend (slope={slope:.3f}/ep)')

    # Shade early vs late halves
    mid = n_ep // 2
    ax.axvspan(0.5, mid + 0.5, alpha=0.05, color='gray', label=f'Early (ep 1-{mid})')
    ax.axvspan(mid + 0.5, n_ep + 0.5, alpha=0.1, color='green', label=f'Late (ep {mid+1}-{n_ep})')

    pen_early = float(np.mean(pens[:mid]))
    pen_late  = float(np.mean(pens[mid:]))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Penalty Zone Hits')
    ax.set_title(f'Penalty Hits per Episode  (red=Mode A, blue=Mode B)\n'
                 f'Early mean={pen_early:.2f}  Late mean={pen_late:.2f}  '
                 f'{"↓" if pen_late < pen_early else "→"}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0.5, n_ep + 0.5)

    # Right: rolling mode A/B accuracy
    ax     = axes[1]
    accs_a = np.array(data['acc_A'], dtype=float)
    accs_b = np.array(data['acc_B'], dtype=float)
    mask_a = ~np.isnan(accs_a)
    mask_b = ~np.isnan(accs_b)

    if mask_a.any():
        ax.plot(ep_idx[mask_a], accs_a[mask_a], 'o-', color='#e6194b',
                linewidth=2, markersize=6, label='Mode A  (NW food)')
    if mask_b.any():
        ax.plot(ep_idx[mask_b], accs_b[mask_b], 's-', color='steelblue',
                linewidth=2, markersize=6, label='Mode B  (SE food)')

    ax.axhline(0.6, color='green', linestyle='--', linewidth=1.3, alpha=0.8,
               label='Threshold (0.6)')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.set_xlabel('Episode  (cumulative)')
    ax.set_ylabel('Rolling Accuracy  (food_eaten ≥ 1)')
    ax.set_ylim(-0.05, 1.15)
    ax.set_title('Rolling Mode A / B Accuracy over 20 Episodes')
    ax.legend(fontsize=9)
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

    print('=== Session 20: 文脈を読まないと死ぬ環境 ===')
    print(f'PenaltyContextGridWorld: penalty={_PENALTY} HP on zone entry, '
          f'food={_CFOOD_VAL} HP, decay={_CHP_DECAY}/step')

    print('\n[Exp A] Evolution in PenaltyContextGridWorld + 4-condition comparison')
    evolved, fit_hist, exp_a = run_exp_a_evolution(seed=_SEED)

    print('\n  Summary (all evaluated in PenaltyContextGridWorld):')
    print(f'  {"condition":>20}  {"mean_steps":>10}  {"acc_A":>6}  {"acc_B":>6}  {"mean_pens":>9}')
    print('  ' + '─' * 60)
    for key in _COND_ORDER:
        d         = exp_a[key]
        modes_arr = np.array(d['modes'])
        food_arr  = np.array(d['food'])
        acc_a = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
        acc_b = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
        print(f'  {key:>20}  {np.mean(d["steps"]):10.1f}  {acc_a:6.3f}  {acc_b:6.3f}  '
              f'{np.mean(d["penalties"]):9.2f}')

    plot_exp_a_evolution(fit_hist, exp_a)

    print(f'\n[Exp B] Context separation (n_agents={_N_AGENTS}, n_ep_per_mode=10)')
    exp_b = run_exp_b_context_separation(evolved, seed=_SEED, n_ep_per_mode=10)
    plot_exp_b_context_separation(exp_b)

    print(f'\n[Exp C] Penalty avoidance learning (n_episodes=20)')
    exp_c = run_exp_c_penalty_avoidance(evolved, seed=_SEED, n_episodes=20)
    plot_exp_c_penalty_avoidance(exp_c)

    # Judgment
    print('\n  ── Judgment Criteria ─────────────────────────────')
    d_pen     = exp_a['penalty_evolved']
    m_arr     = np.array(d_pen['modes'])
    f_arr     = np.array(d_pen['food'])
    acc_a_fin = float(np.mean(f_arr[m_arr == 'A'] >= 1)) if np.any(m_arr == 'A') else 0.0
    acc_b_fin = float(np.mean(f_arr[m_arr == 'B'] >= 1)) if np.any(m_arr == 'B') else 0.0
    cos_mean  = float(np.mean(exp_b['cos_distances'])) if exp_b['cos_distances'] else 0.0
    pen_e     = float(np.mean(exp_c['penalties'][:5]))
    pen_l     = float(np.mean(exp_c['penalties'][-5:]))

    print(f'  acc_A > 0.6   : {acc_a_fin:.3f}  {"✓" if acc_a_fin > 0.6 else "✗"}')
    print(f'  acc_B > 0.6   : {acc_b_fin:.3f}  {"✓" if acc_b_fin > 0.6 else "✗"}')
    print(f'  cos_dist > 0.1: {cos_mean:.4f}  {"✓" if cos_mean > 0.1 else "✗"}')
    print(f'  pen reduction : early={pen_e:.2f} → late={pen_l:.2f}  '
          f'{"✓" if pen_l < pen_e else "✗"}')

    n_pass = sum([
        acc_a_fin > 0.6,
        acc_b_fin > 0.6,
        cos_mean > 0.1,
        pen_l < pen_e,
    ])
    print(f'  → {n_pass}/4 criteria met  '
          f'("文脈を読んでいる" = any criterion satisfied)')

    print('\nDone.')
