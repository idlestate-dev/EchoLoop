"""Session 17: 活動依存的なエッジ生成

Session 16 finding: ep=0.000 (no random edge addition) gave best long-term
  stability — random edge addition was "contaminating" the network.
Hypothesis: replacing random edge addition with activity-dependent edge addition
  can combine long-term stability with exploration ability.

Key change (_s17_hebb vs _s11_hebb):
  Random:       iterates all N² pairs, adds each non-existing edge with p=ep → O(N²)/step
  Activity-dep: samples one pair ∝ current node activity, attempts with p=ep → O(1)/step

Experiments:
  A  Single agent: T=5000 continuous steps, 3 conditions, 1000-step windows
     Track: food gained, edge count, action entropy, Δedge
  B  Evolution: 3 conditions × 2 episode lengths (500, 2000)
     50 gen × 10 agents × 5 ep — fitness = mean survival steps
  C  Long eval: best genomes from Exp B (500-step training) on 2000-step × 5 ep
     4 conditions (random / activity / no-noise / random baseline)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _s10_world_step, _s10_inp4,
    _N, _K, _INIT_W, _LR, _HP_MAX, _HP_DECAY,
    _FOOD_VAL, _RESPAWN, _FOOD_POS, _N_PROP, _MAX_STEPS,
)
from session_11_noise_escape import _s11_hebb, _s11_entropy
from session_12_sleep_consolidation import _s12_consolidation_phase, _BEST_EP, _BEST_AN
from session_16_noise_evolution import _s16_run_random_ep

_SEED            = 42
_T_CONSOLIDATION = 200
_ACTIVITY_NOISE  = _BEST_AN   # 0.05, fixed from S12

_EXP_A_T_TOTAL = 5000
_EXP_A_WINDOW  = 1000

_EXP_B_N_GEN    = 50
_EXP_B_N_AGENTS = 10
_EXP_B_N_EP     = 5
_EXP_B_N_SURV   = 3
_EXP_B_EP_LENGTHS = [500, 2000]

_EXP_C_MAX_STEPS = 2000
_EXP_C_N_EP      = 5

_CONDITIONS = [
    {'name': 'random',   'edge_add_prob': 0.1, 'activity_dependent': False},
    {'name': 'activity', 'edge_add_prob': 0.1, 'activity_dependent': True},
    {'name': 'no_noise', 'edge_add_prob': 0.0, 'activity_dependent': False},
]

_COND_COLORS = {
    'random':           'steelblue',
    'activity':         'mediumseagreen',
    'no_noise':         'darkorange',
    'random_baseline':  'gray',
}
_COND_LABELS = {
    'random':           'Random\n(ep=0.1)',
    'activity':         'Activity-dep\n(ep=0.1)',
    'no_noise':         'No-noise\n(ep=0.0)',
    'random_baseline':  'Random\nbaseline',
}

# fixed seed offsets per condition name to avoid hash() non-determinism
_COND_SEED_OFFSET = {'random': 1000, 'activity': 2000, 'no_noise': 3000}


# ── Core Hebbian function ─────────────────────────────────────────────────────

def _s17_hebb(G, W, activity, rng, edge_add_prob, activity_dependent=False):
    """Hebbian update with configurable edge-addition strategy.

    activity_dependent=False: same O(N²) random-pair loop as _s11_hebb
    activity_dependent=True : one attempt per call, nodes sampled ∝ activity
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

    if activity_dependent:
        if rng.random() < edge_add_prob:
            act_pos = np.clip(activity, 0, None)
            if act_pos.sum() > 1e-9:
                probs = act_pos / act_pos.sum()
                i = int(rng.choice(_N, p=probs))
                j = int(rng.choice(_N, p=probs))
            else:
                i = int(rng.integers(0, _N))
                j = int(rng.integers(0, _N))
            if i != j and not G.has_edge(i, j):
                G.add_edge(i, j, weight=_INIT_W)
                W[i, j] = _INIT_W
    else:
        existing = set(G.edges())
        for i in range(_N):
            for j in range(_N):
                if i != j and (i, j) not in existing and rng.random() < edge_add_prob:
                    G.add_edge(i, j, weight=_INIT_W)
                    W[i, j] = _INIT_W


# ── Episode runner ────────────────────────────────────────────────────────────

def _s17_run_ep(G, W, edge_add_prob, activity_dependent, rng, max_steps,
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
            _s17_hebb(G, W, activity, rng, edge_add_prob, activity_dependent)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food


# ── Genome helpers ────────────────────────────────────────────────────────────

def _s17_make_genome(rng):
    G = _s10_build_graph(rng)
    return {'G': G, 'W': _s10_get_W(G)}


def _s17_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    return {'G': G_new, 'W': _s10_get_W(G_new)}


# ── Experiment A: Single agent continuous run ─────────────────────────────────

def _s17_continuous(edge_add_prob, activity_dependent, T_total, window, rng):
    """Single agent, T_total continuous steps with death-reset. No consolidation.

    Returns list of per-window dicts:
      {'food': int, 'edges': int, 'entropy': float, 'delta_edges': int}
    """
    G = _s10_build_graph(rng)
    W = _s10_get_W(G)

    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]

    food_count    = 0
    action_counts = np.zeros(5)
    edge_prev     = G.number_of_edges()
    records       = []

    for step in range(T_total):
        if hp <= 0:
            row, col   = 2, 2
            hp         = 100
            food_avail = [True, True]
            food_timer = [0, 0]
            activity   = np.zeros(_N)

        inp4 = _s10_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if _ACTIVITY_NOISE > 0.0:
            activity = np.clip(activity + rng.normal(0, _ACTIVITY_NOISE, _N), 0.0, 1.0)

        action = int(np.argmax(activity[4:9]))
        action_counts[action] += 1
        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            food_count += 1

        hp -= _HP_DECAY
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (step + 1) % _K == 0:
            _s17_hebb(G, W, activity, rng, edge_add_prob, activity_dependent)

        if (step + 1) % window == 0:
            e    = G.number_of_edges()
            entr = _s11_entropy(action_counts)
            records.append({
                'food':        food_count,
                'edges':       e,
                'entropy':     entr,
                'delta_edges': e - edge_prev,
            })
            food_count    = 0
            action_counts = np.zeros(5)
            edge_prev     = e

    return records


def run_exp_a_single_agent(seed=_SEED):
    """Run 3 conditions × T=5000 steps. Returns per-window records dict."""
    results = {}
    for cond in _CONDITIONS:
        name = cond['name']
        print(f'  [Exp A: {name}]')
        rng  = np.random.default_rng(seed)
        recs = _s17_continuous(
            cond['edge_add_prob'], cond['activity_dependent'],
            _EXP_A_T_TOTAL, _EXP_A_WINDOW, rng,
        )
        results[name] = recs
        total_food   = sum(r['food'] for r in recs)
        final_edges  = recs[-1]['edges']
        mean_entropy = float(np.mean([r['entropy'] for r in recs]))
        print(f'    total_food={total_food}  final_edges={final_edges}  '
              f'mean_entropy={mean_entropy:.3f}')
    return results


# ── Experiment B: Evolution ───────────────────────────────────────────────────

def run_exp_b_evolution(seed=_SEED):
    """Evolve topology independently for each condition × episode length.

    Returns nested dict: results[cond_name][max_steps] = {
      'gen_means': list[float],
      'gen_bests': list[float],
      'final_best': genome dict,
    }
    """
    results = {cond['name']: {} for cond in _CONDITIONS}

    for cond in _CONDITIONS:
        name    = cond['name']
        ep_prob = cond['edge_add_prob']
        act_dep = cond['activity_dependent']

        for max_steps in _EXP_B_EP_LENGTHS:
            print(f'\n  [Exp B: {name}  max_steps={max_steps}]')
            rng = np.random.default_rng(seed + _COND_SEED_OFFSET[name] + max_steps)

            pop = [_s17_make_genome(rng) for _ in range(_EXP_B_N_AGENTS)]
            gen_means, gen_bests = [], []

            for gen in range(_EXP_B_N_GEN):
                fitnesses = []
                for g in pop:
                    total = 0
                    for _ in range(_EXP_B_N_EP):
                        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                        s, _   = _s17_run_ep(
                            g['G'], g['W'], ep_prob, act_dep, ep_rng, max_steps,
                        )
                        total += s
                    fitnesses.append(total / _EXP_B_N_EP)

                gen_means.append(float(np.mean(fitnesses)))
                gen_bests.append(float(np.max(fitnesses)))

                idx_sorted = np.argsort(fitnesses)[::-1]
                survivors  = [pop[i] for i in idx_sorted[:_EXP_B_N_SURV]]
                new_pop    = list(survivors)
                while len(new_pop) < _EXP_B_N_AGENTS:
                    parent = survivors[int(rng.integers(0, _EXP_B_N_SURV))]
                    new_pop.append(_s17_mutate_genome(parent, rng))
                pop = new_pop

                if (gen + 1) % 10 == 0 or gen == 0:
                    print(f'    gen {gen+1:3d}: mean={gen_means[-1]:7.1f}  '
                          f'best={gen_bests[-1]:7.1f}')

            results[name][max_steps] = {
                'gen_means':  gen_means,
                'gen_bests':  gen_bests,
                'final_best': pop[0],
            }
            print(f'  → final_best edges={pop[0]["G"].number_of_edges()}')

    return results


# ── Experiment C: Long evaluation ─────────────────────────────────────────────

def run_exp_c_long_eval(exp_b_results, seed=_SEED):
    """Evaluate best genomes from Exp B (500-step training) on long episodes.

    Returns dict keyed by condition name (including 'random_baseline'):
      {'steps': ndarray (n_ep,), 'food': ndarray (n_ep,)}
    """
    rng     = np.random.default_rng(seed + 7777)
    results = {}

    for cond in _CONDITIONS:
        name    = cond['name']
        ep_prob = cond['edge_add_prob']
        act_dep = cond['activity_dependent']
        genome  = exp_b_results[name][500]['final_best']

        print(f'\n  [Exp C: {name}]')
        G = genome['G'].copy()
        W = _s10_get_W(G)
        steps_l, food_l = [], []
        for _ in range(_EXP_C_N_EP):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, f   = _s17_run_ep(G, W, ep_prob, act_dep, ep_rng, _EXP_C_MAX_STEPS)
            steps_l.append(s); food_l.append(f)
        results[name] = {
            'steps': np.array(steps_l),
            'food':  np.array(food_l),
        }
        print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    print('\n  [Exp C: random_baseline]')
    steps_l, food_l = [], []
    for _ in range(_EXP_C_N_EP):
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        s, f   = _s16_run_random_ep(ep_rng, _EXP_C_MAX_STEPS)
        steps_l.append(s); food_l.append(f)
    results['random_baseline'] = {
        'steps': np.array(steps_l),
        'food':  np.array(food_l),
    }
    print(f'    mean_steps={np.mean(steps_l):.1f}  mean_food={np.mean(food_l):.2f}')

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_exp_a_single_agent(
        data, fname='images/session_17/results_s17_single_agent.png'):
    """4-row × 3-col: 4 metrics × 3 conditions, 1000-step windows."""
    cond_names = [c['name'] for c in _CONDITIONS]
    windows    = np.arange(_EXP_A_WINDOW, _EXP_A_T_TOTAL + 1, _EXP_A_WINDOW)

    metrics = [
        ('food',        'Food gained',        'Food count / window'),
        ('edges',       'Edge count',          'Total edges'),
        ('entropy',     'Action entropy',      'Entropy (bits)'),
        ('delta_edges', 'Topology change Δedge', 'Edge delta / window'),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(
        'Session 17 Exp A: Single Agent Continuous Run (T=5000, window=1000)\n'
        '3 conditions: random / activity-dependent / no-noise edge addition  '
        f'(activity_noise={_ACTIVITY_NOISE}, seed={_SEED})',
        fontsize=10,
    )

    for col_idx, name in enumerate(cond_names):
        recs = data[name]
        for row_idx, (key, title, ylabel) in enumerate(metrics):
            ax   = axes[row_idx][col_idx]
            vals = [r[key] for r in recs]
            ax.plot(windows, vals,
                    color=_COND_COLORS[name], linewidth=2, marker='o', markersize=5)
            ax.axhline(float(np.mean(vals)), color=_COND_COLORS[name],
                       linestyle='--', linewidth=1, alpha=0.5)
            label = _COND_LABELS[name].replace('\n', ' ')
            ax.set_title(f'{label} — {title}', fontsize=9)
            ax.set_xlabel('Step')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_evolution(
        data, fname='images/session_17/results_s17_evolution.png'):
    """2-row × 3-col: episode lengths × conditions, generation fitness curves."""
    cond_names = [c['name'] for c in _CONDITIONS]

    fig, axes = plt.subplots(len(_EXP_B_EP_LENGTHS), len(cond_names),
                             figsize=(15, 9), sharex=True)
    fig.suptitle(
        f'Session 17 Exp B: Evolution  '
        f'(n_gen={_EXP_B_N_GEN}, n_agents={_EXP_B_N_AGENTS}, n_ep={_EXP_B_N_EP})\n'
        '3 conditions × 2 episode lengths  |  fitness = mean survival steps',
        fontsize=10,
    )

    xs = np.arange(1, _EXP_B_N_GEN + 1)
    for row_idx, max_steps in enumerate(_EXP_B_EP_LENGTHS):
        for col_idx, name in enumerate(cond_names):
            ax = axes[row_idx][col_idx]
            d  = data[name][max_steps]
            c  = _COND_COLORS[name]

            ax.plot(xs, d['gen_bests'], color=c, linewidth=2.0,
                    label='best', zorder=3)
            ax.plot(xs, d['gen_means'], color=c, linewidth=1.2,
                    linestyle='--', alpha=0.55, label='mean', zorder=2)

            label = _COND_LABELS[name].replace('\n', ' ')
            ax.set_title(f'{label}  |  max_steps={max_steps}', fontsize=9)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Survival Steps')
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_long_eval(
        data, fname='images/session_17/results_s17_long_eval.png'):
    """2-panel: survival steps per episode + early vs late food."""
    cond_order = [c['name'] for c in _CONDITIONS] + ['random_baseline']
    n_ep       = _EXP_C_N_EP
    xs         = np.arange(1, n_ep + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Session 17 Exp C: Long Episode Evaluation  '
        f'({_EXP_C_MAX_STEPS}-step × {n_ep} ep)\n'
        '4 conditions: evolved (random / activity-dep / no-noise) + random baseline',
        fontsize=10,
    )

    # Panel 1: survival steps per episode
    ax = axes[0]
    for key in cond_order:
        label = _COND_LABELS[key]
        ax.plot(xs, data[key]['steps'],
                color=_COND_COLORS[key], label=label,
                linewidth=2, marker='o', markersize=5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Survival Steps')
    ax.set_title(f'Survival Steps per Episode\n(max_steps={_EXP_C_MAX_STEPS})')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)

    # Panel 2: early vs late food
    ax     = axes[1]
    x_pos  = np.arange(len(cond_order))
    half_f = 2
    half_l = n_ep - 2

    for gi, key in enumerate(cond_order):
        food  = data[key]['food']
        first = float(np.mean(food[:half_f]))
        last  = float(np.mean(food[half_l:]))
        color = _COND_COLORS[key]
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
    ax.set_xticklabels([_COND_LABELS[k] for k in cond_order], fontsize=8)
    ax.set_ylabel('Mean Food Count per Episode')
    ax.set_title('Food Gain: Early vs Late Episodes\n(degradation check)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 17: 活動依存的なエッジ生成 ===')

    print(f'\n[Exp A] Single agent continuous run '
          f'(T={_EXP_A_T_TOTAL}, window={_EXP_A_WINDOW}, 3 conditions)')
    exp_a = run_exp_a_single_agent(seed=_SEED)
    plot_exp_a_single_agent(exp_a)

    print(f'\n[Exp B] Evolution  '
          f'(n_gen={_EXP_B_N_GEN}, n_agents={_EXP_B_N_AGENTS}, '
          f'n_ep={_EXP_B_N_EP}, ep_lengths={_EXP_B_EP_LENGTHS})')
    exp_b = run_exp_b_evolution(seed=_SEED)
    plot_exp_b_evolution(exp_b)

    print('\n  Summary: final best topology (edges) after evolution')
    print(f'  {"condition":>10}  ' +
          '  '.join(f'{"ms="+str(ms):>12}' for ms in _EXP_B_EP_LENGTHS))
    print('  ' + '─' * (12 + 14 * len(_EXP_B_EP_LENGTHS)))
    for cond in _CONDITIONS:
        name = cond['name']
        vals = '  '.join(
            f'{exp_b[name][ms]["final_best"]["G"].number_of_edges():12d}'
            for ms in _EXP_B_EP_LENGTHS
        )
        print(f'  {name:>10}  {vals}')

    print(f'\n[Exp C] Long evaluation '
          f'({_EXP_C_MAX_STEPS}-step × {_EXP_C_N_EP} ep, best from 500-step training)')
    exp_c = run_exp_c_long_eval(exp_b, seed=_SEED)
    plot_exp_c_long_eval(exp_c)

    print('\n  Summary: Exp C results')
    print(f'  {"condition":>18}  {"mean_steps":>10}  {"mean_food":>9}  '
          f'{"early_food":>10}  {"late_food":>9}')
    print('  ' + '─' * 62)
    for key in [c['name'] for c in _CONDITIONS] + ['random_baseline']:
        d     = exp_c[key]
        food  = d['food']
        early = float(np.mean(food[:2]))
        late  = float(np.mean(food[-2:]))
        print(f'  {key:>18}  {np.mean(d["steps"]):10.1f}  '
              f'{np.mean(food):9.2f}  {early:10.2f}  {late:9.2f}')

    print('\nDone.')
