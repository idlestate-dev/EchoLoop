"""Session 12: 探索と記憶の固定（睡眠仮説）

Session 11 diagnosis:
  Noise diversifies behavior (exploration) → topology varies between episodes
  Evolution cannot lock in gains → gen10→gen50 performance flat or declining

Hypothesis: A consolidation (sleep) phase after each episode
  - Decays weak edges (weak memories fade)
  - Preserves strong edges (strong memories survive)
  - Produces a stable topology for the next episode to build on
  - Enables evolutionary selection to act on meaningful differences

Consolidation phase (after each episode):
  - No external input: inp4 = [0, 0, 0, 0]
  - No activity noise
  - No new edge addition (edge_add_prob = 0)
  - Hebbian with decay only (strengthening only fires if residual activity > 0.5)

Experiments:
  A  Single agent, 20 episodes × 4 T_consolidation conditions {0, 50, 200, 500}
     Track per episode: survival steps, topology change, action entropy
  B  Evolution (50 gen × 10 agents × 5 ep): best T_consolidation vs S11 vs S10 vs random
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _s10_world_step, _s10_inp4,
    _s10_evolve_old, _s10_random_baseline,
    _N, _K, _INIT_W, _LR, _GRID, _HP_MAX, _HP_DECAY,
    _FOOD_VAL, _RESPAWN, _FOOD_POS, _N_PROP, _MAX_STEPS, _ACTION_NAMES,
)
from session_11_noise_escape import _s11_hebb, _s11_entropy, _s11_evolve_new

# Best noise params from Session 11
_BEST_EP = 0.10
_BEST_AN = 0.05


# ─── Consolidation (sleep) phase ──────────────────────────────────────────────

def _s12_consolidation_phase(G, W, activity_end, rng, T_consolidation):
    """Run T_consolidation steps with zero input, no noise, no new edges.

    Decay trims weak connections; strong edges (high residual activity) survive.
    Returns (n_before, n_after, w_mean_before, w_mean_after).
    """
    ws_before = [d['weight'] for _, _, d in G.edges(data=True)]
    n_before  = G.number_of_edges()
    m_before  = float(np.mean(ws_before)) if ws_before else 0.0

    if T_consolidation == 0:
        return n_before, n_before, m_before, m_before

    inp4_zero = np.zeros(4)
    act = activity_end.copy()

    for t in range(T_consolidation):
        for _ in range(_N_PROP):
            act = _s10_propagate(W, act, inp4_zero)
        if (t + 1) % _K == 0:
            _s11_hebb(G, W, act, rng, edge_add_prob=0.0)

    ws_after = [d['weight'] for _, _, d in G.edges(data=True)]
    n_after  = G.number_of_edges()
    m_after  = float(np.mean(ws_after)) if ws_after else 0.0

    return n_before, n_after, m_before, m_after


# ─── Episode + consolidation runner ───────────────────────────────────────────

def _s12_run_ep(G, W, rng, T_consolidation=0,
                edge_add_prob=_BEST_EP, activity_noise=_BEST_AN):
    """Awake phase (episode) followed by sleep phase (consolidation).

    Returns (steps_survived, action_entropy, delta_edges, delta_w_mean).
    Modifies G and W in place.
    """
    activity    = np.zeros(_N)
    row, col    = 2, 2
    hp          = 100
    food_avail  = [True, True]
    food_timer  = [0, 0]
    steps       = 0
    win_actions = np.zeros(5, dtype=int)

    for step in range(_MAX_STEPS):
        if hp <= 0:
            break

        inp4 = _s10_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, size=_N), 0.0, 1.0)

        action = int(np.argmax(activity[4:9]))
        win_actions[action] += 1

        row, col, hp, ate = _s10_world_step(row, col, action, food_avail, hp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0

        hp -= _HP_DECAY
        steps = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (step + 1) % _K == 0:
            _s11_hebb(G, W, activity, rng, edge_add_prob)

    entropy = _s11_entropy(win_actions)
    n_b, n_a, m_b, m_a = _s12_consolidation_phase(G, W, activity, rng, T_consolidation)

    return steps, entropy, n_a - n_b, m_a - m_b


# ─── Experiment A: Consolidation sweep ────────────────────────────────────────

def run_consolidation_sweep(seed=42, n_episodes=20,
                            T_consolidations=(0, 50, 200, 500),
                            edge_add_prob=_BEST_EP, activity_noise=_BEST_AN):
    """Single agent, n_episodes per T_consolidation condition."""
    results = {}
    print(f'  {"T_consol":>8} │ {"mean_steps":>11}  {"last_step":>9}  '
          f'{"mean_Δedge":>10}  {"mean_H":>7}')
    print('  ' + '─' * 55)

    for T_c in T_consolidations:
        rng = np.random.default_rng(seed)
        G   = _s10_build_graph(rng)
        W   = _s10_get_W(G)

        ep_steps, ep_entropy, ep_delta_edges, ep_delta_w = [], [], [], []

        for _ in range(n_episodes):
            rng_ep = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, ent, de, dw = _s12_run_ep(G, W, rng_ep, T_c, edge_add_prob, activity_noise)
            ep_steps.append(s)
            ep_entropy.append(ent)
            ep_delta_edges.append(de)
            ep_delta_w.append(dw)

        results[T_c] = {
            'steps':       ep_steps,
            'entropy':     ep_entropy,
            'delta_edges': ep_delta_edges,
            'delta_w':     ep_delta_w,
        }
        print(f'  {T_c:8d} │ {np.mean(ep_steps):11.1f}  {ep_steps[-1]:9d}  '
              f'{np.mean(ep_delta_edges):10.1f}  {np.mean(ep_entropy):7.3f}')

    best_T = max(T_consolidations, key=lambda T: np.mean(results[T]['steps']))
    results['best_T'] = int(best_T)
    print(f'  → Best: T_consolidation={best_T} '
          f'(mean_steps={np.mean(results[best_T]["steps"]):.1f})')
    return results


# ─── Experiment B: Evolution ───────────────────────────────────────────────────

def _s12_evolve_new(seed, n_gen=50, n_agents=10, n_ep=5, n_surv=3,
                    T_consolidation=200):
    """Evolve new-arch agents with consolidation phase after each episode."""
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_agents):
        G = _s10_build_graph(rng)
        pop.append((G, _s10_get_W(G)))

    gen_means, gen_bests = [], []

    for gen in range(n_gen):
        fitnesses = []
        for G, W in pop:
            rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
            total  = sum(
                _s12_run_ep(G, W,
                            np.random.default_rng(int(rng_ag.integers(0, 2**32))),
                            T_consolidation)[0]
                for _ in range(n_ep)
            )
            fitnesses.append(total / n_ep)

        gen_means.append(float(np.mean(fitnesses)))
        gen_bests.append(float(np.max(fitnesses)))

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:n_surv]]

        new_pop = list(survivors)
        while len(new_pop) < n_agents:
            G_p, _ = survivors[int(rng.integers(0, n_surv))]
            G_c    = _s10_mutate(G_p, rng)
            new_pop.append((G_c, _s10_get_W(G_c)))
        pop = new_pop

        if (gen + 1) % 10 == 0:
            print(f'    gen {gen+1:3d}: mean={gen_means[-1]:.1f}, best={gen_bests[-1]:.1f}')

    return {'gen_means': gen_means, 'gen_bests': gen_bests}


def run_evolution(seed=42, best_T=200,
                  n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    """Exp B: 4 conditions — sleep+noise, noise-only (S11), old arch (S10), random."""
    print(f'  [new arch + noise + sleep (T_consolidation={best_T})]...')
    sleep_data = _s12_evolve_new(seed, n_gen, n_agents, n_ep, n_surv, best_T)

    print('  [new arch + noise, no sleep — S11 baseline]...')
    no_sleep_data = _s11_evolve_new(seed + 1, n_gen, n_agents, n_ep, n_surv,
                                    _BEST_EP, _BEST_AN)

    print('  [old arch, co-evo readout — S10 baseline]...')
    old_data = _s10_evolve_old(seed + 2, n_gen, n_agents, n_ep, n_surv)

    print('  [random baseline]...')
    rnd_data = _s10_random_baseline(seed + 3, n_gen, n_agents, n_ep)

    return {
        'sleep':    sleep_data,
        'no_sleep': no_sleep_data,
        'old':      old_data,
        'rnd':      rnd_data,
        'n_gen':    n_gen,
        'best_T':   best_T,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_consolidation_sweep(
        data, fname='images/session_12/results_s12_consolidation_sweep.png'):
    T_cs   = (0, 50, 200, 500)
    xs     = np.arange(1, len(data[0]['steps']) + 1)
    colors = ['gray', 'steelblue', 'darkorange', 'crimson']
    best_T = data['best_T']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Session 12 Exp A: Consolidation Phase Sweep — 20 episodes per condition\n'
        f'Awake params: edge_add_prob={_BEST_EP}, activity_noise={_BEST_AN}  '
        f'| ★ = best T_consolidation',
        fontsize=10,
    )

    # Panel 1: survival steps per episode
    ax = axes[0]
    for T_c, color in zip(T_cs, colors):
        steps  = data[T_c]['steps']
        lw     = 2.5 if T_c == best_T else 1.5
        label  = f'T={T_c}{"★" if T_c == best_T else ""}'
        ax.plot(xs, steps, label=label, color=color, marker='o',
                markersize=3, linewidth=lw)
        if len(steps) >= 5:
            smooth = np.convolve(steps, np.ones(5) / 5, mode='valid')
            ax.plot(xs[4:], smooth, color=color, linewidth=3.5, alpha=0.25)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps Survived')
    ax.set_title('Survival Steps per Episode')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: delta_edges from consolidation (T>0 only)
    ax = axes[1]
    for T_c, color in zip(T_cs[1:], colors[1:]):
        de    = data[T_c]['delta_edges']
        label = f'T={T_c}{"★" if T_c == best_T else ""}'
        ax.plot(xs, de, label=label, color=color, marker='s',
                markersize=3, linewidth=1.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Δ Edges (after − before consolidation)')
    ax.set_title('Topology Pruning During Sleep Phase\n(−N = N edges removed)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: action entropy per episode
    ax = axes[2]
    max_H = float(np.log2(5))
    for T_c, color in zip(T_cs, colors):
        ent   = data[T_c]['entropy']
        label = f'T={T_c}{"★" if T_c == best_T else ""}'
        ax.plot(xs, ent, label=label, color=color, marker='o',
                markersize=3, linewidth=1.8)
    ax.axhline(max_H, color='black', linestyle='--', linewidth=1,
               alpha=0.35, label=f'max H={max_H:.2f}b')
    ax.set_ylim(0, max_H * 1.15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Action Entropy (bits)')
    ax.set_title('Behavioral Diversity per Episode')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_evolution(data, fname='images/session_12/results_s12_evolution.png'):
    n_gen  = data['n_gen']
    best_T = data['best_T']
    xs     = np.arange(1, n_gen + 1)

    cond_map = [
        ('sleep',    f'New arch + noise + sleep\n(T_consol={best_T})',  'steelblue',  '-'),
        ('no_sleep', 'New arch + noise, no sleep\n(S11 baseline)',       'firebrick',  '--'),
        ('old',      'Old arch, co-evo readout\n(S10 baseline)',         'darkorange', '-'),
        ('rnd',      'Random baseline',                                   'gray',       ':'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Session 12 Exp B: Sleep Consolidation — evolution comparison\n'
        f'T_consolidation={best_T}, noise: ep={_BEST_EP}, noise={_BEST_AN}',
        fontsize=10,
    )

    for ax, metric_key, ylabel in [
        (axes[0], 'gen_means', 'Mean Survival Steps'),
        (axes[1], 'gen_bests', 'Best Survival Steps'),
    ]:
        for cond_key, label, color, ls in cond_map:
            vals = data[cond_key][metric_key]
            ax.plot(xs, vals, label=label, color=color, linestyle=ls, linewidth=2)
            if len(vals) > 5:
                smooth = np.convolve(vals, np.ones(5) / 5, mode='valid')
                ax.plot(xs[4:], smooth, color=color, linewidth=3.5, alpha=0.3)
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' per Generation')
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 12: 探索と記憶の固定（睡眠仮説）===')

    print('\n[Exp A] Consolidation sweep (4 conditions × 20 episodes)...')
    expA = run_consolidation_sweep(seed=42)
    for T_c in (0, 50, 200, 500):
        d = expA[T_c]
        print(f'  T={T_c:3d}: mean_steps={np.mean(d["steps"]):.1f}, '
              f'last={d["steps"][-1]}, mean_Δedge={np.mean(d["delta_edges"]):.1f}')
    plot_consolidation_sweep(expA)

    best_T = expA['best_T']
    print(f'\n[Exp B] Evolution with T_consolidation={best_T}...')
    expB = run_evolution(seed=42, best_T=best_T)
    for key, label in [('sleep',    'Sleep    '), ('no_sleep', 'No-sleep '),
                       ('old',      'Old      '), ('rnd',      'Random   ')]:
        m = expB[key]['gen_means']
        b = expB[key]['gen_bests']
        print(f'  {label}: gen1={m[0]:.1f}, gen25={m[24]:.1f}, gen50={m[-1]:.1f}'
              f' | best50={b[-1]:.1f}')
    plot_evolution(expB)

    print('\nDone.')
