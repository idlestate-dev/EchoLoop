"""Session 11: 自発的ノイズによる局所最適からの脱出

Session 10 diagnosis:
  argmax + Hebbian → winner-take-all → single-action attractor → 100 steps, never improves

Hypothesis:
  Increasing spontaneous activity noise breaks the attractor,
  allowing the new architecture to escape the local optimum.

Noise parameters swept:
  edge_add_prob  : [0.01, 0.05, 0.10]  (stochastic edge sprouting rate)
  activity_noise : [0.00, 0.05, 0.10]  (Gaussian noise added to all activities)

Experiments:
  A  Noise sweep: 9 conditions × T=5000 continuous steps
     Track per 500-step window: food count, action entropy, edge count
  B  Evolution: best noise condition vs no-noise (S10) vs old arch (S10) vs random
     50 gen × 10 agents × 5 episodes
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _s10_world_step, _s10_inp4,
    _s10_evolve_old, _s10_random_baseline,
    _N, _K, _INIT_W, _LR, _GRID, _HP_MAX, _HP_DECAY,
    _FOOD_VAL, _RESPAWN, _FOOD_POS, _N_PROP, _MAX_STEPS, _ACTION_NAMES,
)

_LOG2 = np.log(2)


# ─── Hebbian with configurable edge-add probability ───────────────────────────

def _s11_hebb(G, W, activity, rng, edge_add_prob):
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
    existing = set(G.edges())
    for i in range(_N):
        for j in range(_N):
            if i != j and (i, j) not in existing and rng.random() < edge_add_prob:
                G.add_edge(i, j, weight=_INIT_W)
                W[i, j] = _INIT_W


def _s11_entropy(action_counts):
    counts = np.asarray(action_counts, dtype=float)
    total  = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ─── Episode runner ───────────────────────────────────────────────────────────

def _s11_run_ep(G, W, rng, edge_add_prob=0.01, activity_noise=0.0, use_hebb=True):
    """New-arch episode with noise params. Modifies G/W via Hebbian. Returns steps."""
    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0

    for step in range(_MAX_STEPS):
        if hp <= 0:
            break

        inp4 = _s10_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, size=_N), 0.0, 1.0)

        action = int(np.argmax(activity[4:9]))
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

        if use_hebb and (step + 1) % _K == 0:
            _s11_hebb(G, W, activity, rng, edge_add_prob)

    return steps


# ─── Experiment A: Noise sweep, single agent ──────────────────────────────────

def _s11_continuous(seed, T_total, window, edge_add_prob, activity_noise):
    """Single agent, T_total continuous steps. Returns per-window metrics."""
    rng = np.random.default_rng(seed)
    G   = _s10_build_graph(rng)
    W   = _s10_get_W(G)

    windows_food    = []
    windows_entropy = []
    windows_edges   = []

    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]

    win_food    = 0
    win_actions = np.zeros(5, dtype=int)

    for t in range(T_total):
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
            win_food += 1

        hp -= _HP_DECAY
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (t + 1) % _K == 0:
            _s11_hebb(G, W, activity, rng, edge_add_prob)

        if hp <= 0:
            row, col   = 2, 2
            hp         = 100
            food_avail = [True, True]
            food_timer = [0, 0]

        if (t + 1) % window == 0:
            windows_food.append(win_food)
            windows_entropy.append(_s11_entropy(win_actions))
            windows_edges.append(G.number_of_edges())
            win_food    = 0
            win_actions = np.zeros(5, dtype=int)

    return {
        'windows_food':    windows_food,
        'windows_entropy': windows_entropy,
        'windows_edges':   windows_edges,
        'total_food':      sum(windows_food),
        'mean_entropy':    float(np.mean(windows_entropy)),
    }


def run_noise_sweep(seed=42, T_total=5000, window=500,
                    edge_add_probs=(0.01, 0.05, 0.10),
                    activity_noises=(0.00, 0.05, 0.10)):
    """Exp A: 3×3 sweep. Returns dict keyed by (edge_add_prob, activity_noise)."""
    results = {}
    print(f'  {"ep_prob":>8} {"noise":>6} | {"food":>5} {"entropy":>8} {"edges":>6}')
    print('  ' + '-' * 40)
    for ep in edge_add_probs:
        for an in activity_noises:
            d = _s11_continuous(seed, T_total, window, ep, an)
            results[(ep, an)] = d
            print(f'  {ep:8.2f} {an:6.2f} | {d["total_food"]:5d} '
                  f'{d["mean_entropy"]:8.3f} {d["windows_edges"][-1]:6d}')

    # Best condition by total food (break ties by entropy)
    best = max(results, key=lambda k: (results[k]['total_food'], results[k]['mean_entropy']))
    results['best_condition'] = best
    print(f'  → Best: edge_add_prob={best[0]}, activity_noise={best[1]} '
          f'(food={results[best]["total_food"]}, entropy={results[best]["mean_entropy"]:.3f})')
    return results


# ─── Experiment B: Evolution ───────────────────────────────────────────────────

def _s11_evolve_new(seed, n_gen=50, n_agents=10, n_ep=5, n_surv=3,
                    edge_add_prob=0.01, activity_noise=0.0):
    """Evolve new-arch agents with given noise params."""
    rng = np.random.default_rng(seed)
    pop = [(G := _s10_build_graph(rng), _s10_get_W(G)) for _ in range(n_agents)]

    gen_means, gen_bests = [], []

    for gen in range(n_gen):
        fitnesses = []
        for G, W in pop:
            rng_ag = np.random.default_rng(int(rng.integers(0, 2**32)))
            total  = sum(
                _s11_run_ep(G, W,
                            np.random.default_rng(int(rng_ag.integers(0, 2**32))),
                            edge_add_prob, activity_noise)
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


def run_evolution(seed=42, best_noise=(0.10, 0.10),
                  n_gen=50, n_agents=10, n_ep=5, n_surv=3):
    """Exp B: 4 conditions — noise, no-noise, old-arch, random."""
    best_ep, best_an = best_noise

    print(f'  [new arch + noise (ep={best_ep}, noise={best_an})]...')
    noise_data = _s11_evolve_new(seed, n_gen, n_agents, n_ep, n_surv, best_ep, best_an)

    print('  [new arch, no noise — S10 baseline]...')
    no_noise_data = _s11_evolve_new(seed + 1, n_gen, n_agents, n_ep, n_surv, 0.01, 0.0)

    print('  [old arch, co-evo readout — S10 baseline]...')
    old_data = _s10_evolve_old(seed + 2, n_gen, n_agents, n_ep, n_surv)

    print('  [random baseline]...')
    rnd_data = _s10_random_baseline(seed + 3, n_gen, n_agents, n_ep)

    return {
        'noise':      noise_data,
        'no_noise':   no_noise_data,
        'old':        old_data,
        'rnd':        rnd_data,
        'n_gen':      n_gen,
        'best_noise': best_noise,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_noise_sweep(data, fname='images/session_11/results_s11_noise_sweep.png'):
    edge_add_probs  = (0.01, 0.05, 0.10)
    activity_noises = (0.00, 0.05, 0.10)
    best_cond       = data['best_condition']
    window          = 500
    n_windows       = len(data[(edge_add_probs[0], activity_noises[0])]['windows_food'])
    xs              = [window * (i + 1) for i in range(n_windows)]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    fig.suptitle(
        'Session 11 Exp A: Noise Sweep — new arch, T=5000 continuous steps\n'
        'row = edge_add_prob, col = activity_noise  |  gold frame = best condition',
        fontsize=10,
    )

    for ri, ep in enumerate(edge_add_probs):
        for ci, an in enumerate(activity_noises):
            ax  = axes[ri][ci]
            d   = data[(ep, an)]
            is_best = (ep, an) == best_cond

            # Food count bars
            ax2 = ax.twinx()
            ax.bar(xs, d['windows_food'], width=window * 0.6,
                   color='gold', alpha=0.6, label='Food')
            ax.set_ylabel('Food eaten', fontsize=7, color='goldenrod')
            ax.tick_params(axis='y', labelcolor='goldenrod', labelsize=6)

            # Action entropy line
            ax2.plot(xs, d['windows_entropy'], color='steelblue',
                     marker='o', markersize=3, linewidth=1.5, label='Entropy')
            ax2.axhline(np.log2(5), color='steelblue', linestyle='--',
                        linewidth=0.8, alpha=0.5)
            ax2.set_ylim(0, np.log2(5) * 1.15)
            ax2.set_ylabel('Action entropy (bits)', fontsize=7, color='steelblue')
            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=6)

            title = (f'ep={ep}, noise={an}\n'
                     f'food={d["total_food"]} | H={d["mean_entropy"]:.2f}b')
            ax.set_title(title, fontsize=7.5,
                         color='darkred' if is_best else 'black')
            ax.tick_params(axis='x', labelsize=6)

            if is_best:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gold')
                    spine.set_linewidth(2.5)

            if ri == 2:
                ax.set_xlabel('Steps', fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')

    # ── Supplementary heatmap ─────────────────────────────────────────────────
    hm_food    = np.array([[data[(ep, an)]['total_food']
                            for an in activity_noises] for ep in edge_add_probs],
                          dtype=float)
    hm_entropy = np.array([[data[(ep, an)]['mean_entropy']
                            for an in activity_noises] for ep in edge_add_probs])
    hm_edges   = np.array([[data[(ep, an)]['windows_edges'][-1]
                            for an in activity_noises] for ep in edge_add_probs],
                          dtype=float)

    fig2, ax2s = plt.subplots(1, 3, figsize=(13, 3.5))
    fig2.suptitle('Session 11 Exp A: Aggregate metrics heatmap', fontsize=10)

    hm_specs = [
        (hm_food,    'Total Food Eaten',   'YlOrRd'),
        (hm_entropy, 'Mean Action Entropy (bits)', 'Blues'),
        (hm_edges,   'Final Edge Count',   'Greens'),
    ]
    for ax, (mat, title, cmap) in zip(ax2s, hm_specs):
        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels([f'{an}' for an in activity_noises])
        ax.set_yticks(range(3))
        ax.set_yticklabels([f'{ep}' for ep in edge_add_probs])
        ax.set_xlabel('activity_noise', fontsize=8)
        ax.set_ylabel('edge_add_prob', fontsize=8)
        ax.set_title(title, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for (r, c), val in np.ndenumerate(mat):
            ax.text(c, r, f'{val:.1f}' if isinstance(val, float) else f'{int(val)}',
                    ha='center', va='center', fontsize=8,
                    color='white' if mat[r, c] > mat.max() * 0.7 else 'black')
        # Mark best
        bi = list(edge_add_probs).index(best_cond[0])
        bj = list(activity_noises).index(best_cond[1])
        ax.add_patch(mpatches.FancyBboxPatch((bj - 0.45, bi - 0.45), 0.9, 0.9,
                     boxstyle='round,pad=0.05', linewidth=2.5,
                     edgecolor='gold', facecolor='none'))

    plt.tight_layout()
    hm_fname = fname.replace('.png', '_heatmap.png')
    plt.savefig(hm_fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {hm_fname}')


def plot_evolution(data, fname='images/session_11/results_s11_evolution.png'):
    n_gen     = data['n_gen']
    xs        = np.arange(1, n_gen + 1)
    best_ep, best_an = data['best_noise']

    cond_map = [
        ('noise',    f'New arch + noise\n(ep={best_ep}, noise={best_an})', 'steelblue',   '-'),
        ('no_noise', 'New arch, no noise\n(S10 baseline)',                  'firebrick',   '--'),
        ('old',      'Old arch, co-evo readout\n(S10 baseline)',            'darkorange',  '-'),
        ('rnd',      'Random baseline',                                      'gray',        ':'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Session 11 Exp B: Evolution — noise vs no-noise vs old arch vs random\n'
        f'Best noise: edge_add_prob={best_ep}, activity_noise={best_an}',
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
    print('=== Session 11: 自発的ノイズによる局所最適からの脱出 ===')

    print('\n[Exp A] Noise sweep (9 conditions × T=5000)...')
    expA = run_noise_sweep(seed=42)
    plot_noise_sweep(expA)

    best_noise = expA['best_condition']
    print(f'\n[Exp B] Evolution with best noise {best_noise}...')
    expB = run_evolution(seed=42, best_noise=best_noise)
    for key, label in [('noise', 'Noise'), ('no_noise', 'No-noise'),
                       ('old',   'Old'),   ('rnd',     'Random')]:
        m = expB[key]['gen_means']
        b = expB[key]['gen_bests']
        print(f'  {label:12}: gen1={m[0]:.1f}, gen25={m[24]:.1f}, gen50={m[-1]:.1f} '
              f'| best50={b[-1]:.1f}')
    plot_evolution(expB)

    print('\nDone.')
