"""Session 19: 文脈依存的な行動の創発（再挑戦）

Session 4-7 で繰り返し失敗した問い：
  「北文脈と南文脈で内部活動パターンが異なるか」

Session 18 アーキテクチャで再挑戦：
  Readoutなし（argmax output nodes）
  ep≈0.000 + ar≈0.605（活動依存的な生成）
  睡眠固定（T_consolidation=200）
  ms=1000 で進化した個体

Experiments:
  A  4条件の性能評価（ContextGridWorld, n_agents=10, n_episodes=20）
     evolved_ms1000 / s12_fixed / old_arch / random_baseline
  B  ベスト個体のモードA/B活動パターン分離計測（cosine_distance box plot）
  C  Session 18 evolved vs Session 4-era のスパース性比較
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate, _s10_hebb,
    _N, _K, _N_PROP,
)
from session_12_sleep_consolidation import _s12_consolidation_phase, _BEST_AN
from session_18_ratio_evolution import (
    _s18_hebb, _s18_make_genome, _s18_mutate_genome,
    _ACTIVITY_NOISE, _T_CONSOLIDATION,
    _N_GEN, _N_EP, _N_SURV,
)

_SEED      = 42
_N_AGENTS  = 10
_N_EPISODES = 20

# ContextGridWorld params (from spec)
_CGRID     = 5
_CHP_MAX   = 200
_CHP_DECAY = 1
_CFOOD_VAL = 30
_CRESPAWN  = 40
_CSTEPS    = 500

# S12-fixed params
_S12_EP = 0.1
_S12_AR = 0.0
_S12_AN = _BEST_AN  # 0.05


# ── Episode runners ────────────────────────────────────────────────────────────

def _s19_inp4(row, col, hp, food_avail):
    return np.array([
        col / (_CGRID - 1),
        row / (_CGRID - 1),
        hp  / _CHP_MAX,
        1.0 if food_avail else 0.0,
    ])


def _s19_run_context_ep(G, W, edge_add_prob, activity_ratio, rng,
                         mode=None, activity_noise=_ACTIVITY_NOISE,
                         T_consolidation=_T_CONSOLIDATION,
                         record_activity=False):
    """New-arch ContextGridWorld episode (S18 Hebb + sleep).

    Returns (steps_survived, food_count, mode, act_records or None).
    act_records: list of full _N-activity vectors per step.
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos   = (0, 0) if mode == 'A' else (4, 4)   # (row, col)
    food_avail = True
    food_timer = 0

    activity = np.zeros(_N)
    row, col = 2, 2
    hp       = 100
    steps    = 0
    food     = 0
    records  = [] if record_activity else None

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        inp4 = _s19_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        if record_activity:
            records.append(activity.copy())

        action = int(np.argmax(activity[4:9]))

        fr, fc = food_pos
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

        hp -= _CHP_DECAY
        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, activity, rng, edge_add_prob, activity_ratio)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food, mode, records


def _s19_run_context_ep_old_arch(G, W, readout_w, rng,
                                   mode=None, activity_noise=_S12_AN,
                                   T_consolidation=_T_CONSOLIDATION,
                                   record_activity=False):
    """Old-arch ContextGridWorld episode (softmax readout + standard Hebb)."""
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos   = (0, 0) if mode == 'A' else (4, 4)
    food_avail = True
    food_timer = 0

    activity = np.zeros(_N)
    row, col = 2, 2
    hp       = 100
    steps    = 0
    food     = 0
    records  = [] if record_activity else None

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        inp4 = _s19_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        if activity_noise > 0.0:
            activity = np.clip(activity + rng.normal(0, activity_noise, _N), 0.0, 1.0)

        if record_activity:
            records.append(activity.copy())

        scores = activity[4:20] @ readout_w
        x      = scores - np.max(scores)
        p      = np.exp(x)
        p     /= p.sum()
        action = int(rng.choice(5, p=p))

        fr, fc = food_pos
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

        hp -= _CHP_DECAY
        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s10_hebb(G, W, activity, rng)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)
    return steps, food, mode, records


def _s19_run_context_ep_random(rng, mode=None):
    """Random agent in ContextGridWorld. Returns (steps, food, mode)."""
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos   = (0, 0) if mode == 'A' else (4, 4)
    food_avail = True
    food_timer = 0

    row, col = 2, 2
    hp       = 100
    steps    = 0
    food     = 0

    for step in range(_CSTEPS):
        if hp <= 0:
            break

        action = int(rng.integers(0, 5))
        fr, fc = food_pos
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

        hp -= _CHP_DECAY
        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

    return steps, food, mode


# ── S18 ms=1000 genome re-evolution ───────────────────────────────────────────

def _s19_get_evolved_ms1000_genome(seed=_SEED):
    """Re-run S18 evolution for ms=1000 (seed=42+1000=1042, deterministic).

    Same logic as session_18_ratio_evolution.run_exp_a_genome_evolution
    for max_steps=1000 only.
    """
    from session_18_ratio_evolution import _s18_run_ep

    max_steps  = 1000
    n_agents   = 10
    rng        = np.random.default_rng(seed + max_steps)
    pop        = [_s18_make_genome(rng) for _ in range(n_agents)]

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

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < n_agents:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s18_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            best = pop[0]
            print(f'    gen {gen+1:3d}: best={max(fitnesses):7.1f}  '
                  f'ep={best["edge_add_prob"]:.3f}  ar={best["activity_ratio"]:.3f}')

    return pop[0]


# ── Experiment A: Performance ──────────────────────────────────────────────────

def run_exp_a_performance(evolved_genome, seed=_SEED):
    """4 conditions × n_agents × n_episodes on ContextGridWorld.

    Returns dict keyed by condition:
      { 'steps': ndarray, 'food': ndarray, 'modes': list, 'success': ndarray }
    """
    rng     = np.random.default_rng(seed + 19000)
    results = {}

    cond_list = ['evolved_ms1000', 's12_fixed', 'old_arch', 'random']

    for cond in cond_list:
        print(f'\n  [Exp A: {cond}]')
        all_steps, all_food, all_modes, all_success = [], [], [], []

        for ag in range(_N_AGENTS):
            # Build agent
            if cond == 'evolved_ms1000':
                G   = evolved_genome['G'].copy()
                W   = _s10_get_W(G)
                ep_ = evolved_genome['edge_add_prob']
                ar_ = evolved_genome['activity_ratio']
            elif cond == 's12_fixed':
                init_rng = np.random.default_rng(seed + 19001 + ag)
                G  = _s10_build_graph(init_rng)
                W  = _s10_get_W(G)
                ep_ = _S12_EP
                ar_ = _S12_AR
            elif cond == 'old_arch':
                init_rng = np.random.default_rng(seed + 19100 + ag)
                G        = _s10_build_graph(init_rng)
                W        = _s10_get_W(G)
                readout_ = init_rng.standard_normal((16, 5)) * 0.5

            for ep_idx in range(_N_EPISODES):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))

                if cond == 'evolved_ms1000':
                    s, f, m, _ = _s19_run_context_ep(
                        G, W, ep_, ar_, ep_rng)
                elif cond == 's12_fixed':
                    s, f, m, _ = _s19_run_context_ep(
                        G, W, ep_, ar_, ep_rng, activity_noise=_S12_AN)
                elif cond == 'old_arch':
                    s, f, m, _ = _s19_run_context_ep_old_arch(
                        G, W, readout_, ep_rng)
                else:  # random
                    s, f, m = _s19_run_context_ep_random(ep_rng)

                all_steps.append(s)
                all_food.append(f)
                all_modes.append(m)
                all_success.append(f >= 1)

        modes_arr = np.array(all_modes)
        succ_arr  = np.array(all_success)
        acc_a     = float(np.mean(succ_arr[modes_arr == 'A'])) if np.any(modes_arr == 'A') else 0.0
        acc_b     = float(np.mean(succ_arr[modes_arr == 'B'])) if np.any(modes_arr == 'B') else 0.0
        print(f'    mean_steps={np.mean(all_steps):.1f}  mean_food={np.mean(all_food):.2f}  '
              f'acc_A={acc_a:.3f}  acc_B={acc_b:.3f}')

        results[cond] = {
            'steps':   np.array(all_steps),
            'food':    np.array(all_food),
            'modes':   all_modes,
            'success': succ_arr,
        }

    return results


# ── Experiment B: Context separation ──────────────────────────────────────────

def run_exp_b_context_separation(evolved_genome, seed=_SEED, n_ep_per_mode=10):
    """Record output-node activity (nodes 4-8) per mode, compute cosine distance.

    n_agents trials; each trial: n_ep_per_mode episodes per mode.
    Returns { mean_A, mean_B, ep_means_A, ep_means_B, cos_distances }.
    """
    print('\n  [Exp B: context separation]')
    rng        = np.random.default_rng(seed + 19200)
    ep_means_A = []
    ep_means_B = []
    cos_distances = []

    for trial in range(_N_AGENTS):
        G   = evolved_genome['G'].copy()
        W   = _s10_get_W(G)
        ep_ = evolved_genome['edge_add_prob']
        ar_ = evolved_genome['activity_ratio']

        trial_means_A, trial_means_B = [], []

        for ei in range(n_ep_per_mode * 2):
            mode   = 'A' if ei % 2 == 0 else 'B'
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, recs = _s19_run_context_ep(
                G, W, ep_, ar_, ep_rng, mode=mode, record_activity=True)

            if recs:
                arr      = np.array(recs)        # (steps, _N)
                mean_out = arr[:, 4:9].mean(axis=0)   # output nodes
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
    print(f'    mean_A={np.round(mean_A, 3)}')
    print(f'    mean_B={np.round(mean_B, 3)}')

    return {
        'mean_A':         mean_A,
        'mean_B':         mean_B,
        'ep_means_A':     ep_means_A,
        'ep_means_B':     ep_means_B,
        'cos_distances':  cos_distances,
    }


# ── Experiment C: Sparsity ─────────────────────────────────────────────────────

def run_exp_c_sparsity(evolved_genome, seed=_SEED, n_ep=10):
    """Compare sparsity: S18 evolved vs S4-era (old arch, no sleep).

    Sparsity = active node count (activity > 0.1) among nodes 4-19 per step.
    Returns { 's18': {...}, 's4': {...} }.
    """
    print('\n  [Exp C: sparsity]')
    rng = np.random.default_rng(seed + 19400)

    def _collect(G, W, runner_fn, runner_kw, n_ep):
        active_A, active_B = [], []
        var_A,    var_B    = [], []
        for ei in range(n_ep * 2):
            mode   = 'A' if ei % 2 == 0 else 'B'
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, recs = runner_fn(G, W, ep_rng=ep_rng, mode=mode, **runner_kw)
            if recs:
                arr = np.array(recs)          # (steps, _N)
                internal = arr[:, 4:]         # nodes 4-19, shape (steps, 16)
                per_step_active = (internal > 0.1).sum(axis=1)   # (steps,)
                mean_per_node   = internal.mean(axis=0)           # (16,)
                if mode == 'A':
                    active_A.extend(per_step_active.tolist())
                    var_A.append(float(np.var(mean_per_node)))
                else:
                    active_B.extend(per_step_active.tolist())
                    var_B.append(float(np.var(mean_per_node)))
        return {'active_A': active_A, 'active_B': active_B,
                'var_A': var_A, 'var_B': var_B}

    # S18 evolved
    G_s18 = evolved_genome['G'].copy()
    W_s18 = _s10_get_W(G_s18)

    def runner_s18(G, W, ep_rng, mode, **kw):
        return _s19_run_context_ep(
            G, W, evolved_genome['edge_add_prob'],
            evolved_genome['activity_ratio'],
            ep_rng, mode=mode, record_activity=True)

    s18_data = _collect(G_s18, W_s18, runner_s18, {}, n_ep)

    # S4-era: old arch, no sleep consolidation
    rng_s4   = np.random.default_rng(seed + 19500)
    G_s4     = _s10_build_graph(rng_s4)
    W_s4     = _s10_get_W(G_s4)
    rw_s4    = rng_s4.standard_normal((16, 5)) * 0.5

    def runner_s4(G, W, ep_rng, mode, **kw):
        return _s19_run_context_ep_old_arch(
            G, W, rw_s4, ep_rng, mode=mode,
            T_consolidation=0, record_activity=True)

    s4_data = _collect(G_s4, W_s4, runner_s4, {}, n_ep)

    print(f'    S18 active/step: A={np.mean(s18_data["active_A"]):.2f}  '
          f'B={np.mean(s18_data["active_B"]):.2f}')
    print(f'    S4  active/step: A={np.mean(s4_data["active_A"]):.2f}  '
          f'B={np.mean(s4_data["active_B"]):.2f}')

    return {'s18': s18_data, 's4': s4_data}


# ── Plotting ──────────────────────────────────────────────────────────────────

_COND_ORDER  = ['evolved_ms1000', 's12_fixed', 'old_arch', 'random']
_COND_COLORS = {
    'evolved_ms1000': '#f58231',
    's12_fixed':      'steelblue',
    'old_arch':       'mediumseagreen',
    'random':         'gray',
}
_COND_LABELS = {
    'evolved_ms1000': 'S18 evolved\n(ep≈0,ar≈0.6)',
    's12_fixed':      'S12 fixed\n(ep=0.1,ar=0)',
    'old_arch':       'Old arch\n(Readout)',
    'random':         'Random\nbaseline',
}


def plot_exp_a_performance(data, fname='images/session_19/results_s19_performance.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Session 19 Exp A: ContextGridWorld Performance\n'
        f'4 conditions × n_agents={_N_AGENTS} × n_episodes={_N_EPISODES}',
        fontsize=11,
    )

    # Box plot: survival steps
    ax = axes[0]
    bp = ax.boxplot(
        [data[k]['steps'].tolist() for k in _COND_ORDER],
        patch_artist=True, widths=0.5,
    )
    for patch, key in zip(bp['boxes'], _COND_ORDER):
        patch.set_facecolor(_COND_COLORS[key])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([_COND_LABELS[k] for k in _COND_ORDER], fontsize=9)
    ax.set_ylabel('Survival Steps')
    ax.set_title(f'Survival Steps per Episode  (max={_CSTEPS})')
    ax.grid(True, alpha=0.3, axis='y')

    # Bar: mode A / B accuracy
    ax  = axes[1]
    xs  = np.arange(len(_COND_ORDER))
    w   = 0.35
    for gi, key in enumerate(_COND_ORDER):
        d         = data[key]
        modes_arr = np.array(d['modes'])
        succ_arr  = d['success']
        acc_a = float(np.mean(succ_arr[modes_arr == 'A'])) if np.any(modes_arr == 'A') else 0.0
        acc_b = float(np.mean(succ_arr[modes_arr == 'B'])) if np.any(modes_arr == 'B') else 0.0
        ax.bar(xs[gi] - w/2, acc_a, width=w, color=_COND_COLORS[key], alpha=0.9,
               label='Mode A (NW)' if gi == 0 else '')
        ax.bar(xs[gi] + w/2, acc_b, width=w, color=_COND_COLORS[key], alpha=0.45,
               label='Mode B (SE)' if gi == 0 else '')
        ax.text(xs[gi] - w/2, acc_a + 0.02, f'{acc_a:.2f}', ha='center', fontsize=8)
        ax.text(xs[gi] + w/2, acc_b + 0.02, f'{acc_b:.2f}', ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([_COND_LABELS[k] for k in _COND_ORDER], fontsize=9)
    ax.set_ylabel('Accuracy  (food_eaten ≥ 1)')
    ax.set_ylim(0, 1.2)
    ax.set_title('Mode A (NW) / Mode B (SE) Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_context_separation(
        data, fname='images/session_19/results_s19_context_separation.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Session 19 Exp B: Context Separation\n'
        'Output nodes (node4-8) mean activity: Mode A (NW food) vs Mode B (SE food)',
        fontsize=11,
    )

    action_names = ['North\n(n4)', 'South\n(n5)', 'West\n(n6)',
                    'East\n(n7)', 'Eat\n(n8)']

    # Heatmap: 2 modes × 5 output nodes
    ax      = axes[0]
    hm_data = np.array([data['mean_A'], data['mean_B']])
    vmax    = max(hm_data.max(), 0.01)
    im      = ax.imshow(hm_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(action_names, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Mode A\n(NW food)', 'Mode B\n(SE food)'], fontsize=9)
    ax.set_title('Mean Output Node Activity per Mode')
    for (r, c), val in np.ndenumerate(hm_data):
        ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                fontsize=9, color='black')
    plt.colorbar(im, ax=ax)

    # Box + scatter: cosine distance distribution
    ax     = axes[1]
    cos_d  = data['cos_distances']
    rng_jit = np.random.default_rng(0)
    if cos_d:
        bp = ax.boxplot([cos_d], patch_artist=True,
                        boxprops=dict(facecolor='#f58231', alpha=0.7))
        jitter = rng_jit.uniform(-0.15, 0.15, len(cos_d))
        ax.scatter(1 + jitter, cos_d, color='darkorange', alpha=0.7, s=35, zorder=3)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
                   label='No separation (cos_dist=0)')
        ax.set_ylabel('Cosine Distance (mode A vs B)')
        ax.set_xticks([1])
        ax.set_xticklabels(['evolved_ms1000'])
        ax.set_title(
            f'Cosine Distance Distribution\n'
            f'n_trials={len(cos_d)}  '
            f'mean={np.mean(cos_d):.4f} ± {np.std(cos_d):.4f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_sparsity(data, fname='images/session_19/results_s19_sparsity.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Session 19 Exp C: Sparsity Comparison\n'
        'Session 18 evolved (ar≈0.6) vs Session 4-era (old arch, no sleep)',
        fontsize=11,
    )

    # Box: active node count per step (nodes 4-19)
    ax = axes[0]
    s18_all = data['s18']['active_A'] + data['s18']['active_B']
    s4_all  = data['s4']['active_A']  + data['s4']['active_B']
    bp      = ax.boxplot([s18_all, s4_all], patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#f58231', 'steelblue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['S18 evolved\n(ep≈0, ar≈0.6)', 'S4-era\n(old arch)'], fontsize=9)
    ax.set_ylabel('Active Nodes per Step  (activity > 0.1, nodes 4-19)')
    ax.set_title('Active Node Count Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    for i, arr in enumerate([s18_all, s4_all], 1):
        med = np.median(arr)
        ax.text(i, med + 0.1, f'med={med:.1f}', ha='center', fontsize=8)

    # Bar: activity variance A vs B (shows mode-specific differentiation)
    ax  = axes[1]
    def _safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    cats = ['S18\nMode A', 'S18\nMode B', 'S4\nMode A', 'S4\nMode B']
    vals = [
        _safe_mean(data['s18']['var_A']), _safe_mean(data['s18']['var_B']),
        _safe_mean(data['s4']['var_A']),  _safe_mean(data['s4']['var_B']),
    ]
    colors = ['#f58231', '#f58231', 'steelblue', 'steelblue']
    alphas = [0.95, 0.5, 0.95, 0.5]
    for i, (cat, val, col, alpha) in enumerate(zip(cats, vals, colors, alphas)):
        ax.bar(i, val, color=col, alpha=alpha, edgecolor='white')
        ax.text(i, val + max(vals) * 0.02, f'{val:.4f}', ha='center', fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel('Mean Activity Variance (across output nodes per episode)')
    ax.set_title('Activity Variance: Mode A vs B\n'
                 '(higher = more node differentiation)')
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

    print('=== Session 19: 文脈依存的な行動の創発（再挑戦） ===')

    print('\n[S18 ms=1000 genome re-evolution  (seed=42+1000, 50 gen)]')
    evolved = _s19_get_evolved_ms1000_genome(seed=_SEED)
    print(f'  → ep={evolved["edge_add_prob"]:.3f}  '
          f'ar={evolved["activity_ratio"]:.3f}  '
          f'edges={evolved["G"].number_of_edges()}')

    print(f'\n[Exp A] Performance  '
          f'(n_agents={_N_AGENTS}, n_episodes={_N_EPISODES}, max_steps={_CSTEPS})')
    exp_a = run_exp_a_performance(evolved, seed=_SEED)

    print('\n  Summary:')
    print(f'  {"condition":>20}  {"mean_steps":>10}  {"mean_food":>9}  '
          f'{"acc_A":>6}  {"acc_B":>6}')
    print('  ' + '─' * 60)
    for key in _COND_ORDER:
        d         = exp_a[key]
        modes_arr = np.array(d['modes'])
        succ_arr  = d['success']
        acc_a     = float(np.mean(succ_arr[modes_arr == 'A'])) if np.any(modes_arr == 'A') else 0.0
        acc_b     = float(np.mean(succ_arr[modes_arr == 'B'])) if np.any(modes_arr == 'B') else 0.0
        print(f'  {key:>20}  {np.mean(d["steps"]):10.1f}  {np.mean(d["food"]):9.2f}  '
              f'{acc_a:6.3f}  {acc_b:6.3f}')

    plot_exp_a_performance(exp_a)

    print(f'\n[Exp B] Context separation  '
          f'(n_agents={_N_AGENTS}, n_ep_per_mode=10)')
    exp_b = run_exp_b_context_separation(evolved, seed=_SEED, n_ep_per_mode=10)
    plot_exp_b_context_separation(exp_b)

    print(f'\n[Exp C] Sparsity  (n_ep=10)')
    exp_c = run_exp_c_sparsity(evolved, seed=_SEED, n_ep=10)
    plot_exp_c_sparsity(exp_c)

    print('\nDone.')
