"""Session 6: Dynamic (E/I) vs static (low-weight) inhibition."""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from world_base import ContextGridWorld, _run_context_episode

# ─── Session 6: Dynamic (E/I) vs static (low-weight) inhibition ──────────────
#
# Root cause identified in Session 5 sweep:
#   anatomy edge weight=0.5 → tanh(0.5×0.8)=0.38 < Hebbian threshold 0.5
#   → anatomy edges not reinforced → removed by decay within Phase 1.
#
# Fix: anatomy edges marked fixed=True, weight=1.0 (immune to Hebbian decay).
#   tanh(1.0×0.8)=0.664 > 0.5 → anatomy edges ARE reinforced during training.
#   Probe measures loop_north(6,7,8) / loop_south(9,10,11) mean activity
#   instead of stimulus nodes 1/2 (which had arbitrary random connections).

_S6_EI_GENOME  = {'ei_threshold': 0.821, 'recovery_ratio': 0.503, 'recovery_delay': 0}
_S6_BEST_COND  = {'density': 0.2, 'init_weight': 0.05, 'lr': 0.05}
_S6_LOOP_NORTH = [6, 7, 8]
_S6_LOOP_SOUTH = [9, 10, 11]
_S6_LOOP_FOOD  = [3, 4, 5]


def _build_s6_graph(N, rng, initial_density, initial_weight):
    """Association graph for Session 6 with protected anatomy edges.

    Stimulus: 0=food, 1=north, 2=south
    Loop Food:  3→4→5→3   (anatomy: 0→3)
    Loop North: 6→7→8→6   (anatomy: 1→6)
    Loop South: 9→10→11→9 (anatomy: 2→9)
    Background: 12..N-1

    Anatomy edges: fixed=True, weight=1.0 — immune to Hebbian decay.
    Background edges: fixed=False, weight=initial_weight.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < initial_density:
                G.add_edge(i, j, weight=float(initial_weight), fixed=False)
    for s, d in [(0, 3), (1, 6), (2, 9)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    for s, d in [(3, 4), (4, 5), (5, 3)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    for s, d in [(6, 7), (7, 8), (8, 6)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    for s, d in [(9, 10), (10, 11), (11, 9)]:
        G.add_edge(s, d, weight=1.0, fixed=True)
    return G


def _s6_make_ei_state(N, ei_window=100):
    node_type  = np.ones(N, dtype=float)
    act_hist   = [deque(maxlen=ei_window) for _ in range(N)]
    rec_timers = np.zeros(N, dtype=int)
    return node_type, act_hist, rec_timers


def _s6_step(G, activity, ext, N, use_ei, node_type):
    """Propagate network one step (no Hebbian). Modifies activity in-place."""
    new_act = np.zeros(N)
    for i in range(N):
        if i in ext:
            new_act[i] = ext[i]
        elif not use_ei or node_type[i] == 1:
            s = sum(G[j][i]['weight'] * activity[j]
                    for j in G.predecessors(i)
                    if not use_ei or node_type[j] == 1)
            new_act[i] = np.tanh(s)
        else:
            new_act[i] = activity[i] * 0.9   # inhibitory decay
    activity[:] = new_act


def _s6_hebb(G, activity, ext, N, use_ei, node_type, act_hist, rec_timers,
              lr, init_w, rng, ei_thr, rec_ratio, rec_delay):
    """Hebbian update + optional E/I switching.  Called every K steps.
    Fixed-anatomy edges (fixed=True) are skipped — they never decay."""
    to_rm = []
    for i, j, d in list(G.edges(data=True)):
        if d.get('fixed', False):
            continue          # protect anatomy edges
        if use_ei and (node_type[i] != 1 or node_type[j] != 1):
            continue
        w = d['weight']
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += lr
        w -= 0.01
        if w < 0.01:
            to_rm.append((i, j))
        else:
            G[i][j]['weight'] = min(w, 1.0)
    G.remove_edges_from(to_rm)
    existing = set(G.edges())
    for i in range(N):
        for j in range(N):
            if i != j and (i, j) not in existing and rng.random() < 0.005:
                G.add_edge(i, j, weight=float(init_w), fixed=False)

    if not use_ei:
        return
    for i in range(N):
        act_hist[i].append(float(activity[i]))
        if node_type[i] == -1:
            rec_timers[i] += 1
    recent = {i: float(np.mean(act_hist[i])) if act_hist[i] else 0.0
              for i in range(N)}
    exc_cands = [i for i in range(N)
                 if i not in ext and node_type[i] == 1 and recent[i] > ei_thr]
    if exc_cands:
        sw = max(exc_cands, key=lambda i: recent[i])
        node_type[sw] = -1
        rec_timers[sw] = 0
    inh_cands = [i for i in range(N)
                 if node_type[i] == -1
                 and recent[i] < ei_thr * rec_ratio
                 and rec_timers[i] >= rec_delay]
    if inh_cands:
        sw = min(inh_cands, key=lambda i: recent[i])
        node_type[sw] = 1
        rec_timers[sw] = 0


def _s6_probe(G, activity_snap, N, T_probe=100):
    """Probe with food-only (node0=0.8), frozen graph, pure propagation.
    Returns (mean_loop_north, mean_loop_south) averaged over T_probe steps."""
    act = activity_snap.copy()
    ln_acc, ls_acc = 0.0, 0.0
    for _ in range(T_probe):
        new_act = np.zeros(N)
        new_act[0] = 0.8
        for i in range(1, N):
            s = sum(G[j][i]['weight'] * act[j] for j in G.predecessors(i))
            new_act[i] = np.tanh(s)
        act = new_act
        ln_acc += float(np.mean(act[_S6_LOOP_NORTH]))
        ls_acc += float(np.mean(act[_S6_LOOP_SOUTH]))
    return ln_acc / T_probe, ls_acc / T_probe


def run_ei_vs_static_experiments(N=20, seed=42, K=5, T_probe=100, probe_interval=10):
    """Experiments A & B: overwrite resistance and association switch speed.

    Two conditions — pure Hebbian ('no_ei') vs E/I isolation ('ei') —
    both starting from density=0.2, init_weight=0.05, hebbian_lr=0.05.

    Exp A: train 5 phases, probe (food-only) after each phase.
    Exp B: during Phase 4, probe every probe_interval steps; record when
           mean_south first exceeds mean_north (association switch step).
    """
    density   = _S6_BEST_COND['density']
    init_w    = _S6_BEST_COND['init_weight']
    lr        = _S6_BEST_COND['lr']
    ei_thr    = _S6_EI_GENOME['ei_threshold']
    rec_ratio = _S6_EI_GENOME['recovery_ratio']
    rec_delay = _S6_EI_GENOME['recovery_delay']

    phases = [
        ({1: 0.8},         500,  'North solo'),
        ({0: 0.8},         500,  'Food solo'),
        ({0: 0.8, 1: 0.8}, 500,  'North+Food'),
        ({0: 0.8, 2: 0.8}, 500,  'South+Food'),
        ({0: 0.8, 2: 0.8}, 1000, 'South+Food ×2'),
    ]

    print('=== Session 6: E/I vs static (Exp A & B) ===')
    results = {}

    for cond in ['no_ei', 'ei']:
        use_ei = (cond == 'ei')
        rng = np.random.default_rng(seed)
        G   = _build_s6_graph(N, rng, density, init_w)
        act = np.zeros(N)
        node_type, act_hist, rec_timers = _s6_make_ei_state(N)

        probe_A  = []
        exp_b    = None

        for p_idx, (ext, T, pname) in enumerate(phases):
            if p_idx == 3:    # Phase 4: Exp B — probe at each checkpoint
                b_steps, b_north, b_south = [], [], []
                first_switch = None
                for t in range(T):
                    _s6_step(G, act, ext, N, use_ei, node_type)
                    if (t + 1) % K == 0:
                        _s6_hebb(G, act, ext, N, use_ei, node_type, act_hist,
                                 rec_timers, lr, init_w, rng,
                                 ei_thr, rec_ratio, rec_delay)
                    if (t + 1) % probe_interval == 0:
                        mn, ms = _s6_probe(G, act, N, T_probe)
                        b_steps.append(t + 1)
                        b_north.append(mn)
                        b_south.append(ms)
                        if first_switch is None and ms > mn:
                            first_switch = t + 1
                exp_b = {
                    'steps': b_steps,
                    'north': b_north,
                    'south': b_south,
                    'first_switch': first_switch,
                }
            else:
                for t in range(T):
                    _s6_step(G, act, ext, N, use_ei, node_type)
                    if (t + 1) % K == 0:
                        _s6_hebb(G, act, ext, N, use_ei, node_type, act_hist,
                                 rec_timers, lr, init_w, rng,
                                 ei_thr, rec_ratio, rec_delay)

            mn, ms = _s6_probe(G, act, N, T_probe)
            probe_A.append({'phase': pname, 'mean_north': mn, 'mean_south': ms})
            print(f'  [{cond}] {pname}: loop_north={mn:.4f}  loop_south={ms:.4f}')

        results[cond] = {
            'probe_A': probe_A,
            'exp_b': exp_b,
            'G_final': G.copy(),
            'activity_final': act.copy(),
        }

    return results


def run_ei_vs_static_context(expt_data, N=20, seed=42, n_episodes=10, T_episode=100):
    """Experiment C: context routing test using the Phase5-trained networks.

    Present two context patterns alternately:
      Mode A: {0: 0.8, 1: 0.4}  — food + NW-like hint
      Mode B: {0: 0.8, 2: 0.4}  — food + SE-like hint

    Measure whether loop_north (6,7,8) activates more for mode A and
    loop_south (9,10,11) activates more for mode B.
    """
    loop_north = [6, 7, 8]
    loop_south = [9, 10, 11]
    internal   = list(range(3, N))

    print('\n=== Session 6: E/I vs static (Exp C) ===')
    results = {}

    for cond, cdata in expt_data.items():
        G        = cdata['G_final']
        act_init = cdata['activity_final']

        a_north_list, a_south_list = [], []
        b_north_list, b_south_list = [], []
        mean_A_vecs, mean_B_vecs   = [], []

        rng_ep = np.random.default_rng(seed + 70000)
        modes  = ['A'] * (n_episodes // 2) + ['B'] * (n_episodes - n_episodes // 2)
        rng_ep.shuffle(modes)

        for mode in modes:
            act = act_init.copy()
            ext = {0: 0.8, 1: 0.4} if mode == 'A' else {0: 0.8, 2: 0.4}
            step_vecs = []
            for _ in range(T_episode):
                new_act = np.zeros(N)
                for i in range(N):
                    if i in ext:
                        new_act[i] = ext[i]
                    else:
                        s = sum(G[j][i]['weight'] * act[j]
                                for j in G.predecessors(i))
                        new_act[i] = np.tanh(s)
                act = new_act
                step_vecs.append(act[internal].copy())

            mean_vec = np.mean(step_vecs, axis=0)
            # indices into internal vector: loop_north[k]-3, loop_south[k]-3
            mn = float(np.mean(mean_vec[[n - 3 for n in loop_north]]))
            ms = float(np.mean(mean_vec[[n - 3 for n in loop_south]]))

            if mode == 'A':
                a_north_list.append(mn); a_south_list.append(ms)
                mean_A_vecs.append(mean_vec)
            else:
                b_north_list.append(mn); b_south_list.append(ms)
                mean_B_vecs.append(mean_vec)

        acc_A = float(np.mean(np.array(a_north_list) > np.array(a_south_list)))
        acc_B = float(np.mean(np.array(b_south_list) > np.array(b_north_list)))

        mu_A = np.mean(mean_A_vecs, axis=0) if mean_A_vecs else np.zeros(len(internal))
        mu_B = np.mean(mean_B_vecs, axis=0) if mean_B_vecs else np.zeros(len(internal))
        norm = np.linalg.norm(mu_A) * np.linalg.norm(mu_B)
        cos_dist = 1.0 - float(np.dot(mu_A, mu_B)) / (norm + 1e-10)

        print(f'  [{cond}] acc_A={acc_A:.2f}  acc_B={acc_B:.2f}  cos_dist={cos_dist:.4f}')

        results[cond] = {
            'acc_A': acc_A, 'acc_B': acc_B, 'cos_dist': cos_dist,
            'mean_A_north': float(np.mean(a_north_list)),
            'mean_A_south': float(np.mean(a_south_list)),
            'mean_B_north': float(np.mean(b_north_list)),
            'mean_B_south': float(np.mean(b_south_list)),
            'mean_A_vec': mu_A, 'mean_B_vec': mu_B,
        }

    return results


def plot_ei_vs_static_overwrite(data, fname='images/session_6/results_ei_vs_static_overwrite.png'):
    """Plot experiments A and B."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Session 6 — E/I (dynamic) vs pure Hebbian (static): '
                 'overwrite resistance & switch speed', fontsize=11)

    colors = {'no_ei': {'north': 'steelblue', 'south': 'tomato'},
              'ei':    {'north': '#1f77b4',    'south': '#d62728'}}
    alphas = {'no_ei': 1.0, 'ei': 0.55}
    styles = {'no_ei': '-', 'ei': '--'}
    markers = {'no_ei': 'o', 'ei': 's'}
    phase_names = [r['phase'] for r in data['no_ei']['probe_A']]

    # — Ax 0: Exp A — north probe per phase —
    ax = axes[0]
    for cond in ['no_ei', 'ei']:
        probe = data[cond]['probe_A']
        norths = [r['mean_north'] for r in probe]
        souths = [r['mean_south'] for r in probe]
        xs = list(range(len(norths)))
        kw = dict(ls=styles[cond], marker=markers[cond], ms=6, alpha=alphas[cond])
        ax.plot(xs, norths, color=colors[cond]['north'],
                label=f'{cond} north', **kw)
        ax.plot(xs, souths, color=colors[cond]['south'],
                label=f'{cond} south', **kw)
    ax.axhline(0.3, color='gray', lw=0.8, ls=':', label='thr=0.3')
    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Mean loop activity (food-only probe)')
    ax.set_title('Exp A: loop_north/south after each phase')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # — Ax 1: Exp A — south probe per phase —
    ax = axes[1]
    for cond in ['no_ei', 'ei']:
        probe = data[cond]['probe_A']
        norths = [r['mean_north'] for r in probe]
        souths = [r['mean_south'] for r in probe]
        xs = list(range(len(norths)))
        kw = dict(ls=styles[cond], marker=markers[cond], ms=6, alpha=alphas[cond])
        ax.plot(xs, [s - n for s, n in zip(souths, norths)],
                color=colors[cond]['south'],
                label=f'{cond}  south−north')
    ax.axhline(0, color='gray', lw=1.0, ls=':')
    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('mean_south − mean_north (probe)')
    ax.set_title('Exp A: context conflict (south−north margin)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # — Ax 2: Exp B — Phase 4 switch speed —
    ax = axes[2]
    for cond in ['no_ei', 'ei']:
        b = data[cond]['exp_b']
        diff = [s - n for s, n in zip(b['south'], b['north'])]
        sw = b['first_switch']
        lbl = f'{cond} (switch at step {sw})' if sw else f'{cond} (no switch)'
        ax.plot(b['steps'], diff, ls=styles[cond], color='black' if cond == 'no_ei' else 'crimson',
                lw=1.5, label=lbl)
        if sw:
            ax.axvline(sw, ls=':', lw=0.8,
                       color='black' if cond == 'no_ei' else 'crimson')
    ax.axhline(0, color='gray', lw=1.0, ls='--')
    ax.set_xlabel('Training step in Phase 4 (South+Food)')
    ax.set_ylabel('mean_south − mean_north (probe)')
    ax.set_title('Exp B: association switch speed during Phase 4')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_ei_vs_static_context(data, fname='images/session_6/results_ei_vs_static_context.png'):
    """Plot experiment C: context routing accuracy and loop activity."""
    conds = list(data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Session 6 — Exp C: context routing after Phase5 training', fontsize=11)

    # — Ax 0: accuracy bars —
    ax = axes[0]
    x  = np.arange(len(conds))
    bw = 0.35
    ax.bar(x - bw/2, [data[c]['acc_A'] for c in conds], bw,
           label='Mode A acc\n(north > south)', color='steelblue', alpha=0.85)
    ax.bar(x + bw/2, [data[c]['acc_B'] for c in conds], bw,
           label='Mode B acc\n(south > north)', color='tomato', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy (fraction of trials)')
    ax.set_title('Context routing accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # — Ax 1: loop activity breakdown —
    ax = axes[1]
    bw = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw
    series  = [
        ('A→North', [data[c]['mean_A_north'] for c in conds], 'steelblue'),
        ('A→South', [data[c]['mean_A_south'] for c in conds], 'lightsteelblue'),
        ('B→North', [data[c]['mean_B_north'] for c in conds], 'lightsalmon'),
        ('B→South', [data[c]['mean_B_south'] for c in conds], 'tomato'),
    ]
    for (lbl, vals, col), off in zip(series, offsets):
        ax.bar(x + off, vals, bw, label=lbl, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel('Mean loop node activity')
    ax.set_title('Loop activity by mode')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # — Ax 2: cosine distance —
    ax = axes[2]
    cos_vals = [data[c]['cos_dist'] for c in conds]
    bars = ax.bar(x, cos_vals, color=['steelblue', 'tomato'], alpha=0.85, width=0.5)
    for bar, val in zip(bars, cos_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel('Cosine distance (A vs B internal activity)')
    ax.set_title('Context separation (cosine dist)')
    ax.set_ylim(0, max(max(cos_vals) * 1.3, 0.05))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f'Saved {fname}')


