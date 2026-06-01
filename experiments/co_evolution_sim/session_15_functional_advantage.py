"""Session 15: 構造的優位性の機能的検証

Question: ablation耐性の高い構造（New-Surv型）は
         環境変化・ダメージ・長期エピソードで
         実際に機能的優位性を持つか？

Experiments (priority order):
  B  Long episode:      short (500×5ep), medium (500×20ep), long (2000×5ep)
  C  Environment shift: food position shift at step=250, 3 shift strengths
  A  Edge damage:       random edge removal at rates [0.0, 0.1, 0.3, 0.5]

Evolved individuals reused via collect_all(seed=42) — same seed as Session 12-13
reproduces the same evolved topologies. No new evolution experiments.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from session_10_embodied_output import (
    _s10_propagate, _s10_inp4, _s10_softmax_sample, _s10_hebb,
    _N, _K, _GRID, _HP_MAX, _HP_DECAY,
    _FOOD_VAL, _RESPAWN, _FOOD_POS, _N_PROP, _MAX_STEPS,
)
from session_11_noise_escape import _s11_hebb, _s11_entropy
from session_12_sleep_consolidation import _s12_consolidation_phase, _BEST_EP, _BEST_AN
from session_13_anatomy import collect_all

_SEED = 42

_EXP_B_CONDITIONS = {
    'short':  {'max_steps':  500, 'n_ep':  5},
    'medium': {'max_steps':  500, 'n_ep': 20},
    'long':   {'max_steps': 2000, 'n_ep':  5},
}

_EXP_C_CONDITIONS = [
    {'name': 'full_reversal', 'before': [(0, 0), (4, 4)], 'after': [(0, 4), (4, 0)]},
    {'name': 'minor_shift',   'before': [(0, 0), (4, 4)], 'after': [(1, 1), (3, 3)]},
    {'name': 'no_shift',      'before': [(0, 0), (4, 4)], 'after': [(0, 0), (4, 4)]},
]
_SHIFT_STEP = 250

_DAMAGE_RATES = [0.0, 0.1, 0.3, 0.5]

_ACTION_NAMES = ['North', 'South', 'West', 'East', 'Eat']

_GRP_NEW = 'new_surv'
_GRP_OLD = 'old_surv'
_GRP_RND = 'random'
_GRP_LABELS = {_GRP_NEW: 'New-Surv', _GRP_OLD: 'Old-Surv', _GRP_RND: 'Random'}
_GRP_COLORS = {_GRP_NEW: 'steelblue', _GRP_OLD: 'darkorange', _GRP_RND: 'gray'}


# ── World step with configurable food positions ────────────────────────────────

def _s15_world_step(row, col, action, food_avail, hp, food_pos):
    ate = -1
    if   action == 0: row = max(0, row - 1)
    elif action == 1: row = min(_GRID - 1, row + 1)
    elif action == 2: col = max(0, col - 1)
    elif action == 3: col = min(_GRID - 1, col + 1)
    elif action == 4:
        for idx, (fr, fc) in enumerate(food_pos):
            if row == fr and col == fc and food_avail[idx]:
                hp  = min(_HP_MAX, hp + _FOOD_VAL)
                ate = idx
                break
    return row, col, hp, ate


# ── Episode runners ────────────────────────────────────────────────────────────

def _s15_run_ep_new(G, W, rng, max_steps=_MAX_STEPS, T_consol=200,
                    food_pos=None, shift_step=None, food_pos_after=None):
    """New-arch episode (noise + Hebbian + optional consolidation) with tracking.

    Modifies G and W in place.
    Returns (steps, food_total, entropy, food_pre, food_post, act_pre, act_post).
    act_pre / act_post: length-5 int arrays of action counts before/after shift_step.
    """
    fp       = list(_FOOD_POS) if food_pos is None else list(food_pos)
    fp_after = list(fp)        if food_pos_after is None else list(food_pos_after)
    s_step   = max_steps + 1   if shift_step is None else shift_step

    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    food_pre   = 0
    food_post  = 0
    act_pre    = np.zeros(5, dtype=int)
    act_post   = np.zeros(5, dtype=int)
    cur_fp     = fp

    for step in range(max_steps):
        if hp <= 0:
            break

        if step == s_step:
            cur_fp     = fp_after
            food_avail = [True, True]
            food_timer = [0, 0]

        inp4 = _s10_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)
        activity = np.clip(activity + rng.normal(0, _BEST_AN, _N), 0.0, 1.0)

        action = int(np.argmax(activity[4:9]))
        (act_pre if step < s_step else act_post)[action] += 1

        row, col, hp, ate = _s15_world_step(row, col, action, food_avail, hp, cur_fp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            if step < s_step:
                food_pre += 1
            else:
                food_post += 1

        hp    -= _HP_DECAY
        steps  = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (step + 1) % _K == 0:
            _s11_hebb(G, W, activity, rng, _BEST_EP)

    if T_consol > 0:
        _s12_consolidation_phase(G, W, activity, rng, T_consol)

    entropy = _s11_entropy(act_pre + act_post)
    return steps, food_pre + food_post, entropy, food_pre, food_post, act_pre, act_post


def _s15_run_ep_old(G, W, readout_w, rng, max_steps=_MAX_STEPS,
                    food_pos=None, shift_step=None, food_pos_after=None):
    """Old-arch episode (readout + Hebbian) with tracking.

    Modifies G and W in place.
    Returns (steps, food_total, entropy, food_pre, food_post, act_pre, act_post).
    """
    fp       = list(_FOOD_POS) if food_pos is None else list(food_pos)
    fp_after = list(fp)        if food_pos_after is None else list(food_pos_after)
    s_step   = max_steps + 1   if shift_step is None else shift_step

    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    food_pre   = 0
    food_post  = 0
    act_pre    = np.zeros(5, dtype=int)
    act_post   = np.zeros(5, dtype=int)
    cur_fp     = fp

    for step in range(max_steps):
        if hp <= 0:
            break

        if step == s_step:
            cur_fp     = fp_after
            food_avail = [True, True]
            food_timer = [0, 0]

        inp4 = _s10_inp4(row, col, hp, food_avail)
        for _ in range(_N_PROP):
            activity = _s10_propagate(W, activity, inp4)

        action = _s10_softmax_sample(activity[4:20] @ readout_w, rng)
        (act_pre if step < s_step else act_post)[action] += 1

        row, col, hp, ate = _s15_world_step(row, col, action, food_avail, hp, cur_fp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            if step < s_step:
                food_pre += 1
            else:
                food_post += 1

        hp    -= _HP_DECAY
        steps  = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

        if (step + 1) % _K == 0:
            _s10_hebb(G, W, activity, rng)

    entropy = _s11_entropy(act_pre + act_post)
    return steps, food_pre + food_post, entropy, food_pre, food_post, act_pre, act_post


def _s15_run_ep_random(rng, max_steps=_MAX_STEPS,
                       food_pos=None, shift_step=None, food_pos_after=None):
    """Random-action baseline with tracking.

    Returns (steps, food_total, entropy, food_pre, food_post, act_pre, act_post).
    """
    fp       = list(_FOOD_POS) if food_pos is None else list(food_pos)
    fp_after = list(fp)        if food_pos_after is None else list(food_pos_after)
    s_step   = max_steps + 1   if shift_step is None else shift_step

    row, col   = 2, 2
    hp         = 100
    food_avail = [True, True]
    food_timer = [0, 0]
    steps      = 0
    food_pre   = 0
    food_post  = 0
    act_pre    = np.zeros(5, dtype=int)
    act_post   = np.zeros(5, dtype=int)
    cur_fp     = fp

    for step in range(max_steps):
        if hp <= 0:
            break

        if step == s_step:
            cur_fp     = fp_after
            food_avail = [True, True]
            food_timer = [0, 0]

        action = int(rng.integers(0, 5))
        (act_pre if step < s_step else act_post)[action] += 1

        row, col, hp, ate = _s15_world_step(row, col, action, food_avail, hp, cur_fp)
        if ate >= 0:
            food_avail[ate] = False
            food_timer[ate] = 0
            if step < s_step:
                food_pre += 1
            else:
                food_post += 1

        hp    -= _HP_DECAY
        steps  = step + 1
        for idx in range(2):
            if not food_avail[idx]:
                food_timer[idx] += 1
                if food_timer[idx] >= _RESPAWN:
                    food_avail[idx] = True
                    food_timer[idx] = 0

    entropy = _s11_entropy(act_pre + act_post)
    return steps, food_pre + food_post, entropy, food_pre, food_post, act_pre, act_post


# ── Experiment B: Long episode evaluation ─────────────────────────────────────

def run_exp_b_long_episode(collected, seed=_SEED):
    """Evaluate New-Surv / Old-Surv / Random across short/medium/long episode conditions.

    G and W are shared across all episodes for a single agent run (topology evolves
    via Hebbian and consolidation), matching the Session 12-13 training protocol.

    Returns nested dict:
      results[cond_name][group] = {
        'steps_per_ep':   np.ndarray (n_agents, n_ep),
        'food_per_ep':    np.ndarray (n_agents, n_ep),
        'entropy_per_ep': np.ndarray (n_agents, n_ep),
      }
    """
    rng = np.random.default_rng(seed)
    results = {}

    for cond_name, cond in _EXP_B_CONDITIONS.items():
        max_steps = cond['max_steps']
        n_ep      = cond['n_ep']
        print(f'\n  [Exp B: {cond_name}  max_steps={max_steps} × {n_ep} ep]')
        results[cond_name] = {}

        # New-Surv: topology shared across episodes (evolves in place)
        grp_s, grp_f, grp_h = [], [], []
        for agent in collected[_GRP_NEW]:
            G = agent['G'].copy()
            W = agent['W'].copy()
            ep_s, ep_f, ep_h = [], [], []
            for _ in range(n_ep):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, h, *_ = _s15_run_ep_new(G, W, ep_rng, max_steps, T_consol=200)
                ep_s.append(s); ep_f.append(f); ep_h.append(h)
            grp_s.append(ep_s); grp_f.append(ep_f); grp_h.append(ep_h)
        results[cond_name][_GRP_NEW] = {
            'steps_per_ep':   np.array(grp_s),
            'food_per_ep':    np.array(grp_f),
            'entropy_per_ep': np.array(grp_h),
        }

        # Old-Surv
        grp_s, grp_f, grp_h = [], [], []
        for agent in collected[_GRP_OLD]:
            G  = agent['G'].copy()
            W  = agent['W'].copy()
            rw = agent['rw']
            ep_s, ep_f, ep_h = [], [], []
            for _ in range(n_ep):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, h, *_ = _s15_run_ep_old(G, W, rw, ep_rng, max_steps)
                ep_s.append(s); ep_f.append(f); ep_h.append(h)
            grp_s.append(ep_s); grp_f.append(ep_f); grp_h.append(ep_h)
        results[cond_name][_GRP_OLD] = {
            'steps_per_ep':   np.array(grp_s),
            'food_per_ep':    np.array(grp_f),
            'entropy_per_ep': np.array(grp_h),
        }

        # Random baseline (single run, stored as 1×n_ep)
        ep_s, ep_f, ep_h = [], [], []
        for _ in range(n_ep):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            s, f, h, *_ = _s15_run_ep_random(ep_rng, max_steps)
            ep_s.append(s); ep_f.append(f); ep_h.append(h)
        results[cond_name][_GRP_RND] = {
            'steps_per_ep':   np.array([ep_s]),
            'food_per_ep':    np.array([ep_f]),
            'entropy_per_ep': np.array([ep_h]),
        }

        for gk in (_GRP_NEW, _GRP_OLD, _GRP_RND):
            d = results[cond_name][gk]
            print(f'    {_GRP_LABELS[gk]:12s}: '
                  f'mean_steps={np.mean(d["steps_per_ep"]):7.1f}  '
                  f'mean_food={np.mean(d["food_per_ep"]):5.2f}')

    return results


# ── Experiment C: Environment shift ────────────────────────────────────────────

def run_exp_c_env_shift(collected, seed=_SEED):
    """Evaluate New-Surv / Old-Surv / Random when food positions shift at step=250.

    Each agent starts from a fresh copy of its evolved topology (no carry-over).

    Returns nested dict:
      results[cond_name][group] = {
        'food_pre':      list[int],   one value per agent/run
        'food_post':     list[int],
        'recovery_rate': float,       mean_post / mean_pre (0 if mean_pre == 0)
        'act_pre':       np.ndarray (n_runs, 5),
        'act_post':      np.ndarray (n_runs, 5),
      }
    """
    rng = np.random.default_rng(seed + 100)
    results = {}

    for cond in _EXP_C_CONDITIONS:
        name      = cond['name']
        fp_before = cond['before']
        fp_after  = cond['after']
        print(f'\n  [Exp C: {name}]')
        results[name] = {}

        for gk, label in ((_GRP_NEW, 'New-Surv'), (_GRP_OLD, 'Old-Surv')):
            pre_l, post_l, ap_l, apost_l = [], [], [], []
            for agent in collected[gk]:
                G  = agent['G'].copy()
                W  = agent['W'].copy()
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                if gk == _GRP_NEW:
                    _, _, _, fp, fpost, ap, apost = _s15_run_ep_new(
                        G, W, ep_rng, max_steps=500, T_consol=0,
                        food_pos=fp_before, shift_step=_SHIFT_STEP,
                        food_pos_after=fp_after)
                else:
                    _, _, _, fp, fpost, ap, apost = _s15_run_ep_old(
                        G, W, agent['rw'], ep_rng, max_steps=500,
                        food_pos=fp_before, shift_step=_SHIFT_STEP,
                        food_pos_after=fp_after)
                pre_l.append(fp);   post_l.append(fpost)
                ap_l.append(ap);    apost_l.append(apost)

            mean_pre  = float(np.mean(pre_l))
            mean_post = float(np.mean(post_l))
            rr = mean_post / mean_pre if mean_pre > 1e-6 else 0.0
            results[name][gk] = {
                'food_pre':      pre_l,
                'food_post':     post_l,
                'recovery_rate': rr,
                'act_pre':       np.array(ap_l,    dtype=float),
                'act_post':      np.array(apost_l, dtype=float),
            }
            print(f'    {label:12s}: pre={mean_pre:.2f}  post={mean_post:.2f}  RR={rr:.3f}')

        # Random baseline: 5 independent runs
        pre_l, post_l, ap_l, apost_l = [], [], [], []
        for _ in range(5):
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, fp, fpost, ap, apost = _s15_run_ep_random(
                ep_rng, max_steps=500,
                food_pos=fp_before, shift_step=_SHIFT_STEP, food_pos_after=fp_after)
            pre_l.append(fp);   post_l.append(fpost)
            ap_l.append(ap);    apost_l.append(apost)
        mean_pre  = float(np.mean(pre_l))
        mean_post = float(np.mean(post_l))
        rr = mean_post / mean_pre if mean_pre > 1e-6 else 0.0
        results[name][_GRP_RND] = {
            'food_pre':      pre_l,
            'food_post':     post_l,
            'recovery_rate': rr,
            'act_pre':       np.array(ap_l,    dtype=float),
            'act_post':      np.array(apost_l, dtype=float),
        }
        print(f'    {"Random":12s}: pre={mean_pre:.2f}  post={mean_post:.2f}  RR={rr:.3f}')

    return results


# ── Experiment A: Edge damage ──────────────────────────────────────────────────

def run_exp_a_damage(collected, seed=_SEED):
    """Evaluate New-Surv / Old-Surv after randomly removing edges.

    Each (agent, damage_rate) pair uses a fresh topology copy.
    Hebbian runs during the episode, so topology can partially self-repair.

    Returns nested dict:
      results[rate][group] = {
        'steps':          list[int],
        'food':           list[int],
        'entropy':        list[float],
        'n_edges_orig':   list[int],   before damage
        'n_edges_dmg':    list[int],   after damage, before episode
        'n_edges_final':  list[int],   after episode (Hebbian may re-form edges)
      }
    """
    rng = np.random.default_rng(seed + 200)
    results = {}

    for rate in _DAMAGE_RATES:
        print(f'\n  [Exp A: damage_rate={rate}]')
        results[rate] = {}

        for gk, label in ((_GRP_NEW, 'New-Surv'), (_GRP_OLD, 'Old-Surv')):
            steps_l, food_l, ent_l = [], [], []
            n_orig_l, n_dmg_l, n_fin_l = [], [], []

            for agent in collected[gk]:
                G = agent['G'].copy()
                W = agent['W'].copy()
                n_orig = G.number_of_edges()

                if rate > 0.0:
                    edges  = list(G.edges())
                    n_rm   = int(len(edges) * rate)
                    rm_idx = rng.choice(len(edges), size=n_rm, replace=False)
                    for idx in rm_idx:
                        i, j = edges[idx]
                        G.remove_edge(i, j)
                        W[i, j] = 0.0

                n_dmg  = G.number_of_edges()
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))

                if gk == _GRP_NEW:
                    s, f, h, *_ = _s15_run_ep_new(
                        G, W, ep_rng, max_steps=500, T_consol=0)
                else:
                    s, f, h, *_ = _s15_run_ep_old(
                        G, W, agent['rw'], ep_rng, max_steps=500)

                n_fin = G.number_of_edges()
                steps_l.append(s);  food_l.append(f);   ent_l.append(h)
                n_orig_l.append(n_orig); n_dmg_l.append(n_dmg); n_fin_l.append(n_fin)

            results[rate][gk] = {
                'steps':         steps_l,
                'food':          food_l,
                'entropy':       ent_l,
                'n_edges_orig':  n_orig_l,
                'n_edges_dmg':   n_dmg_l,
                'n_edges_final': n_fin_l,
            }
            recov = [fin - dmg for dmg, fin in zip(n_dmg_l, n_fin_l)]
            print(f'    {label:12s}: steps={steps_l}  '
                  f'edges orig→dmg→fin={list(zip(n_orig_l,n_dmg_l,n_fin_l))}  '
                  f'Δedge={recov}')

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_b_long_episode(
        data, fname='images/session_15/results_s15_long_episode.png'):
    """3-panel figure for Experiment B.

    Panel 1 (left):  Episode learning curve — medium condition (500×20ep).
    Panel 2 (mid):   Mean survival steps per condition and group (bar chart).
    Panel 3 (right): First-half vs last-half food count — medium condition.
    """
    cond_order  = ['short', 'medium', 'long']
    cond_labels = ['Short\n(500×5)', 'Medium\n(500×20)', 'Long\n(2000×5)']
    grp_order   = [_GRP_NEW, _GRP_OLD, _GRP_RND]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Session 15 Exp B: Long Episode Evaluation — New-Surv vs Old-Surv vs Random\n'
        'seed=42, evolved topology reused from Session 12-13',
        fontsize=10,
    )

    # ── Panel 1: Episode learning curve (medium condition) ─────────────────────
    ax  = axes[0]
    med = data['medium']
    for gk in grp_order:
        d    = med[gk]['steps_per_ep']        # (n_agents, n_ep)
        mean = np.mean(d, axis=0)
        xs   = np.arange(1, len(mean) + 1)
        color = _GRP_COLORS[gk]
        ax.plot(xs, mean, color=color, label=_GRP_LABELS[gk],
                linewidth=2, marker='o', markersize=3)
        if d.shape[0] > 1:
            std = np.std(d, axis=0)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Survival Steps')
    ax.set_title('Learning Curve (medium: 500 × 20 ep)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Mean steps per condition × group ──────────────────────────────
    ax    = axes[1]
    width = 0.22
    x     = np.arange(len(cond_order))
    for gi, gk in enumerate(grp_order):
        means  = [np.mean(data[c][gk]['steps_per_ep']) for c in cond_order]
        stds   = [np.std( data[c][gk]['steps_per_ep']) for c in cond_order]
        offset = (gi - 1) * width
        ax.bar(x + offset, means, width=width,
               label=_GRP_LABELS[gk], color=_GRP_COLORS[gk],
               edgecolor='gray', linewidth=0.7,
               yerr=stds, capsize=3, error_kw={'elinewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel('Mean Survival Steps')
    ax.set_title('Summary: Mean Steps per Condition')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: First half vs last half food (medium condition) ───────────────
    ax   = axes[2]
    med  = data['medium']
    n_ep = med[_GRP_NEW]['food_per_ep'].shape[1]
    half = n_ep // 2
    x_g  = np.arange(len(grp_order))

    for gi, gk in enumerate(grp_order):
        food       = med[gk]['food_per_ep']   # (n_agents, n_ep)
        first_half = float(np.mean(food[:, :half]))
        last_half  = float(np.mean(food[:, half:]))
        color      = _GRP_COLORS[gk]
        ax.bar(x_g[gi] - 0.15, first_half, width=0.28,
               color=color, alpha=0.5, edgecolor='gray', linewidth=0.7)
        ax.bar(x_g[gi] + 0.15, last_half,  width=0.28,
               color=color, alpha=1.0, edgecolor='gray', linewidth=0.7)

    handles = [
        Patch(color='gray', alpha=0.5, label=f'ep 1–{half}  (first half)'),
        Patch(color='gray', alpha=1.0, label=f'ep {half+1}–{n_ep} (last half)'),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xticks(x_g)
    ax.set_xticklabels([_GRP_LABELS[gk] for gk in grp_order], fontsize=8)
    ax.set_ylabel('Mean Food Count per Episode')
    ax.set_title(f'Food Gain: First vs Last Half\n(medium: 500 × 20 ep)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_env_shift(
        data, fname='images/session_15/results_s15_env_shift.png'):
    """2-panel figure for Experiment C.

    Panel 1 (left):  Recovery rate per shift condition and group (grouped bars).
    Panel 2 (right): Action distribution change (post − pre) for full_reversal,
                     shown as a heatmap (groups × actions).
    """
    cond_names   = ['full_reversal', 'minor_shift', 'no_shift']
    cond_display = ['Full reversal\n(0,0)↔(4,4)→(0,4)↔(4,0)',
                    'Minor shift\n→(1,1)↔(3,3)',
                    'No shift\n(control)']
    grp_order    = [_GRP_NEW, _GRP_OLD, _GRP_RND]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Session 15 Exp C: Environment Shift at step=250\n'
        'recovery_rate = food_post / food_pre  |  seed=42',
        fontsize=10,
    )

    # ── Panel 1: Recovery rate grouped bars ───────────────────────────────────
    ax    = axes[0]
    width = 0.22
    x     = np.arange(len(cond_names))
    for gi, gk in enumerate(grp_order):
        rrs    = [data[c][gk]['recovery_rate'] for c in cond_names]
        offset = (gi - 1) * width
        ax.bar(x + offset, rrs, width=width,
               label=_GRP_LABELS[gk], color=_GRP_COLORS[gk],
               edgecolor='gray', linewidth=0.7)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.4,
               label='RR = 1 (no change)')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_display, fontsize=7.5)
    ax.set_ylabel('Recovery Rate (food_post / food_pre)')
    ax.set_title('Adaptation to Environment Shift')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Action distribution change heatmap (full_reversal) ───────────
    ax       = axes[1]
    cond_key = 'full_reversal'
    mat      = []
    row_labels = []
    for gk in grp_order:
        d    = data[cond_key][gk]
        pre  = np.mean(d['act_pre'],  axis=0)   # (5,)
        post = np.mean(d['act_post'], axis=0)   # (5,)
        # normalise to relative frequency (avoid /0)
        pre_norm  = pre  / (pre.sum()  + 1e-12)
        post_norm = post / (post.sum() + 1e-12)
        delta     = post_norm - pre_norm
        mat.append(delta)
        row_labels.append(f'{_GRP_LABELS[gk]}\npre food={np.mean(d["food_pre"]):.1f}'
                          f'  post={np.mean(d["food_post"]):.1f}')
    mat = np.array(mat)
    vmax = max(float(np.abs(mat).max()), 0.05)
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(_ACTION_NAMES, fontsize=8)
    ax.set_yticks(range(len(grp_order)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title('Action Distribution Δ (post − pre)\nFull reversal condition')
    for r in range(len(grp_order)):
        for c in range(5):
            ax.text(c, r, f'{mat[r, c]:+.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04,
                 label='Δ relative frequency')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_a_damage(
        data, fname='images/session_15/results_s15_damage.png'):
    """2-panel figure for Experiment A.

    Panel 1 (left):  Damage rate vs mean survival steps (line + scatter).
    Panel 2 (right): Topology recovery — Δedges (final − damaged) per group.
    """
    grp_order = [_GRP_NEW, _GRP_OLD]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Session 15 Exp A: Edge Damage Tolerance\n'
        'Hebbian runs during episode — topology may self-repair  |  seed=42',
        fontsize=10,
    )

    # ── Panel 1: Survival steps vs damage rate ────────────────────────────────
    ax = axes[0]
    for gk in grp_order:
        means  = [np.mean(data[r][gk]['steps']) for r in _DAMAGE_RATES]
        stds   = [np.std( data[r][gk]['steps']) for r in _DAMAGE_RATES]
        color  = _GRP_COLORS[gk]
        ax.plot(_DAMAGE_RATES, means, color=color, label=_GRP_LABELS[gk],
                linewidth=2, marker='o', markersize=6)
        ax.fill_between(_DAMAGE_RATES,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15)
    ax.set_xlabel('Damage Rate (fraction of edges removed)')
    ax.set_ylabel('Survival Steps')
    ax.set_title('Damage Tolerance: Survival Steps')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Topology recovery (Δedges) ───────────────────────────────────
    ax    = axes[1]
    width = 0.15
    x     = np.arange(len(_DAMAGE_RATES))
    for gi, gk in enumerate(grp_order):
        # Δedges = n_edges_final − n_edges_dmg (positive = re-formed)
        delta_means = [
            np.mean([fin - dmg
                     for dmg, fin in zip(data[r][gk]['n_edges_dmg'],
                                         data[r][gk]['n_edges_final'])])
            for r in _DAMAGE_RATES
        ]
        delta_stds = [
            np.std([fin - dmg
                    for dmg, fin in zip(data[r][gk]['n_edges_dmg'],
                                        data[r][gk]['n_edges_final'])])
            for r in _DAMAGE_RATES
        ]
        offset = (gi - 0.5) * width
        ax.bar(x + offset, delta_means, width=width,
               label=_GRP_LABELS[gk], color=_GRP_COLORS[gk],
               edgecolor='gray', linewidth=0.7,
               yerr=delta_stds, capsize=3, error_kw={'elinewidth': 1.2})
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in _DAMAGE_RATES])
    ax.set_xlabel('Damage Rate')
    ax.set_ylabel('Δ Edges (final − damaged)')
    ax.set_title('Topology Self-Repair via Hebbian\n(positive = edges re-formed)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Session 15: 構造的優位性の機能的検証 ===')

    print('\n[Collection] Re-evolving Session 12-13 individuals (seed=42) ...')
    print('  (same seed → same evolved topologies as Session 13)')
    collected = collect_all(seed=42, n_gen=50, n_agents=10, n_ep=5, n_surv=3)

    print('\n  Summary:')
    for gk in ('new_surv', 'old_surv'):
        agents = collected[gk]
        print(f'  {gk}: mean_steps={[round(a["mean_steps"], 1) for a in agents]}'
              f'  max_steps={[int(a["max_steps"]) for a in agents]}')

    print('\n=== Exp B: Long Episode Evaluation ===')
    exp_b = run_exp_b_long_episode(collected)
    print('\n  First-half vs last-half food (medium condition):')
    med  = exp_b['medium']
    n_ep = med[_GRP_NEW]['food_per_ep'].shape[1]
    half = n_ep // 2
    for gk in (_GRP_NEW, _GRP_OLD, _GRP_RND):
        food = med[gk]['food_per_ep']
        print(f'  {_GRP_LABELS[gk]:12s}: '
              f'ep1-{half}={np.mean(food[:,:half]):.2f}  '
              f'ep{half+1}-{n_ep}={np.mean(food[:,half:]):.2f}')
    plot_exp_b_long_episode(exp_b)

    print('\n=== Exp C: Environment Shift ===')
    exp_c = run_exp_c_env_shift(collected)
    plot_exp_c_env_shift(exp_c)

    print('\n=== Exp A: Edge Damage ===')
    exp_a = run_exp_a_damage(collected)
    plot_exp_a_damage(exp_a)

    print('\nDone.')
