"""Session 29: 追跡性を持つ捕食者

Session 28の問題:
  捕食者がランダムウォークでは「北に逃げ続ける」戦略が成立する
  → 文脈を読む必要がない → 文脈依存的な行動が出ない

根本的な設計変更:
  捕食者が pursuit_prob でエージェントを追跡する
  追跡するたびに捕食者の資源が枯渇 → 休眠（ランダムウォーク）
  休眠中は資源が回復 → 再追跡

ネットワーク:
  Session 28と同一 (21ノード、入力5/出力5/内部11)
  tau_s/tau_i/tau_o = 94/43/34 (S27最良値に固定)

実験:
  A  pursuit_prob スイープ [0.0, 0.3, 0.6, 1.0] (50世代進化 × 4条件)
  B  ベスト条件での文脈依存的な行動の計測 (4文脈)
  C  睡眠様状態の観察 (T=2000、捕食者休眠との相関)
  D  捕食者疲労速度スイープ [0.01, 0.05, 0.1, 0.2]

出力:
  images/session_29/results_s29_pursuit_sweep.png
  images/session_29/results_s29_context.png
  images/session_29/results_s29_sleep_pattern.png
  images/session_29/results_s29_predator_fatigue.png
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import _jp_font  # noqa: F401 — sets Japanese font in matplotlib rcParams

from session_10_embodied_output import _N_PROP, _K, _INIT_W, _LR
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_18_ratio_evolution import (
    _EP_INIT_MAX, _AR_INIT_MAX,
    _EP_MUT_STD, _AR_MUT_STD,
    _ACTIVITY_NOISE,
    _N_SURV,
)
from session_27_tm_resources import _s27_update_resources
from session_28_predator import (
    _S28_N, _S28_INP_START, _S28_INP_END,
    _S28_OUT_START, _S28_OUT_END, _S28_INT_START, _S28_INT_END,
    _S28_DENSITY, _S28_MUT_STD, _S28_EDGE_CHNG,
    _S28_HEBB_LR, _S28_HEBB_DECAY,
    _S28_TAU_S, _S28_TAU_I, _S28_TAU_O,
    _S28_DEPL_LO, _S28_DEPL_HI, _S28_DEPL_MUT_STD,
    _S28_GRID, _S28_HP_START, _S28_HP_MAX, _S28_HP_DECAY,
    _S28_FOOD_VALUE, _S28_FOOD_RESPAWN, _S28_MAX_STEPS,
    _S28_N_FOODS, _S28_PRED_DAMAGE, _S28_FOOD_RESOURCE,
    _S28_FOOD_DIST, _S28_PRED_DIST,
    _S28_N_GEN, _S28_N_AGENTS, _S28_N_EP, _S28_N_SURV, _S28_SEED,
    _S28_MR, _S28_ACT_NOISE, _S28_T_CONSOL, _S28_ACT_THRESH,
    _S28_CONTEXTS, _S28_ACTION_NAMES,
    _s28_build_graph, _s28_get_W, _s28_propagate, _s28_mutate_graph,
    _s28_hebb, _s28_make_tau_arr, _s28_make_genome, _s28_mutate_genome,
    _s28_inp5, _s28_init_foods, _s28_init_pred, _s28_cosine_dist,
    _s28_measure_context,
)

# ── Session 29 constants ────────────────────────────────────────────────────────

_S29_N              = _S28_N          # 21 nodes (same network)
_S29_GRID           = _S28_GRID       # 5×5

# Predator pursuit + fatigue defaults
_S29_PRED_SPEED     = 2              # predator moves every 2 steps (S28 best)
_S29_PRED_DEPLETION = 0.05           # resource consumed per pursuit step
_S29_PRED_RECOVERY  = 0.02           # resource recovered per dormant step
_S29_PRED_DORMANT_LO = 0.2          # enter dormancy when resources < this
_S29_PRED_DORMANT_HI = 0.8          # exit dormancy when resources > this

# Experiment A: pursuit_prob sweep
_S29_PURSUIT_PROBS  = [0.0, 0.3, 0.6, 1.0]
_S29_N_GEN          = _S28_N_GEN     # 50 generations

# Experiment B: context measurement
_S29_N_CONTEXT      = 25             # samples per context
_S29_CONTEXT_T      = 100            # steps per measurement

# Experiment C: sleep observation
_S29_SLEEP_T        = 2000
_S29_SLEEP_CHUNK    = 100

# Experiment D: predator fatigue speed sweep
_S29_PRED_DEPLETIONS = [0.01, 0.05, 0.1, 0.2]
_S29_PURSUIT_PROB_DEF = 0.6          # fixed pursuit_prob for Exp D
_S29_N_GEN_D        = 30             # reduced for Exp D

_S29_SEED           = _S28_SEED


# ── Predator movement with pursuit + fatigue ────────────────────────────────────

def _s29_pred_step(pred_pos, agent_pos, pursuit_prob,
                   pred_resources, pred_dormant, rng):
    """Move predator one step: pursue agent or random walk based on state.

    Returns (new_pos, new_pred_resources, new_pred_dormant, is_pursuing).
    """
    pr, pc = pred_pos
    ar, ac = agent_pos

    # Decide whether to pursue
    if pred_dormant:
        is_pursuing = False
    else:
        is_pursuing = bool(rng.random() < pursuit_prob)

    if is_pursuing:
        dr = int(np.sign(ar - pr))
        dc = int(np.sign(ac - pc))
        if dr == 0 and dc == 0:
            # Same cell: fall back to random walk this step
            is_pursuing = False
        elif dr == 0:
            pc = int(np.clip(pc + dc, 0, _S29_GRID - 1))
        elif dc == 0:
            pr = int(np.clip(pr + dr, 0, _S29_GRID - 1))
        else:
            if rng.random() < 0.5:
                pc = int(np.clip(pc + dc, 0, _S29_GRID - 1))
            else:
                pr = int(np.clip(pr + dr, 0, _S29_GRID - 1))

    if not is_pursuing:
        d = int(rng.integers(0, 4))
        if d == 0:   pr = max(0, pr - 1)
        elif d == 1: pr = min(_S29_GRID - 1, pr + 1)
        elif d == 2: pc = max(0, pc - 1)
        else:        pc = min(_S29_GRID - 1, pc + 1)

    # Update predator resources
    if is_pursuing:
        pred_resources -= _S29_PRED_DEPLETION
    else:
        pred_resources += _S29_PRED_RECOVERY
    pred_resources = float(np.clip(pred_resources, 0.0, 1.0))

    # Dormancy transitions (hysteresis)
    if pred_resources < _S29_PRED_DORMANT_LO:
        pred_dormant = True
    elif pred_resources > _S29_PRED_DORMANT_HI:
        pred_dormant = False

    return [pr, pc], pred_resources, pred_dormant, is_pursuing


# ── Episode runner ──────────────────────────────────────────────────────────────

def _s29_run_ep(G, W, genome, rng,
                pursuit_prob=_S29_PURSUIT_PROB_DEF,
                predator_speed=_S29_PRED_SPEED,
                pred_depletion=_S29_PRED_DEPLETION,
                record_activity=False,
                record_resources=False,
                record_pred_state=False):
    """PursuingPredator episode with TM resource model.

    Modifies G and W in place via Hebbian learning.

    Returns dict:
      steps, food, pred_hits,
      act_recs (list[ndarray] or None),
      res_recs (list[ndarray] or None),
      pred_dormant_recs (list[bool] or None).
    """
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S29_N)
    activity  = np.zeros(_S29_N)

    row, col  = 2, 2
    hp        = float(_S28_HP_START)

    food_positions = _s28_init_foods(rng)
    food_avail     = [True] * _S28_N_FOODS
    food_timer     = [0]   * _S28_N_FOODS

    pred_pos      = _s28_init_pred(rng)
    pred_resources = 1.0
    pred_dormant   = False

    steps     = 0
    food      = 0
    pred_hits = 0

    act_recs       = [] if record_activity   else None
    res_recs       = [] if record_resources  else None
    pred_dom_recs  = [] if record_pred_state else None

    for step in range(_S28_MAX_STEPS):
        if hp <= 0:
            break

        # Move predator every `predator_speed` steps
        if step % predator_speed == 0:
            pred_pos, pred_resources, pred_dormant, _ = _s29_pred_step(
                pred_pos, [row, col], pursuit_prob,
                pred_resources, pred_dormant, rng)

        # Check predator collision
        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            pred_hits += 1

        # Compute input
        inp5 = _s28_inp5(row, col, hp, food_positions, food_avail, pred_pos)

        # Propagate network
        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)

        # Effective activity with resources
        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S29_N), 0.0, 1.0)

        # Update agent resources (TM model)
        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)

        if record_activity:
            act_recs.append(eff.copy())
        if record_resources:
            res_recs.append(resources.copy())
        if record_pred_state:
            pred_dom_recs.append(pred_dormant)

        # HP costs
        hp -= _S28_HP_DECAY
        hp -= metabolic_rate * float(np.sum(eff))

        # Agent action
        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(_S29_GRID - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(_S29_GRID - 1, col + 1)
        elif action == 4:
            for fi in range(_S28_N_FOODS):
                if (food_avail[fi]
                        and row == food_positions[fi][0]
                        and col == food_positions[fi][1]):
                    hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                    resources = np.clip(
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources), 0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    food += 1
                    break

        steps = step + 1

        # Food respawn
        for fi in range(_S28_N_FOODS):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        # Hebbian learning
        if (step + 1) % _K == 0:
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

    _s12_consolidation_phase(G, W, activity, rng, _S28_T_CONSOL)

    return {
        'steps':          steps,
        'food':           food,
        'pred_hits':      pred_hits,
        'act_recs':       act_recs,
        'res_recs':       res_recs,
        'pred_dom_recs':  pred_dom_recs,
    }


# ── Evolution ───────────────────────────────────────────────────────────────────

def _s29_evolve(pursuit_prob=_S29_PURSUIT_PROB_DEF,
                predator_speed=_S29_PRED_SPEED,
                pred_depletion=_S29_PRED_DEPLETION,
                seed=_S29_SEED, n_gen=_S29_N_GEN):
    """Evolve genomes in PursuingPredator world.

    Returns (best_genome, history_dict).
    """
    rng = np.random.default_rng(
        seed + 29000 + int(pursuit_prob * 100) + int(pred_depletion * 1000))
    pop = [_s28_make_genome(rng) for _ in range(_S28_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_food_count', 'gen_pred_hits',
        'gen_mean_active', 'gen_depl', 'gen_edge_count',
    )}

    for gen in range(n_gen):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_hits, ep_active = 0, [], [], []
            for _ in range(_S28_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                res = _s29_run_ep(
                    g['G'], g['W'], g, ep_rng,
                    pursuit_prob=pursuit_prob,
                    predator_speed=predator_speed,
                    pred_depletion=pred_depletion,
                    record_activity=True)
                total += res['steps']
                ep_food.append(res['food'])
                ep_hits.append(res['pred_hits'])
                if res['act_recs']:
                    arr = np.array(res['act_recs'])
                    ep_active.append(
                        float(np.sum(np.mean(arr, axis=0) > _S28_ACT_THRESH)))
            fitnesses.append(total / _S28_N_EP)
            g['_ep_food']   = float(np.mean(ep_food))
            g['_ep_hits']   = float(np.mean(ep_hits))
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0
            g['_ep_edges']  = float(g['G'].number_of_edges())

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]

        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_pred_hits'].append(bg['_ep_hits'])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_depl'].append(bg['depletion_rate'])
        hist['gen_edge_count'].append(bg['_ep_edges'])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_S28_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _S28_N_AGENTS:
            parent = survivors[int(rng.integers(0, _S28_N_SURV))]
            new_pop.append(_s28_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'hits={bg["_ep_hits"]:.2f}/ep  '
                  f'depl={bg["depletion_rate"]:.3f}  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        for k in ('_ep_food', '_ep_hits', '_ep_active', '_ep_edges'):
            g.pop(k, None)

    return pop[0], hist


def _s29_mean_cosine_dist(genome, seed_offset, pursuit_prob, pred_depletion):
    """Measure mean pairwise cosine distance across 4 contexts."""
    G_copy = genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng_c  = np.random.default_rng(seed_offset)
    results = []
    for ctx in _S28_CONTEXTS:
        r = _s28_measure_context(G_copy, W_copy, genome, ctx, rng_c, T=_S29_CONTEXT_T)
        results.append(r)
    n_ctx  = len(_S28_CONTEXTS)
    pairs  = [(i, j) for i in range(n_ctx) for j in range(i + 1, n_ctx)]
    vecs   = [r['mean_output'] for r in results]
    return float(np.mean([_s28_cosine_dist(vecs[i], vecs[j]) for i, j in pairs])), results


# ── Experiment A: pursuit_prob sweep ───────────────────────────────────────────

def run_exp_a_pursuit_sweep(seed=_S29_SEED, n_gen=_S29_N_GEN):
    """Evolve under 4 pursuit_prob conditions and measure context dependence.

    Returns list of dicts per pursuit_prob.
    """
    print(f'\n  [Exp A] pursuit_prob スイープ '
          f'({_S29_PURSUIT_PROBS}, {n_gen}世代/条件)')
    sweep = []

    for pp in _S29_PURSUIT_PROBS:
        print(f'\n  pursuit_prob={pp}:')
        best, hist = _s29_evolve(
            pursuit_prob=pp, predator_speed=_S29_PRED_SPEED,
            pred_depletion=_S29_PRED_DEPLETION, seed=seed, n_gen=n_gen)

        mcd, _ = _s29_mean_cosine_dist(
            best, seed + 29400 + int(pp * 100), pp, _S29_PRED_DEPLETION)
        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'depl={best["depletion_rate"]:.3f}  '
              f'mcd={mcd:.4f}')

        sweep.append({
            'pursuit_prob':     pp,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'depl':             best['depletion_rate'],
            'pred_hits':        hist['gen_pred_hits'][-1],
            'food':             hist['gen_food_count'][-1],
        })

    return sweep


def plot_exp_a_pursuit_sweep(sweep,
                              fname='images/session_29/results_s29_pursuit_sweep.png'):
    pps   = [r['pursuit_prob']     for r in sweep]
    mcds  = [r['mean_cosine_dist'] for r in sweep]
    steps = [r['best_steps']       for r in sweep]
    depls = [r['depl']             for r in sweep]
    hits  = [r['pred_hits']        for r in sweep]
    foods = [r['food']             for r in sweep]

    xs    = np.arange(len(pps))
    xlbls = [f'pp={p}' for p in pps]
    colors = ['steelblue', 'seagreen', 'tomato', 'darkorange']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 29 Exp A: pursuit_prob スイープ\n'
        f'{_S29_N_GEN}世代 × {_S28_N_AGENTS}個体 × {_S28_N_EP}ep/個体  '
        f'(pred_depletion={_S29_PRED_DEPLETION}, pred_recovery={_S29_PRED_RECOVERY})',
        fontsize=12,
    )

    # Panel 1: context dependence (main hypothesis)
    ax = axes[0][0]
    for xi, (v, c) in enumerate(zip(mcds, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Mean pairwise cosine dist')
    ax.set_title('文脈依存性の強さ\n(高い = 文脈間の出力差が大きい)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: survival steps
    ax = axes[0][1]
    for xi, (v, c) in enumerate(zip(steps, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Best steps / ep')
    ax.set_title('生存ステップ数')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: depletion_rate
    ax = axes[0][2]
    for xi, (v, c) in enumerate(zip(depls, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('depletion_rate')
    ax.set_ylim(0, _S28_DEPL_HI + 0.05)
    ax.set_title('depletion_rate の収束値\n(高い = エージェント疲労が強い)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: learning curves for all conditions
    ax = axes[1][0]
    for r, c in zip(sweep, colors):
        xs_g = np.arange(1, len(r['hist']['gen_best_steps']) + 1)
        ax.plot(xs_g, r['hist']['gen_best_steps'], color=c, linewidth=2,
                label=f'pp={r["pursuit_prob"]}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best steps / ep')
    ax.set_title('進化曲線（生存ステップ推移）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: pred hits
    ax = axes[1][1]
    for xi, (v, c) in enumerate(zip(hits, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: summary
    ax = axes[1][2]
    best_r = max(sweep, key=lambda r: r['mean_cosine_dist'])
    summary_lines = ['pursuit_prob別サマリー\n']
    for r in sweep:
        marker = '★' if r['pursuit_prob'] == best_r['pursuit_prob'] else '  '
        summary_lines.append(
            f'{marker}pp={r["pursuit_prob"]:.1f}: '
            f'steps={r["best_steps"]:.0f}  '
            f'mcd={r["mean_cosine_dist"]:.4f}  '
            f'depl={r["depl"]:.3f}'
        )
    summary_lines.append(f'\n最大文脈依存性: pp={best_r["pursuit_prob"]}')
    ax.text(0.5, 0.5, '\n'.join(summary_lines), transform=ax.transAxes,
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.set_title('実験サマリー')
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment B: context-dependent behavior ────────────────────────────────────

def run_exp_b_context(best_genome, pursuit_prob, seed=_S29_SEED,
                       n_per_context=_S29_N_CONTEXT):
    """4文脈での行動パターンを計測し文脈依存性を計算。

    Returns dict: context_results, cosine_matrix, mean_cosine_dist, p_values.
    """
    print(f'\n  [Exp B] 文脈依存的な行動の計測 '
          f'(pursuit_prob={pursuit_prob}, n={n_per_context}/文脈)')
    G_copy = best_genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 29200)

    results = []
    all_action_samples = []  # list of (ctx_idx, action_dist_sample)

    for ctx_i, ctx in enumerate(_S28_CONTEXTS):
        samples = []
        for _ in range(n_per_context):
            r = _s28_measure_context(G_copy, W_copy, best_genome, ctx, rng,
                                      T=_S29_CONTEXT_T)
            samples.append(r['mean_output'])
        mean_out = np.mean(samples, axis=0)
        action_dist_arr = np.array([s for s in samples])
        action_count = np.zeros(5)
        for s in samples:
            action_count[int(np.argmax(s))] += 1
        results.append({
            'mean_output':  mean_out,
            'action_dist':  action_count / n_per_context,
            'output_samples': np.array(samples),
        })
        dom = _S28_ACTION_NAMES[int(np.argmax(mean_out))]
        print(f'  [{ctx["label"].replace(chr(10)," ")}]  '
              f'mean_out={mean_out.round(3)}  主行動={dom}')

    # Cosine distance matrix
    n_ctx   = len(_S28_CONTEXTS)
    cos_mat = np.zeros((n_ctx, n_ctx))
    for i in range(n_ctx):
        for j in range(n_ctx):
            cos_mat[i, j] = _s28_cosine_dist(
                results[i]['mean_output'], results[j]['mean_output'])

    pairs  = [(i, j) for i in range(n_ctx) for j in range(i + 1, n_ctx)]
    mean_cd = float(np.mean([cos_mat[i, j] for i, j in pairs]))

    # Statistical significance: t-test between context pairs for dominant output
    p_values = {}
    for (i, j) in pairs:
        samp_i = results[i]['output_samples'].max(axis=1)  # max output value
        samp_j = results[j]['output_samples'].max(axis=1)
        _, p = scipy_stats.ttest_ind(samp_i, samp_j)
        p_values[(i, j)] = float(p)

    min_p = min(p_values.values())
    print(f'  → mean cosine dist = {mean_cd:.4f}  '
          f'(最小p値={min_p:.4f}  '
          f'{"統計的に有意" if min_p < 0.05 else "有意差なし"})')

    return {
        'context_results':  results,
        'cosine_matrix':    cos_mat,
        'mean_cosine_dist': mean_cd,
        'p_values':         p_values,
        'pursuit_prob':     pursuit_prob,
    }


def plot_exp_b_context(exp_b, fname='images/session_29/results_s29_context.png'):
    results  = exp_b['context_results']
    cos_mat  = exp_b['cosine_matrix']
    p_values = exp_b['p_values']
    ctx_lbls = [c['label'] for c in _S28_CONTEXTS]
    n_ctx    = len(_S28_CONTEXTS)
    pp       = exp_b['pursuit_prob']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Session 29 Exp B: 文脈依存的な行動の計測\n'
        f'pursuit_prob={pp}  4文脈 × 5出力ノード  '
        f'mean cosine dist={exp_b["mean_cosine_dist"]:.4f}',
        fontsize=12,
    )

    # Panel 1: output activity heatmap
    ax = axes[0]
    out_matrix = np.array([r['mean_output'] for r in results])
    im = ax.imshow(out_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'node{i+5}\n({a})' for i, a in enumerate(_S28_ACTION_NAMES)],
                        fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=9)
    ax.set_title('出力ノード平均活動\n(文脈 × 出力ノード)')
    for i in range(n_ctx):
        for j in range(5):
            ax.text(j, i, f'{out_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if out_matrix[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: cosine distance matrix with p-values
    ax = axes[1]
    vmax = max(cos_mat.max(), 0.01)
    im2 = ax.imshow(cos_mat, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(ctx_lbls, fontsize=8)
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(ctx_lbls, fontsize=8)
    ax.set_title('Cosine距離行列\n(大=文脈間出力が異なる, p値を併記)')
    for i in range(n_ctx):
        for j in range(n_ctx):
            if i == j:
                label = f'{cos_mat[i,j]:.3f}'
            else:
                pair = (min(i, j), max(i, j))
                p = p_values.get(pair, float('nan'))
                sig = '*' if p < 0.05 else ''
                label = f'{cos_mat[i,j]:.3f}{sig}'
            ax.text(j, i, label, ha='center', va='center', fontsize=7,
                    color='white' if cos_mat[i, j] > vmax * 0.6 else 'black')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Panel 3: action distribution stacked bar
    ax = axes[2]
    action_mat  = np.array([r['action_dist'] for r in results])
    colors_act  = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    bottoms     = np.zeros(n_ctx)
    xs          = np.arange(n_ctx)
    for ai, (color, name) in enumerate(zip(colors_act, _S28_ACTION_NAMES)):
        ax.bar(xs, action_mat[:, ai], bottom=bottoms, color=color, alpha=0.85,
               label=name, width=0.6)
        bottoms += action_mat[:, ai]
    ax.set_xticks(xs)
    ax.set_xticklabels(ctx_lbls, fontsize=9)
    ax.set_ylabel('Action probability')
    ax.set_title('文脈別行動分布\n(*=隣接文脈間で有意差あり p<0.05)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment C: sleep pattern with predator dormancy ─────────────────────────

def _s29_run_long(G, W, genome, rng,
                   pursuit_prob=_S29_PURSUIT_PROB_DEF,
                   predator_speed=_S29_PRED_SPEED,
                   pred_depletion=_S29_PRED_DEPLETION,
                   T=_S29_SLEEP_T):
    """T-step continuous run recording activity and predator dormancy per chunk.

    HP resets when it hits 0 to maintain T steps of observation.
    Returns dict: chunks, food_total, pred_hits_total.
    """
    depletion_rate = genome['depletion_rate']
    edge_add_prob  = genome['edge_add_prob']
    activity_ratio = genome['activity_ratio']
    metabolic_rate = genome['metabolic_rate']

    tau_arr   = _s28_make_tau_arr()
    resources = np.ones(_S29_N)
    activity  = np.zeros(_S29_N)

    row, col  = 2, 2
    hp        = float(_S28_HP_START)

    food_positions = _s28_init_foods(rng)
    food_avail     = [True] * _S28_N_FOODS
    food_timer     = [0]   * _S28_N_FOODS

    pred_pos       = _s28_init_pred(rng)
    pred_resources = 1.0
    pred_dormant   = False

    food_total = 0
    pred_hits  = 0

    chunks      = []
    chunk_acts  = []
    chunk_ress  = []
    chunk_dorms = []  # bool per step: predator dormant?

    for step in range(T):
        if hp <= 0:
            hp = float(_S28_HP_START)

        if step % predator_speed == 0:
            pred_pos, pred_resources, pred_dormant, _ = _s29_pred_step(
                pred_pos, [row, col], pursuit_prob,
                pred_resources, pred_dormant, rng)

        if pred_pos[0] == row and pred_pos[1] == col:
            hp -= _S28_PRED_DAMAGE
            pred_hits += 1

        inp5 = _s28_inp5(row, col, hp, food_positions, food_avail, pred_pos)

        for _ in range(_N_PROP):
            activity = _s28_propagate(W, activity, inp5)

        eff = np.clip(activity * resources, 0.0, 1.0)
        if _S28_ACT_NOISE > 0.0:
            eff = np.clip(eff + rng.normal(0, _S28_ACT_NOISE, _S29_N), 0.0, 1.0)

        resources = _s27_update_resources(resources, activity, tau_arr, depletion_rate)

        chunk_acts.append(eff.copy())
        chunk_ress.append(resources.copy())
        chunk_dorms.append(pred_dormant)

        hp -= _S28_HP_DECAY
        hp -= metabolic_rate * float(np.sum(eff))

        action = int(np.argmax(eff[_S28_OUT_START:_S28_OUT_END]))
        if action == 0:   row = max(0, row - 1)
        elif action == 1: row = min(_S29_GRID - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(_S29_GRID - 1, col + 1)
        elif action == 4:
            for fi in range(_S28_N_FOODS):
                if (food_avail[fi]
                        and row == food_positions[fi][0]
                        and col == food_positions[fi][1]):
                    hp = min(_S28_HP_MAX, hp + _S28_FOOD_VALUE)
                    resources = np.clip(
                        resources + _S28_FOOD_RESOURCE * (1.0 - resources), 0.0, 1.0)
                    food_avail[fi] = False
                    food_timer[fi] = 0
                    food_total += 1
                    break

        for fi in range(_S28_N_FOODS):
            if not food_avail[fi]:
                food_timer[fi] += 1
                if food_timer[fi] >= _S28_FOOD_RESPAWN:
                    food_avail[fi] = True
                    food_timer[fi] = 0

        if (step + 1) % _K == 0:
            _s28_hebb(G, W, eff, rng, edge_add_prob, activity_ratio)

        activity = eff.copy()

        if (step + 1) % _S29_SLEEP_CHUNK == 0:
            arr_a  = np.array(chunk_acts)
            arr_r  = np.array(chunk_ress)
            dorm_ratio = float(np.mean(chunk_dorms))
            chunks.append({
                't_end':              step + 1,
                'sensory_activity':   float(np.mean(arr_a[:, _S28_INP_START:_S28_INP_END])),
                'output_activity':    float(np.mean(arr_a[:, _S28_OUT_START:_S28_OUT_END])),
                'internal_activity':  float(np.mean(arr_a[:, _S28_INT_START:_S28_INT_END])),
                'resource_mean':      float(np.mean(arr_r)),
                'resource_sensory':   float(np.mean(arr_r[:, _S28_INP_START:_S28_INP_END])),
                'resource_internal':  float(np.mean(arr_r[:, _S28_INT_START:_S28_INT_END])),
                'pred_dormant_ratio': dorm_ratio,
            })
            chunk_acts  = []
            chunk_ress  = []
            chunk_dorms = []

    return {
        'chunks':     chunks,
        'food_total': food_total,
        'pred_hits':  pred_hits,
    }


def run_exp_c_sleep_pattern(best_genome, pursuit_prob, seed=_S29_SEED):
    """T=2000ステップ連続実行で睡眠様状態と捕食者休眠の相関を観察。"""
    print(f'\n  [Exp C] 睡眠様状態の観察 (T={_S29_SLEEP_T}ステップ, pp={pursuit_prob})')
    G_copy = best_genome['G'].copy()
    W_copy = _s28_get_W(G_copy)
    rng    = np.random.default_rng(seed + 29100)
    res    = _s29_run_long(G_copy, W_copy, best_genome, rng, pursuit_prob=pursuit_prob)
    n_c    = len(res['chunks'])
    if n_c > 0:
        dorm_mean = float(np.mean([c['pred_dormant_ratio'] for c in res['chunks']]))
        print(f'  → chunks={n_c}  food={res["food_total"]}  '
              f'pred_hits={res["pred_hits"]}  '
              f'pred_dormant_ratio_mean={dorm_mean:.3f}')
    res['pursuit_prob'] = pursuit_prob
    return res


def plot_exp_c_sleep_pattern(exp_c,
                               fname='images/session_29/results_s29_sleep_pattern.png'):
    chunks = exp_c['chunks']
    pp     = exp_c['pursuit_prob']
    if not chunks:
        print('Warning: no chunks recorded for Exp C')
        return

    t_vals  = [c['t_end']              for c in chunks]
    sens    = [c['sensory_activity']   for c in chunks]
    out     = [c['output_activity']    for c in chunks]
    intern  = [c['internal_activity']  for c in chunks]
    res_all = [c['resource_mean']      for c in chunks]
    res_sen = [c['resource_sensory']   for c in chunks]
    res_int = [c['resource_internal']  for c in chunks]
    dorm    = [c['pred_dormant_ratio'] for c in chunks]

    # Correlation: predator dormancy → agent activity
    corr_sens, p_corr = scipy_stats.pearsonr(dorm, sens) if len(dorm) > 2 else (0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'Session 29 Exp C: 睡眠様状態の観察 (T={_S29_SLEEP_T}, pp={pp})\n'
        f'捕食者休眠率と感覚器活動の相関 r={corr_sens:.3f} (p={p_corr:.3f})',
        fontsize=12,
    )

    # Panel 1: layer activity + predator dormancy
    ax = axes[0][0]
    ax.plot(t_vals, sens,   color='steelblue', linewidth=2, marker='o', markersize=4,
            label='感覚器 (node0-4)')
    ax.plot(t_vals, out,    color='tomato',    linewidth=2, marker='s', markersize=4,
            label='出力   (node5-9)')
    ax.plot(t_vals, intern, color='green',     linewidth=2, marker='^', markersize=4,
            label='内部   (node10-20)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean activity')
    ax.set_title('層別平均活動の時系列')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: predator dormancy ratio + resource
    ax = axes[0][1]
    ax2 = ax.twinx()
    ax.fill_between(t_vals, dorm, alpha=0.3, color='gray', label='捕食者休眠率')
    ax.plot(t_vals, dorm, color='gray', linewidth=1, marker='o', markersize=3)
    ax2.plot(t_vals, res_all, color='purple', linewidth=2, marker='s', markersize=4,
             label='エージェント資源(平均)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Pred dormant ratio', color='gray')
    ax2.set_ylabel('Agent resources', color='purple')
    ax.set_ylim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax.set_title('捕食者休眠率 vs エージェント資源量')
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: scatter - pred dormancy vs agent activity
    ax = axes[1][0]
    ax.scatter(dorm, sens, color='steelblue', alpha=0.7, s=40, label='感覚器活動')
    ax.scatter(dorm, intern, color='green', alpha=0.7, s=40, marker='^', label='内部活動')
    # Regression lines
    if len(dorm) > 2:
        dorm_arr = np.array(dorm)
        for y_arr, col, lbl in [(np.array(sens), 'steelblue', '感覚器'),
                                  (np.array(intern), 'green', '内部')]:
            m, b = np.polyfit(dorm_arr, y_arr, 1)
            xs_r = np.array([min(dorm), max(dorm)])
            ax.plot(xs_r, m * xs_r + b, color=col, linewidth=2,
                    label=f'{lbl}: r={np.corrcoef(dorm_arr, y_arr)[0,1]:.3f}')
    ax.set_xlabel('Pred dormant ratio')
    ax.set_ylabel('Agent activity')
    ax.set_title('捕食者休眠率 × エージェント活動\n(正相関 = 捕食者休眠中に活動増加?)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: resource time series
    ax = axes[1][1]
    ax.plot(t_vals, res_all, color='purple',    linewidth=2, marker='o', markersize=4,
            label='全ノード平均')
    ax.plot(t_vals, res_sen, color='steelblue', linewidth=2, linestyle='--', marker='s',
            markersize=4, label='感覚器資源')
    ax.plot(t_vals, res_int, color='green',     linewidth=2, linestyle='--', marker='^',
            markersize=4, label='内部資源')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean resources')
    ax.set_title(f'資源量の時系列\nfood={exp_c["food_total"]}  hits={exp_c["pred_hits"]}')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Experiment D: predator fatigue speed sweep ─────────────────────────────────

def run_exp_d_fatigue_sweep(seed=_S29_SEED, n_gen=_S29_N_GEN_D):
    """捕食者疲労速度スイープ (pursuit_prob=0.6固定, pred_depletion変化).

    Returns list of dicts per pred_depletion value.
    """
    print(f'\n  [Exp D] 捕食者疲労速度スイープ '
          f'(depletions={_S29_PRED_DEPLETIONS}, pp={_S29_PURSUIT_PROB_DEF}, '
          f'{n_gen}世代/条件)')
    sweep = []

    for depl in _S29_PRED_DEPLETIONS:
        print(f'\n  pred_depletion={depl}:')
        best, hist = _s29_evolve(
            pursuit_prob=_S29_PURSUIT_PROB_DEF,
            predator_speed=_S29_PRED_SPEED,
            pred_depletion=depl, seed=seed, n_gen=n_gen)

        # Measure context dependence
        mcd, _ = _s29_mean_cosine_dist(
            best, seed + 29500 + int(depl * 1000), _S29_PURSUIT_PROB_DEF, depl)

        # Estimate sleep opportunity: run a short long-run to get dormancy ratio
        G_copy = best['G'].copy()
        W_copy = _s28_get_W(G_copy)
        rng_d  = np.random.default_rng(seed + 29600 + int(depl * 1000))
        long_res = _s29_run_long(G_copy, W_copy, best, rng_d,
                                  pursuit_prob=_S29_PURSUIT_PROB_DEF,
                                  pred_depletion=depl,
                                  T=_S29_SLEEP_T)
        dorm_ratio = float(np.mean(
            [c['pred_dormant_ratio'] for c in long_res['chunks']])) if long_res['chunks'] else 0.0
        agent_res  = float(np.mean(
            [c['resource_mean'] for c in long_res['chunks']])) if long_res['chunks'] else 0.0

        print(f'  → steps={hist["gen_best_steps"][-1]:.1f}  '
              f'depl_agent={best["depletion_rate"]:.3f}  '
              f'mcd={mcd:.4f}  '
              f'pred_dormant_ratio={dorm_ratio:.3f}')

        sweep.append({
            'pred_depletion':   depl,
            'best_genome':      best,
            'hist':             hist,
            'mean_cosine_dist': mcd,
            'best_steps':       hist['gen_best_steps'][-1],
            'depl_agent':       best['depletion_rate'],
            'pred_hits':        hist['gen_pred_hits'][-1],
            'pred_dormant_ratio': dorm_ratio,
            'agent_resource':   agent_res,
        })

    return sweep


def plot_exp_d_fatigue_sweep(sweep,
                               fname='images/session_29/results_s29_predator_fatigue.png'):
    depls       = [r['pred_depletion']    for r in sweep]
    mcds        = [r['mean_cosine_dist']  for r in sweep]
    steps       = [r['best_steps']        for r in sweep]
    dorm_ratios = [r['pred_dormant_ratio'] for r in sweep]
    hits        = [r['pred_hits']         for r in sweep]
    agent_res   = [r['agent_resource']    for r in sweep]

    xs    = np.arange(len(depls))
    xlbls = [f'depl={d}' for d in depls]
    colors = ['royalblue', 'seagreen', 'tomato', 'darkorange']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Session 29 Exp D: 捕食者疲労速度スイープ\n'
        f'pursuit_prob={_S29_PURSUIT_PROB_DEF}  recovery={_S29_PRED_RECOVERY}  '
        f'{_S29_N_GEN_D}世代/条件',
        fontsize=12,
    )

    # Panel 1: predator dormancy ratio
    ax = axes[0][0]
    for xi, (v, c) in enumerate(zip(dorm_ratios, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Pred dormant ratio')
    ax.set_title('捕食者休眠率\n(高い = 疲れやすい = エージェントの隙が多い)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: context dependence
    ax = axes[0][1]
    for xi, (v, c) in enumerate(zip(mcds, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Mean cosine dist')
    ax.set_title('文脈依存性の強さ\n(高い = 文脈間で出力差が大きい)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: survival steps
    ax = axes[0][2]
    for xi, (v, c) in enumerate(zip(steps, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Best steps / ep')
    ax.set_title('生存ステップ数')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: predator hits
    ax = axes[1][0]
    for xi, (v, c) in enumerate(zip(hits, colors)):
        ax.bar(xi, v, color=c, alpha=0.85, edgecolor='white')
        ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel('Pred hits / ep')
    ax.set_title('捕食者ヒット数')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: agent resource vs dormancy scatter
    ax = axes[1][1]
    for (d, dr, ar, c) in zip(depls, dorm_ratios, agent_res, colors):
        ax.scatter(dr, ar, color=c, s=100, zorder=3, label=f'depl={d}')
    ax.set_xlabel('Pred dormant ratio')
    ax.set_ylabel('Agent resource (mean)')
    ax.set_title('捕食者休眠率 × エージェント資源量\n'
                 '(右上 = 休眠多い → 資源が豊富 = 睡眠機会あり)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: summary
    ax = axes[1][2]
    best_r = max(sweep, key=lambda r: r['pred_dormant_ratio'])
    summary_lines = ['捕食者疲労速度別サマリー\n']
    for r in sweep:
        marker = '★' if r['pred_depletion'] == best_r['pred_depletion'] else '  '
        summary_lines.append(
            f'{marker}depl={r["pred_depletion"]:.2f}: '
            f'dorm={r["pred_dormant_ratio"]:.3f}  '
            f'mcd={r["mean_cosine_dist"]:.4f}  '
            f'steps={r["best_steps"]:.0f}'
        )
    summary_lines.append(f'\n最大休眠率: pred_depl={best_r["pred_depletion"]}')
    ax.text(0.5, 0.5, '\n'.join(summary_lines), transform=ax.transAxes,
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.set_title('実験サマリー')
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 29: 追跡性を持つ捕食者 ===')
    print(f'ネットワーク: {_S29_N}ノード '
          f'(入力{_S28_INP_END}+出力{_S28_OUT_END-_S28_OUT_START}+'
          f'内部{_S28_INT_END-_S28_INT_START})')
    print(f'固定tau: tau_s={_S28_TAU_S}  tau_i={_S28_TAU_I}  tau_o={_S28_TAU_O}')
    print(f'捕食者追跡: depletion={_S29_PRED_DEPLETION}  '
          f'recovery={_S29_PRED_RECOVERY}  speed={_S29_PRED_SPEED}')
    print(f'進化対象: depletion_rate, edge_add_prob, activity_ratio')
    print()

    # Experiment A: pursuit_prob sweep
    print('[Exp A] pursuit_prob スイープ')
    sweep_a = run_exp_a_pursuit_sweep(seed=_S29_SEED)
    plot_exp_a_pursuit_sweep(sweep_a)

    # Find best condition (highest context dependence) and closest to pp=0.6
    best_a = max(sweep_a, key=lambda r: r['mean_cosine_dist'])
    print(f'\n  ベスト条件: pursuit_prob={best_a["pursuit_prob"]}  '
          f'mcd={best_a["mean_cosine_dist"]:.4f}')
    # Use pp=0.6 genome if available (spec prediction), else use best
    cands_06 = [r for r in sweep_a if r['pursuit_prob'] == 0.6]
    exp_b_genome = cands_06[0]['best_genome'] if cands_06 else best_a['best_genome']
    exp_b_pp     = cands_06[0]['pursuit_prob'] if cands_06 else best_a['pursuit_prob']

    # Experiment B: context measurement
    print('\n[Exp B] 文脈依存的な行動の計測')
    exp_b = run_exp_b_context(exp_b_genome, exp_b_pp, seed=_S29_SEED)
    plot_exp_b_context(exp_b)

    # Experiment C: sleep pattern
    print('\n[Exp C] 睡眠様状態の観察')
    exp_c = run_exp_c_sleep_pattern(exp_b_genome, exp_b_pp, seed=_S29_SEED)
    plot_exp_c_sleep_pattern(exp_c)

    # Experiment D: predator fatigue speed sweep
    print('\n[Exp D] 捕食者疲労速度スイープ')
    sweep_d = run_exp_d_fatigue_sweep(seed=_S29_SEED)
    plot_exp_d_fatigue_sweep(sweep_d)

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('=== Session 29 Summary ===')
    print()

    print('[A] pursuit_prob スイープ')
    for r in sweep_a:
        dom = _S28_ACTION_NAMES
        print(f'  pp={r["pursuit_prob"]:.1f}: '
              f'steps={r["best_steps"]:.1f}  '
              f'mcd={r["mean_cosine_dist"]:.4f}  '
              f'depl={r["depl"]:.3f}  '
              f'hits={r["pred_hits"]:.2f}/ep')
    print(f'  → 最大文脈依存性: pp={best_a["pursuit_prob"]}  '
          f'mcd={best_a["mean_cosine_dist"]:.4f}')

    print('\n[B] 文脈依存性 (pp={})'.format(exp_b_pp))
    print(f'  mean cosine dist = {exp_b["mean_cosine_dist"]:.4f}  '
          f'({"✓ 文脈分離あり" if exp_b["mean_cosine_dist"] > 0.05 else "✗ 分離弱い"})')
    for ctx, res in zip(_S28_CONTEXTS, exp_b['context_results']):
        dom_i = int(np.argmax(res['action_dist']))
        dom   = _S28_ACTION_NAMES[dom_i]
        print(f'  [{ctx["label"].replace(chr(10)," ")}]: '
              f'主行動={dom} ({res["action_dist"][dom_i]:.0%})')

    print('\n[C] 睡眠様状態 (pp={})'.format(exp_b_pp))
    if exp_c['chunks']:
        dorm_mean = float(np.mean([c['pred_dormant_ratio'] for c in exp_c['chunks']]))
        dorm_arr  = np.array([c['pred_dormant_ratio'] for c in exp_c['chunks']])
        sens_arr  = np.array([c['sensory_activity']   for c in exp_c['chunks']])
        corr, p   = scipy_stats.pearsonr(dorm_arr, sens_arr) if len(dorm_arr) > 2 else (0, 1)
        print(f'  捕食者休眠率(平均): {dorm_mean:.3f}')
        print(f'  休眠率 × 感覚器活動 相関: r={corr:.3f} p={p:.4f}  '
              f'({"✓ 有意な相関" if p < 0.05 else "✗ 有意差なし"})')
        print(f'  食料獲得: {exp_c["food_total"]}  捕食者ヒット: {exp_c["pred_hits"]}')

    print('\n[D] 捕食者疲労速度スイープ')
    for r in sweep_d:
        print(f'  pred_depl={r["pred_depletion"]:.2f}: '
              f'dorm={r["pred_dormant_ratio"]:.3f}  '
              f'mcd={r["mean_cosine_dist"]:.4f}  '
              f'steps={r["best_steps"]:.1f}')
    best_d = max(sweep_d, key=lambda r: r['pred_dormant_ratio'])
    print(f'  → 最大休眠率: pred_depl={best_d["pred_depletion"]}  '
          f'dorm={best_d["pred_dormant_ratio"]:.3f}')

    print('\nDone.')
