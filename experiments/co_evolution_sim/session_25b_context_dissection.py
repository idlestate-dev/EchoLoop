"""Session 25b: 文脈分離の萌芽の深掘り

Session 25 の発見:
  mr=0.01: acc_B=0.310  (初めて acc_B > 0 が出た)
  mr=0.00: acc_A=0.880, acc_B=0.000
  mr=0.05: acc_B=0.110

問い:
  A. acc_B=0.310 は本物か？（ランダムと有意に違うか）
  B. hunger_threshold のスイープ → 飢餓が強いほど acc_B が上がるか？
  C. mr=0.00 vs mr=0.01 の構造的な違いを解剖する
  D. penalty_damage のスイープ → ペナルティが強いほど文脈分離が起きやすくなるか？

出力:
  results_s25b_significance.png
  results_s25b_hunger_sweep.png
  results_s25b_structure.png
  results_s25b_penalty_sweep.png
"""

import os
import numpy as np
import _jp_font  # noqa: F401 — sets Japanese font in matplotlib rcParams
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from session_10_embodied_output import (
    _s10_build_graph, _s10_get_W, _s10_propagate, _s10_mutate,
    _N, _K, _N_PROP,
)
from session_12_sleep_consolidation import _s12_consolidation_phase
from session_18_ratio_evolution import (
    _s18_hebb,
    _ACTIVITY_NOISE, _T_CONSOLIDATION,
    _N_GEN, _N_SURV,
    _EP_INIT_MAX, _AR_INIT_MAX,
    _EP_MUT_STD, _AR_MUT_STD,
)
from session_19_context_reboot import _CGRID, _CHP_MAX, _CRESPAWN, _CSTEPS
from session_20_penalty_context import _s20_inp4

from session_25_hunger_learning import (
    _SEED, _N_AGENTS, _N_EP, _ACT_THRESHOLD,
    _INP_START, _INP_END, _OUT_START, _OUT_END, _INT_START, _INT_END,
    _S25_HP_START, _S25_HP_DECAY, _S25_FOOD_VALUE,
    _HUNGER_THR_LO, _HUNGER_THR_HI, _HUNGER_PEN_LO, _HUNGER_PEN_HI,
    _HUNGER_THR_STD, _HUNGER_PEN_STD,
    _s25_make_genome, _s25_mutate_genome,
    _s25_run_ep_hunger_dynamic, _s25_run_ep_hunger_penalty,
    _s25_evolve_mr,
)

_S25B_SEED = 42

# Exp B: hunger_threshold sweep
_S25B_HUNGER_THRESHOLDS = [20, 40, 60, 80, 100]
_S25B_HUNGER_COLORS = {20: '#d62728', 40: '#ff7f0e', 60: '#2ca02c',
                        80: '#1f77b4', 100: '#9467bd'}

# Exp D: penalty sweep — override _PENALTY from session_20
_S25B_PENALTY_DAMAGES = [10, 30, 50, 100, 200]
_S25B_PENALTY_COLORS  = {10: '#aec7e8', 30: '#1f77b4', 50: '#ffbb78',
                          100: '#d62728', 200: '#7f0000'}

_DEFAULT_PENALTY = 10  # session_20._PENALTY value

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _s25b_run_ep_penalty_custom(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                                 hunger_threshold, hunger_penalty,
                                 rng, mode=None,
                                 activity_noise=_ACTIVITY_NOISE,
                                 T_consolidation=_T_CONSOLIDATION,
                                 record_activity=False,
                                 penalty_damage=_DEFAULT_PENALTY):
    """PenaltyContextGridWorld with configurable penalty_damage.

    Identical to _s25_run_ep_hunger_penalty except penalty_damage is a parameter.
    """
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'

    food_pos    = (0, 0) if mode == 'A' else (4, 4)
    penalty_pos = (4, 4) if mode == 'A' else (0, 0)
    food_avail  = True
    food_timer  = 0

    activity  = np.zeros(_N)
    row, col  = 2, 2
    hp        = float(_S25_HP_START)
    hunger    = 0
    steps     = 0
    food      = 0
    penalties = 0
    records   = [] if record_activity else None

    fr, fc = food_pos
    pr, pc = penalty_pos

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

        hp -= metabolic_rate * float(np.sum(activity))

        action = int(np.argmax(activity[_OUT_START:_OUT_END]))

        food_eaten = False
        if action in (0, 1, 2, 3):
            if   action == 0: row = max(0, row - 1)
            elif action == 1: row = min(_CGRID - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(_CGRID - 1, col + 1)
            if row == pr and col == pc:
                hp -= penalty_damage
                penalties += 1
        elif action == 4:
            if row == fr and col == fc and food_avail:
                hp = min(_CHP_MAX, hp + _S25_FOOD_VALUE)
                food_avail = False
                food_timer = 0
                food_eaten = True
                food += 1

        hunger += 1
        if food_eaten:
            hunger = 0
        elif hunger > hunger_threshold:
            hp -= hunger_penalty

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


def _s25b_eval_acc(genome, rng, n_per_mode=25, penalty_damage=_DEFAULT_PENALTY):
    """モードA・B それぞれ n_per_mode エピソード評価 → (acc_A, acc_B, food_A_list, food_B_list)."""
    G = genome['G'].copy()
    W = _s10_get_W(G)
    food_A, food_B = [], []
    for ei in range(n_per_mode * 2):
        mode   = 'A' if ei < n_per_mode else 'B'
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        _, f, _, _, _ = _s25b_run_ep_penalty_custom(
            G, W,
            genome['edge_add_prob'], genome['activity_ratio'],
            genome['metabolic_rate'],
            genome['hunger_threshold'], genome['hunger_penalty'],
            ep_rng, mode=mode,
            penalty_damage=penalty_damage)
        if mode == 'A':
            food_A.append(f)
        else:
            food_B.append(f)
    acc_A = float(np.mean(np.array(food_A) >= 1))
    acc_B = float(np.mean(np.array(food_B) >= 1))
    return acc_A, acc_B, food_A, food_B


# ── Experiment A: statistical significance ────────────────────────────────────

def run_exp_a_significance(exp25_results, seed=_S25B_SEED, n_per_mode=25):
    """mr=0.01 のベスト個体を n_per_mode×2 エピソードで評価し acc_B の信頼区間を計算。

    Returns dict: acc_A, acc_B, ci95_B, p_value_B, food_A_list, food_B_list.
    """
    print(f'\n  [Exp A] 統計的有意性: mr=0.01 best / n_per_mode={n_per_mode}')
    genome = exp25_results[0.01]['best']
    rng    = np.random.default_rng(seed + 25001)

    acc_A, acc_B, food_A, food_B = _s25b_eval_acc(genome, rng, n_per_mode=n_per_mode)

    # binomial 95% CI for acc_B — Wilson interval (manual, no statsmodels needed)
    n_B  = len(food_B)
    k_B  = sum(f >= 1 for f in food_B)
    z    = scipy_stats.norm.ppf(0.975)   # 1.96 for 95% CI
    p_hat = k_B / n_B if n_B > 0 else 0.0
    denom = 1 + z**2 / n_B
    center = (p_hat + z**2 / (2 * n_B)) / denom
    margin = z * (p_hat * (1 - p_hat) / n_B + z**2 / (4 * n_B**2))**0.5 / denom
    ci_lo, ci_hi = max(0.0, center - margin), min(1.0, center + margin)

    # one-sided binomial test: acc_B > random_baseline
    # random baseline: we test against p=0.30 (conservative estimate from session 25)
    random_baseline = 0.30
    p_val = scipy_stats.binomtest(k_B, n_B, p=random_baseline, alternative='greater').pvalue

    print(f'    acc_A={acc_A:.3f}  acc_B={acc_B:.3f}  k_B={k_B}/{n_B}')
    print(f'    95% CI for acc_B: [{ci_lo:.3f}, {ci_hi:.3f}]')
    print(f'    one-sided test vs random_baseline={random_baseline}: p={p_val:.4f}')

    return {
        'acc_A': acc_A, 'acc_B': acc_B,
        'food_A': food_A, 'food_B': food_B,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'p_value': p_val, 'random_baseline': random_baseline,
        'k_B': k_B, 'n_B': n_B,
    }


# ── Experiment B: hunger_threshold sweep ─────────────────────────────────────

def _s25b_evolve_fixed_hunger_thr(hunger_threshold, seed):
    """hunger_threshold を固定して mr=0.01 で 50 世代進化。

    hunger_penalty のみ進化させる（threshold は固定）。
    Returns (best_genome, history_dict).
    """
    rng = np.random.default_rng(seed + 25010 + hunger_threshold)
    pop = []
    for _ in range(_N_AGENTS):
        g = _s25_make_genome(rng, metabolic_rate=0.01)
        g['hunger_threshold'] = hunger_threshold  # fix
        pop.append(g)

    hist = {k: [] for k in ('gen_best_steps', 'gen_food_count',
                              'gen_mean_active', 'gen_hunger_penalty')}

    for gen in range(_N_GEN):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_active = 0, [], []
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, _, recs, _ = _s25_run_ep_hunger_dynamic(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'],
                    g['hunger_threshold'], g['hunger_penalty'],
                    ep_rng, record_activity=True)
                total += s
                ep_food.append(f)
                if recs:
                    arr = np.array(recs)
                    ep_active.append(float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD)))
            fitnesses.append(total / _N_EP)
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0
            g['_ep_food']   = float(np.mean(ep_food))

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]
        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_hunger_penalty'].append(float(bg['hunger_penalty']))

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            child  = _s25_mutate_genome(parent, rng)
            child['hunger_threshold'] = hunger_threshold  # keep fixed
            new_pop.append(child)
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'      gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'pen={bg["hunger_penalty"]:.2f}  active={bg["_ep_active"]:.1f}')

    for g in pop:
        g.pop('_ep_active', None)
        g.pop('_ep_food', None)

    return pop[0], hist


def run_exp_b_hunger_sweep(seed=_S25B_SEED, n_per_mode=10):
    """hunger_threshold [20,40,60,80,100] でスイープ、mr=0.01 固定 50 世代。

    Returns dict keyed by hunger_threshold: { best, hist, acc_A, acc_B, food_per_ep }.
    """
    print('\n  [Exp B] hunger_threshold スイープ (mr=0.01, 50 世代)')
    results = {}
    for ht in _S25B_HUNGER_THRESHOLDS:
        print(f'\n    hunger_threshold={ht}')
        best, hist = _s25b_evolve_fixed_hunger_thr(ht, seed)
        rng = np.random.default_rng(seed + 25020 + ht)
        acc_A, acc_B, _, _ = _s25b_eval_acc(best, rng, n_per_mode=n_per_mode)
        food_per_ep = hist['gen_food_count'][-1]
        print(f'    → food/ep={food_per_ep:.2f}  acc_A={acc_A:.3f}  acc_B={acc_B:.3f}')
        results[ht] = {'best': best, 'hist': hist,
                        'acc_A': acc_A, 'acc_B': acc_B, 'food_per_ep': food_per_ep}
    return results


# ── Experiment C: structural comparison mr=0.00 vs mr=0.01 ───────────────────

def _s25b_collect_activity_patterns(genome, seed, n_per_mode=20, penalty_damage=_DEFAULT_PENALTY):
    """モード A / B ごとに output ノードの活動パターンを収集。

    Returns dict: mean_A, mean_B (shape [5]), std_A, std_B,
                  cosine_dist, weight_stats.
    """
    rng    = np.random.default_rng(seed)
    G = genome['G'].copy()
    W = _s10_get_W(G)
    out_A, out_B = [], []
    inp_A, inp_B = [], []

    for ei in range(n_per_mode * 2):
        mode   = 'A' if ei < n_per_mode else 'B'
        ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        _, _, _, _, recs = _s25b_run_ep_penalty_custom(
            G, W,
            genome['edge_add_prob'], genome['activity_ratio'],
            genome['metabolic_rate'],
            genome['hunger_threshold'], genome['hunger_penalty'],
            ep_rng, mode=mode, record_activity=True,
            penalty_damage=penalty_damage)
        if recs:
            arr = np.array(recs)
            mean_out = arr[:, _OUT_START:_OUT_END].mean(axis=0)
            mean_inp = arr[:, _INP_START:_INP_END].mean(axis=0)
            if mode == 'A':
                out_A.append(mean_out)
                inp_A.append(mean_inp)
            else:
                out_B.append(mean_out)
                inp_B.append(mean_inp)

    mA = np.mean(out_A, axis=0) if out_A else np.zeros(_OUT_END - _OUT_START)
    mB = np.mean(out_B, axis=0) if out_B else np.zeros(_OUT_END - _OUT_START)
    sA = np.std(out_A,  axis=0) if out_A else np.zeros(_OUT_END - _OUT_START)
    sB = np.std(out_B,  axis=0) if out_B else np.zeros(_OUT_END - _OUT_START)

    nA, nB = np.linalg.norm(mA), np.linalg.norm(mB)
    cos_dist = 1.0 - float(np.dot(mA, mB) / (nA * nB)) if (nA > 1e-9 and nB > 1e-9) else 0.0

    weights = np.array([d.get('weight', 0.0) for _, _, d in G.edges(data=True)])
    weight_stats = {
        'mean': float(np.mean(weights)) if len(weights) > 0 else 0.0,
        'std':  float(np.std(weights))  if len(weights) > 0 else 0.0,
        'abs_mean': float(np.mean(np.abs(weights))) if len(weights) > 0 else 0.0,
        'n_edges': len(weights),
    }

    return {
        'mean_A': mA, 'mean_B': mB, 'std_A': sA, 'std_B': sB,
        'inp_mean_A': np.mean(inp_A, axis=0) if inp_A else np.zeros(_INP_END - _INP_START),
        'inp_mean_B': np.mean(inp_B, axis=0) if inp_B else np.zeros(_INP_END - _INP_START),
        'cosine_dist': cos_dist,
        'weight_stats': weight_stats,
    }


def run_exp_c_structure(exp25_results, seed=_S25B_SEED, n_per_mode=20):
    """mr=0.00 と mr=0.01 のベスト個体の活動パターン・重み分布を比較。

    Returns dict keyed by mr.
    """
    print('\n  [Exp C] 構造比較: mr=0.00 vs mr=0.01 (n_per_mode={})'.format(n_per_mode))
    results = {}
    for mr in [0.0, 0.01]:
        genome = exp25_results[mr]['best']
        pat = _s25b_collect_activity_patterns(
            genome, seed + 25030 + int(mr * 1000), n_per_mode=n_per_mode)
        ws = pat['weight_stats']
        print(f'    mr={mr:.2f}: cos_dist={pat["cosine_dist"]:.4f}  '
              f'edges={ws["n_edges"]}  w_mean={ws["mean"]:.4f}  w_abs={ws["abs_mean"]:.4f}')
        print(f'      out_A={np.round(pat["mean_A"], 3)}')
        print(f'      out_B={np.round(pat["mean_B"], 3)}')
        results[mr] = pat
    return results


# ── Experiment D: penalty_damage sweep ────────────────────────────────────────

def _s25b_evolve_fixed_penalty(penalty_damage, seed):
    """penalty_damage を固定して mr=0.01, hunger_threshold=50 で 50 世代進化。

    Returns (best_genome, history_dict).
    """
    fixed_ht  = 50
    rng = np.random.default_rng(seed + 25040 + penalty_damage)
    pop = []
    for _ in range(_N_AGENTS):
        g = _s25_make_genome(rng, metabolic_rate=0.01)
        g['hunger_threshold'] = fixed_ht
        pop.append(g)

    hist = {k: [] for k in ('gen_best_steps', 'gen_food_count',
                              'gen_mean_active', 'gen_mean_penalties')}

    for gen in range(_N_GEN):
        fitnesses = []
        for g in pop:
            total, ep_food, ep_active, ep_pen = 0, [], [], []
            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, f, pens, recs, _ = _run_ep_hunger_dynamic_with_penalty(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'],
                    g['hunger_threshold'], g['hunger_penalty'],
                    ep_rng, penalty_damage=penalty_damage, record_activity=True)
                total += s
                ep_food.append(f)
                ep_pen.append(pens)
                if recs:
                    arr = np.array(recs)
                    ep_active.append(float(np.sum(np.mean(arr, axis=0) > _ACT_THRESHOLD)))
            fitnesses.append(total / _N_EP)
            g['_ep_active']     = float(np.mean(ep_active))   if ep_active else 0.0
            g['_ep_food']       = float(np.mean(ep_food))
            g['_ep_penalties']  = float(np.mean(ep_pen))

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]
        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_mean_penalties'].append(bg['_ep_penalties'])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]
        new_pop    = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            child  = _s25_mutate_genome(parent, rng)
            child['hunger_threshold'] = fixed_ht
            new_pop.append(child)
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'      gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'penalties={bg["_ep_penalties"]:.2f}/ep  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        g.pop('_ep_active', None)
        g.pop('_ep_food', None)
        g.pop('_ep_penalties', None)

    return pop[0], hist


def _run_ep_hunger_dynamic_with_penalty(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                                         hunger_threshold, hunger_penalty,
                                         rng, penalty_damage=_DEFAULT_PENALTY,
                                         activity_noise=_ACTIVITY_NOISE,
                                         T_consolidation=_T_CONSOLIDATION,
                                         record_activity=False):
    """HungerDynamicGridWorld with configurable penalty_damage.

    Returns (steps, food, penalties_count, records_or_None, layer_stats_dict).
    Extended return: (steps, food, penalties_int, penalties_int, records, stats).
    """
    all_pos     = [(r, c) for r in range(_CGRID) for c in range(_CGRID)]
    food_pos    = all_pos[int(rng.integers(0, len(all_pos)))]
    remaining   = [p for p in all_pos if p != food_pos]
    penalty_pos = remaining[int(rng.integers(0, len(remaining)))]

    food_avail = True
    food_timer = 0
    activity   = np.zeros(_N)
    row, col   = 2, 2
    hp         = float(_S25_HP_START)
    hunger     = 0
    steps      = 0
    food       = 0
    penalties  = 0
    records    = [] if record_activity else None

    fr, fc = food_pos
    pr, pc = penalty_pos

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

        hp -= metabolic_rate * float(np.sum(activity))

        action = int(np.argmax(activity[_OUT_START:_OUT_END]))

        food_eaten = False
        if action in (0, 1, 2, 3):
            if   action == 0: row = max(0, row - 1)
            elif action == 1: row = min(_CGRID - 1, row + 1)
            elif action == 2: col = max(0, col - 1)
            elif action == 3: col = min(_CGRID - 1, col + 1)
            if row == pr and col == pc:
                hp -= penalty_damage
                penalties += 1
        elif action == 4:
            if row == fr and col == fc and food_avail:
                hp = min(_CHP_MAX, hp + _S25_FOOD_VALUE)
                food_avail = False
                food_timer = 0
                food_eaten = True
                food += 1

        hunger += 1
        if food_eaten:
            hunger = 0
        elif hunger > hunger_threshold:
            hp -= hunger_penalty

        steps = step + 1

        if not food_avail:
            food_timer += 1
            if food_timer >= _CRESPAWN:
                food_avail = True
                food_timer = 0

        if (step + 1) % _K == 0:
            _s18_hebb(G, W, activity, rng, edge_add_prob, activity_ratio)

    _s12_consolidation_phase(G, W, activity, rng, T_consolidation)

    layer_stats = {}
    if records:
        arr = np.array(records)
        layer_stats['output_variance'] = float(np.mean(np.var(arr[:, _OUT_START:_OUT_END], axis=1)))

    return steps, food, penalties, records, layer_stats


def run_exp_d_penalty_sweep(seed=_S25B_SEED, n_per_mode=10):
    """penalty_damage スイープ (mr=0.01, hunger_threshold=50, 50 世代)。

    Returns dict keyed by penalty_damage.
    """
    print('\n  [Exp D] penalty_damage スイープ (mr=0.01, hunger_thr=50, 50 世代)')
    results = {}
    for pd in _S25B_PENALTY_DAMAGES:
        print(f'\n    penalty_damage={pd}')
        best, hist = _s25b_evolve_fixed_penalty(pd, seed)
        rng = np.random.default_rng(seed + 25050 + pd)
        acc_A, acc_B, _, _ = _s25b_eval_acc(best, rng, n_per_mode=n_per_mode,
                                              penalty_damage=pd)
        mean_pen = hist['gen_mean_penalties'][-1]
        food_per_ep = hist['gen_food_count'][-1]
        print(f'    → food/ep={food_per_ep:.2f}  acc_A={acc_A:.3f}  acc_B={acc_B:.3f}  '
              f'mean_pen={mean_pen:.2f}/ep')
        results[pd] = {'best': best, 'hist': hist,
                        'acc_A': acc_A, 'acc_B': acc_B,
                        'food_per_ep': food_per_ep, 'mean_penalties': mean_pen}
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_a_significance(exp_a,
                             fname='images/session_25/results_s25b_significance.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Session 25b Exp A: acc_B=0.310 の統計的有意性\n'
        f'mr=0.01 ベスト個体  n_per_mode={exp_a["n_B"]}  '
        f'ランダム基準={exp_a["random_baseline"]:.2f}  p={exp_a["p_value"]:.4f}',
        fontsize=11,
    )

    food_A = np.array(exp_a['food_A'])
    food_B = np.array(exp_a['food_B'])
    n_B    = exp_a['n_B']

    # Panel 1: food count distribution
    ax = axes[0]
    bins = np.arange(-0.5, max(max(food_A), max(food_B)) + 1.5, 1)
    ax.hist(food_A, bins=bins, alpha=0.6, color='steelblue', label=f'Mode A  acc={exp_a["acc_A"]:.3f}')
    ax.hist(food_B, bins=bins, alpha=0.6, color='tomato',    label=f'Mode B  acc={exp_a["acc_B"]:.3f}')
    ax.set_xlabel('Food count per episode')
    ax.set_ylabel('Frequency')
    ax.set_title('Food count distribution\n(A vs B)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: acc_B with CI
    ax = axes[1]
    ax.bar(['acc_B'], [exp_a['acc_B']], color='tomato', alpha=0.8, width=0.4,
           label=f'acc_B={exp_a["acc_B"]:.3f}')
    ax.errorbar(['acc_B'], [exp_a['acc_B']],
                yerr=[[exp_a['acc_B'] - exp_a['ci_lo']],
                       [exp_a['ci_hi'] - exp_a['acc_B']]],
                fmt='none', color='black', capsize=8, linewidth=2, label='95% CI (Wilson)')
    ax.axhline(exp_a['random_baseline'], color='green', linestyle='--', linewidth=1.8,
               label=f'random baseline={exp_a["random_baseline"]:.2f}')
    ax.axhline(0.6, color='purple', linestyle=':', linewidth=1.5,
               label='context sep. threshold=0.6')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'acc_B 95% CI\n[{exp_a["ci_lo"]:.3f}, {exp_a["ci_hi"]:.3f}]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: cumulative acc_B over episodes
    ax = axes[2]
    cumB = np.cumsum(np.array(food_B) >= 1) / (np.arange(n_B) + 1)
    ax.plot(np.arange(1, n_B + 1), cumB, color='tomato', linewidth=2, label='cumulative acc_B')
    ax.axhline(exp_a['random_baseline'], color='green', linestyle='--', linewidth=1.5,
               label=f'random={exp_a["random_baseline"]:.2f}')
    ax.axhline(exp_a['acc_B'], color='tomato', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.set_xlabel('Episode (Mode B)')
    ax.set_ylabel('Cumulative accuracy')
    ax.set_title('Cumulative acc_B over episodes')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    sig = '有意 (p < 0.05)' if exp_a['p_value'] < 0.05 else '非有意 (p ≥ 0.05)'
    fig.text(0.5, 0.01,
             f'判定: acc_B={exp_a["acc_B"]:.3f}  95%CI=[{exp_a["ci_lo"]:.3f},{exp_a["ci_hi"]:.3f}]  {sig}',
             ha='center', fontsize=11, color='darkred' if exp_a['p_value'] < 0.05 else 'gray')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_hunger_sweep(exp_b,
                             fname='images/session_25/results_s25b_hunger_sweep.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Session 25b Exp B: hunger_threshold スイープ (mr=0.01, 50 世代)\n'
        '飢餓が強い（threshold 低）ほど文脈分離が進むか？',
        fontsize=11,
    )

    thresholds = _S25B_HUNGER_THRESHOLDS
    colors     = [_S25B_HUNGER_COLORS[ht] for ht in thresholds]
    acc_A_vals = [exp_b[ht]['acc_A']       for ht in thresholds]
    acc_B_vals = [exp_b[ht]['acc_B']       for ht in thresholds]
    food_vals  = [exp_b[ht]['food_per_ep'] for ht in thresholds]

    x = np.arange(len(thresholds))
    xlbls = [f'ht={ht}' for ht in thresholds]

    for ai, (vals, title, threshold, ylabel) in enumerate([
        (acc_A_vals, 'Mode A Accuracy', 0.6, 'acc_A'),
        (acc_B_vals, 'Mode B Accuracy', 0.6, 'acc_B'),
        (food_vals,  'Food / episode',  1.0, 'food count/ep'),
    ]):
        ax   = axes[ai]
        bars = ax.bar(x, vals, color=colors, alpha=0.82, edgecolor='white', linewidth=1.2)
        ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.85,
                   label=f'threshold={threshold}')
        for bar, val in zip(bars, vals):
            ymax = max(vals) * 1.3 + 0.05
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(vals) * 1.4, threshold * 1.5) + 0.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_structure(exp_c,
                          fname='images/session_25/results_s25b_structure.png'):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Session 25b Exp C: mr=0.00 vs mr=0.01 の構造比較\n'
        '出力ノード活動パターン (Mode A vs B) / 入力ノード / 重み分布',
        fontsize=11,
    )

    node_labels_out = [f'out{i}' for i in range(_OUT_END - _OUT_START)]
    node_labels_inp = [f'inp{i}' for i in range(_INP_END - _INP_START)]
    mr_pairs = [(0.0, '#888888'), (0.01, 'steelblue')]

    for row_idx, (mr, color) in enumerate(mr_pairs):
        pat = exp_c[mr]
        ws  = pat['weight_stats']

        # Output node activity A vs B
        ax = axes[row_idx][0]
        x  = np.arange(len(node_labels_out))
        ax.bar(x - 0.2, pat['mean_A'], 0.35, color='steelblue', alpha=0.8, label='Mode A')
        ax.bar(x + 0.2, pat['mean_B'], 0.35, color='tomato',    alpha=0.8, label='Mode B')
        ax.errorbar(x - 0.2, pat['mean_A'], yerr=pat['std_A'], fmt='none',
                    color='darkblue', capsize=4, linewidth=1.2)
        ax.errorbar(x + 0.2, pat['mean_B'], yerr=pat['std_B'], fmt='none',
                    color='darkred',  capsize=4, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(node_labels_out)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Mean activity')
        ax.set_title(f'mr={mr:.2f}  Output nodes A vs B\ncos_dist={pat["cosine_dist"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Input node activity A vs B
        ax = axes[row_idx][1]
        x2 = np.arange(len(node_labels_inp))
        ax.bar(x2 - 0.2, pat['inp_mean_A'], 0.35, color='steelblue', alpha=0.8, label='Mode A')
        ax.bar(x2 + 0.2, pat['inp_mean_B'], 0.35, color='tomato',    alpha=0.8, label='Mode B')
        ax.set_xticks(x2)
        ax.set_xticklabels(node_labels_inp)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Mean activity')
        ax.set_title(f'mr={mr:.2f}  Input nodes A vs B')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Weight distribution
        ax = axes[row_idx][2]
        info_text = (f'edges={ws["n_edges"]}\n'
                     f'w_mean={ws["mean"]:.4f}\n'
                     f'w_std={ws["std"]:.4f}\n'
                     f'|w|_mean={ws["abs_mean"]:.4f}')
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        ax.set_title(f'mr={mr:.2f}  Weight stats')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_d_penalty_sweep(exp_d,
                              fname='images/session_25/results_s25b_penalty_sweep.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Session 25b Exp D: penalty_damage スイープ (mr=0.01, hunger_thr=50, 50 世代)\n'
        'ペナルティが強いほど文脈分離が起きやすくなるか？',
        fontsize=11,
    )

    pds    = _S25B_PENALTY_DAMAGES
    colors = [_S25B_PENALTY_COLORS[pd] for pd in pds]
    acc_A_vals = [exp_d[pd]['acc_A']          for pd in pds]
    acc_B_vals = [exp_d[pd]['acc_B']          for pd in pds]
    pen_vals   = [exp_d[pd]['mean_penalties'] for pd in pds]

    x     = np.arange(len(pds))
    xlbls = [f'pen={pd}' for pd in pds]

    for ai, (vals, title, threshold, ylabel) in enumerate([
        (acc_A_vals, 'Mode A Accuracy', 0.6, 'acc_A'),
        (acc_B_vals, 'Mode B Accuracy', 0.6, 'acc_B'),
        (pen_vals,   'Mean penalties / ep', None, 'penalties/ep'),
    ]):
        ax   = axes[ai]
        bars = ax.bar(x, vals, color=colors, alpha=0.82, edgecolor='white', linewidth=1.2)
        if threshold is not None:
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.85,
                       label=f'threshold={threshold}')
            ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ymax_local = max(max(vals) * 1.4, 0.1)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax_local * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(vals) * 1.45, 0.1))
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 25b: 文脈分離の萌芽の深掘り ===')
    print('前提: Session 25 で mr=0.01 → acc_B=0.310 が初めて確認された')
    print()

    # Session 25 の進化を再実行してベスト個体を取得
    print('[前処理] Session 25 の進化を再実行してベスト個体を取得...')
    from session_25_hunger_learning import run_exp_a_genome_convergence
    exp25 = run_exp_a_genome_convergence(seed=_SEED)
    print()

    print(f'[Exp A] 統計的有意性: mr=0.01 ベスト個体 / n_per_mode=25')
    exp_a_s25b = run_exp_a_significance(exp25, seed=_S25B_SEED, n_per_mode=25)
    exp_a_s25b['n_B'] = 25
    plot_exp_a_significance(exp_a_s25b)

    print(f'\n[Exp B] hunger_threshold スイープ (mr=0.01 固定、{_N_GEN} 世代)')
    exp_b_s25b = run_exp_b_hunger_sweep(seed=_S25B_SEED, n_per_mode=10)
    plot_exp_b_hunger_sweep(exp_b_s25b)

    print('\n[Exp C] 構造比較: mr=0.00 vs mr=0.01 (n_per_mode=20)')
    exp_c_s25b = run_exp_c_structure(exp25, seed=_S25B_SEED, n_per_mode=20)
    plot_exp_c_structure(exp_c_s25b)

    print(f'\n[Exp D] penalty_damage スイープ (mr=0.01, hunger_thr=50 固定、{_N_GEN} 世代)')
    exp_d_s25b = run_exp_d_penalty_sweep(seed=_S25B_SEED, n_per_mode=10)
    plot_exp_d_penalty_sweep(exp_d_s25b)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n=== Session 25b Summary ===')

    print('\n[A] 統計的有意性')
    sig = exp_a_s25b['p_value'] < 0.05
    print(f'  acc_B={exp_a_s25b["acc_B"]:.3f}  '
          f'95%CI=[{exp_a_s25b["ci_lo"]:.3f},{exp_a_s25b["ci_hi"]:.3f}]  '
          f'p={exp_a_s25b["p_value"]:.4f}')
    print(f'  → {"✓ 有意: ランダムを上回る" if sig else "✗ 非有意: ランダムと区別できない"}')

    print('\n[B] hunger_threshold スイープ')
    print(f'  {"ht":>4}  {"acc_A":>6}  {"acc_B":>6}  {"food/ep":>8}')
    print('  ' + '─' * 32)
    for ht in _S25B_HUNGER_THRESHOLDS:
        r = exp_b_s25b[ht]
        print(f'  {ht:4d}  {r["acc_A"]:6.3f}  {r["acc_B"]:6.3f}  {r["food_per_ep"]:8.2f}')

    print('\n[C] 構造比較')
    for mr in [0.0, 0.01]:
        pat = exp_c_s25b[mr]
        ws  = pat['weight_stats']
        print(f'  mr={mr:.2f}: cos_dist={pat["cosine_dist"]:.4f}  '
              f'edges={ws["n_edges"]}  |w|_mean={ws["abs_mean"]:.4f}')

    print('\n[D] penalty_damage スイープ')
    print(f'  {"pen":>5}  {"acc_A":>6}  {"acc_B":>6}  {"penalties/ep":>13}')
    print('  ' + '─' * 35)
    for pd in _S25B_PENALTY_DAMAGES:
        r = exp_d_s25b[pd]
        print(f'  {pd:5d}  {r["acc_A"]:6.3f}  {r["acc_B"]:6.3f}  {r["mean_penalties"]:13.2f}')

    print('\nDone.')
