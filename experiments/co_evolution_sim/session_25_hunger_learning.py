"""Session 25: 生物学的な飢餓による学習の必然化

Session 24 の問題:
  hp_decayを進化させても下限（0.50）に収束
  「長寿命」が最適解になり学習が不要
  根本問題: 生存ステップ数を適応度にすると
            「何もしなくても長生きする」戦略が勝つ

設計思想:
  「食料を取らないと生き残れない」という生物学的な飢餓を実装。
  hp_decay=0（飢餓で代替）。
  飢餓状態になると急速にHPが減る → 食料獲得が必須 → 学習が必要になる。

飢餓メカニズム:
  hunger += 1 each step
  if hunger > hunger_threshold: hp -= hunger_penalty
  if food eaten: hp += food_value; hunger = 0

ゲノム（進化対象）:
  hunger_threshold  int   [20, 100]
  hunger_penalty    float [1.0, 10.0]
  edge_add_prob     float [0.0, 0.2]
  activity_ratio    float [0.0, 1.0]

固定:
  metabolic_rate    swept over [0.0, 0.01, 0.05]
  hp_start=100, hp_decay=0, food_value=50
  T_consolidation=200, activity_noise=0.05

Node layout (N=20):
  node  0-3  : input    (x, y, HP, food_flag)
  node  4-8  : output   (argmax selects action)
  node  9-19 : internal (recurrent)

Experiments:
  A  飢餓パラメータの進化 (mr=0.00/0.01/0.05、各50世代)
     世代別: hunger_threshold/penalty/edge_add_prob/activity_ratio/
             food_count/エッジ数/活動ノード数/best_steps
  B  ネットワークの生死確認 (入出力/内部活動、out_var、ablation)
  C  活動ノード数の分布 (Session 22-23-24との比較)
  D  文脈分離の再計測 (PenaltyContextGridWorld: acc_A, acc_B, cosine_dist)

判定基準:
  「飢餓により学習が必然化された」:
    food_count が世代とともに増加 かつ 静止個体(food=0)が淘汰される
  「ネットワークが機能している」:
    edges > 10, out_var > 0.1, active > 4
  「文脈分離が起きた」:
    acc_A > 0.6 かつ acc_B > 0.6
"""

import os
import numpy as np
import _jp_font  # noqa: F401 — sets Japanese font in matplotlib rcParams
import matplotlib.pyplot as plt

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
from session_19_context_reboot import (
    _CGRID, _CHP_MAX, _CRESPAWN, _CSTEPS,
)
from session_20_penalty_context import _s20_inp4, _PENALTY

_SEED      = 42
_N_AGENTS  = 10
_N_EP      = 5
_ACT_THRESHOLD = 0.1

_INP_START = 0
_INP_END   = 4
_OUT_START = 4
_OUT_END   = 9
_INT_START = 9
_INT_END   = 20

# Fixed survival params
_S25_HP_START   = 100
_S25_HP_DECAY   = 0.0
_S25_FOOD_VALUE = 50

# Hunger genome bounds
_HUNGER_THR_LO  = 20
_HUNGER_THR_HI  = 100
_HUNGER_PEN_LO  = 1.0
_HUNGER_PEN_HI  = 10.0
_HUNGER_THR_STD = 5.0
_HUNGER_PEN_STD = 0.5

_S25_MR_VALUES = [0.0, 0.01, 0.05]
_S25_MR_COLORS = {0.0: 'gray', 0.01: 'steelblue', 0.05: '#f58231'}
_S25_MR_LABELS = {
    0.0:  'mr=0.00\n(no cost)',
    0.01: 'mr=0.01\n(light)',
    0.05: 'mr=0.05\n(medium)',
}


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s25_make_genome(rng, metabolic_rate):
    G = _s10_build_graph(rng)
    return {
        'G':                G,
        'W':                _s10_get_W(G),
        'edge_add_prob':    float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio':   float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate':   float(metabolic_rate),
        'hunger_threshold': int(rng.integers(_HUNGER_THR_LO, _HUNGER_THR_HI + 1)),
        'hunger_penalty':   float(rng.uniform(_HUNGER_PEN_LO, _HUNGER_PEN_HI)),
    }


def _s25_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    ep  = float(np.clip(
        genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD),
        0.0, _EP_INIT_MAX))
    ar  = float(np.clip(
        genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD),
        0.0, _AR_INIT_MAX))
    ht  = int(np.clip(
        round(genome['hunger_threshold'] + rng.normal(0, _HUNGER_THR_STD)),
        _HUNGER_THR_LO, _HUNGER_THR_HI))
    hp  = float(np.clip(
        genome['hunger_penalty'] + rng.normal(0, _HUNGER_PEN_STD),
        _HUNGER_PEN_LO, _HUNGER_PEN_HI))
    return {
        'G':                G_new,
        'W':                _s10_get_W(G_new),
        'edge_add_prob':    ep,
        'activity_ratio':   ar,
        'metabolic_rate':   genome['metabolic_rate'],
        'hunger_threshold': ht,
        'hunger_penalty':   hp,
    }


# ── Episode runners ────────────────────────────────────────────────────────────

def _s25_run_ep_hunger_dynamic(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                                hunger_threshold, hunger_penalty,
                                rng,
                                activity_noise=_ACTIVITY_NOISE,
                                T_consolidation=_T_CONSOLIDATION,
                                record_activity=False):
    """HungerDynamicGridWorld: 飢餓メカニズムつきの動的グリッド環境。

    hp_decay=0、food_value=_S25_FOOD_VALUE 固定。
    hunger がリセット閾値を超えると hunger_penalty/step が加算される。
    Returns (steps, food, penalties, records_or_None, layer_stats_dict).
    """
    all_pos     = [(r, c) for r in range(_CGRID) for c in range(_CGRID)]
    food_pos    = all_pos[int(rng.integers(0, len(all_pos)))]
    remaining   = [p for p in all_pos if p != food_pos]
    penalty_pos = remaining[int(rng.integers(0, len(remaining)))]

    food_avail = True
    food_timer = 0

    activity = np.zeros(_N)
    row, col = 2, 2
    hp       = float(_S25_HP_START)
    hunger   = 0
    steps    = 0
    food     = 0
    penalties = 0
    records  = [] if record_activity else None

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
                hp -= _PENALTY
                penalties += 1
        elif action == 4:
            if row == fr and col == fc and food_avail:
                hp = min(_CHP_MAX, hp + _S25_FOOD_VALUE)
                food_avail = False
                food_timer = 0
                food_eaten = True
                food += 1

        # Hunger mechanic
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
        layer_stats['input_mean']      = float(np.mean(arr[:, _INP_START:_INP_END]))
        layer_stats['output_mean']     = float(np.mean(arr[:, _OUT_START:_OUT_END]))
        layer_stats['internal_mean']   = float(np.mean(arr[:, _INT_START:_INT_END]))
        node_means                     = np.mean(arr, axis=0)
        layer_stats['input_active']    = float(
            np.sum(node_means[_INP_START:_INP_END] > _ACT_THRESHOLD))
        layer_stats['output_active']   = float(
            np.sum(node_means[_OUT_START:_OUT_END] > _ACT_THRESHOLD))
        layer_stats['internal_active'] = float(
            np.sum(node_means[_INT_START:_INT_END] > _ACT_THRESHOLD))
        layer_stats['output_variance'] = float(
            np.mean(np.var(arr[:, _OUT_START:_OUT_END], axis=1)))

    return steps, food, penalties, records, layer_stats


def _s25_run_ep_hunger_penalty(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                                hunger_threshold, hunger_penalty,
                                rng, mode=None,
                                activity_noise=_ACTIVITY_NOISE,
                                T_consolidation=_T_CONSOLIDATION,
                                record_activity=False):
    """PenaltyContextGridWorld with hunger mechanics.

    Returns (steps, food, mode, penalties, records_or_None).
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
                hp -= _PENALTY
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


# ── Inner evolution loop ───────────────────────────────────────────────────────

def _s25_evolve_mr(metabolic_rate, seed):
    """飢餓パラメータを含むゲノムを固定 mr で進化させる。

    Returns (best_genome, history_dict).
    history keys: gen_best_steps, gen_hunger_threshold, gen_hunger_penalty,
                  gen_edge_add_prob, gen_activity_ratio,
                  gen_food_count, gen_edges, gen_mean_active.
    """
    mr_idx = _S25_MR_VALUES.index(metabolic_rate)
    rng    = np.random.default_rng(seed + 25000 + mr_idx * 1000)
    pop    = [_s25_make_genome(rng, metabolic_rate) for _ in range(_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_hunger_threshold', 'gen_hunger_penalty',
        'gen_edge_add_prob', 'gen_activity_ratio',
        'gen_food_count', 'gen_edges', 'gen_mean_active',
    )}

    for gen in range(_N_GEN):
        fitnesses = []

        for g in pop:
            total     = 0
            ep_active = []
            ep_food   = []

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
                    arr        = np.array(recs)
                    node_means = np.mean(arr, axis=0)
                    ep_active.append(float(np.sum(node_means > _ACT_THRESHOLD)))

            fitnesses.append(total / _N_EP)
            g['_ep_active'] = float(np.mean(ep_active)) if ep_active else 0.0
            g['_ep_food']   = float(np.mean(ep_food))

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]

        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_hunger_threshold'].append(float(bg['hunger_threshold']))
        hist['gen_hunger_penalty'].append(float(bg['hunger_penalty']))
        hist['gen_edge_add_prob'].append(float(bg['edge_add_prob']))
        hist['gen_activity_ratio'].append(float(bg['activity_ratio']))
        hist['gen_food_count'].append(bg['_ep_food'])
        hist['gen_edges'].append(bg['G'].number_of_edges())
        hist['gen_mean_active'].append(bg['_ep_active'])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s25_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'    gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'hunger_thr={bg["hunger_threshold"]:3d}  '
                  f'hunger_pen={bg["hunger_penalty"]:.2f}  '
                  f'food={bg["_ep_food"]:.2f}/ep  '
                  f'edges={bg["G"].number_of_edges():4d}  '
                  f'active={bg["_ep_active"]:.1f}')

    for g in pop:
        g.pop('_ep_active', None)
        g.pop('_ep_food',   None)

    return pop[0], hist


# ── Experiment A: genome convergence ──────────────────────────────────────────

def run_exp_a_genome_convergence(seed=_SEED):
    """3条件の mr で飢餓パラメータを進化させ、収束推移を記録。

    Returns dict keyed by mr: { 'best': genome, 'hist': history_dict }
    """
    results = {}
    for mr in _S25_MR_VALUES:
        print(f'\n  [mr={mr:.2f}] 飢餓パラメータを進化中 ({_N_GEN} 世代)...')
        best, hist = _s25_evolve_mr(mr, seed)
        results[mr] = {'best': best, 'hist': hist}
        print(f'  → hunger_thr={best["hunger_threshold"]}  '
              f'hunger_pen={best["hunger_penalty"]:.2f}  '
              f'food/ep={hist["gen_food_count"][-1]:.2f}  '
              f'edges={best["G"].number_of_edges()}  '
              f'active={hist["gen_mean_active"][-1]:.1f}  '
              f'steps={hist["gen_best_steps"][-1]:.1f}')
    return results


# ── Experiment B: network vitality ────────────────────────────────────────────

def _s25_measure_vitality(genome, seed, n_episodes=20):
    """ベスト個体のネットワーク生死を確認（ablation 含む）。

    Returns dict: input/output/internal means, output_variance, ablation_effect, edges.
    """
    mr_idx = _S25_MR_VALUES.index(genome['metabolic_rate'])
    rng    = np.random.default_rng(seed + 25100 + mr_idx * 100)

    out_vars         = []
    ablation_effects = []
    layer_inp        = []
    layer_out        = []
    layer_int        = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**32))

        ep_rng = np.random.default_rng(ep_seed)
        G = genome['G'].copy()
        W = _s10_get_W(G)
        _, _, _, recs, ls = _s25_run_ep_hunger_dynamic(
            G, W,
            genome['edge_add_prob'], genome['activity_ratio'],
            genome['metabolic_rate'],
            genome['hunger_threshold'], genome['hunger_penalty'],
            ep_rng, record_activity=True)

        if recs and ls:
            layer_inp.append(ls['input_mean'])
            layer_out.append(ls['output_mean'])
            layer_int.append(ls['internal_mean'])
            out_vars.append(ls['output_variance'])

        rng_tmp  = np.random.default_rng(ep_seed + 1)
        act_norm = np.zeros(_N)
        act_abl  = np.zeros(_N)
        normal_outs   = []
        ablated_outs  = []
        for _ in range(min(100, _CSTEPS)):
            inp_norm = np.array([0.5, 0.5, 0.5, 1.0])
            inp_zero = np.zeros(4)
            for _ in range(_N_PROP):
                act_norm = _s10_propagate(W, act_norm, inp_norm)
                act_abl  = _s10_propagate(W, act_abl,  inp_zero)
            if _ACTIVITY_NOISE > 0.0:
                noise    = rng_tmp.normal(0, _ACTIVITY_NOISE, _N)
                act_norm = np.clip(act_norm + noise, 0.0, 1.0)
                act_abl  = np.clip(act_abl  + noise, 0.0, 1.0)
            normal_outs.append(act_norm[_OUT_START:_OUT_END].copy())
            ablated_outs.append(act_abl[_OUT_START:_OUT_END].copy())

        if normal_outs:
            nm = np.array(normal_outs)
            am = np.array(ablated_outs)
            ablation_effects.append(float(np.mean(np.abs(nm - am))))

    return {
        'input_mean':      float(np.mean(layer_inp))        if layer_inp        else 0.0,
        'output_mean':     float(np.mean(layer_out))        if layer_out        else 0.0,
        'internal_mean':   float(np.mean(layer_int))        if layer_int        else 0.0,
        'output_variance': float(np.mean(out_vars))         if out_vars         else 0.0,
        'ablation_effect': float(np.mean(ablation_effects)) if ablation_effects else 0.0,
        'edges':           genome['G'].number_of_edges(),
    }


def run_exp_b_network_vitality(exp_a_results, seed=_SEED):
    """実験Aのベスト個体のネットワーク生死を確認。

    Returns dict keyed by mr.
    """
    print('\n  [Exp B: network vitality check]')
    results = {}
    for mr in _S25_MR_VALUES:
        v = _s25_measure_vitality(exp_a_results[mr]['best'], seed, n_episodes=20)
        results[mr] = v
        print(f'    mr={mr:.2f}: inp={v["input_mean"]:.3f}  '
              f'out={v["output_mean"]:.3f}  int={v["internal_mean"]:.3f}  '
              f'out_var={v["output_variance"]:.4f}  '
              f'ablation={v["ablation_effect"]:.4f}  '
              f'edges={v["edges"]}')
    return results


# ── Experiment C: sparsity ─────────────────────────────────────────────────────

def run_exp_c_sparsity(exp_a_results, seed=_SEED, n_episodes=20):
    """活動ノード数の分布を計測（Session 22-23-24 との比較用）。

    Returns dict keyed by mr.
    """
    print('\n  [Exp C: sparsity check]')
    results = {}

    for mr in _S25_MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _S25_MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 25200 + mr_idx * 100)

        active_counts = []
        for _ in range(n_episodes):
            G = g['G'].copy()
            W = _s10_get_W(G)
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, recs, _ = _s25_run_ep_hunger_dynamic(
                G, W,
                g['edge_add_prob'], g['activity_ratio'],
                g['metabolic_rate'],
                g['hunger_threshold'], g['hunger_penalty'],
                ep_rng, record_activity=True)
            if recs:
                arr        = np.array(recs)
                node_means = np.mean(arr, axis=0)
                active_counts.append(float(np.sum(node_means > _ACT_THRESHOLD)))

        mean_active = float(np.mean(active_counts)) if active_counts else 0.0
        print(f'    mr={mr:.2f}: mean_active={mean_active:.1f}')
        results[mr] = {'active_counts': active_counts, 'mean_active': mean_active}

    return results


# ── Experiment D: context separation ──────────────────────────────────────────

def run_exp_d_context(exp_a_results, seed=_SEED, n_ep_per_mode=10):
    """実験Aのベスト個体を PenaltyContextGridWorld（飢餓つき）で評価。

    Returns dict keyed by mr: acc_A, acc_B, mean_cos_dist.
    """
    print('\n  [Exp D: context separation (PenaltyContextGridWorld + hunger)]')
    results = {}

    for mr in _S25_MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _S25_MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 25300 + mr_idx * 100)

        cos_distances = []
        all_food      = []
        all_modes     = []

        for _ in range(_N_AGENTS):
            G = g['G'].copy()
            W = _s10_get_W(G)
            trial_means_A, trial_means_B = [], []

            for ei in range(n_ep_per_mode * 2):
                mode   = 'A' if ei % 2 == 0 else 'B'
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                _, f, _, _, recs = _s25_run_ep_hunger_penalty(
                    G, W,
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'],
                    g['hunger_threshold'], g['hunger_penalty'],
                    ep_rng, mode=mode, record_activity=True)
                all_food.append(f)
                all_modes.append(mode)

                if recs:
                    arr      = np.array(recs)
                    mean_out = arr[:, _OUT_START:_OUT_END].mean(axis=0)
                    if mode == 'A':
                        trial_means_A.append(mean_out)
                    else:
                        trial_means_B.append(mean_out)

            if trial_means_A and trial_means_B:
                mA = np.mean(trial_means_A, axis=0)
                mB = np.mean(trial_means_B, axis=0)
                nA, nB = np.linalg.norm(mA), np.linalg.norm(mB)
                if nA > 1e-9 and nB > 1e-9:
                    cos_distances.append(1.0 - float(np.dot(mA, mB) / (nA * nB)))

        modes_arr = np.array(all_modes)
        food_arr  = np.array(all_food)
        acc_a   = float(np.mean(food_arr[modes_arr == 'A'] >= 1)) if np.any(modes_arr == 'A') else 0.0
        acc_b   = float(np.mean(food_arr[modes_arr == 'B'] >= 1)) if np.any(modes_arr == 'B') else 0.0
        mean_cd = float(np.mean(cos_distances)) if cos_distances else 0.0

        print(f'    mr={mr:.2f}: acc_A={acc_a:.3f}  acc_B={acc_b:.3f}  '
              f'cos_dist={mean_cd:.4f}')

        results[mr] = {'acc_A': acc_a, 'acc_B': acc_b, 'mean_cos_dist': mean_cd,
                       'cos_distances': cos_distances}

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_a_genome_convergence(exp_a_results,
                                   fname='images/session_25/results_s25_genome_convergence.png'):
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        'Session 25 Exp A: 飢餓パラメータの進化\n'
        'hunger_threshold / hunger_penalty / edge_add_prob / activity_ratio /\n'
        '食料獲得回数 / エッジ数 / 活動ノード数 / best_steps の世代別推移\n'
        f'n_agents={_N_AGENTS}  n_ep={_N_EP}  n_gen={_N_GEN}  '
        f'hp_start={_S25_HP_START}  food_value={_S25_FOOD_VALUE}  hp_decay=0',
        fontsize=10,
    )

    gens = np.arange(1, _N_GEN + 1)

    panels = [
        ('gen_hunger_threshold', 'hunger_threshold (evolved)',
         'Hunger threshold (steps)',   _HUNGER_THR_LO, _HUNGER_THR_HI),
        ('gen_hunger_penalty',   'hunger_penalty (evolved)',
         'Hunger penalty (HP/step)',   _HUNGER_PEN_LO, _HUNGER_PEN_HI),
        ('gen_edge_add_prob',    'edge_add_prob (evolved)',
         'Edge add probability',       0.0, _EP_INIT_MAX),
        ('gen_activity_ratio',   'activity_ratio (evolved)',
         'Activity ratio',             0.0, _AR_INIT_MAX),
        ('gen_food_count',       '食料獲得回数 (best agent / ep)',
         'Food count per episode',     0.0, None),
        ('gen_edges',            'Edge count (best genome)',
         'Number of edges',            0, None),
        ('gen_mean_active',      f'Active nodes (activity > {_ACT_THRESHOLD})',
         'Active nodes (all layers)',  0, _N + 1),
        ('gen_best_steps',       'Best fitness (mean survival steps)',
         'Mean steps',                 0, None),
    ]

    for idx, (key, title, ylabel, ylo, yhi) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        for mr in _S25_MR_VALUES:
            hist = exp_a_results[mr]['hist']
            ax.plot(gens, hist[key],
                    color=_S25_MR_COLORS[mr], linewidth=2,
                    label=_S25_MR_LABELS[mr].replace('\n', ' '))
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if ylo is not None:
            ax.set_ylim(bottom=ylo)
        if yhi is not None:
            ax.set_ylim(top=yhi)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if key == 'gen_food_count':
            ax.axhline(1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7,
                       label='≥1 food/ep (学習の証拠)')
            ax.legend(fontsize=8)
        elif key == 'gen_mean_active':
            ax.axhline(4.0, color='purple', linestyle=':', linewidth=1.2, alpha=0.7,
                       label='静止ライン active=4')
            ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_network_vitality(exp_b_results,
                                  fname='images/session_25/results_s25_network_vitality.png'):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Session 25 Exp B: ネットワークの生死確認\n'
        '入力/出力/内部ノードの平均活動、出力ノード分散、入力Ablation効果、エッジ数\n'
        'Session 24 比較: mr=0.01 out_var=0.1385',
        fontsize=10,
    )

    x      = np.arange(len(_S25_MR_VALUES))
    colors = [_S25_MR_COLORS[mr] for mr in _S25_MR_VALUES]
    xlbls  = [_S25_MR_LABELS[mr] for mr in _S25_MR_VALUES]

    # S24 reference values for out_var and ablation (mr=0.01)
    _S24_REF_OUT_VAR  = 0.1385
    _S24_REF_ABLATION = None  # not recalled, skipped

    panels = [
        ('input_mean',      'Input Node Mean Activity\n(nodes 0-3)',          None),
        ('output_mean',     'Output Node Mean Activity\n(nodes 4-8)',          None),
        ('internal_mean',   'Internal Node Mean Activity\n(nodes 9-19)',       None),
        ('output_variance', 'Output Node Variance\n(per-step var, nodes 4-8)', 0.1),
        ('ablation_effect', 'Input Ablation Effect\n(|normal - zeroed|)',       0.05),
        ('edges',           'Edge Count (final best genome)',                   10),
    ]

    for idx, (key, title, threshold) in enumerate(panels):
        ax   = axes[idx // 3][idx % 3]
        vals = [exp_b_results[mr][key] for mr in _S25_MR_VALUES]
        ymax = max(max(vals) * 1.4, 0.01) if max(vals) > 0 else 0.5

        bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
        if threshold is not None:
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.85,
                       label=f'threshold={threshold}')
            ax.legend(fontsize=8)
        if key == 'output_variance':
            ax.axhline(_S24_REF_OUT_VAR, color='orange', linestyle=':', linewidth=1.2, alpha=0.7,
                       label=f'S24 mr=0.01 ref={_S24_REF_OUT_VAR}')
            ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f'{val:.3f}' if isinstance(val, float) else str(val),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_c_sparsity(exp_c_results,
                         fname='images/session_25/results_s25_sparsity.png'):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        'Session 25 Exp C: 活動ノード数の分布\n'
        f'Active node threshold = activity > {_ACT_THRESHOLD}  (N={_N} total nodes)\n'
        'S22/S23/S24 参照: mr=0.01 → active≈4（静止）',
        fontsize=10,
    )

    x      = np.arange(len(_S25_MR_VALUES))
    colors = [_S25_MR_COLORS[mr] for mr in _S25_MR_VALUES]
    means  = [exp_c_results[mr]['mean_active'] for mr in _S25_MR_VALUES]

    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
    ax.axhline(4.0, color='purple', linestyle=':', linewidth=1.5, alpha=0.8,
               label='S22/S23/S24 静止状態 active≈4')
    ax.axhline(15.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='飽和ライン ≈ 15 nodes')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_S25_MR_LABELS[mr] for mr in _S25_MR_VALUES], fontsize=10)
    ax.set_ylabel(f'Mean Active Nodes (activity > {_ACT_THRESHOLD})')
    ax.set_title('Mean Active Nodes: S25 HungerDynamicGridWorld（飢餓必然化後）')
    ax.set_ylim(0, _N + 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_d_context(exp_d_results,
                        fname='images/session_25/results_s25_context.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Session 25 Exp D: 文脈分離の再計測 (PenaltyContextGridWorld + 飢餓)\n'
        '飢餓メカニズムで学習が必然化された個体が文脈依存的な行動を獲得しているか\n'
        'acc_A/B: food eaten ≥ 1  |  cosine_dist: output nodes (mode A vs B)',
        fontsize=10,
    )

    x      = np.arange(len(_S25_MR_VALUES))
    colors = [_S25_MR_COLORS[mr] for mr in _S25_MR_VALUES]
    xlbls  = [_S25_MR_LABELS[mr] for mr in _S25_MR_VALUES]

    metrics = [
        ('acc_A',         'Mode A Accuracy\n(food ≥ 1 in NW-food episodes)', 0.6),
        ('acc_B',         'Mode B Accuracy\n(food ≥ 1 in SE-food episodes)', 0.6),
        ('mean_cos_dist', 'Cosine Distance\n(output: Mode A vs Mode B)',     0.1),
    ]

    for ai, (key, title, threshold) in enumerate(metrics):
        ax   = axes[ai]
        vals = [exp_d_results[mr][key] for mr in _S25_MR_VALUES]
        ymax = max(max(vals) * 1.3, threshold * 1.5) + 0.05

        bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
        ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.85,
                   label=f'Threshold = {threshold}')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_ylabel(title.split('\n')[0])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=9)
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

    print('=== Session 25: 生物学的な飢餓による学習の必然化 ===')
    print('飢餓メカニズム: hunger += 1/step; if hunger > thr: hp -= penalty')
    print(f'固定: hp_start={_S25_HP_START}  hp_decay=0  food_value={_S25_FOOD_VALUE}')
    print(f'進化: hunger_threshold [{_HUNGER_THR_LO},{_HUNGER_THR_HI}]  '
          f'hunger_penalty [{_HUNGER_PEN_LO},{_HUNGER_PEN_HI}]')
    print(f'mr スイープ: {_S25_MR_VALUES}')
    print(f'n_agents={_N_AGENTS}  n_ep={_N_EP}  n_gen={_N_GEN}')

    print(f'\n[Exp A] 飢餓パラメータの進化（各 mr 条件 {_N_GEN} 世代）')
    exp_a = run_exp_a_genome_convergence(seed=_SEED)

    print('\n  Summary (converged best genomes):')
    hdr = (f'  {"mr":>6}  {"h_thr":>5}  {"h_pen":>5}  {"food/ep":>7}  '
           f'{"edges":>6}  {"active":>7}  {"steps":>8}')
    print(hdr)
    print('  ' + '─' * 60)
    for mr in _S25_MR_VALUES:
        bg   = exp_a[mr]['best']
        hist = exp_a[mr]['hist']
        print(f'  {mr:6.3f}  {bg["hunger_threshold"]:5d}  {bg["hunger_penalty"]:5.2f}  '
              f'{hist["gen_food_count"][-1]:7.2f}  '
              f'{bg["G"].number_of_edges():6d}  '
              f'{hist["gen_mean_active"][-1]:7.1f}  '
              f'{hist["gen_best_steps"][-1]:8.1f}')
    plot_exp_a_genome_convergence(exp_a)

    print('\n[Exp B] ネットワークの生死確認 (n_episodes=20)')
    exp_b = run_exp_b_network_vitality(exp_a, seed=_SEED)
    plot_exp_b_network_vitality(exp_b)

    print('\n[Exp C] 活動ノード数の分布 (n_episodes=20)')
    exp_c = run_exp_c_sparsity(exp_a, seed=_SEED, n_episodes=20)
    plot_exp_c_sparsity(exp_c)

    print('\n[Exp D] 文脈分離の再計測 (PenaltyContextGridWorld + 飢餓, n_ep_per_mode=10)')
    exp_d = run_exp_d_context(exp_a, seed=_SEED, n_ep_per_mode=10)
    plot_exp_d_context(exp_d)

    # ── Judgment ──────────────────────────────────────────────────────────────
    print('\n  ── Judgment Criteria (Session 25) ─────────────────────────────')
    print('  判定基準:')
    print('  J1. 「飢餓により学習が必然化された」: food_count/ep > 1.0')
    print('  J2. 「ネットワークが機能している」: edges > 10 かつ out_var > 0.1 かつ active > 4')
    print('  J3. 「文脈分離が起きた」: acc_A > 0.6 かつ acc_B > 0.6')
    print()

    print(f'  {"mr":>6}  {"food/ep":>7}  J1  '
          f'{"edges":>6}  {"out_v":>6}  {"act":>5}  J2  '
          f'{"accA":>6}  {"accB":>6}  J3')
    print('  ' + '─' * 70)

    for mr in _S25_MR_VALUES:
        bg   = exp_a[mr]['best']
        hist = exp_a[mr]['hist']
        v    = exp_b[mr]
        c    = exp_c[mr]
        d    = exp_d[mr]

        food_per_ep = hist['gen_food_count'][-1]
        j1 = food_per_ep > 1.0
        j2 = v['edges'] > 10 and v['output_variance'] > 0.1 and c['mean_active'] > 4.0
        j3 = d['acc_A'] > 0.6 and d['acc_B'] > 0.6

        print(f'  {mr:6.3f}  {food_per_ep:7.2f}  {"✓" if j1 else "✗"}   '
              f'{v["edges"]:6d}  {v["output_variance"]:6.3f}  {c["mean_active"]:5.1f}  '
              f'{"✓" if j2 else "✗"}   '
              f'{d["acc_A"]:6.3f}  {d["acc_B"]:6.3f}  {"✓" if j3 else "✗"}')

    print()
    n_pass_j1 = sum(
        1 for mr in _S25_MR_VALUES
        if exp_a[mr]['hist']['gen_food_count'][-1] > 1.0
    )
    if n_pass_j1 > 0:
        print(f'  → J1: {n_pass_j1}/{len(_S25_MR_VALUES)} 条件で「飢餓により学習が必然化された」 ✓')
        print('    food_count が 1.0/ep を超え、学習が必然化された')
    else:
        print('  → J1: 全条件で food_count/ep ≤ 1.0 — 飢餓後も食料獲得は不十分')
        print('    考えられる理由: hunger_threshold が最大値に進化し飢餓が遅延')

    print('\nDone.')
