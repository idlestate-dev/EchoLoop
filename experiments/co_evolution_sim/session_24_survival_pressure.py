"""Session 24: 生存圧パラメータの進化（改訂版）

Session 23 の問題:
  hp_start=100, hp_decay=1 が固定 → 100ステップ何もせず生き延びられる
  → 代謝コストを払ってでも学習する必要がない
  → active=4 の静止が最適解

本セッションの問い:
  1. hp_start / hp_decay / food_value を進化させたとき
     「学習が必要な生存圧」（hp_decay > 2 または寿命 < 30）に収束するか
  2. その生存圧のもとで出力ノードが機能する豊かなトポロジーが選ばれるか
  3. edges=1 の「死んだネットワーク」から脱出できるか

Node layout (N=20):
  node  0-3  : input     (4 nodes) — x, y, HP, food_flag
  node  4-8  : output    (5 nodes) — argmax selects action
  node  9-19 : internal  (11 nodes) — recurrent

Experiments:
  A  生存圧パラメータの進化 (mr=0.00/0.01/0.05、各50世代)
     世代別: hp_start/hp_decay/food_value/寿命/エッジ数/層別活動ノード数
  B  ネットワークの生死確認 (入力/出力/内部層の活動、出力分散、入力ablation効果)
  C  活動ノード数の分布 (Session 22-23との比較)
  D  文脈分離の再計測 (PenaltyContextGridWorld: acc_A, acc_B, cosine_dist)
  E  寿命と活動量・出力分散の関係 (散布図)

判定基準:
  「学習が必要な生存圧が選ばれた」: hp_decay > 2.0 または 寿命 < 30
  「ネットワークが機能している」: edges > 10, output_variance > 0.1,
                                   input_ablation_effect > 0.05
  「スパース性が創発した」: 4 < active < 15
  「文脈分離が起きた」: acc_A > 0.6 かつ acc_B > 0.6
"""

import os
import numpy as np
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

_SEED          = 42
_N_AGENTS      = 10
_N_EP          = 5
_ACT_THRESHOLD = 0.1

# Node layer boundaries
_INP_START  = 0
_INP_END    = 4   # nodes 0-3
_OUT_START  = 4
_OUT_END    = 9   # nodes 4-8
_INT_START  = 9
_INT_END    = 20  # nodes 9-19

# Survival-pressure parameter bounds (new in S24)
_HP_START_LO   = 30
_HP_START_HI   = 150
_HP_DECAY_LO   = 0.5
_HP_DECAY_HI   = 5.0
_FOOD_VAL_LO   = 10
_FOOD_VAL_HI   = 50
_HP_START_STD  = 10.0
_HP_DECAY_STD  = 0.3
_FOOD_VAL_STD  = 3.0

_S24_MR_VALUES = [0.0, 0.01, 0.05]
_S24_MR_COLORS = {0.0: 'gray', 0.01: 'steelblue', 0.05: '#f58231'}
_S24_MR_LABELS = {
    0.0:  'mr=0.00\n(no cost)',
    0.01: 'mr=0.01\n(light)',
    0.05: 'mr=0.05\n(medium)',
}


# ── Genome helpers ─────────────────────────────────────────────────────────────

def _s24_make_genome(rng, metabolic_rate):
    G = _s10_build_graph(rng)
    return {
        'G':              G,
        'W':              _s10_get_W(G),
        'edge_add_prob':  float(rng.uniform(0.0, _EP_INIT_MAX)),
        'activity_ratio': float(rng.uniform(0.0, _AR_INIT_MAX)),
        'metabolic_rate': float(metabolic_rate),
        'hp_start':       int(rng.integers(_HP_START_LO, _HP_START_HI + 1)),
        'hp_decay':       float(rng.uniform(_HP_DECAY_LO, _HP_DECAY_HI)),
        'food_value':     int(rng.integers(_FOOD_VAL_LO, _FOOD_VAL_HI + 1)),
    }


def _s24_mutate_genome(genome, rng):
    G_new = _s10_mutate(genome['G'], rng)
    ep  = float(np.clip(
        genome['edge_add_prob']  + rng.normal(0, _EP_MUT_STD),
        0.0, _EP_INIT_MAX))
    ar  = float(np.clip(
        genome['activity_ratio'] + rng.normal(0, _AR_MUT_STD),
        0.0, _AR_INIT_MAX))
    hps = int(np.clip(
        round(genome['hp_start'] + rng.normal(0, _HP_START_STD)),
        _HP_START_LO, _HP_START_HI))
    hpd = float(np.clip(
        genome['hp_decay']   + rng.normal(0, _HP_DECAY_STD),
        _HP_DECAY_LO, _HP_DECAY_HI))
    fv  = int(np.clip(
        round(genome['food_value'] + rng.normal(0, _FOOD_VAL_STD)),
        _FOOD_VAL_LO, _FOOD_VAL_HI))
    return {
        'G':              G_new,
        'W':              _s10_get_W(G_new),
        'edge_add_prob':  ep,
        'activity_ratio': ar,
        'metabolic_rate': genome['metabolic_rate'],
        'hp_start':       hps,
        'hp_decay':       hpd,
        'food_value':     fv,
    }


# ── Episode runners ────────────────────────────────────────────────────────────

def _s24_run_ep_dynamic(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                         hp_start, hp_decay, food_value,
                         rng,
                         activity_noise=_ACTIVITY_NOISE,
                         T_consolidation=_T_CONSOLIDATION,
                         record_activity=False):
    """DynamicGridWorld with genome-parameterized survival pressure.

    hp_start/hp_decay/food_value are per-agent (evolved) values.
    Returns (steps, food, penalties, records_or_None, layer_stats_dict).
    layer_stats keys: input_mean, output_mean, internal_mean,
                      input_active, output_active, internal_active,
                      output_variance.
    """
    all_pos     = [(r, c) for r in range(_CGRID) for c in range(_CGRID)]
    food_pos    = all_pos[int(rng.integers(0, len(all_pos)))]
    remaining   = [p for p in all_pos if p != food_pos]
    penalty_pos = remaining[int(rng.integers(0, len(remaining)))]

    food_avail = True
    food_timer = 0

    activity  = np.zeros(_N)
    row, col  = 2, 2
    hp        = float(hp_start)
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
                hp = min(_CHP_MAX, hp + food_value)
                food_avail = False
                food_timer = 0
                food += 1

        hp    -= hp_decay
        steps  = step + 1

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
        # mean per-step variance across the 5 output nodes
        layer_stats['output_variance'] = float(
            np.mean(np.var(arr[:, _OUT_START:_OUT_END], axis=1)))

    return steps, food, penalties, records, layer_stats


def _s24_run_ep_penalty(G, W, edge_add_prob, activity_ratio, metabolic_rate,
                          hp_start, hp_decay, food_value,
                          rng, mode=None,
                          activity_noise=_ACTIVITY_NOISE,
                          T_consolidation=_T_CONSOLIDATION,
                          record_activity=False):
    """PenaltyContextGridWorld with genome-parameterized survival pressure.

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
    hp        = float(hp_start)
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
                hp = min(_CHP_MAX, hp + food_value)
                food_avail = False
                food_timer = 0
                food += 1

        hp   -= hp_decay
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

def _s24_evolve_mr(metabolic_rate, seed):
    """Evolve all genome params (incl. hp_start/hp_decay/food_value) with fixed mr.

    Returns (best_genome, history_dict).
    history keys: gen_best_steps, gen_mean_active,
                  gen_hp_start, gen_hp_decay, gen_food_value, gen_lifespan,
                  gen_edges,
                  gen_active_input, gen_active_output, gen_active_internal.
    """
    mr_idx = _S24_MR_VALUES.index(metabolic_rate)
    rng    = np.random.default_rng(seed + 24000 + mr_idx * 1000)
    pop    = [_s24_make_genome(rng, metabolic_rate) for _ in range(_N_AGENTS)]

    hist = {k: [] for k in (
        'gen_best_steps', 'gen_mean_active',
        'gen_hp_start', 'gen_hp_decay', 'gen_food_value', 'gen_lifespan',
        'gen_edges',
        'gen_active_input', 'gen_active_output', 'gen_active_internal',
    )}

    for gen in range(_N_GEN):
        fitnesses   = []
        best_layers = {}

        for g in pop:
            total      = 0
            ep_active  = []
            ep_inp_act = []
            ep_out_act = []
            ep_int_act = []
            last_ls    = {}

            for _ in range(_N_EP):
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                s, _, _, recs, ls = _s24_run_ep_dynamic(
                    g['G'], g['W'],
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'],
                    g['hp_start'], g['hp_decay'], g['food_value'],
                    ep_rng, record_activity=True)
                total += s
                last_ls = ls
                if recs:
                    arr = np.array(recs)
                    node_means = np.mean(arr, axis=0)
                    ep_active.append(
                        float(np.sum(node_means > _ACT_THRESHOLD)))
                    ep_inp_act.append(ls.get('input_active',  0.0))
                    ep_out_act.append(ls.get('output_active', 0.0))
                    ep_int_act.append(ls.get('internal_active', 0.0))

            fitnesses.append(total / _N_EP)

            # store layers for best genome detection later
            g['_ep_active']  = float(np.mean(ep_active))  if ep_active  else 0.0
            g['_ep_inp_act'] = float(np.mean(ep_inp_act)) if ep_inp_act else 0.0
            g['_ep_out_act'] = float(np.mean(ep_out_act)) if ep_out_act else 0.0
            g['_ep_int_act'] = float(np.mean(ep_int_act)) if ep_int_act else 0.0

        best_idx = int(np.argmax(fitnesses))
        bg = pop[best_idx]

        hist['gen_best_steps'].append(fitnesses[best_idx])
        hist['gen_mean_active'].append(bg['_ep_active'])
        hist['gen_hp_start'].append(float(bg['hp_start']))
        hist['gen_hp_decay'].append(float(bg['hp_decay']))
        hist['gen_food_value'].append(float(bg['food_value']))
        hist['gen_lifespan'].append(float(bg['hp_start']) / float(bg['hp_decay']))
        hist['gen_edges'].append(bg['G'].number_of_edges())
        hist['gen_active_input'].append(bg['_ep_inp_act'])
        hist['gen_active_output'].append(bg['_ep_out_act'])
        hist['gen_active_internal'].append(bg['_ep_int_act'])

        idx_sorted = np.argsort(fitnesses)[::-1]
        survivors  = [pop[i] for i in idx_sorted[:_N_SURV]]

        new_pop = list(survivors)
        while len(new_pop) < _N_AGENTS:
            parent = survivors[int(rng.integers(0, _N_SURV))]
            new_pop.append(_s24_mutate_genome(parent, rng))
        pop = new_pop

        if (gen + 1) % 10 == 0 or gen == 0:
            lifespan = bg['hp_start'] / bg['hp_decay']
            print(f'    gen {gen+1:3d}: best={fitnesses[best_idx]:7.1f}  '
                  f'hp_start={bg["hp_start"]:4d}  hp_decay={bg["hp_decay"]:.2f}  '
                  f'food={bg["food_value"]:3d}  lifespan={lifespan:.1f}  '
                  f'edges={bg["G"].number_of_edges():4d}  '
                  f'active={bg["_ep_active"]:.1f}  '
                  f'out={bg["_ep_out_act"]:.1f}')

    # clean up temp keys
    for g in pop:
        for k in ('_ep_active', '_ep_inp_act', '_ep_out_act', '_ep_int_act'):
            g.pop(k, None)

    return pop[0], hist


# ── Experiment A: genome convergence ──────────────────────────────────────────

def run_exp_a_genome_convergence(seed=_SEED):
    """3条件の mr で生存圧パラメータを進化させ、収束推移を記録。

    Returns dict keyed by mr: { 'best': genome, 'hist': history_dict }
    """
    results = {}
    for mr in _S24_MR_VALUES:
        print(f'\n  [mr={mr:.2f}] 生存圧パラメータを進化中 ({_N_GEN} 世代)...')
        best, hist = _s24_evolve_mr(mr, seed)
        results[mr] = {'best': best, 'hist': hist}
        lifespan = best['hp_start'] / best['hp_decay']
        print(f'  → hp_start={best["hp_start"]}  hp_decay={best["hp_decay"]:.2f}  '
              f'food={best["food_value"]}  lifespan={lifespan:.1f}  '
              f'edges={best["G"].number_of_edges()}  '
              f'active(last)={hist["gen_mean_active"][-1]:.1f}')
    return results


# ── Experiment B: network vitality ────────────────────────────────────────────

def _s24_measure_vitality(genome, seed, n_episodes=20):
    """ベスト個体のネットワーク生死を確認。

    入力ablation効果: 各ステップで inp4=実値 vs inp4=0 の出力差を計測。
    Returns dict with layer activities, output_variance, ablation_effect.
    """
    mr_idx = _S24_MR_VALUES.index(genome['metabolic_rate'])
    rng    = np.random.default_rng(seed + 24100 + mr_idx * 100)

    out_vars    = []
    ablation_effects = []
    layer_inp   = []
    layer_out   = []
    layer_int   = []

    for _ in range(n_episodes):
        ep_rng_seed = int(rng.integers(0, 2**32))

        # Normal episode
        ep_rng = np.random.default_rng(ep_rng_seed)
        G = genome['G'].copy()
        W = _s10_get_W(G)
        _, _, _, recs, ls = _s24_run_ep_dynamic(
            G, W,
            genome['edge_add_prob'], genome['activity_ratio'],
            genome['metabolic_rate'],
            genome['hp_start'], genome['hp_decay'], genome['food_value'],
            ep_rng, record_activity=True)

        if recs and ls:
            layer_inp.append(ls['input_mean'])
            layer_out.append(ls['output_mean'])
            layer_int.append(ls['internal_mean'])
            out_vars.append(ls['output_variance'])

        # Ablation episode (same seed, zero input)
        ep_rng_abl = np.random.default_rng(ep_rng_seed)
        G_abl = genome['G'].copy()
        W_abl = _s10_get_W(G_abl)
        activity_abl = np.zeros(_N)
        normal_out_means = []
        ablated_out_means = []

        # Parallel forward pass: measure output difference with inp4=0
        # Run a fixed-length forward measurement (not full episode) for clean comparison
        rng_tmp = np.random.default_rng(ep_rng_seed + 1)
        act_norm = np.zeros(_N)
        act_abl  = np.zeros(_N)
        for step in range(min(100, _CSTEPS)):
            # Normal: use a synthetic inp4 based on position (2,2) as baseline
            inp_norm = np.array([0.5, 0.5, 0.5, 1.0])
            inp_zero = np.zeros(4)
            for _ in range(_N_PROP):
                act_norm = _s10_propagate(W, act_norm, inp_norm)
                act_abl  = _s10_propagate(W, act_abl,  inp_zero)
            if _ACTIVITY_NOISE > 0.0:
                noise = rng_tmp.normal(0, _ACTIVITY_NOISE, _N)
                act_norm = np.clip(act_norm + noise, 0.0, 1.0)
                act_abl  = np.clip(act_abl  + noise, 0.0, 1.0)
            normal_out_means.append(act_norm[_OUT_START:_OUT_END].copy())
            ablated_out_means.append(act_abl[_OUT_START:_OUT_END].copy())

        if normal_out_means:
            nm = np.array(normal_out_means)
            am = np.array(ablated_out_means)
            ablation_effects.append(float(np.mean(np.abs(nm - am))))

    return {
        'input_mean':       float(np.mean(layer_inp))    if layer_inp  else 0.0,
        'output_mean':      float(np.mean(layer_out))    if layer_out  else 0.0,
        'internal_mean':    float(np.mean(layer_int))    if layer_int  else 0.0,
        'output_variance':  float(np.mean(out_vars))     if out_vars   else 0.0,
        'ablation_effect':  float(np.mean(ablation_effects)) if ablation_effects else 0.0,
        'edges':            genome['G'].number_of_edges(),
    }


def run_exp_b_network_vitality(exp_a_results, seed=_SEED):
    """実験Aのベスト個体のネットワーク生死を確認。

    Returns dict keyed by mr.
    """
    print('\n  [Exp B: network vitality check]')
    results = {}
    for mr in _S24_MR_VALUES:
        v = _s24_measure_vitality(exp_a_results[mr]['best'], seed, n_episodes=20)
        results[mr] = v
        print(f'    mr={mr:.2f}: inp={v["input_mean"]:.3f}  '
              f'out={v["output_mean"]:.3f}  int={v["internal_mean"]:.3f}  '
              f'out_var={v["output_variance"]:.4f}  '
              f'ablation={v["ablation_effect"]:.4f}  '
              f'edges={v["edges"]}')
    return results


# ── Experiment C: sparsity ─────────────────────────────────────────────────────

def run_exp_c_sparsity(exp_a_results, seed=_SEED, n_episodes=20):
    """活動ノード数の分布を計測（Session 22-23 との比較用）。

    Returns dict keyed by mr.
    """
    print('\n  [Exp C: sparsity check]')
    results = {}

    for mr in _S24_MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _S24_MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 24200 + mr_idx * 100)

        active_counts = []

        for _ in range(n_episodes):
            G = g['G'].copy()
            W = _s10_get_W(G)
            ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            _, _, _, recs, _ = _s24_run_ep_dynamic(
                G, W,
                g['edge_add_prob'], g['activity_ratio'],
                g['metabolic_rate'],
                g['hp_start'], g['hp_decay'], g['food_value'],
                ep_rng, record_activity=True)
            if recs:
                arr = np.array(recs)
                node_means = np.mean(arr, axis=0)
                active_counts.append(float(np.sum(node_means > _ACT_THRESHOLD)))

        mean_active = float(np.mean(active_counts)) if active_counts else 0.0
        print(f'    mr={mr:.2f}: mean_active={mean_active:.1f}')
        results[mr] = {'active_counts': active_counts, 'mean_active': mean_active}

    return results


# ── Experiment D: context separation ──────────────────────────────────────────

def run_exp_d_context(exp_a_results, seed=_SEED, n_ep_per_mode=10):
    """実験Aのベスト個体を PenaltyContextGridWorld で評価。

    Returns dict keyed by mr: acc_A, acc_B, mean_cos_dist.
    """
    print('\n  [Exp D: context separation (PenaltyContextGridWorld)]')
    results = {}

    for mr in _S24_MR_VALUES:
        g      = exp_a_results[mr]['best']
        mr_idx = _S24_MR_VALUES.index(mr)
        rng    = np.random.default_rng(seed + 24300 + mr_idx * 100)

        cos_distances = []
        all_food      = []
        all_modes     = []

        for _ in range(_N_AGENTS):
            G  = g['G'].copy()
            W  = _s10_get_W(G)
            trial_means_A, trial_means_B = [], []

            for ei in range(n_ep_per_mode * 2):
                mode   = 'A' if ei % 2 == 0 else 'B'
                ep_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
                _, f, _, _, recs = _s24_run_ep_penalty(
                    G, W,
                    g['edge_add_prob'], g['activity_ratio'],
                    g['metabolic_rate'],
                    g['hp_start'], g['hp_decay'], g['food_value'],
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


# ── Experiment E: lifespan vs metrics ─────────────────────────────────────────

def run_exp_e_lifespan(exp_a_results, exp_b_results, exp_c_results):
    """収束した寿命と各指標の関係をまとめる。

    Returns list of dicts with: mr, lifespan, edge_add_prob, mean_active,
                                output_variance, ablation_effect.
    """
    print('\n  [Exp E: lifespan vs network metrics]')
    rows = []
    for mr in _S24_MR_VALUES:
        bg       = exp_a_results[mr]['best']
        hist     = exp_a_results[mr]['hist']
        lifespan = float(bg['hp_start']) / float(bg['hp_decay'])
        v        = exp_b_results[mr]
        c        = exp_c_results[mr]
        row = {
            'mr':              mr,
            'hp_start':        bg['hp_start'],
            'hp_decay':        bg['hp_decay'],
            'food_value':      bg['food_value'],
            'lifespan':        lifespan,
            'edge_add_prob':   bg['edge_add_prob'],
            'mean_active':     c['mean_active'],
            'output_variance': v['output_variance'],
            'ablation_effect': v['ablation_effect'],
            'edges':           bg['G'].number_of_edges(),
            'best_steps_last': hist['gen_best_steps'][-1],
        }
        rows.append(row)
        print(f'    mr={mr:.2f}: lifespan={lifespan:.1f}  '
              f'edges={row["edges"]}  active={row["mean_active"]:.1f}  '
              f'out_var={row["output_variance"]:.4f}  '
              f'ablation={row["ablation_effect"]:.4f}')
    return rows


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_exp_a_genome_convergence(exp_a_results,
                                   fname='images/session_24/results_s24_genome_convergence.png'):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        'Session 24 Exp A: 生存圧パラメータの進化\n'
        'hp_start / hp_decay / food_value / 寿命 / エッジ数 / 活動ノード数の世代別推移\n'
        f'n_agents={_N_AGENTS}  n_ep={_N_EP}  n_gen={_N_GEN}  DynamicGridWorld',
        fontsize=11,
    )

    gens = np.arange(1, _N_GEN + 1)

    panels = [
        ('gen_hp_start',    'hp_start (evolved)',          'HP start value',
         _HP_START_LO, _HP_START_HI),
        ('gen_hp_decay',    'hp_decay (evolved)',           'HP decay per step',
         _HP_DECAY_LO, _HP_DECAY_HI),
        ('gen_food_value',  'food_value (evolved)',         'Food HP reward',
         _FOOD_VAL_LO, _FOOD_VAL_HI),
        ('gen_lifespan',    '寿命 = hp_start / hp_decay',  'Lifespan (steps w/o food)',
         0, None),
        ('gen_edges',       'Edge count (best genome)',     'Number of edges',
         0, None),
        ('gen_mean_active', f'Active nodes (activity > {_ACT_THRESHOLD})',
         'Active nodes (all layers)', 0, _N + 1),
    ]

    for idx, (key, title, ylabel, ylo, yhi) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        for mr in _S24_MR_VALUES:
            hist = exp_a_results[mr]['hist']
            ax.plot(gens, hist[key],
                    color=_S24_MR_COLORS[mr], linewidth=2,
                    label=_S24_MR_LABELS[mr].replace('\n', ' '))
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylo is not None:
            ax.set_ylim(bottom=ylo)
        if yhi is not None:
            ax.set_ylim(top=yhi)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Threshold lines
        if key == 'gen_hp_decay':
            ax.axhline(2.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7,
                       label='threshold=2.0')
            ax.legend(fontsize=9)
        elif key == 'gen_lifespan':
            ax.axhline(30.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7,
                       label='threshold=30')
            ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_b_network_vitality(exp_b_results,
                                  fname='images/session_24/results_s24_network_vitality.png'):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Session 24 Exp B: ネットワークの生死確認\n'
        '入力/出力/内部ノードの平均活動、出力ノード分散、入力Ablation効果、エッジ数',
        fontsize=11,
    )

    x      = np.arange(len(_S24_MR_VALUES))
    colors = [_S24_MR_COLORS[mr] for mr in _S24_MR_VALUES]
    xlbls  = [_S24_MR_LABELS[mr] for mr in _S24_MR_VALUES]

    panels = [
        ('input_mean',      'Input Node Mean Activity\n(nodes 0-3)',  None),
        ('output_mean',     'Output Node Mean Activity\n(nodes 4-8)', None),
        ('internal_mean',   'Internal Node Mean Activity\n(nodes 9-19)', None),
        ('output_variance', 'Output Node Variance\n(per-step var across nodes 4-8)', 0.1),
        ('ablation_effect', 'Input Ablation Effect\n(mean |normal - zeroed_input|)', 0.05),
        ('edges',           'Edge Count (final best genome)', 10),
    ]

    for idx, (key, title, threshold) in enumerate(panels):
        ax   = axes[idx // 3][idx % 3]
        vals = [exp_b_results[mr][key] for mr in _S24_MR_VALUES]
        ymax = max(max(vals) * 1.4, 0.01) if max(vals) > 0 else 0.5

        bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
        if threshold is not None:
            ax.axhline(threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.85,
                       label=f'threshold={threshold}')
            ax.legend(fontsize=9)
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
                         fname='images/session_24/results_s24_sparsity.png'):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        'Session 24 Exp C: 活動ノード数の分布\n'
        f'Active node threshold = activity > {_ACT_THRESHOLD}  (N={_N} total nodes)\n'
        'S22参照: mr=0.01静止→active≈4  S23参照: 動的環境→active≈4',
        fontsize=10,
    )

    x      = np.arange(len(_S24_MR_VALUES))
    colors = [_S24_MR_COLORS[mr] for mr in _S24_MR_VALUES]
    means  = [exp_c_results[mr]['mean_active'] for mr in _S24_MR_VALUES]

    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
    ax.axhline(4.0, color='purple', linestyle=':', linewidth=1.5, alpha=0.8,
               label='S22/S23 静止状態 active≈4')
    ax.axhline(15.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='飽和ライン ≈ 15 nodes')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_S24_MR_LABELS[mr] for mr in _S24_MR_VALUES], fontsize=10)
    ax.set_ylabel(f'Mean Active Nodes (activity > {_ACT_THRESHOLD})')
    ax.set_title('Mean Active Nodes: S24 DynamicGridWorld (生存圧パラメータ進化後)')
    ax.set_ylim(0, _N + 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_d_context(exp_d_results,
                        fname='images/session_24/results_s24_context.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Session 24 Exp D: 文脈分離の再計測 (PenaltyContextGridWorld)\n'
        '生存圧パラメータを進化させた個体が文脈依存的な行動を獲得しているか\n'
        'acc_A/B: food eaten ≥ 1  |  cosine_dist: output nodes (mode A vs B)',
        fontsize=10,
    )

    x      = np.arange(len(_S24_MR_VALUES))
    colors = [_S24_MR_COLORS[mr] for mr in _S24_MR_VALUES]
    xlbls  = [_S24_MR_LABELS[mr] for mr in _S24_MR_VALUES]

    metrics = [
        ('acc_A',         'Mode A Accuracy\n(food ≥ 1 in NW-food episodes)', 0.6),
        ('acc_B',         'Mode B Accuracy\n(food ≥ 1 in SE-food episodes)', 0.6),
        ('mean_cos_dist', 'Cosine Distance\n(output nodes: Mode A vs Mode B)', 0.1),
    ]

    for ai, (key, title, threshold) in enumerate(metrics):
        ax   = axes[ai]
        vals = [exp_d_results[mr][key] for mr in _S24_MR_VALUES]
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
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


def plot_exp_e_lifespan(exp_e_rows,
                         fname='images/session_24/results_s24_lifespan.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Session 24 Exp E: 収束した寿命 vs ネットワーク指標\n'
        '寿命 = hp_start / hp_decay  (短いほど「学習が必要な生存圧」)',
        fontsize=11,
    )

    lifespans = [r['lifespan']        for r in exp_e_rows]
    colors    = [_S24_MR_COLORS[r['mr']] for r in exp_e_rows]
    labels    = [_S24_MR_LABELS[r['mr']].replace('\n', ' ') for r in exp_e_rows]

    panels = [
        ([r['edge_add_prob']   for r in exp_e_rows],
         'edge_add_prob (converged)',           'Edge Add Prob'),
        ([r['mean_active']     for r in exp_e_rows],
         f'Active nodes (>{_ACT_THRESHOLD})',   'Active Nodes'),
        ([r['output_variance'] for r in exp_e_rows],
         'Output node variance',                'Output Variance'),
        ([r['ablation_effect'] for r in exp_e_rows],
         'Input ablation effect',               'Ablation Effect'),
    ]

    thresholds = [None, None, 0.1, 0.05]

    for idx, ((yvals, title, ylabel), thr) in enumerate(zip(panels, thresholds)):
        ax = axes[idx // 2][idx % 2]
        for x_, y_, c_, lbl_ in zip(lifespans, yvals, colors, labels):
            ax.scatter([x_], [y_], color=c_, s=150, zorder=5, label=lbl_)
        ax.axvline(30.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7,
                   label='lifespan threshold=30')
        if thr is not None:
            ax.axhline(thr, color='red', linestyle=':', linewidth=1.2, alpha=0.7,
                       label=f'threshold={thr}')
        ax.set_xlabel('Lifespan (hp_start / hp_decay)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

    print('=== Session 24: 生存圧パラメータの進化（改訂版） ===')
    print('環境: DynamicGridWorld（hp_start/hp_decay/food_value を進化対象に追加）')
    print(f'mr 固定スイープ: {_S24_MR_VALUES}')
    print(f'n_agents={_N_AGENTS}  n_ep={_N_EP}  n_gen={_N_GEN}')
    print()
    print('ゲノム新パラメータ:')
    print(f'  hp_start:   [{_HP_START_LO}, {_HP_START_HI}]')
    print(f'  hp_decay:   [{_HP_DECAY_LO}, {_HP_DECAY_HI}]')
    print(f'  food_value: [{_FOOD_VAL_LO}, {_FOOD_VAL_HI}]')

    print(f'\n[Exp A] 生存圧パラメータの進化（各 mr 条件 {_N_GEN} 世代）')
    exp_a = run_exp_a_genome_convergence(seed=_SEED)

    print('\n  Summary (converged best genomes):')
    hdr = f'  {"mr":>6}  {"hp_s":>5}  {"hp_d":>5}  {"fv":>4}  {"life":>6}  ' \
          f'{"edges":>6}  {"active":>7}  {"out_act":>8}  {"steps":>8}'
    print(hdr)
    print('  ' + '─' * 70)
    for mr in _S24_MR_VALUES:
        bg   = exp_a[mr]['best']
        hist = exp_a[mr]['hist']
        life = bg['hp_start'] / bg['hp_decay']
        print(f'  {mr:6.3f}  {bg["hp_start"]:5d}  {bg["hp_decay"]:5.2f}  '
              f'{bg["food_value"]:4d}  {life:6.1f}  '
              f'{bg["G"].number_of_edges():6d}  '
              f'{hist["gen_mean_active"][-1]:7.1f}  '
              f'{hist["gen_active_output"][-1]:8.1f}  '
              f'{hist["gen_best_steps"][-1]:8.1f}')
    plot_exp_a_genome_convergence(exp_a)

    print('\n[Exp B] ネットワークの生死確認 (n_episodes=20)')
    exp_b = run_exp_b_network_vitality(exp_a, seed=_SEED)
    plot_exp_b_network_vitality(exp_b)

    print('\n[Exp C] 活動ノード数の分布 (n_episodes=20)')
    exp_c = run_exp_c_sparsity(exp_a, seed=_SEED, n_episodes=20)
    plot_exp_c_sparsity(exp_c)

    print('\n[Exp D] 文脈分離の再計測 (PenaltyContextGridWorld, n_ep_per_mode=10)')
    exp_d = run_exp_d_context(exp_a, seed=_SEED, n_ep_per_mode=10)
    plot_exp_d_context(exp_d)

    print('\n[Exp E] 寿命とネットワーク指標の関係')
    exp_e = run_exp_e_lifespan(exp_a, exp_b, exp_c)
    plot_exp_e_lifespan(exp_e)

    # ── Judgment ──────────────────────────────────────────────────────────────
    print('\n  ── Judgment Criteria (Session 24) ─────────────────────────────')
    print('  判定基準:')
    print('  J1. 「学習が必要な生存圧が選ばれた」: hp_decay > 2.0 または 寿命 < 30')
    print('  J2. 「ネットワークが機能している」: edges > 10 かつ out_var > 0.1 かつ abl > 0.05')
    print('  J3. 「スパース性が創発した」: 4 < active < 15')
    print('  J4. 「文脈分離が起きた」: acc_A > 0.6 かつ acc_B > 0.6')
    print()

    print(f'  {"mr":>6}  {"hp_d":>5}  {"life":>6}  J1  '
          f'{"edges":>6}  {"out_v":>6}  {"abl":>6}  J2  '
          f'{"act":>5}  J3  '
          f'{"accA":>6}  {"accB":>6}  J4')
    print('  ' + '─' * 80)

    for mr in _S24_MR_VALUES:
        bg   = exp_a[mr]['best']
        hist = exp_a[mr]['hist']
        life = bg['hp_start'] / bg['hp_decay']
        v    = exp_b[mr]
        c    = exp_c[mr]
        d    = exp_d[mr]

        j1 = bg['hp_decay'] > 2.0 or life < 30.0
        j2 = v['edges'] > 10 and v['output_variance'] > 0.1 and v['ablation_effect'] > 0.05
        j3 = 4.0 < c['mean_active'] < 15.0
        j4 = d['acc_A'] > 0.6 and d['acc_B'] > 0.6

        print(f'  {mr:6.3f}  {bg["hp_decay"]:5.2f}  {life:6.1f}  '
              f'{"✓" if j1 else "✗"}   '
              f'{v["edges"]:6d}  {v["output_variance"]:6.3f}  {v["ablation_effect"]:6.3f}  '
              f'{"✓" if j2 else "✗"}   '
              f'{c["mean_active"]:5.1f}  {"✓" if j3 else "✗"}   '
              f'{d["acc_A"]:6.3f}  {d["acc_B"]:6.3f}  {"✓" if j4 else "✗"}')

    print()
    n_pass_j1 = sum(1 for mr in _S24_MR_VALUES
                    if exp_a[mr]['best']['hp_decay'] > 2.0
                    or exp_a[mr]['best']['hp_start'] / exp_a[mr]['best']['hp_decay'] < 30.0)
    if n_pass_j1 > 0:
        print(f'  → J1: {n_pass_j1}/{len(_S24_MR_VALUES)} 条件で「学習が必要な生存圧」が選ばれた ✓')
        print('    edges=1 の「静止」問題から脱出できた可能性がある')
    else:
        print('  → J1: 全条件で生存圧は低いまま — hp_decay < 2.0, 寿命 ≥ 30')
        print('    生存圧パラメータを進化させても低い値に収束した')
        print('    考えられる理由: 食料ランダム配置でも「何もしない」戦略が最適')

    print('\nDone.')
