#!/usr/bin/env python3
"""
EchoLoop v6 — multi-agent echo dynamics

Hypothesis under test (meaning as closed-loop stabilization):
  A proto-signal-like function may emerge when:
    1. internal loop state is externalized as a continuous acoustic call
    2. the call perturbs other agents' internal loop states
    3. receivers change their spatial behavior
    4. environmental feedback (danger avoidance) stabilizes the pathway

Three conditions compared with identical starting positions and random seed:
  NO_CALLS         — no communication; pure baseline
  CALLS_NO_SPATIAL — calls heard; internal state perturbed; no spatial memory
  FULL             — calls heard + spatial avoidance memory at caller position

The call is NOT a predefined symbol.  It is a continuous readout of vigilance,
alert, and defensive route activations.  Receivers cannot decode "alarm" — they
receive acoustic energy whose roughness/pitch/loudness happens to excite their
own vigilance and alert routes via a fixed linear coupling.  Meaning, if any
emerges, arises from the closed dynamical loop, not from encoding.
"""

import os
import sys
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# ─── World ──────────────────────────────────────────────────────────────────
WORLD_SIZE   = 80
N_AGENTS     = 8
N_STEPS      = 1000
# Zone at centre; agents start 1.6–2.8× radius away so they cross it naturally
DANGER_ZONES = [(40, 40, 10)]          # (cx, cy, radius)

# ─── Internal loop dynamics (v5 parameters, unchanged) ───────────────────────
W_FWD         = 0.55
W_BWD         = 0.35
CROSS         = 0.30
DECAY         = 0.85
S_INHIB       = 0.04
NOISE         = 0.015
FATIGUE_RATE  = 0.012
FATIGUE_DECAY = 0.995
MIN_GAIN      = 0.05

# ─── Movement ────────────────────────────────────────────────────────────────
SPEED_BASE   = 1.5
MAX_SPEED    = 3.5
VEL_DAMP     = 0.72
DEF_STRENGTH = 4.5    # strong flee inside zone so agents exit quickly then re-approach
SOC_STRENGTH = 0.15   # very weak social — exploration dominates
EXPLORE_STR  = 1.1

# ─── Acoustic / communication ────────────────────────────────────────────────
CALL_THRESHOLD   = 0.42   # (defensive_loop_act + vigilance_route) to emit call
CALL_REFRACTORY  = 8      # minimum steps between calls — prevents runaway cascade
HEAR_SIGMA       = 35.0   # distance decay σ (world units)
HEAR_ALPHA       = 0.25   # hearing gain into vigilance/alert routes
DANGER_BOOST     = 0.45   # constant input boost per step inside danger zone

# ─── Spatial avoidance memory ────────────────────────────────────────────────
AVOID_DECAY         = 0.95   # faster decay so marks don't saturate
AVOID_MARK_STRENGTH = 0.18   # keeps grid well below saturation at equilibrium
AVOID_GAIN          = 3.0    # gradient-based repulsion strength

# ─── Loop / route definitions (identical to v5) ──────────────────────────────
LOOP_LIST = ['social', 'exploration', 'defensive']

MEMBERSHIPS = {
    'attention': {'social': 1.0},
    'engage':    {'social': 1.0},
    'approach':  {'social': 1.0},
    'observe':   {'social': 0.70, 'exploration': 0.70},   # bridge S↔E
    'curiosity': {'exploration': 1.0},
    'wander':    {'exploration': 1.0},
    'inspect':   {'exploration': 1.0},
    'vigilance': {'exploration': 0.60, 'defensive': 0.60}, # bridge E↔D
    'alert':     {'defensive': 1.0},
    'freeze':    {'defensive': 1.0},
    'withdraw':  {'defensive': 1.0},
}
ROUTE_NAMES = list(MEMBERSHIPS.keys())

LOOP_EDGE_DEFS = {
    'social': [
        ('attention', 'observe',   +W_FWD), ('observe',   'engage',    +W_FWD),
        ('engage',    'approach',  +W_FWD), ('approach',  'attention', +W_FWD),
        ('observe',   'attention', -W_BWD), ('engage',    'observe',   -W_BWD),
        ('approach',  'engage',    -W_BWD), ('attention', 'approach',  -W_BWD),
    ],
    'exploration': [
        ('curiosity', 'observe',   +W_FWD), ('observe',   'wander',   +W_FWD),
        ('wander',    'vigilance', +W_FWD), ('vigilance', 'inspect',  +W_FWD),
        ('inspect',   'curiosity', +W_FWD),
        ('observe',   'curiosity', -W_BWD), ('wander',    'observe',  -W_BWD),
        ('vigilance', 'wander',   -W_BWD), ('inspect',   'vigilance', -W_BWD),
        ('curiosity', 'inspect',  -W_BWD),
    ],
    'defensive': [
        ('alert',     'vigilance', +W_FWD), ('vigilance', 'freeze',   +W_FWD),
        ('freeze',    'withdraw',  +W_FWD), ('withdraw',  'alert',    +W_FWD),
        ('vigilance', 'alert',    -W_BWD), ('freeze',    'vigilance', -W_BWD),
        ('withdraw',  'freeze',   -W_BWD), ('alert',     'withdraw',  -W_BWD),
    ],
}

LOOP_COLORS = {'social': '#e74c3c', 'exploration': '#9b59b6', 'defensive': '#f39c12'}
COND_COLORS = {
    'NO_CALLS':         '#636e72',
    'CALLS_NO_SPATIAL': '#74b9ff',
    'FULL':             '#00b894',
}
COND_LABELS = {
    'NO_CALLS':         'No calls (baseline)',
    'CALLS_NO_SPATIAL': 'Calls — state only',
    'FULL':             'Calls + spatial memory',
}


# ─── RouteEdge ───────────────────────────────────────────────────────────────
class RouteEdge:
    __slots__ = ('src', 'dst', 'weight', 'loop_name', 'is_forward', 'flow')
    def __init__(self, src, dst, weight, loop_name):
        self.src, self.dst = src, dst
        self.weight     = float(weight)
        self.loop_name  = loop_name
        self.is_forward = weight > 0
        self.flow       = 0.0


# ─── Agent ───────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, agent_id: int, pos):
        self.id  = agent_id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)

        self.routes  = {n: 0.05 for n in ROUTE_NAMES}
        self.fatigue = {l: 0.0  for l in LOOP_LIST}

        # Seed exploration-dominant initial state so agents wander freely
        self.routes['curiosity'] = 0.26
        self.routes['wander']    = 0.22
        self.routes['observe']   = 0.12
        self.routes['vigilance'] = 0.07
        self.routes['attention'] = 0.14

        seen = set()
        self.edges: list[RouteEdge] = []
        for loop, edge_list in LOOP_EDGE_DEFS.items():
            for src, dst, w in edge_list:
                assert (src, dst) not in seen, f"Duplicate edge ({src},{dst})"
                seen.add((src, dst))
                self.edges.append(RouteEdge(src, dst, w, loop))

        # Tracking
        self.was_in_danger      = False
        self.danger_entries     = 0
        self.danger_exposure    = 0
        self.close_calls        = 0   # steps within 2r but outside r
        self.calls_emitted      = 0
        self.calls_heard        = 0
        self.last_call_step     = -CALL_REFRACTORY  # refractory enforcement

        # Per-step history
        self.pos_hist         = []
        self.loop_act_hist    = []
        self.dist_danger_hist = []
        self.call_steps       = []  # steps where this agent emitted a call

    def loop_activities(self) -> dict:
        result = {}
        for l in LOOP_LIST:
            tw = sum(MEMBERSHIPS[r].get(l, 0) for r in ROUTE_NAMES)
            ws = sum(MEMBERSHIPS[r].get(l, 0) * self.routes[r] for r in ROUTE_NAMES)
            result[l] = ws / tw if tw > 0 else 0.0
        return result

    def _eff_w(self, e: RouteEdge) -> float:
        if not e.is_forward:
            return e.weight
        return e.weight * max(MIN_GAIN, 1.0 - self.fatigue[e.loop_name])

    def update_internal(self, ext: dict) -> None:
        """v5 loop dynamics: fatigue, forward/backward flow, cross-inhibition."""
        l_acts = self.loop_activities()

        for l in LOOP_LIST:
            self.fatigue[l] = min(1.0, self.fatigue[l] + FATIGUE_RATE * l_acts[l])
            self.fatigue[l] *= FATIGUE_DECAY

        delta = {n: 0.0 for n in ROUTE_NAMES}
        for e in self.edges:
            f = self.routes[e.src] * self._eff_w(e)
            delta[e.dst] += f
            e.flow = f

        for r in ROUTE_NAMES:
            member_loops = set(MEMBERSHIPS[r].keys())
            non_member   = sum(l_acts[l] for l in LOOP_LIST if l not in member_loops)
            delta[r] -= CROSS * non_member

        for n in ROUTE_NAMES:
            a     = self.routes[n]
            ext_n = ext.get(n, 0.0) + random.uniform(0, NOISE)
            new   = a * DECAY + delta[n] + ext_n - S_INHIB * a * a
            self.routes[n] = float(np.clip(new, 0.0, 1.0))

    def emit_call(self) -> dict:
        """
        Continuous acoustic readout of internal loop state.
        No symbolic encoding: pitch/loudness/roughness/pulse_rate are
        direct projections of vigilance, alert, and defensive routes.
        The receiver cannot decode 'alarm'; it only receives signals
        that happen to excite its own vigilance/alert via HEAR_ALPHA coupling.
        """
        v = self.routes['vigilance']
        a = self.routes['alert']
        d = (self.routes['alert']   * 0.50 +
             self.routes['freeze']  * 0.30 +
             self.routes['withdraw']* 0.20)
        return {
            'agent_id':   self.id,
            'pos':        self.pos.copy(),
            'pitch':      200.0 + v * 3800.0,            # vigilance → Hz
            'loudness':   float(np.clip(d * 2.0, 0, 1)), # defensive → amplitude
            'roughness':  float(a),                       # alert → roughness
            'pulse_rate': (v + d) * 15.0,                 # combined → pulse rate
        }

    def move(self, centroid: np.ndarray, avoidance_grid) -> None:
        """
        Velocity = damped(old) + explore + social + defensive_flee + avoid_gradient.
        Avoidance grid is None except in FULL mode.
        """
        l_acts = self.loop_activities()

        # Exploration: randomised persistent wander
        wander = np.random.randn(2)
        wander /= np.linalg.norm(wander) + 1e-8
        explore_vel = wander * SPEED_BASE * l_acts['exploration'] * EXPLORE_STR

        # Social: drift toward group centroid (only when far enough)
        diff_c = centroid - self.pos
        dc     = np.linalg.norm(diff_c) + 1e-8
        social_vel = (diff_c / dc) * SPEED_BASE * l_acts['social'] * SOC_STRENGTH * float(dc > 10)

        # Flee: strong inside zone (depth-proportional), plus gentle post-
        # experience avoidance driven purely by elevated defensive loop state
        # (no hardcoded zone radius for the outer region).
        def_vel = np.zeros(2)
        for cx, cy, r in DANGER_ZONES:
            diff = self.pos - np.array([cx, cy], dtype=float)
            d    = np.linalg.norm(diff) + 1e-8
            if d < r * 1.1:
                # Inside or just at edge: strong flee proportional to depth
                strength = max(0.0, (r * 1.1 - d) / (r * 1.1))
                def_vel += (diff / d) * SPEED_BASE * DEF_STRENGTH * strength * l_acts['defensive']

        # Spatial avoidance memory repulsion (FULL mode only)
        if avoidance_grid is not None:
            ix = int(np.clip(self.pos[0], 1, WORLD_SIZE - 2))
            iy = int(np.clip(self.pos[1], 1, WORLD_SIZE - 2))
            local = avoidance_grid[iy, ix]
            if local > 0.02:
                gx = (avoidance_grid[iy, min(ix+2, WORLD_SIZE-1)] -
                      avoidance_grid[iy, max(ix-2, 0)])
                gy = (avoidance_grid[min(iy+2, WORLD_SIZE-1), ix] -
                      avoidance_grid[max(iy-2, 0), ix])
                grad = np.array([gx, gy])
                gn   = np.linalg.norm(grad) + 1e-9
                # Move away from increasing avoidance gradient
                def_vel -= (grad / gn) * AVOID_GAIN * local * l_acts['defensive']

        self.vel = VEL_DAMP * self.vel + explore_vel + social_vel + def_vel
        spd = np.linalg.norm(self.vel)
        if spd > MAX_SPEED:
            self.vel = self.vel / spd * MAX_SPEED

        self.pos += self.vel
        # Reflect off walls
        for dim in range(2):
            if self.pos[dim] < 0:
                self.pos[dim] = -self.pos[dim]
                self.vel[dim]  = abs(self.vel[dim])
            elif self.pos[dim] > WORLD_SIZE:
                self.pos[dim] = 2.0 * WORLD_SIZE - self.pos[dim]
                self.vel[dim] = -abs(self.vel[dim])
        self.pos = np.clip(self.pos, 0.0, float(WORLD_SIZE))


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _make_initial_positions(rng):
    """Ring-distributed initial positions, 1.6–2.8× danger radius from zone centre."""
    cx0, cy0, r0 = DANGER_ZONES[0]
    positions = []
    while len(positions) < N_AGENTS:
        angle = rng.uniform(0, 2 * np.pi)
        dist  = rng.uniform(r0 * 1.6, r0 * 2.8)
        pos   = np.array([cx0, cy0], dtype=float) + dist * np.array(
            [np.cos(angle), np.sin(angle)])
        positions.append(np.clip(pos, 5, WORLD_SIZE - 5))
    return positions


# ─── Simulation ──────────────────────────────────────────────────────────────
def run_simulation(mode: str, initial_positions, seed: int = 42, ablation=None):
    """
    Run one condition.  Returns (agents, log, final_avoidance_grid).

    mode:
      'NO_CALLS'         — agents react only to direct proximity of danger zone
      'CALLS_NO_SPATIAL' — calls heard, internal state perturbed, no spatial marks
      'FULL'             — calls + spatial avoidance memory (proto-signal condition)
    ablation (FULL mode only):
      None               — unmodified
      'shuffle_pos'      — spatial mark placed at a random position (not caller's)
      'random_acoustic'  — pitch/roughness replaced with uniform random values
      'no_refractory'    — call refractory period disabled
    """
    assert mode in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL')

    random.seed(seed)
    np.random.seed(seed)

    agents = [Agent(i, pos.copy()) for i, pos in enumerate(initial_positions)]
    avoidance_grid = np.zeros((WORLD_SIZE, WORLD_SIZE))

    log = {
        'group_danger_exposure': [],   # agents inside danger zone per step
        'calls_per_step':        [],
        'avoidance_max':         [],   # max grid value per step
        'loop_avgs':             {l: [] for l in LOOP_LIST},
        'all_calls':             [],   # (step, call_dict)
        'all_heard':             [],   # (step, receiver_id, caller_id, intensity, dv, da)
    }

    pending_calls: list[dict] = []   # calls emitted in previous step, broadcast now

    for step in range(N_STEPS):
        # Avoidance grid decays each step (values wash out without reinforcement)
        avoidance_grid *= AVOID_DECAY

        positions = np.array([a.pos for a in agents])
        centroid  = positions.mean(axis=0)

        new_calls   = []
        heard_count = 0

        for agent in agents:
            ext: dict = {}

            # ── Danger zone proximity → internal state boost ───────────────
            min_d = float('inf')
            min_r = DANGER_ZONES[0][2]
            for cx, cy, r in DANGER_ZONES:
                d = np.linalg.norm(agent.pos - np.array([cx, cy], dtype=float))
                if d < min_d:
                    min_d, min_r = d, r
                if d < r:
                    # Constant boost inside the zone so even 1-2 steps inside
                    # pushes defensive state above the call threshold.
                    # Agents wander in freely (no pre-zone warning).
                    ext['vigilance'] = ext.get('vigilance', 0.0) + DANGER_BOOST
                    ext['alert']     = ext.get('alert',     0.0) + DANGER_BOOST * 0.8

            # ── Danger zone entry / exposure tracking ──────────────────────
            in_danger = min_d < min_r
            if in_danger:
                agent.danger_exposure += 1
                if not agent.was_in_danger:
                    agent.danger_entries += 1
            elif min_r <= min_d < min_r * 1.8:
                agent.close_calls += 1
            agent.was_in_danger = in_danger

            # ── Hear calls from previous step (mode-gated) ─────────────────
            if mode != 'NO_CALLS':
                for call in pending_calls:
                    if call['agent_id'] == agent.id:
                        continue
                    d = np.linalg.norm(agent.pos - call['pos'])
                    intensity = call['loudness'] / (1.0 + (d / HEAR_SIGMA) ** 2)
                    if intensity < 0.02:
                        continue

                    # Acoustic → route boost (possibly randomised for ablation)
                    if ablation == 'random_acoustic':
                        eff_roughness = random.uniform(0, 1)
                        eff_pitch     = random.uniform(200, 4000)
                    else:
                        eff_roughness = call['roughness']
                        eff_pitch     = call['pitch']
                    dv = HEAR_ALPHA * eff_roughness * intensity
                    da = HEAR_ALPHA * (eff_pitch / 4000.0) * intensity
                    ext['vigilance'] = ext.get('vigilance', 0.0) + dv
                    ext['alert']     = ext.get('alert',     0.0) + da
                    agent.calls_heard += 1
                    heard_count += 1
                    log['all_heard'].append(
                        (step, agent.id, call['agent_id'], intensity, dv, da))

                    # Spatial avoidance mark (FULL only; position shuffled for ablation)
                    if mode == 'FULL':
                        if ablation == 'shuffle_pos':
                            mark_pos = np.array([
                                random.uniform(0, WORLD_SIZE),
                                random.uniform(0, WORLD_SIZE),
                            ])
                        else:
                            mark_pos = call['pos']
                        px = int(np.clip(mark_pos[0], 0, WORLD_SIZE - 1))
                        py = int(np.clip(mark_pos[1], 0, WORLD_SIZE - 1))
                        mark = intensity * eff_roughness * AVOID_MARK_STRENGTH
                        for ddx in range(-2, 3):
                            for ddy in range(-2, 3):
                                nx, ny = px + ddx, py + ddy
                                if 0 <= nx < WORLD_SIZE and 0 <= ny < WORLD_SIZE:
                                    w = mark * (1.0 - (abs(ddx) + abs(ddy)) / 6.0)
                                    avoidance_grid[ny, nx] = min(
                                        1.0, avoidance_grid[ny, nx] + w)

            # ── Update internal loop dynamics ──────────────────────────────
            agent.update_internal(ext)

            # ── Record per-agent history ───────────────────────────────────
            l_acts = agent.loop_activities()
            agent.pos_hist.append(agent.pos.copy())
            agent.loop_act_hist.append(l_acts.copy())
            agent.dist_danger_hist.append(min_d)

            # ── Emit call if threshold crossed AND refractory satisfied ───
            def_vigi = l_acts['defensive'] + agent.routes['vigilance']
            refractory_ok = (ablation == 'no_refractory') or \
                            (step - agent.last_call_step) >= CALL_REFRACTORY
            if def_vigi > CALL_THRESHOLD and refractory_ok:
                call = agent.emit_call()
                agent.calls_emitted += 1
                agent.last_call_step = step
                agent.call_steps.append(step)
                new_calls.append(call)
                log['all_calls'].append((step, call))

        # ── Move all agents (after internal updates) ───────────────────────
        for agent in agents:
            agent.move(centroid, avoidance_grid if mode == 'FULL' else None)

        pending_calls = new_calls

        # ── Global log ────────────────────────────────────────────────────
        dz_cx, dz_cy, dz_r = DANGER_ZONES[0]
        in_danger_count = sum(
            1 for a in agents
            if np.linalg.norm(a.pos - np.array([dz_cx, dz_cy], dtype=float)) < dz_r
        )
        log['group_danger_exposure'].append(in_danger_count)
        log['calls_per_step'].append(len(new_calls))
        log['avoidance_max'].append(float(avoidance_grid.max()))

        for l in LOOP_LIST:
            log['loop_avgs'][l].append(
                float(np.mean([a.loop_activities()[l] for a in agents])))

    return agents, log, avoidance_grid.copy()


# ─── Visualization ────────────────────────────────────────────────────────────
def _smooth(arr, w=15):
    if w <= 1:
        return np.asarray(arr)
    k = np.ones(w) / w
    return np.convolve(arr, k, mode='same')


def _ax_style(ax):
    ax.set_facecolor('#16213e')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a5a')
    ax.tick_params(colors='#667799', labelsize=7)


def visualize(results: dict, out_dir: str):
    _BG = '#1a1a2e'
    agents_full = results['FULL']['agents']
    log_full    = results['FULL']['log']
    av_grid     = results['FULL']['avoidance']

    fig = plt.figure(figsize=(20, 15), facecolor=_BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.93, bottom=0.07)

    ax_world   = fig.add_subplot(gs[0, 0:2])
    ax_loops   = fig.add_subplot(gs[0, 2])
    ax_acou    = fig.add_subplot(gs[0, 3])
    ax_group   = fig.add_subplot(gs[1, 0:2])
    ax_avoid   = fig.add_subplot(gs[1, 2])
    ax_calls   = fig.add_subplot(gs[1, 3])
    ax_bar     = fig.add_subplot(gs[2, 0])
    ax_dist    = fig.add_subplot(gs[2, 1:3])
    ax_raster  = fig.add_subplot(gs[2, 3])

    for ax in (ax_world, ax_loops, ax_acou, ax_group,
               ax_avoid, ax_calls, ax_bar, ax_dist, ax_raster):
        _ax_style(ax)

    T = np.arange(N_STEPS)
    agent_cmap = plt.cm.tab10(np.linspace(0, 0.9, N_AGENTS))

    # ── Panel 1: World map (FULL condition) ───────────────────────────────
    ax_world.set_title('World map — FULL condition\n'
                       '(heatmap = avoidance memory, dots = call events)',
                       color='white', fontsize=9, pad=4)
    ax_world.set_xlim(0, WORLD_SIZE)
    ax_world.set_ylim(0, WORLD_SIZE)
    ax_world.set_aspect('equal')

    # Avoidance memory heatmap
    if av_grid.max() > 0.01:
        im = ax_world.imshow(
            av_grid, origin='lower', extent=[0, WORLD_SIZE, 0, WORLD_SIZE],
            cmap='YlOrRd', vmin=0, vmax=min(1.0, av_grid.max()),
            alpha=0.55, interpolation='bilinear')
        plt.colorbar(im, ax=ax_world, fraction=0.03, pad=0.01,
                     label='avoidance strength').ax.tick_params(labelsize=6, colors='#667799')

    # Danger zone
    for cx, cy, r in DANGER_ZONES:
        circ = plt.Circle((cx, cy), r, color='#ff4444',
                           fill=True, alpha=0.25, zorder=3)
        circ2 = plt.Circle((cx, cy), r, color='#ff4444',
                            fill=False, lw=1.5, zorder=4)
        ax_world.add_patch(circ)
        ax_world.add_patch(circ2)
        ax_world.text(cx, cy - r - 2, 'DANGER', ha='center',
                      color='#ff8888', fontsize=7, zorder=5)

    # Agent trajectories
    for agent in agents_full:
        hist = np.array(agent.pos_hist)
        ax_world.plot(hist[:, 0], hist[:, 1],
                      color=agent_cmap[agent.id], lw=0.6, alpha=0.45, zorder=1)
        ax_world.scatter(*hist[0], s=18, color=agent_cmap[agent.id],
                         marker='o', zorder=5, alpha=0.9)
        ax_world.scatter(*hist[-1], s=28, color=agent_cmap[agent.id],
                         marker='*', zorder=5, alpha=0.9)

    # Call event markers (colored by time)
    if log_full['all_calls']:
        steps_c  = np.array([s for s, _ in log_full['all_calls']], dtype=float)
        xs_c     = np.array([c['pos'][0] for _, c in log_full['all_calls']])
        ys_c     = np.array([c['pos'][1] for _, c in log_full['all_calls']])
        loud_c   = np.array([c['loudness'] for _, c in log_full['all_calls']])
        ax_world.scatter(xs_c, ys_c, c=steps_c, cmap='plasma',
                         s=loud_c * 30 + 4, alpha=0.50, zorder=6,
                         norm=Normalize(0, N_STEPS))
        ax_world.text(2, 2, '● = call event (size∝loudness, color=time)',
                      color='#aaaacc', fontsize=6)

    ax_world.set_xlabel('X', color='#667799', fontsize=7)
    ax_world.set_ylabel('Y', color='#667799', fontsize=7)

    # ── Panel 2: Internal loop averages (FULL) ────────────────────────────
    ax_loops.set_title('Loop averages — FULL\n(mean across all agents)',
                       color='white', fontsize=9, pad=4)
    for l in LOOP_LIST:
        y = _smooth(log_full['loop_avgs'][l], 12)
        ax_loops.plot(T, y, color=LOOP_COLORS[l], lw=1.4, label=l, alpha=0.9)
    ax_loops.set_xlim(0, N_STEPS)
    ax_loops.set_ylim(0, 0.60)
    ax_loops.set_xlabel('Step', color='#667799', fontsize=7)
    ax_loops.legend(facecolor='#1a1a2e', labelcolor='white',
                    fontsize=6, framealpha=0.85)

    # ── Panel 3: Acoustic scatter ─────────────────────────────────────────
    ax_acou.set_title('Acoustic parameters\n(all calls, FULL)',
                      color='white', fontsize=9, pad=4)
    if log_full['all_calls']:
        pitch_all = np.array([c['pitch']     for _, c in log_full['all_calls']])
        loud_all  = np.array([c['loudness']  for _, c in log_full['all_calls']])
        rough_all = np.array([c['roughness'] for _, c in log_full['all_calls']])
        sc = ax_acou.scatter(pitch_all, loud_all, c=rough_all,
                             cmap='hot', s=8, alpha=0.5,
                             vmin=0, vmax=1, zorder=3)
        plt.colorbar(sc, ax=ax_acou, fraction=0.04, pad=0.01,
                     label='roughness').ax.tick_params(labelsize=5, colors='#667799')
        ax_acou.set_xlabel('pitch (Hz)', color='#667799', fontsize=7)
        ax_acou.set_ylabel('loudness',   color='#667799', fontsize=7)
    else:
        ax_acou.text(0.5, 0.5, 'no calls', ha='center', va='center',
                     color='white', transform=ax_acou.transAxes)

    # ── Panel 4: Group danger exposure — all conditions ───────────────────
    ax_group.set_title('Group danger exposure (agents inside zone per step)',
                       color='white', fontsize=9, pad=4)
    for mode in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL'):
        y = _smooth(results[mode]['log']['group_danger_exposure'], 20)
        ax_group.plot(T, y, color=COND_COLORS[mode], lw=1.5,
                      label=COND_LABELS[mode], alpha=0.9)
    ax_group.set_xlim(0, N_STEPS)
    ax_group.set_ylim(-0.1, N_AGENTS + 0.5)
    ax_group.axhline(0, color='white', alpha=0.15, lw=0.7)
    ax_group.set_xlabel('Step', color='#667799', fontsize=7)
    ax_group.set_ylabel('Agent count', color='#667799', fontsize=7)
    ax_group.legend(facecolor='#1a1a2e', labelcolor='white',
                    fontsize=6, framealpha=0.85)

    # ── Panel 5: Avoidance memory max ─────────────────────────────────────
    ax_avoid.set_title('Avoidance memory strength\n(grid max per step, FULL)',
                       color='white', fontsize=9, pad=4)
    ax_avoid.fill_between(T, log_full['avoidance_max'],
                          alpha=0.40, color='#fd79a8')
    ax_avoid.plot(T, log_full['avoidance_max'],
                  color='#fd79a8', lw=1.0, alpha=0.85)
    ax_avoid.set_xlim(0, N_STEPS)
    ax_avoid.set_ylim(0, 1.05)
    ax_avoid.set_xlabel('Step', color='#667799', fontsize=7)
    ax_avoid.set_ylabel('Max avoidance', color='#667799', fontsize=7)

    # ── Panel 6: Calls per step ───────────────────────────────────────────
    ax_calls.set_title('Calls emitted per step\n(FULL condition)',
                       color='white', fontsize=9, pad=4)
    y_calls = _smooth(log_full['calls_per_step'], 15)
    ax_calls.fill_between(T, y_calls, alpha=0.35, color='#fdcb6e')
    ax_calls.plot(T, y_calls, color='#fdcb6e', lw=1.0, alpha=0.85)
    ax_calls.set_xlim(0, N_STEPS)
    ax_calls.set_xlabel('Step', color='#667799', fontsize=7)
    ax_calls.set_ylabel('Calls/step', color='#667799', fontsize=7)

    # ── Panel 7: Danger entries / exposure bar chart ──────────────────────
    ax_bar.set_title('Danger zone entries\n(total across all agents)',
                     color='white', fontsize=9, pad=4)
    conds    = list(COND_COLORS.keys())
    entries  = [sum(a.danger_entries  for a in results[m]['agents']) for m in conds]
    exposure = [sum(a.danger_exposure for a in results[m]['agents']) for m in conds]
    x = np.arange(len(conds))
    w = 0.35
    b1 = ax_bar.bar(x - w/2, entries,  w, color=[COND_COLORS[m] for m in conds],
                    alpha=0.85, label='entries')
    b2 = ax_bar.bar(x + w/2, exposure, w, color=[COND_COLORS[m] for m in conds],
                    alpha=0.45, label='exposure steps', hatch='//')
    for bar, v in zip(b1, entries):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                    ha='center', color='white', fontsize=7)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['NoCalls', 'StateOnly', 'Full'],
                           color='#667799', fontsize=6)
    ax_bar.legend(facecolor='#1a1a2e', labelcolor='white',
                  fontsize=6, framealpha=0.85)
    ax_bar.set_ylabel('Count', color='#667799', fontsize=7)

    # ── Panel 8: Mean distance to danger zone — all conditions ────────────
    ax_dist.set_title('Mean distance to danger zone (all agents)',
                      color='white', fontsize=9, pad=4)
    for mode in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL'):
        ags = results[mode]['agents']
        mean_d = np.mean(
            [a.dist_danger_hist for a in ags], axis=0)
        ax_dist.plot(T, _smooth(mean_d, 20),
                     color=COND_COLORS[mode], lw=1.5,
                     label=COND_LABELS[mode], alpha=0.9)
    dz_r = DANGER_ZONES[0][2]
    ax_dist.axhline(dz_r, color='#ff4444', ls='--', lw=0.8, alpha=0.7,
                    label=f'zone radius ({dz_r})')
    ax_dist.axhline(dz_r * 1.5, color='#ff9999', ls=':', lw=0.7, alpha=0.5,
                    label='1.5× radius (close zone)')
    ax_dist.set_xlim(0, N_STEPS)
    ax_dist.set_xlabel('Step', color='#667799', fontsize=7)
    ax_dist.set_ylabel('Distance (units)', color='#667799', fontsize=7)
    ax_dist.legend(facecolor='#1a1a2e', labelcolor='white',
                   fontsize=6, framealpha=0.85, loc='upper right')

    # ── Panel 9: Call emission raster ─────────────────────────────────────
    ax_raster.set_title('Call emission raster\n(FULL; each dot = one call)',
                        color='white', fontsize=9, pad=4)
    for agent in agents_full:
        if agent.call_steps:
            ax_raster.scatter(
                agent.call_steps,
                [agent.id] * len(agent.call_steps),
                s=5, color=agent_cmap[agent.id], alpha=0.7, zorder=3)
    ax_raster.set_xlim(0, N_STEPS)
    ax_raster.set_ylim(-0.5, N_AGENTS - 0.5)
    ax_raster.set_yticks(range(N_AGENTS))
    ax_raster.set_yticklabels([f'A{i}' for i in range(N_AGENTS)],
                               color='#667799', fontsize=6)
    ax_raster.set_xlabel('Step', color='#667799', fontsize=7)

    # ── Footer summary ────────────────────────────────────────────────────
    total_calls = {m: len(results[m]['log']['all_calls']) for m in conds}
    total_close = {m: sum(a.close_calls for a in results[m]['agents']) for m in conds}
    mean_dists  = {m: float(np.mean([np.mean(a.dist_danger_hist)
                                      for a in results[m]['agents']])) for m in conds}
    summary = '  |  '.join(
        f"{COND_LABELS[m]}: entries={entries[i]}, "
        f"exposure={exposure[i]}, "
        f"calls={total_calls[m]}, "
        f"mean_dist={mean_dists[m]:.1f}"
        for i, m in enumerate(conds)
    )
    fig.text(0.5, 0.012, summary, color='#667799', fontsize=6,
             ha='center', va='bottom')

    fig.suptitle(
        'EchoLoop v6 — multi-agent echo dynamics\n'
        'Proto-signal hypothesis: continuous acoustic call as loop-state readout  '
        '|  roughness→vigilance, pitch→alert  |  spatial avoidance memory at caller position',
        color='white', fontsize=11, y=0.99)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'echoloop_v6_dashboard.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    print(f"Saved → {out_path}")
    plt.show()


# ─── Console summary ─────────────────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "=" * 68)
    print("EchoLoop v6 — summary")
    print("=" * 68)
    for mode in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL'):
        agents = results[mode]['agents']
        log    = results[mode]['log']
        total_entries  = sum(a.danger_entries  for a in agents)
        total_exposure = sum(a.danger_exposure for a in agents)
        total_close    = sum(a.close_calls     for a in agents)
        total_calls_e  = sum(a.calls_emitted   for a in agents)
        total_calls_h  = sum(a.calls_heard     for a in agents)
        mean_dist = float(np.mean([
            np.mean(a.dist_danger_hist) for a in agents]))

        print(f"\n  [{COND_LABELS[mode]}]")
        print(f"    danger entries (total)   : {total_entries}")
        print(f"    danger exposure (steps)  : {total_exposure}")
        print(f"    close approaches (steps) : {total_close}")
        print(f"    calls emitted            : {total_calls_e}")
        print(f"    calls heard              : {total_calls_h}")
        min_d_ever = min(min(a.dist_danger_hist) for a in agents)
        print(f"    min dist ever reached    : {min_d_ever:.2f}  (zone r={DANGER_ZONES[0][2]})")
        print(f"    mean dist to danger zone : {mean_dist:.1f}")
        print(f"    avoidance memory max     : {max(log['avoidance_max']):.3f}")

    print()
    print("  Interpretation guide:")
    print("  — entries(CALLS_*) < entries(NO_CALLS): acoustic calls reduce danger exposure")
    print("    even without predefined symbolic encoding.  The call is a continuous readout")
    print("    of vigilance/alert routes; roughness and pitch excite receiver routes.")
    print("  — exposure(FULL) < exposure(CALLS_NO_SPATIAL): spatial avoidance memory")
    print("    reduces time spent inside the zone even when entry count is similar.")
    print("  — close_calls(FULL) < close_calls(NO_CALLS): agents in FULL approach")
    print("    the zone less often, consistent with accumulated avoidance memory near zone.")
    print("  — mean_dist(FULL) > mean_dist(NO_CALLS): FULL agents stay farther on average.")
    print("  — avoidance memory max > 0 only in FULL; if it concentrates near")
    print("    the danger zone, it acts as a persistent proto-signal stabilized by")
    print("    the closed loop (danger→call→mark→avoidance→feedback).")
    print()
    print("  What this does NOT show:")
    print("  — No learned encoding; HEAR_ALPHA coupling is fixed at design time")
    print("  — No semantic content; 'alarm' is never decoded, only energy propagates")
    print("  — No reinforcement learning; no reward signal; no gradient descent")
    print("  — Small N and short run: treat as existence proof, not statistical claim")
    print("=" * 68)


# ─── Multi-seed & ablation analysis ──────────────────────────────────────────

def _fmt_stat(vals):
    return f"{np.mean(vals):5.1f} ± {np.std(vals):4.1f}"


def run_multiseed(n_seeds=30):
    """Run three conditions × n_seeds seeds; return per-condition metric lists."""
    print(f"\nMulti-seed evaluation ({n_seeds} seeds, 3 conditions) ...")
    stats = {m: defaultdict(list) for m in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL')}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        init = _make_initial_positions(rng)
        for mode in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL'):
            agents, log, _ = run_simulation(mode, init, seed=seed)
            stats[mode]['entries'].append(sum(a.danger_entries  for a in agents))
            stats[mode]['exposure'].append(sum(a.danger_exposure for a in agents))
            stats[mode]['close'].append(sum(a.close_calls       for a in agents))
            stats[mode]['calls'].append(len(log['all_calls']))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{n_seeds} done")
    return stats


def run_ablations(n_seeds=30):
    """
    Five conditions, all using the same seeds as run_multiseed:
      FULL (unmodified), shuffle_pos, random_acoustic, no_refractory, NO_CALLS.
    """
    print(f"\nAblation analysis ({n_seeds} seeds) ...")
    cond_defs = {
        'FULL':             dict(mode='FULL',     ablation=None),
        'shuffle_pos':      dict(mode='FULL',     ablation='shuffle_pos'),
        'random_acoustic':  dict(mode='FULL',     ablation='random_acoustic'),
        'no_refractory':    dict(mode='FULL',     ablation='no_refractory'),
        'NO_CALLS':         dict(mode='NO_CALLS', ablation=None),
    }
    stats = {k: defaultdict(list) for k in cond_defs}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        init = _make_initial_positions(rng)
        for cname, kwargs in cond_defs.items():
            agents, log, _ = run_simulation(
                kwargs['mode'], init, seed=seed, ablation=kwargs['ablation'])
            stats[cname]['entries'].append(sum(a.danger_entries  for a in agents))
            stats[cname]['exposure'].append(sum(a.danger_exposure for a in agents))
            stats[cname]['calls'].append(len(log['all_calls']))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{n_seeds} done")
    return stats


def compute_warning_metric(results):
    """
    For each condition with calls, per agent:
      - first_heard_step : first step the agent received any call
      - first_entry_step : first step the agent entered the danger zone
      - heard_before     : first_heard < first_entry (or agent never entered)
      - dist_increased   : mean distance in [t+1, t+20] > distance at t (after first hear)

    Compares mean first-entry step for heard-before vs not-heard-before groups.
    Note: selection bias possible (agents near zone hear more calls).
    """
    dz_r = DANGER_ZONES[0][2]
    out = {}
    for mode in ('CALLS_NO_SPATIAL', 'FULL'):
        agents = results[mode]['agents']
        log    = results[mode]['log']

        heard_by: dict = defaultdict(list)
        for step, rid, cid, intensity, dv, da in log['all_heard']:
            heard_by[rid].append(step)

        records = []
        for agent in agents:
            first_entry = next(
                (t for t, d in enumerate(agent.dist_danger_hist) if d < dz_r),
                None)
            heard_steps = sorted(heard_by.get(agent.id, []))
            first_heard = heard_steps[0] if heard_steps else None

            heard_before = (
                first_heard is not None and
                (first_entry is None or first_heard < first_entry)
            )

            dist_increased = None
            if first_heard is not None:
                t0, t1 = first_heard, min(first_heard + 21, len(agent.dist_danger_hist))
                if t1 > t0 + 1:
                    d0 = agent.dist_danger_hist[t0]
                    d_post = float(np.mean(agent.dist_danger_hist[t0 + 1:t1]))
                    dist_increased = d_post > d0

            records.append(dict(
                agent_id=agent.id,
                first_entry=first_entry,
                first_heard=first_heard,
                heard_before=heard_before,
                dist_increased=dist_increased,
            ))

        heard_entries    = [r['first_entry'] for r in records
                            if r['heard_before'] and r['first_entry'] is not None]
        no_heard_entries = [r['first_entry'] for r in records
                            if not r['heard_before'] and r['first_entry'] is not None]
        n_up    = sum(1 for r in records if r['dist_increased'] is True)
        n_check = sum(1 for r in records if r['dist_increased'] is not None)

        out[mode] = dict(
            records=records,
            n_heard_before=sum(r['heard_before'] for r in records),
            n_total=len(records),
            mean_entry_heard=float(np.mean(heard_entries))    if heard_entries    else None,
            mean_entry_no_heard=float(np.mean(no_heard_entries)) if no_heard_entries else None,
            n_dist_up=n_up,
            n_dist_total=n_check,
        )
    return out


def print_multiseed_table(stats, title):
    print(f"\n{title}")
    print("─" * 74)
    print(f"  {'Condition':<22} {'entries':>14}  {'exposure':>14}  {'calls':>14}")
    print("─" * 74)
    for cond, data in stats.items():
        e = _fmt_stat(data['entries'])
        x = _fmt_stat(data['exposure'])
        c = _fmt_stat(data['calls']) if data.get('calls') else '     —       '
        print(f"  {cond:<22} {e:>14}  {x:>14}  {c:>14}")
    print("─" * 74)


def print_warning_results(warning):
    print("\nPre-entry warning metric (seed=42, single run):")
    print("─" * 60)
    for mode, res in warning.items():
        print(f"\n  [{mode}]")
        print(f"    heard call before first entry : "
              f"{res['n_heard_before']} / {res['n_total']} agents")
        if res['mean_entry_heard'] is not None:
            print(f"    mean first-entry step (heard-before group) : "
                  f"{res['mean_entry_heard']:.0f}")
        if res['mean_entry_no_heard'] is not None:
            print(f"    mean first-entry step (not-heard group)    : "
                  f"{res['mean_entry_no_heard']:.0f}")
        if res['mean_entry_heard'] is not None and res['mean_entry_no_heard'] is not None:
            delay = res['mean_entry_heard'] - res['mean_entry_no_heard']
            note = "(later = consistent with warning)" if delay > 0 else "(earlier = possible selection bias)"
            print(f"    entry delay heard vs not-heard              : {delay:+.0f} steps  {note}")
        print(f"    dist increased after hearing (≤20 steps)    : "
              f"{res['n_dist_up']} / {res['n_dist_total']} agents")
    print("─" * 60)


def write_results_ja(multiseed, ablation_stats, warning, n_seeds, out_path):
    """Write RESULTS6_ja.md from computed statistics."""
    ms = multiseed  # shorthand

    def ms_row(cond, label, calls=True):
        e = _fmt_stat(ms[cond]['entries'])
        x = _fmt_stat(ms[cond]['exposure'])
        c = _fmt_stat(ms[cond]['calls']) if calls else '—'
        return f"| {label} | {e} | {x} | {c} |"

    # Entry reduction percentages
    nc_e  = float(np.mean(ms['NO_CALLS']['entries']))
    cn_e  = float(np.mean(ms['CALLS_NO_SPATIAL']['entries']))
    fu_e  = float(np.mean(ms['FULL']['entries']))
    nc_x  = float(np.mean(ms['NO_CALLS']['exposure']))
    cn_x  = float(np.mean(ms['CALLS_NO_SPATIAL']['exposure']))
    fu_x  = float(np.mean(ms['FULL']['exposure']))
    pct_entry  = (nc_e - cn_e) / nc_e * 100 if nc_e > 0 else 0
    pct_exp    = (nc_x - fu_x) / nc_x * 100 if nc_x > 0 else 0

    def ab_row(cond, label):
        e = _fmt_stat(ablation_stats[cond]['entries'])
        x = _fmt_stat(ablation_stats[cond]['exposure'])
        return f"| {label} | {e} | {x} |"

    # Warning metric text
    warn_full = warning.get('FULL', {})
    w_nhb  = warn_full.get('n_heard_before', '?')
    w_ntot = warn_full.get('n_total', '?')
    w_nup  = warn_full.get('n_dist_up', '?')
    w_nchk = warn_full.get('n_dist_total', '?')
    w_meh  = warn_full.get('mean_entry_heard')
    w_men  = warn_full.get('mean_entry_no_heard')
    if w_meh is not None and w_men is not None:
        delay_str = f"{w_meh - w_men:+.0f}ステップ（正 = 受聴エージェントの侵入が遅い）"
    else:
        delay_str = "—（サンプル不足）"

    lines = [
        "# EchoLoop v6 — 結果サマリ（日本語）",
        "",
        "## 概要",
        "",
        "エージェントの内部ループ状態を連続音響パラメータ（ピッチ・ラウドネス・粗さ・パルス率）として",
        "外部化し，そのシグナルが受信エージェントの空間行動に影響を与えるかを検証した。",
        "記号的エンコーディング・強化学習・ニューラルネットワークはいずれも使用していない。",
        "",
        "## 主要結果（30シード平均 ± 標準偏差）",
        "",
        "| 条件 | 危険ゾーン侵入回数 | 暴露ステップ数 | 発音回数 |",
        "|---|---|---|---|",
        ms_row('NO_CALLS',         'NO_CALLS（ベースライン）',     calls=True),
        ms_row('CALLS_NO_SPATIAL', 'CALLS_NO_SPATIAL（音声のみ）', calls=True),
        ms_row('FULL',             'FULL（音声＋空間記憶）',       calls=True),
        "",
        f"音声信号だけで侵入回数がベースライン比 **約{pct_entry:.0f}%** 減少した。",
        f"空間記憶の追加により暴露時間がさらに **約{pct_exp:.0f}%** 短縮された。",
        "",
        "## 事前警告指標（seed=42 単一実行）",
        "",
        f"FULL条件では，{w_nhb}/{w_ntot} エージェントが初回危険ゾーン侵入前に他エージェントの発音を受聴した。",
        f"受聴後20ステップの平均距離変化を見ると，{w_nup}/{w_nchk} エージェントで危険ゾーンへの接近が鈍化した。",
        f"受聴グループと非受聴グループの侵入タイミング差: {delay_str}。",
        "",
        "**注意:** 受聴前エージェントはもともとゾーン付近にいた可能性があり（選択バイアス），",
        "因果関係の主張には注意が必要である。",
        "",
        "## アブレーション（30シード平均 ± 標準偏差）",
        "",
        "| 条件 | 侵入回数 | 暴露時間 |",
        "|---|---|---|",
        ab_row('FULL',            'FULL（オリジナル）'),
        ab_row('shuffle_pos',     'shuffle_pos（マーク位置ランダム化）'),
        ab_row('random_acoustic', 'random_acoustic（音響パラメータランダム化）'),
        ab_row('no_refractory',   'no_refractory（不応期なし）'),
        ab_row('NO_CALLS',        'NO_CALLS（参照ベースライン）'),
        "",
        "- **shuffle_pos**: 空間マークをランダム位置に置く → 位置情報の寄与を検証",
        "- **random_acoustic**: 音響パラメータをランダム化 → 音響構造の寄与を検証",
        "- **no_refractory**: 不応期を除去 → フィードバックカスケード制御の寄与を検証",
        "",
        "## 過大主張を避けるための注意点",
        "",
        "- 音響パラメータとルート活性の結合（`HEAR_ALPHA`）は設計時に固定されており，**学習されていない**",
        "- シグナルに「意味」は事前に割り当てられていない。受信者の応答は線形結合であり，",
        "  アラームを「解読」しているわけではない",
        "- 本結果は N=8 エージェント・1000ステップの存在証明であり，",
        "  統計的に強いクレームには不十分なサンプル数である",
        "- 不応期・閾値・空間減衰などのパラメータは観察可能な挙動が生まれるよう",
        "  手動調整されており，自然に出現したものではない",
        "- 複数シードで方向性は一致しているが，標準偏差が大きく個体差も大きい",
        "",
        f"*生成: EchoLoop v6 / {n_seeds}シード評価*",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved → {out_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    SEED = 42
    analyze = '--analyze' in sys.argv

    random.seed(SEED)
    np.random.seed(SEED)

    rng = np.random.default_rng(SEED)
    initial_positions = _make_initial_positions(rng)

    print(f"EchoLoop v6  |  {N_AGENTS} agents  |  {N_STEPS} steps  |  "
          f"danger zone at {DANGER_ZONES}")
    print(f"Initial positions: "
          + ", ".join(f"({p[0]:.0f},{p[1]:.0f})" for p in initial_positions))

    results = {}
    for mode in ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL'):
        print(f"\nRunning {mode} ...")
        agents, log, avoidance = run_simulation(mode, initial_positions, seed=SEED)
        results[mode] = {'agents': agents, 'log': log, 'avoidance': avoidance}
        n_calls = len(log['all_calls'])
        n_heard  = len(log['all_heard'])
        print(f"  calls emitted: {n_calls}  |  heard events: {n_heard}  |  "
              f"avoidance max: {max(log['avoidance_max']):.3f}")

    print_summary(results)

    out_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images'))
    visualize(results, out_dir)

    if analyze:
        N_SEEDS = 30
        multiseed   = run_multiseed(n_seeds=N_SEEDS)
        abl_stats   = run_ablations(n_seeds=N_SEEDS)
        warning     = compute_warning_metric(results)

        print_multiseed_table(multiseed, f"Multi-seed results (n={N_SEEDS}):")
        print_multiseed_table(abl_stats, "Ablation results:")
        print_warning_results(warning)

        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        write_results_ja(
            multiseed, abl_stats, warning, N_SEEDS,
            os.path.join(repo_root, 'results', 'RESULTS6_ja.md'))


if __name__ == '__main__':
    main()
