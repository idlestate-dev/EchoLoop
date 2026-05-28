#!/usr/bin/env python3
"""
EchoLoop v5 — shared routes による blended attractor dynamics

社会/探索/防御ループが bridge route (observe, vigilance) を共有する。
完全分離でなく mixed internal state が emergent に出る。

構造:
  social (4-cycle):      attention → observe → engage → approach → (→attention)
  exploration (5-cycle): curiosity → observe → wander → vigilance → inspect → (→curiosity)
  defensive (4-cycle):   alert     → vigilance → freeze → withdraw → (→alert)

  observe    = social ↔ exploration bridge
  vigilance  = exploration ↔ defensive bridge

shared route の性質:
  - 複数ループの edge が同一 node に集まる → 干渉・同期
  - non-member loop からの cross-inhibition が 50% 減 → ブリッジが消えにくい
  - 疲労は edge の所属 loop で決まる → loop 間疲労伝播

LCM(4,5) = 20 step のビーティングが partial synchronization を生む

観測: blended states / oscillatory ambiguity / smooth transition / hybrid phases
"""

import random
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# ── 定数 ─────────────────────────────────────────────────────────────────────
N_STEPS       = 700
W_FWD         = 0.55
W_BWD         = 0.35
CROSS         = 0.30
DECAY         = 0.85
S_INHIB       = 0.04
NOISE         = 0.018
A_THRESH      = 0.36
FATIGUE_RATE  = 0.014
FATIGUE_DECAY = 0.995
MIN_GAIN      = 0.05

# ── ループ構成 ────────────────────────────────────────────────────────────────
LOOP_LIST = ['social', 'exploration', 'defensive']

# route → ループ所属 (共有 route は複数)
MEMBERSHIPS = {
    'attention':  {'social': 1.0},
    'engage':     {'social': 1.0},
    'approach':   {'social': 1.0},
    'observe':    {'social': 0.70, 'exploration': 0.70},   # bridge S↔E
    'curiosity':  {'exploration': 1.0},
    'wander':     {'exploration': 1.0},
    'inspect':    {'exploration': 1.0},
    'vigilance':  {'exploration': 0.60, 'defensive': 0.60}, # bridge E↔D
    'alert':      {'defensive': 1.0},
    'freeze':     {'defensive': 1.0},
    'withdraw':   {'defensive': 1.0},
}
ROUTE_NAMES  = list(MEMBERSHIPS.keys())
SHARED_ROUTES = [r for r, m in MEMBERSHIPS.items() if len(m) > 1]  # observe, vigilance

# ループごとの edge 定義 (各 edge は所属 loop を持つ)
# social 4-cycle: attention → observe → engage → approach → attention
# exploration 5-cycle: curiosity → observe → wander → vigilance → inspect → curiosity
# defensive 4-cycle: alert → vigilance → freeze → withdraw → alert
LOOP_EDGE_DEFS = {
    'social': [
        ('attention', 'observe',  +W_FWD), ('observe',  'engage',   +W_FWD),
        ('engage',    'approach', +W_FWD), ('approach', 'attention', +W_FWD),
        ('observe',   'attention', -W_BWD), ('engage',  'observe',   -W_BWD),
        ('approach',  'engage',   -W_BWD), ('attention', 'approach', -W_BWD),
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

LOOP_COLORS  = {'social': '#e74c3c', 'exploration': '#9b59b6', 'defensive': '#f39c12'}
ROUTE_COLORS = {
    'attention': '#ff9999', 'engage':    '#e74c3c', 'approach':  '#922b21',
    'observe':   '#00cec9',                                                  # shared: teal
    'curiosity': '#d7bde2', 'wander':    '#9b59b6', 'inspect':   '#6c3483',
    'vigilance': '#ffeaa7',                                                  # shared: yellow
    'alert':     '#fde3a7', 'freeze':    '#f39c12', 'withdraw':  '#b7770d',
}

ACTION_MAP = {
    'attention': 'look_at_user', 'engage':    'gesture',    'approach':  'move_toward',
    'observe':   'observe',      'curiosity': 'look_around', 'wander':   'move_to_obj',
    'inspect':   'point',        'vigilance': 'scan_wide',  'alert':     'scan',
    'freeze':    'freeze',       'withdraw':  'step_back',
}

# 実験フェーズ
PHASES = [
    (  0, 100, 'internal-1',  {}),
    (100, 160, 'soc push',    {'attention': 0.20, 'engage': 0.15}),
    (160, 270, 'internal-2',  {}),
    (270, 340, 'exp push',    {'curiosity': 0.20, 'wander': 0.15}),
    (340, 470, 'internal-3',  {}),
    (470, 540, 'def push',    {'alert': 0.20, 'freeze': 0.15}),
    (540, 700, 'internal-4',  {}),
]


# ── RouteEdge ─────────────────────────────────────────────────────────────────
class RouteEdge:
    __slots__ = ('src', 'dst', 'weight', 'loop_name', 'is_forward', 'flow')

    def __init__(self, src: str, dst: str, weight: float, loop_name: str):
        self.src, self.dst = src, dst
        self.weight     = float(weight)
        self.loop_name  = loop_name
        self.is_forward = (weight > 0)
        self.flow       = 0.0


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.routes  = {n: 0.05 for n in ROUTE_NAMES}
        self.fatigue = {l: 0.0  for l in LOOP_LIST}

        # social を少し強めに seed
        for r in ['attention', 'engage', 'approach']:
            self.routes[r] = 0.28
        self.routes['observe']   = 0.16   # bridge: partial activation
        self.routes['vigilance'] = 0.08

        # Build edges from loop definitions
        seen = set()
        self.edges: list[RouteEdge] = []
        for loop_name, edge_list in LOOP_EDGE_DEFS.items():
            for src, dst, w in edge_list:
                key = (src, dst)
                assert key not in seen, f"Duplicate edge {key}"
                seen.add(key)
                self.edges.append(RouteEdge(src, dst, w, loop_name))

        self.action = 'idle'

        # 記録
        self.route_hist    = {n: [] for n in ROUTE_NAMES}
        self.loop_hist     = {l: [] for l in LOOP_LIST}
        self.fatigue_hist  = {l: [] for l in LOOP_LIST}
        self.ambiguity_log = []   # 0=clear dominance, 1=perfectly tied
        self.blend_log     = []   # fraction of non-dominant activity
        self.dominant_log  : list[str]  = []
        self.loop_act_log  : list[dict] = []
        self.action_log    : list[str]  = []

    # ── loop activity (soft membership weighted average) ─────────────────
    def loop_activities(self) -> dict:
        result = {}
        for l in LOOP_LIST:
            tw = sum(MEMBERSHIPS[r][l] for r in ROUTE_NAMES if l in MEMBERSHIPS[r])
            ws = sum(MEMBERSHIPS[r][l] * self.routes[r]
                     for r in ROUTE_NAMES if l in MEMBERSHIPS[r])
            result[l] = ws / tw if tw > 0 else 0.0
        return result

    # ── effective forward weight (fatigue from edge's own loop) ──────────
    def _eff_w(self, e: RouteEdge) -> float:
        if not e.is_forward:
            return e.weight
        return e.weight * max(MIN_GAIN, 1.0 - self.fatigue[e.loop_name])

    # ── 1 step ──────────────────────────────────────────────────────────────
    def update(self, ext: dict, reward: float = 0.0) -> str:
        l_acts = self.loop_activities()

        # 1. 疲労更新 (loop 活性度に比例)
        for l in LOOP_LIST:
            self.fatigue[l] = min(1.0, self.fatigue[l] + FATIGUE_RATE * l_acts[l])
            self.fatigue[l] *= FATIGUE_DECAY

        # 2. intra-loop + inter-loop-bridge フロー
        delta = {n: 0.0 for n in ROUTE_NAMES}
        for e in self.edges:
            f = self.routes[e.src] * self._eff_w(e)
            delta[e.dst] += f
            e.flow = f

        # 3. cross-loop 抑制 (shared route は non-member loop からのみ抑制)
        for r in ROUTE_NAMES:
            member_loops = set(MEMBERSHIPS[r].keys())
            non_member   = sum(l_acts[l] for l in LOOP_LIST if l not in member_loops)
            delta[r] -= CROSS * non_member

        # 4. activation 更新
        for n in ROUTE_NAMES:
            a     = self.routes[n]
            ext_n = ext.get(n, 0.0) + random.uniform(0, NOISE)
            new   = a * DECAY + delta[n] + ext_n - S_INHIB * a * a
            self.routes[n] = float(np.clip(new, 0.0, 1.0))

        # 5. reward → forward edge を微調整
        if abs(reward) > 0.01:
            for e in self.edges:
                if e.is_forward and abs(e.flow) > 0.03:
                    e.weight = float(np.clip(e.weight + reward * 0.002, 0.10, 0.80))

        # 6. アクション (dominant loop 内の最高活性 route から読み出し)
        l_acts_now = self.loop_activities()
        dominant   = max(l_acts_now, key=l_acts_now.get)
        # dominant loop の routes で最高のものを選ぶ (bridge route も候補に含む)
        dom_routes = [r for r, m in MEMBERSHIPS.items() if dominant in m]
        best_route = max(dom_routes, key=lambda r: self.routes[r])
        self.action = ACTION_MAP[best_route] if self.routes[best_route] > A_THRESH else 'idle'

        # 7. dominance ambiguity と blend 係数
        acts_sorted = sorted(l_acts_now.values(), reverse=True)
        total_act   = sum(acts_sorted) + 1e-9
        ambiguity   = 1.0 - (acts_sorted[0] - acts_sorted[1]) / (acts_sorted[0] + acts_sorted[1] + 1e-9)
        blend       = sum(v for l, v in l_acts_now.items() if l != dominant) / total_act

        # 8. 記録
        for n in ROUTE_NAMES:
            self.route_hist[n].append(self.routes[n])
        for l in LOOP_LIST:
            self.loop_hist[l].append(l_acts_now[l])
            self.fatigue_hist[l].append(self.fatigue[l])
        self.ambiguity_log.append(float(ambiguity))
        self.blend_log.append(float(blend))
        self.dominant_log.append(dominant)
        self.loop_act_log.append(l_acts_now.copy())
        self.action_log.append(self.action)

        return dominant


# ── シミュレーション ──────────────────────────────────────────────────────────
def simulate() -> Agent:
    agent = Agent()
    for step in range(N_STEPS):
        ext = {}
        for start, end, _, boosts in PHASES:
            if start <= step < end:
                ext = boosts
                break
        agent.update(ext)
    return agent


# ── 可視化 ────────────────────────────────────────────────────────────────────
def visualize(agent: Agent):
    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    ax_main    = fig.add_subplot(gs[0, :])
    ax_ternary = fig.add_subplot(gs[1, 0])
    ax_ambig   = fig.add_subplot(gs[1, 1])
    ax_shared  = fig.add_subplot(gs[1, 2])
    ax_sync    = fig.add_subplot(gs[1, 3])
    ax_ps1     = fig.add_subplot(gs[2, 0])
    ax_ps2     = fig.add_subplot(gs[2, 1])
    ax_routes  = fig.add_subplot(gs[2, 2])
    ax_action  = fig.add_subplot(gs[2, 3])

    _dark = '#16213e'
    for ax in (ax_main, ax_ternary, ax_ambig, ax_shared, ax_sync,
               ax_ps1, ax_ps2, ax_routes, ax_action):
        ax.set_facecolor(_dark)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a5a')
        ax.tick_params(colors='#667799', labelsize=7)

    T = np.arange(N_STEPS)

    # ── Main: Loop Activity + ambiguity background ─────────────────────────
    ax_main.set_title(
        'Loop Activity   (background opacity = dominance ambiguity)',
        color='white', fontsize=10, pad=5)
    ax_main.set_xlim(0, N_STEPS)
    ax_main.set_ylim(-0.02, 1.08)

    for t in range(N_STEPS - 1):
        alpha = agent.ambiguity_log[t] * 0.18 + 0.02
        ax_main.axvspan(t, t+1, alpha=alpha, color='white')

    for l in LOOP_LIST:
        ax_main.plot(T, agent.loop_hist[l], color=LOOP_COLORS[l],
                     lw=1.5, alpha=0.90, label=l)

    # shared route activations (dashed overlay)
    for r in SHARED_ROUTES:
        ax_main.plot(T, agent.route_hist[r], color=ROUTE_COLORS[r],
                     lw=0.8, alpha=0.55, ls='--', label=f'{r} (shared)')

    for start, _, label, _ in PHASES:
        ax_main.axvline(start, color='#333355', lw=0.6, alpha=0.7)
        ax_main.text(start + 2, 1.04, label, color='#8888aa', fontsize=6)

    ax_main.legend(loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=7, framealpha=0.85, ncol=2)
    ax_main.set_xlabel('Step', color='#667799', fontsize=8)
    ax_main.set_ylabel('Loop Activity', color='#667799', fontsize=8)

    # ── Ternary Diagram ────────────────────────────────────────────────────
    ax_ternary.set_title('State Ternary Diagram\n(corners = pure attractors)',
                          color='white', fontsize=9, pad=4)
    ax_ternary.set_aspect('equal')
    ax_ternary.set_xlim(-0.08, 1.08)
    ax_ternary.set_ylim(-0.08, 0.98)
    ax_ternary.axis('off')

    # draw triangle
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax_ternary.plot(tri[:, 0], tri[:, 1], color='#555577', lw=1.0, alpha=0.7)
    # corner labels
    ax_ternary.text(0, -0.05, 'exploration', ha='center', va='top',
                    color=LOOP_COLORS['exploration'], fontsize=7)
    ax_ternary.text(1, -0.05, 'defensive', ha='center', va='top',
                    color=LOOP_COLORS['defensive'], fontsize=7)
    ax_ternary.text(0.5, np.sqrt(3)/2 + 0.04, 'social', ha='center', va='bottom',
                    color=LOOP_COLORS['social'], fontsize=7)
    # centroid (blended state)
    ax_ternary.scatter([0.5], [np.sqrt(3)/6], s=20, c='white', marker='+',
                       alpha=0.30, zorder=3)

    # trajectory
    def to_ternary(s, e, d):
        total = s + e + d + 1e-9
        s, e, d = s/total, e/total, d/total
        x = d + 0.5 * s
        y = (np.sqrt(3)/2) * s
        return x, y

    tx, ty, tc = [], [], []
    for t in range(N_STEPS):
        la = agent.loop_act_log[t]
        x, y = to_ternary(la['social'], la['exploration'], la['defensive'])
        tx.append(x); ty.append(y)
        tc.append(t / N_STEPS)

    for i in range(N_STEPS - 1):
        c = plt.cm.plasma(tc[i])
        ax_ternary.plot(tx[i:i+2], ty[i:i+2], color=c, alpha=0.35, lw=0.7)

    # markers every 50 steps
    for i in range(0, N_STEPS, 50):
        dom = agent.dominant_log[i]
        ax_ternary.scatter(tx[i], ty[i], s=12, c=LOOP_COLORS[dom], alpha=0.7, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=Normalize(0, N_STEPS))
    cb = plt.colorbar(sm, ax=ax_ternary, fraction=0.04, pad=0.01)
    cb.set_label('time', color='#667799', fontsize=5)
    cb.ax.tick_params(labelsize=5, colors='#667799')

    # ── Ambiguity and Blending ─────────────────────────────────────────────
    ax_ambig.set_title('Dominance Ambiguity + Blend Coefficient', color='white', fontsize=9, pad=4)
    W_SMOOTH = 8
    amb_sm = np.convolve(agent.ambiguity_log, np.ones(W_SMOOTH)/W_SMOOTH, mode='same')
    bld_sm = np.convolve(agent.blend_log, np.ones(W_SMOOTH)/W_SMOOTH, mode='same')
    ax_ambig.fill_between(T, amb_sm, alpha=0.40, color='#74b9ff', label='ambiguity')
    ax_ambig.fill_between(T, bld_sm, alpha=0.35, color='#fd79a8', label='blend coeff')
    ax_ambig.plot(T, amb_sm, color='#74b9ff', lw=0.9, alpha=0.80)
    ax_ambig.plot(T, bld_sm, color='#fd79a8', lw=0.9, alpha=0.80)
    ax_ambig.axhline(0.5, color='white', alpha=0.15, ls='--', lw=0.7)
    for start, _, _, _ in PHASES:
        ax_ambig.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_ambig.set_xlim(0, N_STEPS)
    ax_ambig.set_ylim(0, 1.05)
    ax_ambig.set_xlabel('Step', color='#667799', fontsize=7)
    ax_ambig.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=6, framealpha=0.85)

    # ── Shared Route Activations ───────────────────────────────────────────
    ax_shared.set_title('Shared Bridge Routes vs Exclusive Routes', color='white', fontsize=9, pad=4)
    # shared
    for r in SHARED_ROUTES:
        ax_shared.plot(T, agent.route_hist[r], color=ROUTE_COLORS[r],
                       lw=1.8, alpha=0.90, label=f'{r} (bridge)')
    # exclusive samples (one per loop)
    for r in ['attention', 'curiosity', 'alert']:
        ax_shared.plot(T, agent.route_hist[r], color=ROUTE_COLORS[r],
                       lw=0.8, alpha=0.55, ls=':', label=r)
    for start, _, _, _ in PHASES:
        ax_shared.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_shared.set_xlim(0, N_STEPS)
    ax_shared.set_ylim(-0.02, 1.05)
    ax_shared.set_xlabel('Step', color='#667799', fontsize=7)
    ax_shared.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=6,
                     framealpha=0.85, loc='upper right')

    # ── Rolling Synchronization (Pearson r between loop pairs) ────────────
    ax_sync.set_title('Rolling Loop Synchronization (r, w=30)', color='white', fontsize=9, pad=4)
    W_CORR = 30
    r_se, r_ed, r_sd = [], [], []
    for t in range(W_CORR, N_STEPS):
        s  = np.array(agent.loop_hist['social'][t-W_CORR:t])
        e  = np.array(agent.loop_hist['exploration'][t-W_CORR:t])
        d  = np.array(agent.loop_hist['defensive'][t-W_CORR:t])
        def _r(a, b):
            if a.std() < 1e-6 or b.std() < 1e-6: return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        r_se.append(_r(s, e))
        r_ed.append(_r(e, d))
        r_sd.append(_r(s, d))

    t_corr = np.arange(W_CORR, N_STEPS)
    ax_sync.fill_between(t_corr, r_se, alpha=0.35, color='#00cec9',
                         label='social↔exploration (share observe)')
    ax_sync.fill_between(t_corr, r_ed, alpha=0.35, color='#ffeaa7',
                         label='exploration↔defensive (share vigilance)')
    ax_sync.fill_between(t_corr, r_sd, alpha=0.25, color='#636e72',
                         label='social↔defensive (no share)')
    ax_sync.plot(t_corr, r_se, color='#00cec9', lw=0.9, alpha=0.8)
    ax_sync.plot(t_corr, r_ed, color='#ffeaa7', lw=0.9, alpha=0.8)
    ax_sync.plot(t_corr, r_sd, color='#636e72', lw=0.7, alpha=0.7)
    ax_sync.axhline(0, color='white', alpha=0.25, lw=0.8)
    for start, _, _, _ in PHASES:
        if start >= W_CORR:
            ax_sync.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_sync.set_xlim(W_CORR, N_STEPS)
    ax_sync.set_ylim(-1.05, 1.05)
    ax_sync.set_xlabel('Step', color='#667799', fontsize=7)
    ax_sync.set_ylabel('Pearson r', color='#667799', fontsize=7)
    ax_sync.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=5,
                   framealpha=0.85, loc='lower right')

    # ── Phase Spaces ──────────────────────────────────────────────────────
    def plot_phase(ax, lx, ly, title):
        xs = [d[lx] for d in agent.loop_act_log]
        ys = [d[ly] for d in agent.loop_act_log]
        n  = len(xs)
        # ambiguity as bubble size
        sizes = [4 + agent.ambiguity_log[t] * 20 for t in range(n)]
        colors_t = [LOOP_COLORS[agent.dominant_log[t]] for t in range(n)]
        for i in range(n - 1):
            c = plt.cm.plasma(i / n)
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, alpha=0.25, lw=0.6)
        for i in range(0, n, 4):
            ax.scatter(xs[i], ys[i], s=sizes[i], c=[colors_t[i]], alpha=0.45, zorder=3)
        # attractor markers
        attrs = {'social×exploration': [(0.90, 0.05, 'social'), (0.05, 0.90, 'exploration')],
                 'exploration×defensive': [(0.90, 0.05, 'exploration'), (0.05, 0.90, 'defensive')]}
        for ax_val, ay_val, l in attrs.get(f'{lx}×{ly}', []):
            ax.scatter([ax_val], [ay_val], s=55, c=LOOP_COLORS[l], marker='*',
                       zorder=6, alpha=0.7)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel(lx, color=LOOP_COLORS[lx], fontsize=7)
        ax.set_ylabel(ly, color=LOOP_COLORS[ly], fontsize=7)
        ax.set_title(title + '\n(bubble=ambiguity)', color='white', fontsize=8, pad=3)
        ax.tick_params(labelbottom=False, labelleft=False)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=Normalize(0, N_STEPS))
        cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=5, colors='#667799')

    plot_phase(ax_ps1, 'social', 'exploration', 'social × exploration')
    plot_phase(ax_ps2, 'exploration', 'defensive', 'exploration × defensive')

    # ── All Route Activations ─────────────────────────────────────────────
    ax_routes.set_title('Route Activations (bridge routes bold)', color='white', fontsize=9, pad=4)
    ax_routes.set_xlim(0, N_STEPS)
    ax_routes.set_ylim(-0.02, 1.05)
    for r in ROUTE_NAMES:
        is_bridge = r in SHARED_ROUTES
        ax_routes.plot(T, agent.route_hist[r], color=ROUTE_COLORS[r],
                       lw=1.5 if is_bridge else 0.65,
                       alpha=0.90 if is_bridge else 0.55)
    handles = ([mpatches.Patch(color=LOOP_COLORS[l], label=l) for l in LOOP_LIST]
               + [mpatches.Patch(color=ROUTE_COLORS[r], label=f'{r}*') for r in SHARED_ROUTES])
    ax_routes.legend(handles=handles, loc='upper right', facecolor='#1a1a2e',
                     labelcolor='white', fontsize=6, framealpha=0.85)
    for start, _, _, _ in PHASES:
        ax_routes.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_routes.set_xlabel('Step', color='#667799', fontsize=7)

    # ── Action Distribution ────────────────────────────────────────────────
    ax_action.set_title('Action Distribution', color='white', fontsize=9, pad=4)
    counts = Counter(agent.action_log)
    total  = len(agent.action_log)
    acts   = [a for a, _ in counts.most_common()]
    vals   = [counts[a] / total for a in acts]
    def _act_color(a):
        for r, v in ACTION_MAP.items():
            if v == a:
                l = max(MEMBERSHIPS[r], key=MEMBERSHIPS[r].get)
                return LOOP_COLORS[l]
        return '#888888'
    clrs = [_act_color(a) for a in acts]
    bars = ax_action.barh(acts, vals, color=clrs, alpha=0.85, height=0.55)
    ax_action.set_xlim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax_action.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{v*100:.0f}%', va='center', color='white', fontsize=7)
    ax_action.set_xlabel('Fraction', color='#667799', fontsize=7)

    # ── footer ────────────────────────────────────────────────────────────
    sw_total = sum(1 for t in range(1, N_STEPS)
                   if agent.dominant_log[t] != agent.dominant_log[t-1])
    mean_amb = np.mean(agent.ambiguity_log)
    mean_bld = np.mean(agent.blend_log)
    dom_cnt  = Counter(agent.dominant_log)
    fig.text(0.50, 0.012,
             f'Switches: {sw_total}  |  '
             f'Mean ambiguity: {mean_amb:.3f}  |  '
             f'Mean blend: {mean_bld:.3f}  |  '
             + ' / '.join(f'{l}: {dom_cnt[l]/N_STEPS*100:.0f}%' for l in LOOP_LIST),
             color='#667799', fontsize=7, ha='center')

    fig.suptitle('EchoLoop v5 — Shared Routes / Blended Attractor Dynamics  '
                 '(observe: S↔E bridge  |  vigilance: E↔D bridge)',
                 color='white', fontsize=12, y=0.98)

    out = 'echoloop_v5_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved -> {out}")
    plt.show()


# ── エントリポイント ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("EchoLoop v5: running ...")
    agent = simulate()

    dom_cnt = Counter(agent.dominant_log)
    sw_total = sum(1 for t in range(1, N_STEPS)
                   if agent.dominant_log[t] != agent.dominant_log[t-1])
    mean_amb = np.mean(agent.ambiguity_log)
    mean_bld = np.mean(agent.blend_log)

    print(f"\nSwitches: {sw_total}")
    print("Dominance:")
    for l in LOOP_LIST:
        print(f"  {l:12s}  {dom_cnt[l]:3d} steps  ({dom_cnt[l]/N_STEPS*100:.0f}%)")

    print(f"\nMean dominance ambiguity: {mean_amb:.4f}  (0=clear, 1=tied)")
    print(f"Mean blend coefficient:   {mean_bld:.4f}  (0=pure, 1=fully shared)")

    # dwell
    dwell = defaultdict(list)
    cur, start = agent.dominant_log[0], 0
    for t in range(1, N_STEPS):
        if agent.dominant_log[t] != cur:
            dwell[cur].append(t - start)
            cur, start = agent.dominant_log[t], t
    dwell[cur].append(N_STEPS - start)

    print("\nDwell times:")
    for l in LOOP_LIST:
        d = dwell.get(l, [])
        if d:
            print(f"  {l:12s}  n={len(d)}  mean={np.mean(d):.1f}")

    print("\nShared route mean activations:")
    for r in SHARED_ROUTES:
        h = agent.route_hist[r]
        print(f"  {r:10s}  mean={np.mean(h):.3f}  max={max(h):.3f}")

    print("\nActions:")
    for act, cnt in Counter(agent.action_log).most_common():
        print(f"  {act:15s}  {cnt:3d}  ({cnt/N_STEPS*100:.0f}%)")

    visualize(agent)
