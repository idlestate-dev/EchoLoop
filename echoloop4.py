#!/usr/bin/env python3
"""
EchoLoop v4 — fatigue/adaptation による spontaneous switching

状態履歴が dynamics を変える:
  dominant loop は疲労 → effective gain 低下 → 自然崩壊 → 次の loop へ
  非支配 loop はゆっくり回復 → 次の dominance 候補になる

観測現象: spontaneous switching / mood-like oscillation / metastable wandering /
          hysteresis / stuck state / recovery

設計メモ:
  fatigue=0.80 で |λ_rot| < 1 (ループが自律維持できなくなる境界)
  ドウェル推定 ~80 steps / 回復 ~320 steps
"""

import random
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── 定数 ─────────────────────────────────────────────────────────────────────
N_STEPS        = 800
W_FWD          = 0.55
W_BWD          = 0.35
CROSS          = 0.30
DECAY          = 0.85
S_INHIB        = 0.04
NOISE          = 0.020
A_THRESH       = 0.38
FATIGUE_RATE   = 0.015   # dominant loop が 1 step で蓄積する疲労量 (×loop_activity)
FATIGUE_DECAY  = 0.995   # 毎 step の疲労回復率
MIN_GAIN       = 0.05    # 最大疲労時でも残す最小 forward weight 係数

LOOPS = {
    'social':      ['attention', 'engage',   'approach'],
    'exploration': ['curiosity', 'wander',   'inspect'],
    'defensive':   ['alert',     'freeze',   'withdraw'],
}
LOOP_LIST    = list(LOOPS.keys())
ROUTE_NAMES  = [r for routes in LOOPS.values() for r in routes]
ROUTE_TO_LOOP = {r: l for l, routes in LOOPS.items() for r in routes}

LOOP_COLORS  = {'social': '#e74c3c', 'exploration': '#9b59b6', 'defensive': '#f39c12'}
ROUTE_COLORS = {
    'attention': '#ff9999', 'engage':   '#e74c3c', 'approach':  '#922b21',
    'curiosity': '#d7bde2', 'wander':   '#9b59b6', 'inspect':   '#6c3483',
    'alert':     '#fde3a7', 'freeze':   '#f39c12', 'withdraw':  '#b7770d',
}
ACTION_MAP = {
    'attention': 'look_at_user', 'engage':   'gesture',    'approach':  'move_toward',
    'curiosity': 'look_around',  'wander':   'move_to_obj', 'inspect':   'point',
    'alert':     'scan',         'freeze':   'freeze',      'withdraw':  'step_back',
}

# 外界入力フェーズ (start, end, label, {route: boost})
PHASES = [
    (  0, 120, 'internal-1', {}),
    (120, 160, 'exp trigger',{r: 0.22 for r in LOOPS['exploration']}),
    (160, 450, 'internal-2', {}),
    (450, 490, 'def trigger',{r: 0.22 for r in LOOPS['defensive']}),
    (490, 700, 'internal-3', {}),
    (700, 800, 'mixed',      {'attention': 0.08, 'curiosity': 0.10, 'alert': 0.06}),
]
PHASE_BG = ['#1a1a2e', '#1a2a1a', '#1a1a2e', '#2a1a1a', '#1a1a2e', '#1a2020']


# ── RouteEdge ─────────────────────────────────────────────────────────────────
class RouteEdge:
    __slots__ = ('src', 'dst', 'weight', 'flow', 'is_forward')

    def __init__(self, src, dst, weight):
        self.src, self.dst = src, dst
        self.weight     = float(weight)
        self.flow       = 0.0
        self.is_forward = (weight > 0)


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.routes  = {n: 0.05 for n in ROUTE_NAMES}
        self.fatigue = {l: 0.0  for l in LOOP_LIST}

        # social loop を初期 seed
        for r in LOOPS['social']:
            self.routes[r] = 0.32

        # intra-loop edges (前向き + 後ろ向き)
        self.edges: list[RouteEdge] = []
        for loop_name, routes in LOOPS.items():
            n = len(routes)
            for i in range(n):
                self.edges.append(RouteEdge(routes[i], routes[(i+1) % n], +W_FWD))
                self.edges.append(RouteEdge(routes[i], routes[(i-1) % n], -W_BWD))

        self.action = 'idle'

        # 記録
        self.route_hist   = {n: [] for n in ROUTE_NAMES}
        self.loop_hist    = {l: [] for l in LOOP_LIST}
        self.fatigue_hist = {l: [] for l in LOOP_LIST}
        self.gain_hist    = {l: [] for l in LOOP_LIST}  # effective rotating gain
        self.dominant_log : list[str] = []
        self.loop_act_log : list[dict] = []
        self.action_log   : list[str] = []

    # ── helpers ──────────────────────────────────────────────────────────────
    def loop_activities(self) -> dict:
        return {l: float(np.mean([self.routes[r] for r in routes]))
                for l, routes in LOOPS.items()}

    def rotating_gain(self, loop_name: str) -> float:
        """|λ_rot| - 1 : 正なら自律維持、負なら崩壊中"""
        eff_fwd = W_FWD * max(MIN_GAIN, 1.0 - self.fatigue[loop_name])
        return float(np.sqrt(DECAY**2 + (eff_fwd + W_BWD)**2) - 1.0)

    # ── 1 step ──────────────────────────────────────────────────────────────
    def update(self, ext: dict, reward: float = 0.0) -> str:
        l_acts   = self.loop_activities()
        dominant = max(l_acts, key=l_acts.get)

        # 1. 疲労更新: 活性度に比例して蓄積、全ループが decay
        for l in LOOP_LIST:
            self.fatigue[l] = min(1.0,
                self.fatigue[l] + FATIGUE_RATE * l_acts[l])
            self.fatigue[l] *= FATIGUE_DECAY

        # 2. intra-loop フロー (前向き edge に疲労を適用)
        delta = {n: 0.0 for n in ROUTE_NAMES}
        for e in self.edges:
            loop = ROUTE_TO_LOOP[e.src]
            if e.is_forward:
                eff_w = e.weight * max(MIN_GAIN, 1.0 - self.fatigue[loop])
            else:
                eff_w = e.weight   # 後ろ向き抑制は疲労に影響されない
            f = self.routes[e.src] * eff_w
            delta[e.dst] += f
            e.flow = f

        # 3. inter-loop 相互抑制
        for loop_name, routes in LOOPS.items():
            competing = sum(l_acts[l] for l in LOOP_LIST if l != loop_name)
            for r in routes:
                delta[r] -= CROSS * competing

        # 4. activation 更新
        for n in ROUTE_NAMES:
            a    = self.routes[n]
            ext_n = ext.get(n, 0.0) + random.uniform(0, NOISE)
            new  = a * DECAY + delta[n] + ext_n - S_INHIB * a * a
            self.routes[n] = float(np.clip(new, 0.0, 1.0))

        # 5. reward → forward edge を微調整 (主役にはしない)
        if abs(reward) > 0.01:
            for e in self.edges:
                if e.is_forward and abs(e.flow) > 0.03:
                    e.weight = float(np.clip(e.weight + reward * 0.002, 0.10, 0.80))

        # 6. アクション発火
        l_acts_now = self.loop_activities()
        dominant   = max(l_acts_now, key=l_acts_now.get)
        dom_routes = LOOPS[dominant]
        best_route = max(dom_routes, key=lambda r: self.routes[r])
        self.action = ACTION_MAP[best_route] if self.routes[best_route] > A_THRESH else 'idle'

        # 7. 記録
        for n in ROUTE_NAMES:
            self.route_hist[n].append(self.routes[n])
        for l in LOOP_LIST:
            self.loop_hist[l].append(l_acts_now[l])
            self.fatigue_hist[l].append(self.fatigue[l])
            self.gain_hist[l].append(self.rotating_gain(l))
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

    ax_main  = fig.add_subplot(gs[0, :])      # loop activity + fatigue
    ax_gain  = fig.add_subplot(gs[1, 0])      # effective gain per loop
    ax_rast  = fig.add_subplot(gs[1, 1])      # dominance raster
    ax_dwell = fig.add_subplot(gs[1, 2])      # dwell distribution
    ax_ent   = fig.add_subplot(gs[1, 3])      # entropy
    ax_ps1   = fig.add_subplot(gs[2, 0])      # phase space soc×exp (fatigue bubbles)
    ax_ps2   = fig.add_subplot(gs[2, 1])      # phase space exp×def
    ax_rout  = fig.add_subplot(gs[2, 2])      # all 9 route activations
    ax_act   = fig.add_subplot(gs[2, 3])      # action distribution

    _dark = '#16213e'
    for ax in (ax_main, ax_gain, ax_rast, ax_dwell, ax_ent,
               ax_ps1, ax_ps2, ax_rout, ax_act):
        ax.set_facecolor(_dark)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a5a')
        ax.tick_params(colors='#667799', labelsize=7)

    T = np.arange(N_STEPS)

    def shade_phases(ax, ymax=1.0):
        for i, (start, end, label, _) in enumerate(PHASES):
            ax.axvline(start, color='#333355', lw=0.5, alpha=0.7)
            ax.text((start+end)/2, ymax*0.97, label,
                    ha='center', va='top', color='#8888aa', fontsize=6)

    # ── Main: Loop Activity + Fatigue ──────────────────────────────────────
    ax_main.set_title(
        'Loop Activity (solid) and Fatigue (dashed, right axis)',
        color='white', fontsize=10, pad=5)
    ax_main.set_xlim(0, N_STEPS)
    ax_main.set_ylim(-0.02, 1.08)

    # dominant loop background tint
    for t in range(N_STEPS - 1):
        ax_main.axvspan(t, t+1, alpha=0.05, color=LOOP_COLORS[agent.dominant_log[t]])

    for l in LOOP_LIST:
        ax_main.plot(T, agent.loop_hist[l], color=LOOP_COLORS[l],
                     lw=1.5, alpha=0.90, label=l)

    # fatigue on twin axis
    ax_fat = ax_main.twinx()
    ax_fat.set_facecolor('none')
    ax_fat.set_ylim(0, 1.1)
    ax_fat.tick_params(colors='#667799', labelsize=6)
    ax_fat.set_ylabel('Fatigue', color='#667799', fontsize=7)
    for l in LOOP_LIST:
        ax_fat.plot(T, agent.fatigue_hist[l], color=LOOP_COLORS[l],
                    lw=0.8, alpha=0.50, ls='--')
    ax_fat.axhline(0.80, color='white', alpha=0.15, ls=':', lw=0.8)
    ax_fat.text(5, 0.82, 'fatigue threshold (0.80)', color='white', alpha=0.25, fontsize=6)

    shade_phases(ax_main)
    ax_main.legend(loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=8, framealpha=0.85)
    ax_main.set_xlabel('Step', color='#667799', fontsize=8)
    ax_main.set_ylabel('Loop Activity', color='#667799', fontsize=8)

    # ── Effective Gain ────────────────────────────────────────────────────
    ax_gain.set_title('Effective Rotating Gain (|λ|−1)', color='white', fontsize=9, pad=4)
    ax_gain.set_xlim(0, N_STEPS)
    ax_gain.axhline(0, color='white', alpha=0.30, lw=1.0)
    ax_gain.text(5, 0.02, 'self-sustaining', color='white', alpha=0.25, fontsize=6)
    ax_gain.text(5, -0.04, 'decaying', color='white', alpha=0.25, fontsize=6)
    for l in LOOP_LIST:
        ax_gain.plot(T, agent.gain_hist[l], color=LOOP_COLORS[l],
                     lw=1.0, alpha=0.85, label=l)
    shade_phases(ax_gain, ymax=0.25)
    ax_gain.set_xlabel('Step', color='#667799', fontsize=7)
    ax_gain.set_ylabel('|λ_rot| − 1', color='#667799', fontsize=7)
    ax_gain.legend(loc='lower right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=6, framealpha=0.85)

    # ── Dominance Raster ──────────────────────────────────────────────────
    ax_rast.set_title('Dominant Loop per Step', color='white', fontsize=9, pad=4)
    loop_idx  = {l: i for i, l in enumerate(LOOP_LIST)}
    rdata = np.array([loop_idx[d] for d in agent.dominant_log], dtype=float)
    cmap_r = plt.matplotlib.colors.ListedColormap(
        [LOOP_COLORS[l] for l in LOOP_LIST])
    ax_rast.imshow(rdata.reshape(1, -1), aspect='auto', cmap=cmap_r,
                   vmin=0, vmax=2, interpolation='nearest',
                   extent=[0, N_STEPS, 0, 1])
    ax_rast.set_yticks([])
    ax_rast.set_xlabel('Step', color='#667799', fontsize=7)
    for start, _, _, _ in PHASES:
        ax_rast.axvline(start, color='white', lw=0.4, alpha=0.4)
    # switch events
    for t in range(1, N_STEPS):
        if agent.dominant_log[t] != agent.dominant_log[t-1]:
            ax_rast.axvline(t, color='white', lw=1.2, alpha=0.6)
    patches = [mpatches.Patch(color=LOOP_COLORS[l], label=l) for l in LOOP_LIST]
    ax_rast.legend(handles=patches, loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=6, framealpha=0.85)

    # ── Dwell Distribution ────────────────────────────────────────────────
    ax_dwell.set_title('Dwell Time Distribution', color='white', fontsize=9, pad=4)
    dwell_data = defaultdict(list)
    cur, start = agent.dominant_log[0], 0
    for t in range(1, N_STEPS):
        if agent.dominant_log[t] != cur:
            dwell_data[cur].append(t - start)
            cur, start = agent.dominant_log[t], t
    dwell_data[cur].append(N_STEPS - start)

    # stacked histogram
    bins = np.arange(0, max(max(v) for v in dwell_data.values()) + 20, 15)
    for l in LOOP_LIST:
        if dwell_data[l]:
            ax_dwell.hist(dwell_data[l], bins=bins, alpha=0.65,
                          color=LOOP_COLORS[l], label=l, histtype='stepfilled')
    ax_dwell.axvline(80, color='white', alpha=0.25, ls='--', lw=0.8)
    ax_dwell.text(82, ax_dwell.get_ylim()[1]*0.9 if ax_dwell.get_ylim()[1] > 0 else 1,
                  '~80 est.', color='white', alpha=0.30, fontsize=6)
    ax_dwell.set_xlabel('Dwell (steps)', color='#667799', fontsize=7)
    ax_dwell.set_ylabel('Count', color='#667799', fontsize=7)
    ax_dwell.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=6, framealpha=0.85)

    # ── Entropy ────────────────────────────────────────────────────────────
    ax_ent.set_title('Dominance Entropy (w=30)', color='white', fontsize=9, pad=4)
    W_ENT = 30
    ent_vals = []
    for t in range(W_ENT, N_STEPS):
        cnt = Counter(agent.dominant_log[t-W_ENT:t])
        ps  = np.array([cnt.get(l, 0) / W_ENT for l in LOOP_LIST])
        ent_vals.append(-np.sum(ps * np.log2(ps + 1e-9)))

    t_ent = np.arange(W_ENT, N_STEPS)
    ax_ent.fill_between(t_ent, ent_vals, alpha=0.55, color='#74b9ff')
    ax_ent.plot(t_ent, ent_vals, color='#74b9ff', lw=0.9, alpha=0.8)
    ax_ent.axhline(np.log2(3), color='white', alpha=0.18, ls='--', lw=0.7)
    ax_ent.text(W_ENT+3, np.log2(3)+0.06, 'max', color='white', alpha=0.25, fontsize=6)
    for start, _, _, _ in PHASES:
        if start >= W_ENT:
            ax_ent.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_ent.set_xlim(W_ENT, N_STEPS)
    ax_ent.set_ylim(0, 1.8)
    ax_ent.set_xlabel('Step', color='#667799', fontsize=7)
    ax_ent.set_ylabel('H (bits)', color='#667799', fontsize=7)

    # ── Phase Spaces (fatigue encoded as bubble size) ─────────────────────
    def plot_phase(ax, lx, ly, title):
        xs = [d[lx] for d in agent.loop_act_log]
        ys = [d[ly] for d in agent.loop_act_log]
        n  = len(xs)
        # bubble size = fatigue of dominant loop at each step
        dom_fatigue = [agent.fatigue_hist[agent.dominant_log[t]][t] for t in range(n)]
        sizes = [4 + f * 18 for f in dom_fatigue]
        colors = [LOOP_COLORS[agent.dominant_log[t]] for t in range(n)]

        # trajectory (time-colored line, thin)
        for i in range(n-1):
            c = plt.cm.plasma(i / n)
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, alpha=0.25, lw=0.6)

        # bubbles (every 3 steps)
        for i in range(0, n, 3):
            ax.scatter(xs[i], ys[i], s=sizes[i], c=[colors[i]], alpha=0.50, zorder=3)

        # attractor markers
        attr_map = {
            ('social', 'exploration'): [
                (0.90, 0.05, 'social'), (0.05, 0.90, 'exploration')],
            ('exploration', 'defensive'): [
                (0.90, 0.05, 'exploration'), (0.05, 0.90, 'defensive')],
        }
        for ax_val, ay_val, l in attr_map.get((lx, ly), []):
            ax.scatter([ax_val], [ay_val], s=60, c=LOOP_COLORS[l],
                       marker='*', zorder=6, alpha=0.7)

        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel(lx, color=LOOP_COLORS[lx], fontsize=7)
        ax.set_ylabel(ly, color=LOOP_COLORS[ly], fontsize=7)
        ax.set_title(title + '\n(bubble size = fatigue)', color='white', fontsize=8, pad=3)
        ax.tick_params(labelbottom=False, labelleft=False)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, N_STEPS))
        cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('time', color='#667799', fontsize=5)
        cb.ax.tick_params(labelsize=5, colors='#667799')

    plot_phase(ax_ps1, 'social', 'exploration', 'social × exploration')
    plot_phase(ax_ps2, 'exploration', 'defensive', 'exploration × defensive')

    # ── All Route Activations ─────────────────────────────────────────────
    ax_rout.set_title('Route Activations (grouped)', color='white', fontsize=9, pad=4)
    ax_rout.set_xlim(0, N_STEPS)
    ax_rout.set_ylim(-0.02, 1.05)
    for l in LOOP_LIST:
        for r in LOOPS[l]:
            ax_rout.plot(T, agent.route_hist[r],
                         color=ROUTE_COLORS[r], lw=0.65, alpha=0.70, label=r)
    handles = [mpatches.Patch(color=LOOP_COLORS[l], label=l) for l in LOOP_LIST]
    ax_rout.legend(handles=handles, loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=6, framealpha=0.85)
    for start, _, _, _ in PHASES:
        ax_rout.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_rout.set_xlabel('Step', color='#667799', fontsize=7)
    ax_rout.set_ylabel('Activation', color='#667799', fontsize=7)

    # ── Action Distribution ────────────────────────────────────────────────
    ax_act.set_title('Action Distribution', color='white', fontsize=9, pad=4)
    counts = Counter(agent.action_log)
    total  = len(agent.action_log)
    acts   = [a for a, _ in counts.most_common()]
    vals   = [counts[a] / total for a in acts]
    def _act_color(a):
        for r, v in ACTION_MAP.items():
            if v == a:
                return LOOP_COLORS.get(ROUTE_TO_LOOP.get(r, ''), '#888888')
        return '#888888'
    clrs = [_act_color(a) for a in acts]
    bars = ax_act.barh(acts, vals, color=clrs, alpha=0.85, height=0.6)
    ax_act.set_xlim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax_act.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v*100:.0f}%', va='center', color='white', fontsize=7)

    # ── footer ────────────────────────────────────────────────────────────
    dom_cnt  = Counter(agent.dominant_log)
    sw_total = sum(1 for t in range(1, N_STEPS)
                   if agent.dominant_log[t] != agent.dominant_log[t-1])
    fig.text(0.50, 0.012,
             f'Switches: {sw_total}  |  Dominance: ' +
             ' / '.join(f'{l}: {dom_cnt[l]/N_STEPS*100:.0f}%' for l in LOOP_LIST),
             color='#667799', fontsize=7, ha='center')

    fig.suptitle('EchoLoop v4 — Fatigue-Driven Spontaneous Switching  '
                 '(mood-like dynamics)',
                 color='white', fontsize=13, y=0.98)

    out = 'echoloop_v4_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved -> {out}")
    plt.show()


# ── エントリポイント ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("EchoLoop v4: running ...")
    agent = simulate()

    dom_cnt  = Counter(agent.dominant_log)
    sw_total = sum(1 for t in range(1, N_STEPS)
                   if agent.dominant_log[t] != agent.dominant_log[t-1])

    print(f"\nTotal switches: {sw_total}")
    print("Loop dominance:")
    for l in LOOP_LIST:
        print(f"  {l:12s}  {dom_cnt[l]:3d} steps  ({dom_cnt[l]/N_STEPS*100:.0f}%)")

    # dwell times
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
            print(f"  {l:12s}  n={len(d):2d}  mean={np.mean(d):.1f}  "
                  f"min={min(d)}  max={max(d)}")

    print("\nFinal fatigue:")
    for l in LOOP_LIST:
        print(f"  {l:12s}  {agent.fatigue[l]:.3f}")

    print("\nActions:")
    for act, cnt in Counter(agent.action_log).most_common():
        print(f"  {act:15s}  {cnt:3d}  ({cnt/N_STEPS*100:.0f}%)")

    visualize(agent)
