#!/usr/bin/env python3
"""
EchoLoop v3 — 複数ループ競合・attractor 切り替えダイナミクス

3 閉回路 (social / exploration / defensive) が相互抑制で dominance を争う。
外界入力は dominance を偏らせるだけ。行動は route state の副産物。

観測現象:
  attractor switching / hysteresis / phase transition /
  metastable states / spontaneous dominance change / loop entropy

設計:
  intra-loop: 前向き励起(+0.55) + 後ろ向き抑制(-0.35) → 自律的 limit cycle
  inter-loop: loop 活性度に比例した相互抑制 → winner-take-all competition
  external:   特定 route への加算だけ (reward は edge weight の微調整のみ)
"""

import random
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── 定数 ─────────────────────────────────────────────────────────────────────
N_STEPS  = 400
W_FWD    = 0.55   # intra-loop 前向き励起
W_BWD    = 0.35   # intra-loop 後ろ向き抑制 (回転波のため)
CROSS    = 0.30   # inter-loop 相互抑制係数
DECAY    = 0.85
S_INHIB  = 0.04   # 自己抑制
A_THRESH = 0.38   # action 発火閾値
NOISE    = 0.025  # 毎 step の微小ランダム注入 (spontaneous 切り替えを促す)

# ── ループ構成 ────────────────────────────────────────────────────────────────
LOOPS = {
    'social':      ['attention', 'engage',   'approach'],
    'exploration': ['curiosity', 'wander',   'inspect'],
    'defensive':   ['alert',     'freeze',   'withdraw'],
}
LOOP_LIST   = list(LOOPS.keys())
ROUTE_NAMES = [r for routes in LOOPS.values() for r in routes]
ROUTE_TO_LOOP = {r: l for l, routes in LOOPS.items() for r in routes}

LOOP_COLORS  = {'social': '#e74c3c', 'exploration': '#9b59b6', 'defensive': '#f39c12'}
ROUTE_COLORS = {
    'attention': '#ff9999', 'engage':  '#e74c3c', 'approach':  '#922b21',
    'curiosity': '#d7bde2', 'wander':  '#9b59b6', 'inspect':   '#6c3483',
    'alert':     '#fde3a7', 'freeze':  '#f39c12', 'withdraw':  '#b7770d',
}

ACTION_MAP = {
    'attention': 'look_at_user',  'engage':   'gesture',
    'approach':  'move_toward',   'curiosity': 'look_around',
    'wander':    'move_to_obj',   'inspect':   'point',
    'alert':     'scan',          'freeze':    'freeze',
    'withdraw':  'step_back',
}

# ── 実験フェーズ ──────────────────────────────────────────────────────────────
# (start, end, label, {route: ext_boost})
# 外界入力: 特定 route への加算 (0.35 で dormant loop を seeding できる)
PHASES = [
    (  0,  70, 'internal',     {}),
    ( 70, 150, 'exp trigger',  {r: 0.35 for r in LOOPS['exploration']}),
    (150, 230, 'decay',        {}),
    (230, 290, 'def trigger',  {r: 0.35 for r in LOOPS['defensive']}),
    (290, 360, 'hysteresis',   {}),
    (360, 400, 'mixed',        {'attention': 0.12, 'curiosity': 0.14, 'alert': 0.08}),
]

PHASE_COLORS = ['#1a1a3a', '#1a2a1a', '#1a1a3a', '#2a1a1a', '#1a1a3a', '#1a2a2a']


# ── RouteEdge ─────────────────────────────────────────────────────────────────
class RouteEdge:
    __slots__ = ('src', 'dst', 'weight', 'flow')

    def __init__(self, src, dst, weight):
        self.src, self.dst = src, dst
        self.weight = float(weight)
        self.flow   = 0.0


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.routes = {n: 0.05 for n in ROUTE_NAMES}
        # social loop を初期 seed として起こす
        for r in LOOPS['social']:
            self.routes[r] = 0.32

        # intra-loop edges: 前向き励起 + 後ろ向き抑制
        self.edges: list[RouteEdge] = []
        for loop_name, routes in LOOPS.items():
            n = len(routes)
            for i in range(n):
                self.edges.append(RouteEdge(routes[i], routes[(i+1) % n], +W_FWD))
                self.edges.append(RouteEdge(routes[i], routes[(i-1) % n], -W_BWD))

        self.action = 'idle'

        # 記録
        self.route_hist    = {n: [] for n in ROUTE_NAMES}
        self.loop_hist     = {l: [] for l in LOOP_LIST}
        self.dominant_log  : list[str]   = []
        self.loop_act_log  : list[dict]  = []
        self.action_log    : list[str]   = []

    # ── loop activity ─────────────────────────────────────────────────────
    def loop_activities(self) -> dict:
        return {l: float(np.mean([self.routes[r] for r in routes]))
                for l, routes in LOOPS.items()}

    # ── 1 step ──────────────────────────────────────────────────────────────
    def update(self, ext: dict, reward: float = 0.0) -> str:
        l_acts = self.loop_activities()

        # 1. intra-loop フロー
        delta = {n: 0.0 for n in ROUTE_NAMES}
        for e in self.edges:
            f = self.routes[e.src] * e.weight
            delta[e.dst] += f
            e.flow = f

        # 2. inter-loop 相互抑制 (competing loop 活性度の総和)
        for loop_name, routes in LOOPS.items():
            competing = sum(l_acts[l] for l in LOOP_LIST if l != loop_name)
            for r in routes:
                delta[r] -= CROSS * competing

        # 3. activation 更新
        for n in ROUTE_NAMES:
            a    = self.routes[n]
            ext_n = ext.get(n, 0.0) + random.uniform(0, NOISE)
            new  = a * DECAY + delta[n] + ext_n - S_INHIB * a * a
            self.routes[n] = float(np.clip(new, 0.0, 1.0))

        # 4. reward → active intra-loop edge を微調整 (行動を選ばない)
        if abs(reward) > 0.01:
            for e in self.edges:
                if abs(e.flow) > 0.03 and e.weight > 0:
                    e.weight = float(np.clip(e.weight + reward * 0.002, 0.10, 0.80))

        # 5. アクション: 最大活性 route を dominant loop 内で読み出す
        l_acts_now = self.loop_activities()
        dominant   = max(l_acts_now, key=l_acts_now.get)
        dom_routes = LOOPS[dominant]
        best_route = max(dom_routes, key=lambda r: self.routes[r])
        self.action = ACTION_MAP[best_route] if self.routes[best_route] > A_THRESH else 'idle'

        # 記録
        for n in ROUTE_NAMES:
            self.route_hist[n].append(self.routes[n])
        for l in LOOP_LIST:
            self.loop_hist[l].append(l_acts_now[l])
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

    ax_loop    = fig.add_subplot(gs[0, :])       # loop activity (full width)
    ax_ps1     = fig.add_subplot(gs[1, 0])       # phase space social×exploration
    ax_ps2     = fig.add_subplot(gs[1, 1])       # phase space exploration×defensive
    ax_raster  = fig.add_subplot(gs[1, 2])       # dominance raster
    ax_entropy = fig.add_subplot(gs[1, 3])       # loop entropy
    ax_trans   = fig.add_subplot(gs[2, 0])       # transition matrix
    ax_dwell   = fig.add_subplot(gs[2, 1])       # dwell times
    ax_routes  = fig.add_subplot(gs[2, 2])       # all 9 route activations
    ax_action  = fig.add_subplot(gs[2, 3])       # action distribution

    _dark = '#16213e'
    for ax in (ax_loop, ax_ps1, ax_ps2, ax_raster, ax_entropy,
               ax_trans, ax_dwell, ax_routes, ax_action):
        ax.set_facecolor(_dark)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a5a')
        ax.tick_params(colors='#667799', labelsize=7)

    T = np.arange(N_STEPS)

    # ── helpers ───────────────────────────────────────────────────────────
    def shade_phases(ax):
        for start, end, label, _ in PHASES:
            mid = (start + end) / 2
            ax.axvline(start, color='#333355', lw=0.5, alpha=0.7)
            ax.text(mid, ax.get_ylim()[1] * 0.97, label,
                    ha='center', va='top', color='#888899', fontsize=6)

    # ── Loop Activity ──────────────────────────────────────────────────────
    ax_loop.set_title('Loop Activity over Time', color='white', fontsize=11, pad=5)
    ax_loop.set_xlim(0, N_STEPS)
    ax_loop.set_ylim(-0.02, 1.08)
    for start, end, _, _ in PHASES:
        ax_loop.axvline(start, color='#333355', lw=0.7, alpha=0.6)
    for _, end, label, _ in PHASES:
        pass  # labeled below

    for l in LOOP_LIST:
        ax_loop.plot(T, agent.loop_hist[l], color=LOOP_COLORS[l],
                     lw=1.5, alpha=0.90, label=l)

    # dominant loop shading
    for t in range(N_STEPS - 1):
        d = agent.dominant_log[t]
        ax_loop.axvspan(t, t+1, alpha=0.06, color=LOOP_COLORS[d])

    ax_loop.axhline(A_THRESH / 3, color='white', alpha=0.15, ls='--', lw=0.8)
    ax_loop.legend(loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=9, framealpha=0.85)
    ax_loop.set_xlabel('Step', color='#667799', fontsize=8)
    ax_loop.set_ylabel('Loop Activity (mean of 3 routes)', color='#667799', fontsize=8)

    # phase labels at top
    for start, end, label, _ in PHASES:
        ax_loop.text((start+end)/2, 1.04, label, ha='center', va='bottom',
                     color='#aaaacc', fontsize=7)

    # ── Phase Spaces ──────────────────────────────────────────────────────
    def plot_phase_space(ax, lx, ly, title):
        xs = [d[lx] for d in agent.loop_act_log]
        ys = [d[ly] for d in agent.loop_act_log]
        n  = len(xs)
        cmap = plt.cm.plasma
        for i in range(n - 1):
            c = cmap(i / n)
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, alpha=0.45, lw=0.8)
        # direction arrows
        for i in range(0, n-1, 30):
            dx, dy = xs[i+1]-xs[i], ys[i+1]-ys[i]
            if abs(dx)+abs(dy) > 0.005:
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle='->', color='white',
                                            alpha=0.30, lw=0.7))
        # attractor markers (expected stable points for each axis pair)
        attractors = {
            ('social', 'exploration'): [(0.90, 0.05, 'social'), (0.05, 0.90, 'exploration')],
            ('exploration', 'defensive'): [(0.90, 0.05, 'exploration'), (0.05, 0.90, 'defensive')],
        }
        for ax_val, ay_val, loop_name in attractors.get((lx, ly), []):
            ax.scatter([ax_val], [ay_val], s=50, c=LOOP_COLORS[loop_name],
                       marker='*', zorder=6, alpha=0.75)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel(lx, color=LOOP_COLORS[lx], fontsize=7)
        ax.set_ylabel(ly, color=LOOP_COLORS[ly], fontsize=7)
        ax.set_title(title, color='white', fontsize=9, pad=4)
        ax.tick_params(labelbottom=False, labelleft=False)
        # colorbar hint
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_STEPS))
        cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('time', color='#667799', fontsize=6)
        cb.ax.tick_params(labelsize=5, colors='#667799')

    plot_phase_space(ax_ps1, 'social', 'exploration', 'social x exploration')
    plot_phase_space(ax_ps2, 'exploration', 'defensive', 'exploration x defensive')

    # ── Dominance Raster ──────────────────────────────────────────────────
    ax_raster.set_title('Dominant Loop per Step', color='white', fontsize=9, pad=4)
    loop_idx = {l: i for i, l in enumerate(LOOP_LIST)}
    raster_data = np.array([loop_idx[d] for d in agent.dominant_log], dtype=float)
    cmap_raster = plt.cm.colors.ListedColormap(
        [LOOP_COLORS[l] for l in LOOP_LIST])
    ax_raster.imshow(raster_data.reshape(1, -1), aspect='auto', cmap=cmap_raster,
                     vmin=0, vmax=2, interpolation='nearest',
                     extent=[0, N_STEPS, 0, 1])
    ax_raster.set_yticks([])
    ax_raster.set_xlabel('Step', color='#667799', fontsize=7)
    for start, _, _, _ in PHASES:
        ax_raster.axvline(start, color='white', lw=0.5, alpha=0.4)
    # legend
    patches = [mpatches.Patch(color=LOOP_COLORS[l], label=l) for l in LOOP_LIST]
    ax_raster.legend(handles=patches, loc='upper right', facecolor='#1a1a2e',
                     labelcolor='white', fontsize=6, framealpha=0.85)

    # ── Entropy ────────────────────────────────────────────────────────────
    ax_entropy.set_title('Loop Dominance Entropy (w=20)', color='white', fontsize=9, pad=4)
    W_ENT = 20
    entropy_vals = []
    for t in range(W_ENT, N_STEPS):
        cnt = Counter(agent.dominant_log[t-W_ENT:t])
        ps  = np.array([cnt.get(l, 0) / W_ENT for l in LOOP_LIST])
        H   = -np.sum(ps * np.log2(ps + 1e-9))
        entropy_vals.append(H)

    t_ent = np.arange(W_ENT, N_STEPS)
    ax_entropy.fill_between(t_ent, entropy_vals, alpha=0.55, color='#3498db')
    ax_entropy.plot(t_ent, entropy_vals, color='#74b9ff', lw=0.9, alpha=0.8)
    ax_entropy.axhline(np.log2(3), color='white', alpha=0.20, ls='--', lw=0.8)
    ax_entropy.text(W_ENT+3, np.log2(3)+0.04, 'max (uniform)', color='white',
                    alpha=0.30, fontsize=6)
    for start, _, _, _ in PHASES:
        if start >= W_ENT:
            ax_entropy.axvline(start, color='#333366', lw=0.6, alpha=0.7)
    ax_entropy.set_xlim(W_ENT, N_STEPS)
    ax_entropy.set_ylim(0, 1.80)
    ax_entropy.set_xlabel('Step', color='#667799', fontsize=7)
    ax_entropy.set_ylabel('H (bits)', color='#667799', fontsize=7)

    # ── Transition Matrix ──────────────────────────────────────────────────
    ax_trans.set_title('Loop Transition Counts', color='white', fontsize=9, pad=4)
    n_l   = len(LOOP_LIST)
    tmat  = np.zeros((n_l, n_l))
    for t in range(1, N_STEPS):
        i = LOOP_LIST.index(agent.dominant_log[t-1])
        j = LOOP_LIST.index(agent.dominant_log[t])
        if i != j:
            tmat[i, j] += 1

    im = ax_trans.imshow(tmat, cmap='plasma', aspect='auto')
    ax_trans.set_xticks(range(n_l)); ax_trans.set_xticklabels(LOOP_LIST, rotation=30, fontsize=7, color='#aaaacc')
    ax_trans.set_yticks(range(n_l)); ax_trans.set_yticklabels(LOOP_LIST, fontsize=7, color='#aaaacc')
    ax_trans.set_xlabel('to', color='#667799', fontsize=7)
    ax_trans.set_ylabel('from', color='#667799', fontsize=7)
    for i in range(n_l):
        for j in range(n_l):
            if tmat[i, j] > 0:
                ax_trans.text(j, i, int(tmat[i, j]), ha='center', va='center',
                              color='white', fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax_trans, fraction=0.046, pad=0.04)

    # ── Dwell Times ────────────────────────────────────────────────────────
    ax_dwell.set_title('Dwell Times per Loop', color='white', fontsize=9, pad=4)
    dwell_data = defaultdict(list)
    cur_loop, cur_start = agent.dominant_log[0], 0
    for t in range(1, N_STEPS):
        if agent.dominant_log[t] != cur_loop:
            dwell_data[cur_loop].append(t - cur_start)
            cur_loop, cur_start = agent.dominant_log[t], t
    dwell_data[cur_loop].append(N_STEPS - cur_start)

    for i, l in enumerate(LOOP_LIST):
        if dwell_data[l]:
            d = dwell_data[l]
            ax_dwell.boxplot(d, positions=[i], widths=0.5, patch_artist=True,
                             boxprops=dict(facecolor=LOOP_COLORS[l], alpha=0.7),
                             medianprops=dict(color='white', lw=1.5),
                             whiskerprops=dict(color='#888888'),
                             capprops=dict(color='#888888'),
                             flierprops=dict(marker='o', color=LOOP_COLORS[l],
                                            alpha=0.3, markersize=4))
            ax_dwell.text(i, max(d) + 2, f'n={len(d)}',
                          ha='center', color=LOOP_COLORS[l], fontsize=6)

    ax_dwell.set_xticks(range(n_l))
    ax_dwell.set_xticklabels(LOOP_LIST, fontsize=7, color='#aaaacc')
    ax_dwell.set_ylabel('Steps', color='#667799', fontsize=7)
    ax_dwell.set_xlim(-0.6, n_l - 0.4)

    # ── Route Activations (全9ルート) ────────────────────────────────────
    ax_routes.set_title('All Route Activations (grouped by loop)', color='white', fontsize=9, pad=4)
    ax_routes.set_xlim(0, N_STEPS)
    ax_routes.set_ylim(-0.02, 1.05)
    for loop_name in LOOP_LIST:
        for r in LOOPS[loop_name]:
            ax_routes.plot(T, agent.route_hist[r],
                           color=ROUTE_COLORS[r], lw=0.7, alpha=0.75, label=r)
    # loop labels
    for loop_name in LOOP_LIST:
        ax_routes.axhline(-0.5, color=LOOP_COLORS[loop_name], lw=0)  # invisible (legend only)
    handles = [mpatches.Patch(color=LOOP_COLORS[l], label=l) for l in LOOP_LIST]
    ax_routes.legend(handles=handles, loc='upper right', facecolor='#1a1a2e',
                     labelcolor='white', fontsize=6, framealpha=0.85)
    for start, _, _, _ in PHASES:
        ax_routes.axvline(start, color='#333366', lw=0.5, alpha=0.6)
    ax_routes.set_xlabel('Step', color='#667799', fontsize=7)
    ax_routes.set_ylabel('Activation', color='#667799', fontsize=7)

    # ── Action Distribution ────────────────────────────────────────────────
    ax_action.set_title('Action Distribution', color='white', fontsize=9, pad=4)
    counts = Counter(agent.action_log)
    total  = len(agent.action_log)
    acts   = [a for a, _ in counts.most_common()]
    vals   = [counts[a] / total for a in acts]
    colors = [LOOP_COLORS.get(ROUTE_TO_LOOP.get(
                  next((r for r, v in ACTION_MAP.items() if v == a), ''), ''), '#888888')
              for a in acts]
    bars = ax_action.barh(acts, vals, color=colors, alpha=0.85, height=0.6)
    ax_action.set_xlim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax_action.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{v*100:.0f}%', va='center', color='white', fontsize=7)
    ax_action.set_xlabel('Fraction', color='#667799', fontsize=7)

    # ── footer ────────────────────────────────────────────────────────────
    dom_cnt   = Counter(agent.dominant_log)
    total_sw  = sum(1 for t in range(1, N_STEPS)
                    if agent.dominant_log[t] != agent.dominant_log[t-1])
    ent_mean  = float(np.mean(entropy_vals))
    fig.text(0.50, 0.012,
             f'Switches: {total_sw}  |  '
             f'Dominant distribution: ' +
             ' / '.join(f'{l}: {dom_cnt[l]/N_STEPS*100:.0f}%' for l in LOOP_LIST) +
             f'  |  Mean entropy: {ent_mean:.3f} bits',
             color='#667799', fontsize=7, ha='center')

    fig.suptitle('EchoLoop v3 — Competing Loop Dynamics  '
                 '(social / exploration / defensive)',
                 color='white', fontsize=13, y=0.98)

    out = 'echoloop_v3_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved -> {out}")
    plt.show()


# ── エントリポイント ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("EchoLoop v3: running ...")
    agent = simulate()

    dom_cnt  = Counter(agent.dominant_log)
    total_sw = sum(1 for t in range(1, N_STEPS)
                   if agent.dominant_log[t] != agent.dominant_log[t-1])

    print("\nLoop dominance distribution:")
    for l in LOOP_LIST:
        print(f"  {l:12s}  {dom_cnt[l]:3d} steps  ({dom_cnt[l]/N_STEPS*100:.0f}%)")

    print(f"\nTotal dominance switches: {total_sw}")

    # dwell times
    dwell = defaultdict(list)
    cur, start = agent.dominant_log[0], 0
    for t in range(1, N_STEPS):
        if agent.dominant_log[t] != cur:
            dwell[cur].append(t - start)
            cur, start = agent.dominant_log[t], t
    dwell[cur].append(N_STEPS - start)

    print("\nDwell times (steps):")
    for l in LOOP_LIST:
        if dwell[l]:
            d = dwell[l]
            print(f"  {l:12s}  n={len(d):2d}  mean={np.mean(d):.1f}  "
                  f"min={min(d)}  max={max(d)}")

    print("\nFinal loop states:")
    for l, routes in LOOPS.items():
        acts = [agent.routes[r] for r in routes]
        print(f"  {l:12s}  " +
              " ".join(f"{r}={v:.3f}" for r, v in zip(routes, acts)))

    print("\nAction distribution:")
    for act, cnt in Counter(agent.action_log).most_common():
        print(f"  {act:15s}  {cnt:3d}  ({cnt/N_STEPS*100:.0f}%)")

    visualize(agent)
