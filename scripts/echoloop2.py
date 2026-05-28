#!/usr/bin/env python3
"""
EchoLoop v2 — 内部循環ダイナミクスが主役

仮説: 知能 = 継続的状態循環
外界入力は循環を歪めるだけ。行動は route state の読み出しとして emergent に出る。

設計のキー:
  前向き励起 (att→obs→cur→app→att) + 後ろ向き抑制 (obs→att, cur→obs, ...)
  → 回転波の固有値が同期モードより大きい → limit cycle が自発的に出現

実験構成:
  Phase 1 [  0–120]: 通常の外界入力あり
  Phase 2 [120–220]: blackout (外界入力ゼロ) — 内部循環の自律性を確認
  Phase 3 [220–400]: curiosity 寄り文脈 (attractor が切り替わるか)
"""

import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WORLD_SIZE     = 10.0
N_STEPS        = 400
ACTION_THRESH  = 0.35
DECAY          = 0.85
SELF_INHIB     = 0.05   # 小さめ (回転は backward inhibition が担う)

BLACKOUT_START = 120
BLACKOUT_END   = 220
PHASE3_START   = BLACKOUT_END

ROUTE_NAMES  = ['attention', 'observe', 'curiosity', 'approach', 'idle']
ROUTE_COLORS = {
    'attention': '#e74c3c',
    'observe':   '#3498db',
    'curiosity': '#9b59b6',
    'approach':  '#2ecc71',
    'idle':      '#95a5a6',
}

# ── 閉ループ ──────────────────────────────────────────────────────────────────
# 前向き励起 (+) と後ろ向き抑制 (−) の組み合わせで回転波を生む
# 回転固有値の大きさ = sqrt(DECAY^2 + (w_f + w_b)^2)
#   DECAY=0.85, w_f=0.55, w_b=0.35 → |λ_rot| = sqrt(0.72 + 0.81) ≈ 1.24
# 同期固有値 = DECAY + w_f − w_b = 0.85 + 0.55 − 0.35 = 1.05
# → 回転波が支配的になる
LOOP_EDGES = [
    ('attention', 'observe',   +0.55),
    ('observe',   'curiosity', +0.55),
    ('curiosity', 'approach',  +0.55),
    ('approach',  'attention', +0.55),
    # 後ろ向き抑制 → 活性化済みルートが前者を抑え、次へバトンを渡す
    ('observe',   'attention', -0.35),
    ('curiosity', 'observe',   -0.35),
    ('approach',  'curiosity', -0.35),
    ('attention', 'approach',  -0.35),
]

# 小さなクロス接続
CROSS_EDGES = [
    ('idle', 'attention', 0.10),  # idle がループを seed する
    ('idle', 'curiosity', 0.10),
    ('approach', 'idle', -0.25),
    ('observe',  'idle', -0.18),
]

ALL_EDGES = LOOP_EDGES + CROSS_EDGES

ACTION_MAP = {
    'attention': 'look_at_user',
    'observe':   'look_at_object',
    'curiosity': 'point',
    'approach':  'move_toward',
    'idle':      'idle',
}

ACTION_COLORS = {
    'look_at_user':   '#e74c3c',
    'look_at_object': '#3498db',
    'move_toward':    '#2ecc71',
    'point':          '#9b59b6',
    'idle':           '#555577',
}


# ── Vec2 ──────────────────────────────────────────────────────────────────────
class Vec2:
    __slots__ = ('x', 'y')
    def __init__(self, x, y): self.x, self.y = float(x), float(y)
    def dist(self, o): return float(np.hypot(self.x - o.x, self.y - o.y))
    def arr(self): return np.array([self.x, self.y])


class WorldObject:
    def __init__(self, x, y, oid):
        self.pos = Vec2(x, y)
        self.id = oid
        self.interest = random.uniform(0.6, 1.4)


class User:
    def __init__(self, x, y):
        self.pos      = Vec2(x, y)
        self.gaze     = np.array([1.0, 0.0])
        self.reaction = 0.0

    def look_at(self, t: Vec2):
        d = t.arr() - self.pos.arr()
        n = np.linalg.norm(d)
        if n > 1e-6: self.gaze = d / n

    def step_toward(self, t: Vec2, speed: float = 0.12):
        d = t.arr() - self.pos.arr()
        n = np.linalg.norm(d)
        if n > speed:
            v = d / n * speed
            self.pos.x += v[0]; self.pos.y += v[1]

    def gaze_align(self, t: Vec2) -> float:
        d = t.arr() - self.pos.arr()
        n = np.linalg.norm(d)
        if n < 1e-6: return 0.0
        return float(np.clip(np.dot(self.gaze, d / n), 0.0, 1.0))


class RouteEdge:
    __slots__ = ('src', 'dst', 'weight', 'flow')
    def __init__(self, src, dst, weight):
        self.src    = src
        self.dst    = dst
        self.weight = float(weight)
        self.flow   = 0.0


class Agent:
    def __init__(self, x, y):
        self.pos    = Vec2(x, y)
        self.routes = {n: 0.0 for n in ROUTE_NAMES}
        self.routes['idle']      = 0.40
        self.routes['attention'] = 0.35   # ループ開始 seed

        self.edges = [RouteEdge(s, d, w) for s, d, w in ALL_EDGES]

        self.action = 'idle'
        self.history         = {n: [] for n in ROUTE_NAMES}
        self.action_log      : list[str]  = []
        self.flow_log        : list[dict] = []
        self.edge_weight_log : list[dict] = []
        self.above_thresh    : list[bool] = []
        self.dominant_log    : list[str]  = []

    def update(self, user: User, objects: list, reward: float = 0.0,
               phase: int = 0) -> str:

        # ── 1. 外界入力 (phase 1 = blackout はゼロ) ──────────────────────
        if phase == 1:
            ext = {'idle': 0.03}
        elif phase == 0:
            udist = self.pos.dist(user.pos)
            ext = {
                'attention': user.gaze_align(self.pos) * np.exp(-udist / 5.0) * 0.12,
                'curiosity': float(np.clip(
                    sum(o.interest * np.exp(-self.pos.dist(o.pos) / 4.0) for o in objects)
                    / max(1, len(objects)), 0.0, 1.0)) * 0.08,
                'approach':  float(np.clip(1.0 - udist / 8.0, 0.0, 1.0))
                             * (0.5 + user.reaction * 0.5) * 0.07,
                'idle': 0.03,
            }
        else:  # phase 3: curiosity 寄りに文脈切り替え
            udist = self.pos.dist(user.pos)
            ext = {
                'attention': user.gaze_align(self.pos) * np.exp(-udist / 5.0) * 0.04,
                'curiosity': float(np.clip(
                    sum(o.interest * np.exp(-self.pos.dist(o.pos) / 3.5) for o in objects)
                    / max(1, len(objects)), 0.0, 1.0)) * 0.18,
                'observe':   float(np.clip(
                    sum(o.interest * np.exp(-self.pos.dist(o.pos) / 5.0) for o in objects)
                    / max(1, len(objects)), 0.0, 1.0)) * 0.10,
                'idle': 0.03,
            }

        # ── 2. route 間フロー ────────────────────────────────────────────
        delta = {n: 0.0 for n in ROUTE_NAMES}
        flows = {}
        for e in self.edges:
            flow = self.routes[e.src] * e.weight
            delta[e.dst] += flow
            e.flow = flow
            flows[(e.src, e.dst)] = flow

        # ── 3. activation 更新 (decay + flow + ext − 自己抑制) ──────────
        for n in ROUTE_NAMES:
            act     = self.routes[n]
            new_act = act * DECAY + delta[n] + ext.get(n, 0.0) - SELF_INHIB * act * act
            self.routes[n] = float(np.clip(new_act, 0.0, 1.0))

        # ── 4. reward → ループエッジ weight を微調整 (行動を直接選ばない) ─
        if abs(reward) > 0.01:
            for e in self.edges:
                if abs(e.flow) > 0.03 and e.weight > 0.0:
                    e.weight = float(np.clip(e.weight + reward * 0.002, 0.05, 0.80))

        # ── 5. アクション発火 (route activation の読み出し) ────────────
        best = max(ROUTE_NAMES, key=lambda n: self.routes[n])
        self.action = ACTION_MAP[best] if self.routes[best] > ACTION_THRESH else 'idle'

        # ── 6. 物理移動 ──────────────────────────────────────────────────
        if self.action == 'move_toward':
            d = user.pos.arr() - self.pos.arr()
            n = np.linalg.norm(d)
            if n > 1.2:
                v = d / n * 0.10
                self.pos.x += v[0]; self.pos.y += v[1]
            else:
                user.reaction = min(user.reaction, -0.15)

        # ── 記録 ──────────────────────────────────────────────────────────
        for name in ROUTE_NAMES:
            self.history[name].append(self.routes[name])
        self.action_log.append(self.action)
        self.flow_log.append(flows.copy())
        self.edge_weight_log.append(
            {(e.src, e.dst): e.weight for e in self.edges}
        )
        self.above_thresh.append(
            any(self.routes[n] > ACTION_THRESH for n in ROUTE_NAMES[:-1])
        )
        self.dominant_log.append(best)
        return self.action


# ── シミュレーション ──────────────────────────────────────────────────────────
def simulate():
    objects = [WorldObject(random.uniform(2.0, 8.0), random.uniform(2.0, 8.0), i)
               for i in range(4)]
    user  = User(2.0, 5.0)
    agent = Agent(7.0, 5.0)

    waypoints = [Vec2(8.5, 2.0), Vec2(5.0, 8.5), Vec2(1.5, 2.0), Vec2(5.0, 5.0)]
    wp_i      = 0
    traj_u, traj_a = [], []

    GAZE_CYCLE = 22

    for step in range(N_STEPS):
        user.step_toward(waypoints[wp_i])
        if user.pos.dist(waypoints[wp_i]) < 0.3:
            wp_i = (wp_i + 1) % len(waypoints)

        phase_f = (step % GAZE_CYCLE) / GAZE_CYCLE
        if phase_f < 0.30:
            user.look_at(agent.pos); user.reaction = 0.45
        elif phase_f < 0.55:
            nearest = min(objects, key=lambda o: user.pos.dist(o.pos))
            user.look_at(nearest.pos); user.reaction = 0.0
        elif phase_f < 0.75:
            user.gaze = np.array([np.cos(step * 0.17), np.sin(step * 0.17)])
            user.reaction = -0.08
        else:
            user.look_at(agent.pos); user.reaction = 0.30

        sim_phase = (1 if BLACKOUT_START <= step < BLACKOUT_END
                     else 2 if step >= PHASE3_START
                     else 0)

        reward = 0.0
        if agent.action == 'look_at_user' and user.gaze_align(agent.pos) > 0.7:
            reward = 0.20
        elif agent.action == 'move_toward':
            reward = 0.15 if agent.pos.dist(user.pos) < 4.0 else -0.10

        agent.update(user, objects, reward=reward, phase=sim_phase)

        traj_u.append((user.pos.x, user.pos.y))
        traj_a.append((agent.pos.x, agent.pos.y))

    return agent, objects, traj_u, traj_a


# ── 可視化 ────────────────────────────────────────────────────────────────────
def visualize(agent: Agent, objects: list, traj_u: list, traj_a: list):
    fig = plt.figure(figsize=(20, 13), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    ax_acts    = fig.add_subplot(gs[0, :])
    ax_phase1  = fig.add_subplot(gs[1, 0])
    ax_phase2  = fig.add_subplot(gs[1, 1])
    ax_phase3  = fig.add_subplot(gs[1, 2])
    ax_flow    = fig.add_subplot(gs[1, 3])
    ax_persist = fig.add_subplot(gs[2, 0])
    ax_edges   = fig.add_subplot(gs[2, 1])
    ax_world   = fig.add_subplot(gs[2, 2])
    ax_actions = fig.add_subplot(gs[2, 3])

    _dark = '#16213e'
    for ax in (ax_acts, ax_phase1, ax_phase2, ax_phase3, ax_flow,
               ax_persist, ax_edges, ax_world, ax_actions):
        ax.set_facecolor(_dark)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a5a')
        ax.tick_params(colors='#667799', labelsize=7)

    # ── Route Activations ─────────────────────────────────────────────────
    ax_acts.set_title(
        'Route Activations   '
        '(gray shade: blackout  |  blue shade: phase 3 / curiosity context)',
        color='white', fontsize=10, pad=5)
    ax_acts.set_xlim(0, N_STEPS)
    ax_acts.set_ylim(-0.02, 1.08)
    ax_acts.axvspan(BLACKOUT_START, BLACKOUT_END, alpha=0.12, color='#cccccc', label='blackout')
    ax_acts.axvspan(PHASE3_START, N_STEPS,        alpha=0.07, color='#3498db', label='phase 3')
    ax_acts.axhline(ACTION_THRESH, color='white', alpha=0.20, ls='--', lw=0.8)
    ax_acts.text(4, ACTION_THRESH + 0.02, f'threshold ({ACTION_THRESH})',
                 color='white', alpha=0.28, fontsize=7)

    for name in ROUTE_NAMES:
        ax_acts.plot(agent.history[name], color=ROUTE_COLORS[name],
                     lw=1.2, alpha=0.90, label=name)

    ax_acts.legend(loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=8, framealpha=0.85)
    ax_acts.set_xlabel('Step', color='#667799', fontsize=8)
    ax_acts.set_ylabel('Activation', color='#667799', fontsize=8)

    # ── Phase Portraits ────────────────────────────────────────────────────
    def plot_phase(ax, x_name, y_name, title):
        xs = np.array(agent.history[x_name])
        ys = np.array(agent.history[y_name])
        n  = len(xs)
        t  = np.linspace(0, 1, n)

        # time-colored trajectory
        for i in range(n - 1):
            c = plt.cm.plasma(t[i])
            ax.plot(xs[i:i+2], ys[i:i+2], color=c, alpha=0.45, lw=0.8)

        # direction arrows every 25 steps
        for i in range(0, n - 1, 25):
            dx, dy = xs[i+1] - xs[i], ys[i+1] - ys[i]
            if abs(dx) + abs(dy) > 0.005:
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle='->', color='white',
                                            alpha=0.35, lw=0.7))

        ax.scatter(xs[0], ys[0], s=35, c='white', zorder=5, alpha=0.5)
        ax.scatter(xs[-1], ys[-1], s=35, c='yellow', zorder=5, alpha=0.7)
        ax.axvline(ACTION_THRESH, color='white', alpha=0.12, ls='--', lw=0.6)
        ax.axhline(ACTION_THRESH, color='white', alpha=0.12, ls='--', lw=0.6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel(x_name, color='#667799', fontsize=7)
        ax.set_ylabel(y_name, color='#667799', fontsize=7)
        ax.set_title(title, color='white', fontsize=9, pad=4)

    plot_phase(ax_phase1, 'attention', 'observe',   'attention x observe')
    plot_phase(ax_phase2, 'curiosity', 'approach',  'curiosity x approach')
    plot_phase(ax_phase3, 'observe',   'curiosity', 'observe x curiosity')

    # ── Flow Matrix ────────────────────────────────────────────────────────
    ax_flow.set_title('Route-to-Route Flow (avg, signed)', color='white', fontsize=9, pad=4)
    idx  = {n: i for i, n in enumerate(ROUTE_NAMES)}
    n_r  = len(ROUTE_NAMES)
    fmat = np.zeros((n_r, n_r))
    for step_flows in agent.flow_log:
        for (src, dst), f in step_flows.items():
            if src in idx and dst in idx:
                fmat[idx[src], idx[dst]] += f   # signed (excitatory +, inhibitory -)
    fmat /= (N_STEPS + 1e-9)

    vmax = max(abs(fmat).max(), 1e-6)
    im   = ax_flow.imshow(fmat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    lbl  = [n[:4] for n in ROUTE_NAMES]
    ax_flow.set_xticks(range(n_r)); ax_flow.set_xticklabels(lbl, rotation=40, fontsize=6, color='#aaaacc')
    ax_flow.set_yticks(range(n_r)); ax_flow.set_yticklabels(lbl, fontsize=6, color='#aaaacc')
    ax_flow.set_xlabel('to ->', color='#667799', fontsize=7)
    ax_flow.set_ylabel('from v', color='#667799', fontsize=7)
    for i in range(n_r):
        for j in range(n_r):
            v = fmat[i, j]
            if abs(v) > 0.001:
                ax_flow.text(j, i, f'{v:+.3f}', ha='center', va='center',
                             color='white', fontsize=5, alpha=0.85)
    plt.colorbar(im, ax=ax_flow, fraction=0.046, pad=0.04)

    # ── Loop Persistence ───────────────────────────────────────────────────
    ax_persist.set_title('Loop Persistence  (10-step rolling avg)', color='white', fontsize=9, pad=4)
    persist_arr = np.array(agent.above_thresh, dtype=float)
    smoothed    = np.convolve(persist_arr, np.ones(10) / 10, mode='same')
    ax_persist.fill_between(range(N_STEPS), smoothed, alpha=0.65, color='#e74c3c')
    ax_persist.plot(range(N_STEPS), smoothed, color='#ff9999', lw=0.8, alpha=0.7)
    ax_persist.axvspan(BLACKOUT_START, BLACKOUT_END, alpha=0.10, color='#cccccc')
    ax_persist.axvspan(PHASE3_START,   N_STEPS,       alpha=0.06, color='#3498db')
    ax_persist.axhline(0.5, color='white', alpha=0.15, ls='--', lw=0.7)
    ax_persist.set_xlim(0, N_STEPS)
    ax_persist.set_ylim(0, 1.12)
    ax_persist.set_xlabel('Step', color='#667799', fontsize=7)
    ax_persist.set_ylabel('Fraction active', color='#667799', fontsize=7)

    for label, start, end, c in [
        ('p1', 0, BLACKOUT_START, '#e74c3c'),
        ('blackout', BLACKOUT_START, BLACKOUT_END, '#aaaaaa'),
        ('p3', PHASE3_START, N_STEPS, '#3498db'),
    ]:
        pct = persist_arr[start:end].mean() * 100
        ax_persist.text((start + end) / 2, 1.07, f'{pct:.0f}%',
                        ha='center', color=c, fontsize=7, alpha=0.8)

    # ── Loop Edge Weights ──────────────────────────────────────────────────
    ax_edges.set_title('Loop Edge Weights (forward only)', color='white', fontsize=9, pad=4)
    fwd_edges = [('attention', 'observe'), ('observe', 'curiosity'),
                 ('curiosity', 'approach'), ('approach', 'attention')]
    fwd_colors = [ROUTE_COLORS['attention'], ROUTE_COLORS['observe'],
                  ROUTE_COLORS['curiosity'], ROUTE_COLORS['approach']]
    for (src, dst), c in zip(fwd_edges, fwd_colors):
        ws = [d.get((src, dst), float('nan')) for d in agent.edge_weight_log]
        ax_edges.plot(ws, color=c, lw=1.2, alpha=0.9, label=f'{src[:3]}->{dst[:3]}')
    ax_edges.axvspan(BLACKOUT_START, BLACKOUT_END, alpha=0.08, color='#cccccc')
    ax_edges.axvspan(PHASE3_START,   N_STEPS,       alpha=0.05, color='#3498db')
    ax_edges.set_xlim(0, N_STEPS)
    ax_edges.set_xlabel('Step', color='#667799', fontsize=7)
    ax_edges.set_ylabel('Weight', color='#667799', fontsize=7)
    ax_edges.legend(loc='lower right', facecolor='#1a1a2e',
                    labelcolor='white', fontsize=6, framealpha=0.85)

    # ── World ──────────────────────────────────────────────────────────────
    ax_world.set_xlim(0, WORLD_SIZE); ax_world.set_ylim(0, WORLD_SIZE)
    ax_world.set_aspect('equal')
    ax_world.set_title('World', color='white', fontsize=9, pad=4)
    ux, uy     = zip(*traj_u)
    ag_xs, ag_ys = zip(*traj_a)
    ax_world.plot(ux, uy, color='#3498db', alpha=0.18, lw=1)
    ax_world.plot(ag_xs, ag_ys, color='#2ecc71', alpha=0.18, lw=1)
    ax_world.scatter(traj_u[-1][0], traj_u[-1][1], s=130, c='#3498db', zorder=5)
    ax_world.scatter(traj_a[-1][0], traj_a[-1][1], s=130, c='#2ecc71', zorder=5)
    for obj in objects:
        ax_world.scatter(obj.pos.x, obj.pos.y, s=70, c='#f39c12', marker='s', zorder=4)
        ax_world.text(obj.pos.x, obj.pos.y + 0.28, f'O{obj.id}',
                      color='#f39c12', fontsize=6, ha='center')
    for i in range(0, N_STEPS, 10):
        c = ACTION_COLORS.get(agent.action_log[i], '#888888')
        ax_world.scatter(traj_a[i][0], traj_a[i][1], s=11, c=c, alpha=0.60, zorder=3)

    # ── Action Distribution ────────────────────────────────────────────────
    ax_actions.set_title('Action Distribution', color='white', fontsize=9, pad=4)
    counts  = Counter(agent.action_log)
    total   = len(agent.action_log)
    acts    = [a for a, _ in counts.most_common()]
    vals    = [counts[a] / total for a in acts]
    clrs    = [ACTION_COLORS.get(a, '#888888') for a in acts]
    bars    = ax_actions.barh(acts, vals, color=clrs, alpha=0.85, height=0.6)
    ax_actions.set_xlim(0, 1.05)
    for bar, v in zip(bars, vals):
        ax_actions.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{v*100:.0f}%', va='center', color='white', fontsize=7)

    # ── footer ─────────────────────────────────────────────────────────────
    p3_cnt = Counter(agent.action_log[PHASE3_START:])
    p3_top = p3_cnt.most_common(1)[0]
    b_pct  = np.array(agent.above_thresh[BLACKOUT_START:BLACKOUT_END]).mean() * 100
    fig.text(0.50, 0.012,
             f'Blackout persistence: {b_pct:.0f}%  |  '
             f'Phase3 top action: {p3_top[0]} '
             f'({p3_top[1]/(N_STEPS-PHASE3_START)*100:.0f}%)',
             color='#667799', fontsize=7, ha='center')

    fig.suptitle('EchoLoop v2 — Internal Circulation Dynamics',
                 color='white', fontsize=13, y=0.98)

    _img = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images'))
    os.makedirs(_img, exist_ok=True)
    out = os.path.join(_img, 'echoloop_v2_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved -> {out}")
    plt.show()


# ── エントリポイント ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("EchoLoop v2: running simulation ...")
    agent, objects, traj_u, traj_a = simulate()

    print("\nFinal route states:")
    for name in ROUTE_NAMES:
        print(f"  {name:12s}  act={agent.routes[name]:.3f}")

    print("\nLoop edge weights (final):")
    for e in agent.edges:
        sign = '+' if e.weight >= 0 else ''
        print(f"  {e.src:12s} -> {e.dst:12s}  {sign}{e.weight:.4f}")

    b_arr  = np.array(agent.above_thresh)
    p_pcts = [
        b_arr[:BLACKOUT_START].mean(),
        b_arr[BLACKOUT_START:BLACKOUT_END].mean(),
        b_arr[PHASE3_START:].mean(),
    ]
    print("\nLoop persistence:")
    for label, pct in zip(['Phase 1 (normal)', 'Phase 2 (blackout)', 'Phase 3 (ctx swap)'], p_pcts):
        print(f"  {label:22s}  {pct*100:.0f}%")

    print("\nAction distribution:")
    counts = Counter(agent.action_log)
    for act, cnt in counts.most_common():
        print(f"  {act:20s}  {cnt:3d}  ({cnt/N_STEPS*100:.0f}%)")

    dom_counts = Counter(agent.dominant_log)
    print("\nDominant route distribution:")
    for route, cnt in dom_counts.most_common():
        print(f"  {route:12s}  {cnt:3d}  ({cnt/N_STEPS*100:.0f}%)")

    visualize(agent, objects, traj_u, traj_a)
