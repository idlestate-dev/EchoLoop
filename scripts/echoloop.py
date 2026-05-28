#!/usr/bin/env python3
"""
EchoLoop — 経路が本体のエージェントシミュレーション

仮説: 行動はニューラルネットではなく「内部経路」の読み出し。
外界入力は循環ルートを歪めるだけ。よく通る経路は強化、使われない経路は減衰。
"""
import os
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── 定数 ─────────────────────────────────────────────────────────────────────
WORLD_SIZE     = 10.0
N_STEPS        = 300
ACTION_THRESH  = 0.22   # この閾値を超えたルートがアクションを発火
CONNECT_RADIUS = 0.70   # ノード接続半径
GROW_PROB      = 0.06   # 毎stepのノード生成確率
GROW_EXPLOIT   = 0.80   # 80%: active edge周辺, 20%: ランダム

ROUTE_NAMES  = ['attention', 'curiosity', 'approach', 'observe', 'idle']
ROUTE_COLORS = dict(zip(ROUTE_NAMES,
                        ['#e74c3c', '#9b59b6', '#2ecc71', '#3498db', '#95a5a6']))

# ルート間影響 (src → dst): weight  ※負値は抑制
ROUTE_FLOW = {
    ('attention', 'observe'):   0.40,
    ('attention', 'approach'):  0.15,
    ('curiosity', 'observe'):   0.50,
    ('curiosity', 'approach'):  0.20,
    ('observe',   'curiosity'): 0.25,
    ('approach',  'attention'): 0.30,
    ('idle',      'curiosity'): 0.08,
    ('idle',      'attention'): 0.08,
    ('approach',  'idle'):     -0.30,
    ('observe',   'idle'):     -0.20,
    ('attention', 'idle'):     -0.15,
}

ACTION_MAP = {
    'attention': 'look_at_user',
    'curiosity': 'point',
    'approach':  'move_toward',
    'observe':   'look_at_object',
    'idle':      'idle',
}

ACTION_COLORS = {
    'look_at_user':   '#e74c3c',
    'look_at_object': '#3498db',
    'move_toward':    '#2ecc71',
    'point':          '#9b59b6',
    'idle':           '#555577',
}

GAZE_PHASE    = 20    # ユーザー視線サイクル長 (steps)
GAZE_ON_AGENT = 0.30  # サイクルのうち agent を見ている割合


# ── Vec2 ──────────────────────────────────────────────────────────────────────
class Vec2:
    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float):
        self.x, self.y = float(x), float(y)

    def dist(self, other: 'Vec2') -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def arr(self) -> np.ndarray:
        return np.array([self.x, self.y])


# ── WorldObject ───────────────────────────────────────────────────────────────
class WorldObject:
    def __init__(self, x: float, y: float, oid: int):
        self.pos      = Vec2(x, y)
        self.id       = oid
        self.interest = random.uniform(0.6, 1.4)


# ── User ──────────────────────────────────────────────────────────────────────
class User:
    def __init__(self, x: float, y: float):
        self.pos      = Vec2(x, y)
        self.gaze     = np.array([1.0, 0.0])  # 単位ベクトル
        self.reaction = 0.0                    # -1 (拒絶) … +1 (歓迎)

    def look_at(self, target: Vec2):
        d = target.arr() - self.pos.arr()
        n = np.linalg.norm(d)
        if n > 1e-6:
            self.gaze = d / n

    def step_toward(self, target: Vec2, speed: float = 0.12):
        d = target.arr() - self.pos.arr()
        n = np.linalg.norm(d)
        if n > speed:
            v = d / n * speed
            self.pos.x += v[0]
            self.pos.y += v[1]

    def gaze_align(self, target: Vec2) -> float:
        """0..1 — ターゲット方向と gaze の一致度"""
        d = target.arr() - self.pos.arr()
        n = np.linalg.norm(d)
        if n < 1e-6:
            return 0.0
        return float(np.clip(np.dot(self.gaze, d / n), 0.0, 1.0))


# ── Node / Edge ───────────────────────────────────────────────────────────────
class Node:
    _counter = 0

    def __init__(self, x: float, y: float, tag: str):
        self.id         = Node._counter
        Node._counter  += 1
        self.x, self.y  = float(x), float(y)
        self.tag        = tag
        self.activation = 0.0
        self.edges: list['Edge'] = []


class Edge:
    __slots__ = ('a', 'b', 'strength', 'usage')

    def __init__(self, a: Node, b: Node):
        self.a        = a
        self.b        = b
        self.strength = 1.0
        self.usage    = 0.0


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, x: float, y: float):
        self.pos    = Vec2(x, y)
        # 各ルート: act=活性度 (0..1), str=経路強度 (0.2..2.5)
        self.routes = {n: {'act': 0.0, 'str': 1.0} for n in ROUTE_NAMES}
        self.routes['idle']['act'] = 0.30  # 初期ベースライン

        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._init_graph()

        self.action     = 'idle'
        self.history    = {n: [] for n in ROUTE_NAMES}
        self.action_log: list[str] = []

    # ── 内部グラフ初期化 ──────────────────────────────────────────────────────
    def _init_graph(self):
        centers = {
            'attention': ( 0.5,  0.8),
            'curiosity': (-0.8,  0.1),
            'approach':  ( 0.8, -0.1),
            'observe':   ( 0.0, -0.8),
            'idle':      ( 0.0,  0.0),
        }
        for tag, (cx, cy) in centers.items():
            for _ in range(4):
                self.nodes.append(
                    Node(cx + random.gauss(0, 0.15), cy + random.gauss(0, 0.15), tag)
                )
        self._connect()

    def _connect(self):
        """CONNECT_RADIUS 以内のノードペアにエッジを張る (重複除外)"""
        seen = {(e.a.id, e.b.id) for e in self.edges}
        seen |= {(e.b.id, e.a.id) for e in self.edges}
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1:]:
                if (a.id, b.id) in seen:
                    continue
                if np.hypot(a.x - b.x, a.y - b.y) < CONNECT_RADIUS:
                    e = Edge(a, b)
                    self.edges.append(e)
                    a.edges.append(e)
                    b.edges.append(e)
                    seen.add((a.id, b.id))

    def _grow(self):
        """80% よく使われる経路周辺 / 20% 完全ランダムでノードを生やす"""
        if random.random() > GROW_PROB:
            return
        if random.random() < GROW_EXPLOIT and self.edges:
            top = max(self.edges, key=lambda e: e.usage)
            cx  = (top.a.x + top.b.x) / 2 + random.gauss(0, 0.10)
            cy  = (top.a.y + top.b.y) / 2 + random.gauss(0, 0.10)
            tag = top.a.tag
        else:
            cx  = random.gauss(0, 0.70)
            cy  = random.gauss(0, 0.70)
            tag = random.choice(ROUTE_NAMES)
        self.nodes.append(Node(cx, cy, tag))
        self._connect()

    # ── ノードグラフ内の活性化フロー ──────────────────────────────────────────
    def _flow(self):
        """ノードグラフ内の活性化フロー。ルートには逆伝播しない。
        ノードは構造的学習の記録 (edge.usage / edge.strength) に特化する。"""
        delta = {nd.id: 0.0 for nd in self.nodes}
        for e in self.edges:
            flow = e.a.activation * e.strength
            delta[e.b.id] += flow * 0.12
            delta[e.a.id] += e.b.activation * e.strength * 0.12
            e.usage += flow * 0.5
        for nd in self.nodes:
            nd.activation = float(np.clip(nd.activation + delta[nd.id], 0.0, 2.0))

    # ── 毎 step の更新 ────────────────────────────────────────────────────────
    def update(self, user: User, objects: list[WorldObject]) -> str:
        udist = self.pos.dist(user.pos)

        # 1. 外界入力を算出
        attention_in = user.gaze_align(self.pos) * float(np.exp(-udist / 5.0))
        curiosity_in = float(np.clip(
            sum(o.interest * np.exp(-self.pos.dist(o.pos) / 4.0) for o in objects)
            / max(1, len(objects)), 0.0, 1.0
        ))
        approach_in = float(np.clip(1.0 - udist / 8.0, 0.0, 1.0)) * (0.5 + user.reaction * 0.5)

        # 2. 外界入力 → ルート活性化を押し上げ (distortion)
        self.routes['attention']['act'] += attention_in  * 0.22
        self.routes['curiosity']['act'] += curiosity_in  * 0.16
        self.routes['approach']['act']  += approach_in   * 0.14
        self.routes['idle']['act']      += 0.04           # 常時ベースライン

        # 3. ルート間相互作用 (内部循環) ※str は学習の読み出しに使うが流量には掛けない
        delta = {k: 0.0 for k in ROUTE_NAMES}
        for (src, dst), w in ROUTE_FLOW.items():
            delta[dst] += self.routes[src]['act'] * w * 0.06
        for tag, dv in delta.items():
            self.routes[tag]['act'] = float(np.clip(self.routes[tag]['act'] + dv, 0.0, 1.0))

        # 4. ルート活性化 → タグ付きノードへ注入 (グラフは構造学習レイヤ)
        for nd in self.nodes:
            nd.activation += self.routes[nd.tag]['act'] * 0.06

        # 5. グラフ内フロー (ルートへは逆伝播しない)
        self._flow()

        # 6. 減衰 (使われない経路は弱まる)
        for r in self.routes.values():
            r['act'] = float(np.clip(r['act'] * 0.84, 0.0, 1.0))
        for nd in self.nodes:
            nd.activation *= 0.78

        # 7. ノード生成
        self._grow()

        # 8. アクション発火: 最高活性ルートが閾値超え
        # str (学習強度) を読み出しに掛けることで強化された経路が優先されやすい
        best_tag = max(self.routes,
                       key=lambda t: self.routes[t]['act'] * self.routes[t]['str'])
        best_act = self.routes[best_tag]['act']
        action   = ACTION_MAP[best_tag] if best_act > ACTION_THRESH else 'idle'
        self.action = action

        # 9. ユーザー reaction → ルート強度を強化/減衰
        if abs(user.reaction) > 0.05:
            self.routes[best_tag]['str'] = float(np.clip(
                self.routes[best_tag]['str'] + user.reaction * 0.02, 0.2, 2.5
            ))

        # 10. エッジ強度更新 (よく通る経路は強化)
        for e in self.edges:
            if e.usage > 0.05:
                e.strength  = float(np.clip(e.strength + 0.006, 0.0, 3.0))
                e.usage    *= 0.85
            else:
                e.strength  = float(np.clip(e.strength * 0.998, 0.1, 3.0))

        for tag in ROUTE_NAMES:
            self.history[tag].append(self.routes[tag]['act'])
        self.action_log.append(action)
        return action


# ── シミュレーション実行 ──────────────────────────────────────────────────────
def simulate():
    objects = [WorldObject(random.uniform(1.5, 8.5), random.uniform(1.5, 8.5), i)
               for i in range(4)]
    user  = User(2.0, 5.0)
    agent = Agent(8.0, 5.0)

    waypoints = [Vec2(8.5, 2.0), Vec2(5.0, 8.5), Vec2(1.5, 2.0), Vec2(5.0, 5.0)]
    wp_i = 0
    traj_u: list[tuple] = []
    traj_a: list[tuple] = []

    # 粗い視線シナリオ: GAZE_PHASE ごとに 4 フェーズを循環
    # 0: agent を見る(注目)  1: 物を見る(共同注意)
    # 2: 視線そらし(無視)    3: 近づく(接近)
    PHASES = [
        ('gaze_agent',  0.30),
        ('gaze_object', 0.30),
        ('neutral',     0.20),
        ('approach',    0.20),
    ]
    cum = np.cumsum([p for _, p in PHASES])

    for step in range(N_STEPS):
        # ユーザー移動
        user.step_toward(waypoints[wp_i])
        if user.pos.dist(waypoints[wp_i]) < 0.3:
            wp_i = (wp_i + 1) % len(waypoints)

        phase = (step % GAZE_PHASE) / GAZE_PHASE
        if phase < cum[0]:                 # agent を見て反応する
            user.look_at(agent.pos)
            user.reaction = 0.5
        elif phase < cum[1]:               # 最近オブジェクトを見る
            nearest = min(objects, key=lambda o: user.pos.dist(o.pos))
            user.look_at(nearest.pos)
            user.reaction = 0.0
        elif phase < cum[2]:               # 視線そらし
            angle = step * 0.18
            user.gaze     = np.array([np.cos(angle), np.sin(angle)])
            user.reaction = -0.15
        else:                              # 肯定的な近距離接触
            user.look_at(agent.pos)
            user.reaction = 0.4

        action = agent.update(user, objects)

        # move_toward 時に物理移動
        if action == 'move_toward':
            d = user.pos.arr() - agent.pos.arr()
            n = np.linalg.norm(d)
            if n > 1.2:
                v = d / n * 0.10
                agent.pos.x += v[0]
                agent.pos.y += v[1]
            else:
                user.reaction = min(user.reaction, -0.2)  # 近づきすぎ → 負の反応

        traj_u.append((user.pos.x,  user.pos.y))
        traj_a.append((agent.pos.x, agent.pos.y))

    return agent, objects, traj_u, traj_a


# ── 可視化 ────────────────────────────────────────────────────────────────────
def visualize(agent: Agent, objects: list[WorldObject],
              traj_u: list[tuple], traj_a: list[tuple]):
    fig = plt.figure(figsize=(16, 9), facecolor='#1a1a2e')
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                            left=0.06, right=0.97, top=0.93, bottom=0.09)
    ax_world = fig.add_subplot(gs[:, 0])
    ax_acts  = fig.add_subplot(gs[0, 1:])
    ax_graph = fig.add_subplot(gs[1, 1])
    ax_str   = fig.add_subplot(gs[1, 2])

    for ax in (ax_world, ax_acts, ax_graph, ax_str):
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a5a')
        ax.tick_params(colors='#667799', labelsize=7)

    # ── World ──────────────────────────────────────────────────────────────
    ax_world.set_xlim(0, WORLD_SIZE)
    ax_world.set_ylim(0, WORLD_SIZE)
    ax_world.set_aspect('equal')
    ax_world.set_title('World', color='white', fontsize=11, pad=6)

    ux, uy = zip(*traj_u)
    agent_xs, agent_ys = zip(*traj_a)
    ax_world.plot(ux, uy, color='#3498db', alpha=0.22, lw=1)
    ax_world.plot(agent_xs, agent_ys, color='#2ecc71', alpha=0.22, lw=1)
    ax_world.scatter(*traj_u[-1],  s=150, c='#3498db', zorder=5)
    ax_world.scatter(*traj_a[-1], s=150, c='#2ecc71', zorder=5)

    for obj in objects:
        ax_world.scatter(obj.pos.x, obj.pos.y, s=90, c='#f39c12', marker='s', zorder=4)
        ax_world.text(obj.pos.x, obj.pos.y + 0.30, f'O{obj.id}',
                      color='#f39c12', fontsize=7, ha='center')

    # アクション種別を色付きドットで軌跡上に表示
    for i in range(0, N_STEPS, 10):
        c = ACTION_COLORS.get(agent.action_log[i], '#ffffff')
        ax_world.scatter(*traj_a[i], s=14, c=c, alpha=0.65, zorder=3)

    legend_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#3498db', ms=8, label='User'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2ecc71', ms=8, label='Agent'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#f39c12', ms=8, label='Object'),
    ] + [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=c, ms=6, label=a)
        for a, c in ACTION_COLORS.items()
    ]
    ax_world.legend(handles=legend_handles, loc='upper left', facecolor='#0a0a1e',
                    labelcolor='white', fontsize=6, framealpha=0.85)

    # ── Route Activations ──────────────────────────────────────────────────
    ax_acts.set_title('Route Activations over Time', color='white', fontsize=11, pad=6)
    ax_acts.set_xlim(0, N_STEPS)
    ax_acts.set_ylim(-0.02, 1.05)
    ax_acts.axhline(ACTION_THRESH, color='white', alpha=0.22, ls='--', lw=0.8)
    ax_acts.text(4, ACTION_THRESH + 0.02, f'action threshold ({ACTION_THRESH})',
                 color='white', alpha=0.35, fontsize=7)

    # ユーザーが agent を見ているフェーズを赤くシェード
    for s in range(0, N_STEPS, GAZE_PHASE):
        end = min(s + int(GAZE_PHASE * GAZE_ON_AGENT), N_STEPS)
        ax_acts.axvspan(s, end, alpha=0.07, color='#e74c3c',
                        label='user gaze on agent' if s == 0 else '')

    for name in ROUTE_NAMES:
        ax_acts.plot(agent.history[name], label=name,
                     color=ROUTE_COLORS[name], lw=1.3, alpha=0.92)

    ax_acts.legend(loc='upper right', facecolor='#1a1a2e',
                   labelcolor='white', fontsize=8, framealpha=0.85)
    ax_acts.set_xlabel('Step', color='#667799', fontsize=8)
    ax_acts.set_ylabel('Activation', color='#667799', fontsize=8)

    # ── Internal Route Graph ───────────────────────────────────────────────
    ax_graph.set_title('Internal Route Graph', color='white', fontsize=10, pad=6)
    ax_graph.set_aspect('equal')
    xs = [nd.x for nd in agent.nodes]
    ys = [nd.y for nd in agent.nodes]
    margin = 0.35
    ax_graph.set_xlim(min(xs) - margin, max(xs) + margin)
    ax_graph.set_ylim(min(ys) - margin, max(ys) + margin)
    ax_graph.tick_params(labelbottom=False, labelleft=False)

    for e in agent.edges:
        alpha = float(np.clip(e.strength / 2.5, 0.04, 0.55))
        lw    = float(np.clip(e.strength * 0.4, 0.2, 1.5))
        ax_graph.plot([e.a.x, e.b.x], [e.a.y, e.b.y],
                      color='#aaaacc', alpha=alpha, lw=lw)

    for nd in agent.nodes:
        c  = ROUTE_COLORS.get(nd.tag, '#ffffff')
        sz = 12 + nd.activation * 25
        ax_graph.scatter(nd.x, nd.y, s=sz, c=c, alpha=0.75, zorder=3)

    ax_graph.legend(
        handles=[mpatches.Patch(color=ROUTE_COLORS[n], label=n) for n in ROUTE_NAMES],
        loc='lower right', facecolor='#0a0a1e', labelcolor='white',
        fontsize=6, framealpha=0.85,
    )

    # ── Route Strengths ────────────────────────────────────────────────────
    ax_str.set_title('Route Strengths (final)', color='white', fontsize=10, pad=6)
    strengths = [agent.routes[n]['str'] for n in ROUTE_NAMES]
    colors_b  = [ROUTE_COLORS[n] for n in ROUTE_NAMES]
    bars = ax_str.bar(ROUTE_NAMES, strengths, color=colors_b, alpha=0.85, width=0.6)
    ax_str.axhline(1.0, color='white', alpha=0.22, ls='--', lw=0.8)
    ax_str.set_ylim(0, 2.8)
    ax_str.set_ylabel('Strength', color='#667799', fontsize=8)
    ax_str.tick_params(axis='x', rotation=28, labelsize=7)
    for bar, s in zip(bars, strengths):
        ax_str.text(bar.get_x() + bar.get_width() / 2, s + 0.05,
                    f'{s:.2f}', ha='center', va='bottom', color='white', fontsize=7)

    counts  = Counter(agent.action_log)
    total   = len(agent.action_log)
    summary = '  '.join(f'{k}: {v/total*100:.0f}%' for k, v in counts.most_common())
    fig.text(0.50, 0.008, f'Actions — {summary}',
             color='#667799', fontsize=7, ha='center')

    fig.suptitle('EchoLoop: Path-Based Agent Simulation',
                 color='white', fontsize=13, y=0.98)

    _img = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images'))
    os.makedirs(_img, exist_ok=True)
    out = os.path.join(_img, 'echoloop_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved → {out}")
    plt.show()


# ── エントリポイント ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    Node._counter = 0  # 再現性のためリセット

    print("EchoLoop: running simulation ...")
    agent, objects, traj_u, traj_a = simulate()

    print(f"\nNodes: {len(agent.nodes)}  (started: 20)   Edges: {len(agent.edges)}")
    print("Final route states:")
    for name in ROUTE_NAMES:
        r = agent.routes[name]
        print(f"  {name:12s}  act={r['act']:.3f}  str={r['str']:.3f}")

    counts = Counter(agent.action_log)
    total  = len(agent.action_log)
    print("\nAction distribution:")
    for act, cnt in counts.most_common():
        print(f"  {act:20s}  {cnt:3d}  ({cnt / total * 100:.0f}%)")

    visualize(agent, objects, traj_u, traj_a)
