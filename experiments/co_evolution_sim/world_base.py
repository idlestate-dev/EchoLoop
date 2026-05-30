"""world_base.py — Shared GridWorld infrastructure for EchoLoop experiments.

Classes and utility functions imported by session_* modules.
No run_* or plot_* functions here — only reusable building blocks.
"""
import numpy as np
import networkx as nx


class GridWorld:
    grid_size = 5
    hp_max = 200
    hp_decay = 1
    food_value = 30
    food_respawn = 30
    max_steps = 500
    food_positions = [(0, 0), (4, 4)]
    start_pos = (2, 2)
    start_hp = 100


class ContextGridWorld:
    """GridWorld where food position depends on episode context (A=NW, B=SE)."""
    grid_size = 5
    hp_max = 200
    hp_decay = 1
    food_value = 30
    food_respawn = 40
    max_steps = 500
    start_pos = (2, 2)
    start_hp = 100


def _make_graph(N, rng):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.2:
                G.add_edge(i, j, weight=float(rng.random()))
    return G


def _mutate_graph(G, N, rng, mutation_std=0.05, edge_change_prob=0.05):
    """Return a mutated copy of G."""
    G_new = G.copy()
    for i, j in list(G_new.edges()):
        w = G_new[i][j]['weight'] + rng.normal(0, mutation_std)
        G_new[i][j]['weight'] = float(np.clip(w, 0.01, 1.0))
    existing = set(G_new.edges())
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (i, j) in existing:
                if rng.random() < edge_change_prob:
                    G_new.remove_edge(i, j)
            else:
                if rng.random() < edge_change_prob:
                    G_new.add_edge(i, j, weight=float(rng.uniform(0.01, 1.0)))
    return G_new


def _hebbian_step(G, activity, N, rng):
    """Apply Hebbian update. Returns sum of absolute weight changes."""
    delta = 0.0
    edges_to_remove = []
    for i, j, data in list(G.edges(data=True)):
        old_w = data['weight']
        w = old_w
        if activity[i] > 0.5 and activity[j] > 0.5:
            w += 0.05
        w -= 0.01
        if w < 0.01:
            edges_to_remove.append((i, j))
            delta += old_w
        else:
            w = min(w, 1.0)
            G[i][j]['weight'] = w
            delta += abs(w - old_w)
    G.remove_edges_from(edges_to_remove)
    existing = set(G.edges())
    for i in range(N):
        for j in range(N):
            if i != j and (i, j) not in existing and rng.random() < 0.01:
                G.add_edge(i, j, weight=0.05)
                delta += 0.05
    return delta


def _softmax_sample(x, temperature=0.5):
    x = x - np.max(x)
    probs = np.exp(x / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(len(x), p=probs))


def _run_episode(G, activity, rng, N=25, K=10, n_propagation_steps=3,
                 temperature=0.5, readout_weights=None, topology_frozen=False,
                 debug=False):
    """Run one GridWorld episode. Mutates G and activity in place."""
    gw = GridWorld()
    row, col = gw.start_pos
    hp = gw.start_hp
    food_available = [True] * len(gw.food_positions)
    food_timers = [0] * len(gw.food_positions)
    food_eaten = 0
    total_delta = 0.0
    steps_survived = 0

    for step in range(gw.max_steps):
        if hp <= 0:
            break

        x_norm = col / (gw.grid_size - 1)
        y_norm = row / (gw.grid_size - 1)
        hp_norm = hp / gw.hp_max
        food_flag = 1.0 if any(food_available) else 0.0
        input_vals = {0: x_norm, 1: y_norm, 2: hp_norm, 3: food_flag}

        for _ in range(n_propagation_steps):
            new_activity = np.zeros(N)
            for i in range(N):
                if i in input_vals:
                    new_activity[i] = input_vals[i]
                else:
                    s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                    new_activity[i] = np.tanh(s)
            activity[:] = new_activity

        if debug and step == 0:
            n_input_edges = sum(1 for i, j in G.edges() if i in {0, 1, 2, 3})
            print(f'Input node activities: {activity[0:4].round(3)}')
            print(f'Internal node activities (mean): {activity[4:20].mean():.3f}')
            print(f'Number of edges from input nodes (0-3): {n_input_edges}')
            if readout_weights is None:
                n_output_edges = sum(1 for i, j in G.edges() if j in {20, 21, 22, 23, 24})
                print(f'Output node activities: {activity[20:25].round(3)}')
                print(f'Number of edges to output nodes (20-24): {n_output_edges}')
            else:
                scores = activity[4:20] @ readout_weights
                print(f'Readout action scores: {scores.round(3)}')

        if readout_weights is not None:
            action_scores = activity[4:20] @ readout_weights
            action = _softmax_sample(action_scores, temperature=temperature)
        else:
            action = _softmax_sample(activity[20:25], temperature=temperature)

        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(gw.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(gw.grid_size - 1, col + 1)
        elif action == 4:
            for idx, (fr, fc) in enumerate(gw.food_positions):
                if row == fr and col == fc and food_available[idx]:
                    hp = min(gw.hp_max, hp + gw.food_value)
                    food_available[idx] = False
                    food_timers[idx] = 0
                    food_eaten += 1
                    break

        if debug and step % 10 == 0:
            if readout_weights is not None:
                scores = activity[4:20] @ readout_weights
                print(f'Step {step}: pos=({col},{row}), HP={hp}, action={action}, '
                      f'action_scores={scores.round(3)}')
            else:
                print(f'Step {step}: pos=({col},{row}), HP={hp}, action={action}, '
                      f'activity_output={activity[20:25].round(3)}')

        hp -= gw.hp_decay
        steps_survived = step + 1

        for idx in range(len(gw.food_positions)):
            if not food_available[idx]:
                food_timers[idx] += 1
                if food_timers[idx] >= gw.food_respawn:
                    food_available[idx] = True
                    food_timers[idx] = 0

        if not topology_frozen and (step + 1) % K == 0:
            total_delta += _hebbian_step(G, activity, N, rng)

    if debug:
        print(f'Episode end: food_eaten={food_eaten}, steps={steps_survived}')

    return steps_survived, total_delta, G.number_of_edges(), food_eaten


def _run_random_episode(rng):
    """Run one episode with a uniformly random agent."""
    gw = GridWorld()
    row, col = gw.start_pos
    hp = gw.start_hp
    food_available = [True] * len(gw.food_positions)
    food_timers = [0] * len(gw.food_positions)
    steps_survived = 0

    for step in range(gw.max_steps):
        if hp <= 0:
            break

        action = int(rng.integers(0, 5))

        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(gw.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(gw.grid_size - 1, col + 1)
        elif action == 4:
            for idx, (fr, fc) in enumerate(gw.food_positions):
                if row == fr and col == fc and food_available[idx]:
                    hp = min(gw.hp_max, hp + gw.food_value)
                    food_available[idx] = False
                    food_timers[idx] = 0
                    break

        hp -= gw.hp_decay
        steps_survived = step + 1

        for idx in range(len(gw.food_positions)):
            if not food_available[idx]:
                food_timers[idx] += 1
                if food_timers[idx] >= gw.food_respawn:
                    food_available[idx] = True
                    food_timers[idx] = 0

    return steps_survived


def _run_context_episode(G, activity, rng, N=20, K=10, n_propagation_steps=3,
                          temperature=1.0, readout_weights=None,
                          topology_frozen=False, mode=None):
    """ContextGridWorld episode. mode='A' → food at (0,0), 'B' → food at (4,4), None → random."""
    gw = ContextGridWorld()
    if mode is None:
        mode = 'A' if rng.random() < 0.5 else 'B'
    food_pos = (0, 0) if mode == 'A' else (4, 4)
    food_available = [True]
    food_timers = [0]

    row, col = gw.start_pos
    hp = gw.start_hp
    food_eaten = 0
    total_delta = 0.0
    steps_survived = 0

    for step in range(gw.max_steps):
        if hp <= 0:
            break

        input_vals = {
            0: col / (gw.grid_size - 1),
            1: row / (gw.grid_size - 1),
            2: hp / gw.hp_max,
            3: 1.0 if food_available[0] else 0.0,
        }

        for _ in range(n_propagation_steps):
            new_activity = np.zeros(N)
            for i in range(N):
                if i in input_vals:
                    new_activity[i] = input_vals[i]
                else:
                    s = sum(G[j][i]['weight'] * activity[j] for j in G.predecessors(i))
                    new_activity[i] = np.tanh(s)
            activity[:] = new_activity

        if readout_weights is not None:
            action = _softmax_sample(activity[4:20] @ readout_weights,
                                     temperature=temperature)
        else:
            action = int(rng.integers(0, 5))

        fr, fc = food_pos
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(gw.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(gw.grid_size - 1, col + 1)
        elif action == 4:
            if row == fr and col == fc and food_available[0]:
                hp = min(gw.hp_max, hp + gw.food_value)
                food_available[0] = False
                food_timers[0] = 0
                food_eaten += 1

        hp -= gw.hp_decay
        steps_survived = step + 1
        if not food_available[0]:
            food_timers[0] += 1
            if food_timers[0] >= gw.food_respawn:
                food_available[0] = True
                food_timers[0] = 0

        if not topology_frozen and (step + 1) % K == 0:
            total_delta += _hebbian_step(G, activity, N, rng)

    return steps_survived, total_delta, G.number_of_edges(), food_eaten, mode
