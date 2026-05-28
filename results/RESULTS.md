# EchoLoop — Simulation Results

## Hypothesis

> The route — not the neural network — is what matters.
> External input merely distorts the internal circulation. Actions fire as readouts of internal state.

---

## Model Structure

### World (2D Space)

| Entity | Description |
|---|---|
| `User` | Position + gaze direction + reaction (−1 to +1) |
| `Agent` | Position + internal route graph |
| `WorldObject` | Position + interest value (0.6–1.4), 4 static objects |

### Internal Routes (5 types)

| Route | Fired Action | Primary Input Source |
|---|---|---|
| `attention` | `look_at_user` | User's gaze is directed at the agent |
| `curiosity` | `point` | Sum of nearby object interest values |
| `approach` | `move_toward` | User is nearby and reaction is positive |
| `observe` | `look_at_object` | Inflow from attention / curiosity |
| `idle` | `idle` | Constant baseline (+0.04/step) |

Each route has an `activation` and a `strength`.
`strength` is reinforced or weakened by the user's reaction (Hebbian learning).

### Node/Edge Graph

```
class Node:
    x, y       # 2D position
    tag        # owning route name
    activation # activation level

class Edge:
    a, b       # node pair
    strength   # path strength (increases with usage)
    usage      # cumulative flow through this edge
```

- **80%**: new nodes generated near frequently used edges
- **20%**: generated at random

The node graph does not feed back into route dynamics.
It serves as a structural learning record — a layer for logging and visualizing which paths have been reinforced.

### Processing Order per Step

```
1. Compute external inputs
      attention_in  = gaze_align × exp(−dist/5)
      curiosity_in  = Σ(interest × exp(−dist/4)) / n_objects
      approach_in   = clip(1 − dist/8) × (0.5 + reaction × 0.5)

2. External input → push up route activations (distortion)

3. Inter-route interactions (internal circulation)
      e.g.: attention → observe (+0.40), approach → idle (−0.30)

4. Route activation → injected into tagged nodes

5. Flow within node graph (weighted by edge.strength)

6. Decay (activation × 0.84 / step)

7. Node generation (6% probability)

8. Action firing
      score = activation × strength
      Route with highest score above ACTION_THRESH (0.22) wins

9. user.reaction → update winning route's strength by ±0.02

10. Edge reinforcement / decay (Hebbian)
```

---

## Simulation Settings

| Parameter | Value |
|---|---|
| Steps | 300 |
| World size | 10 × 10 |
| Action threshold | 0.22 |
| Gaze cycle length | 20 steps |

**User Gaze Phases (20-step cycle)**

| Phase | Proportion | Behavior | Reaction |
|---|---|---|---|
| gaze_agent | 30% | Looking at the agent | +0.5 |
| gaze_object | 30% | Looking at the nearest object | 0.0 |
| neutral | 20% | Looking away | −0.15 |
| close_positive | 20% | Looking at the agent, positively | +0.4 |

---

## Results (seed=42, 300 steps)

### Final Route State

| Route | activation | strength | Change |
|---|---|---|---|
| `attention` | 0.342 | **2.169** | ↑ Reinforced (user frequently attended to the agent) |
| `curiosity` | 0.304 | 0.985 | → Flat |
| `approach` | 0.266 | 0.996 | → Flat |
| `observe` | 0.106 | 1.000 | → Flat |
| `idle` | 0.151 | 1.050 | → Slight increase |

### Action Distribution

| Action | Count | Proportion |
|---|---|---|
| `look_at_user` | 233 | **78%** |
| `point` | 38 | 13% |
| `idle` | 22 | 7% |
| `move_toward` | 7 | 2% |

### Node Graph

| Item | Initial | Final |
|---|---|---|
| Node count | 20 | **44** (24 generated) |
| Edge count | — | 326 |

---

## Observations

### 1. Attention Route Dominance

`attention` strength grew from **1.0 → 2.17**.
Each time the user looked at the agent, a reaction of +0.5 was applied, establishing a positive feedback loop: "attention wins → strength increases → attention wins more easily."
This is exactly the behavior the hypothesis predicts — frequently-used routes get reinforced.

### 2. External Input as Distortion

During neutral phases (when the user looks away), `attention_in` drops, and `curiosity` — which receives a constant background input — becomes relatively stronger, triggering `point` (13%).
This suggests that external input shifts the competitive balance between routes rather than directly driving behavior.

### 3. Bias in Node Generation

The 80% exploit / 20% random mixed strategy caused nodes tagged to attention and curiosity to be generated more frequently, and their edges were reinforced into dense clusters.

### 4. Why `observe` Fires Rarely

`observe` has no direct external input; it depends entirely on inflow from attention and curiosity.
Its action (`look_at_object`) never built up enough activation to cross the threshold.
A possible improvement: add direct external input to observe when objects are nearby, or boost it when objects enter the agent's field of view.

---

## Visualization

`echoloop_result.png` contains 4 panels:

| Panel | Contents |
|---|---|
| World (left) | 2D trajectory + color-coded action dots |
| Route Activations (top right) | Route activation over 300 steps, threshold line, user gaze phase shading |
| Internal Route Graph (bottom right, left) | Node/edge structure (edge.strength encoded as alpha) |
| Route Strengths (bottom right, right) | Final strength bar chart |

---

## Next Experiments

- **Boost observe**: Add direct input to observe when objects are in the field of view
- **Asymmetric strength decay**: Amplify negative-reaction decay to make "undesirable" behaviors more clearly penalized
- **Multiple agents**: Interaction between agents that each have their own route dynamics
- **Animated node visualization**: Watch node activation changes step by step as a video
- **With vs. without memory comparison**: Compare action distributions when strength is held fixed
