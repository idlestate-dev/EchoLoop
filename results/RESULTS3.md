# EchoLoop v3 — Simulation Results

## Hypothesis

> Multiple internal closed loops compete for dominance through mutual inhibition.
> External input only triggers attractor switching.
> Behavior emerges as a natural byproduct of reading out the dominant loop's state.

---

## Changes from v2

| Item | v2 | v3 |
|---|---|---|
| Number of loops | 1 (attention→observe→curiosity→approach) | **3 competing loops** |
| Loop type | Single limit cycle | social / exploration / defensive |
| Inter-loop relationship | None | **Mutual inhibition (CROSS=0.30)** |
| External input | Added to individual routes | Injected as per-loop bias |
| Observation goal | Autonomous persistence of limit cycle | **Attractor switching / hysteresis** |

---

## Architecture

### 3 Competing Loops

```
social loop:
  attention → engage → approach → (→ attention)
  (backward inhibition: engage→attention, approach→engage, attention→approach)

exploration loop:
  curiosity → wander → inspect → (→ curiosity)

defensive loop:
  alert → freeze → withdraw → (→ alert)
```

### Cross-Inhibition Between Loops

```python
# Each loop's routes are suppressed in proportion to total activity of competing loops
for loop in loops:
    competing_activity = sum(loop_act[other] for other in loops if other != loop)
    for route in loop.routes:
        delta[route] -= CROSS * competing_activity  # CROSS = 0.30
```

### Conditions for Autonomous Persistence (same eigenvalues as v2)

| Mode | Eigenvalue | Result |
|---|---|---|
| Rotating wave | √(0.85² + 0.90²) ≈ **1.24** | Each loop self-sustains in isolation |
| Cross-inhibition | CROSS=0.30 > net_gain=0.05 | Non-dominant loops are suppressed |
| Seeding | ext=0.35 > CROSS=0.30 | External input can trigger a switch |

---

## Experiment Phases

| Phase | Steps | External Input | Intent |
|---|---|---|---|
| internal | 0–70 | None | Self-sustained from initial seed (social dominant) |
| exp trigger | 70–150 | curiosity/wander/inspect += 0.35 | Force transition to exploration attractor |
| decay | 150–230 | None | Observe hysteresis after input is removed |
| def trigger | 230–290 | alert/freeze/withdraw += 0.35 | Transition to defensive attractor |
| hysteresis | 290–360 | None | Does defensive persist? |
| mixed | 360–400 | social+0.12 / exploration+0.14 / defensive+0.08 | Metastable behavior under competing signals |

---

## Results (seed=42, 400 steps)

### Attractor Switching

| Phase | Dominant Loop | Duration | Cause of Switch |
|---|---|---|---|
| 0–76 | **social** | 77 steps | Initial seed (social loop started at 0.32) |
| 77–236 | **exploration** | 160 steps | exp trigger (step 70) seeding → delayed switch |
| 237–399 | **defensive** | 163 steps | def trigger (step 230) seeding |
| Total switches | 2 | — | Clean attractor transitions |

### Notable: Delayed Switching

- exp trigger started: step **70**
- Actual switch: step **77** (7-step delay)
- Why: The external input (0.35) only slightly exceeds the social loop's cross-inhibition (0.30), so the seed takes time to grow. The sequence is: seeding → within-loop self-amplification → bifurcation.

### Hysteresis Confirmed

- def trigger ended: step **290**
- Defensive loop continued for **163 steps** after step 290
- Even during the mixed phase (360–400) with weak social/exploration input, defensive remained dominant
- → **The attractor persists after the input is removed** (hysteresis ✓)

### Loop Dominance Distribution

| Loop | Steps | Proportion |
|---|---|---|
| `social` | 77 | 19% |
| `exploration` | 160 | **40%** |
| `defensive` | 163 | **41%** |

### Action Distribution

| Action | Proportion | Originating Loop |
|---|---|---|
| `scan` | 41% | defensive |
| `look_around` | 38% | exploration |
| `look_at_user` | 14% | social |
| others | 7% | — |

Action proportions closely track loop dominance durations — supporting the view that **behavior is a readout of loop state**.

---

## Observations

### 1. Attractor Switching (confirmed)

Transitions occurred in the order social → exploration → defensive.
Each switch followed a 3-stage process: external seeding → within-loop self-amplification → bifurcation.
Switches were sharp (roughly 7 steps) — a characteristic of nonlinear dynamics.

### 2. Hysteresis (confirmed)

Defensive dominance persisted after the trigger ended.
This "residual dominance after input is gone" suggests that the attractor forms a deep potential well.

### 3. Nonlinear Threshold Behavior

External input below 0.30 (the CROSS value) produced no switch.
Above 0.35, seeding began and the switch was rapid.
→ **Threshold response to input strength** — a characteristic of phase transitions.

### 4. Emergent Action Selection

No action policy was directly learned, yet each phase produced contextually appropriate actions: social → eye contact, exploration → looking around, defensive → scanning.
Behavior emerges naturally from what the dominant loop *is*.

---

## Visualization (`echoloop_v3_result.png`)

| Panel | Contents |
|---|---|
| Loop Activity (top, full width) | Activity of 3 loops over time; background color shows dominant loop |
| Phase Space social×exploration (middle) | Switching trajectory between attractors; ★ = theoretical attractor positions |
| Phase Space exploration×defensive (middle) | Same |
| Dominance Raster (middle) | Dominant loop per step, color-coded |
| Loop Entropy (middle) | Dominance uncertainty (spikes at transition points) |
| Transition Matrix (bottom) | Number of transitions between loops (2 total: social→exp, exp→def) |
| Dwell Times (bottom) | Distribution of time spent in each loop |
| Route Activations (bottom) | Activation of all 9 routes over time |
| Action Distribution (bottom) | Actions color-coded by loop |

---

## Next Experiments

- **Tune competitive balance**: Does lowering CROSS lead to spontaneous switching? (metastable state)
- **Asymmetric CROSS**: social→exploration inhibition ≠ exploration→social → directional transitions
- **More loops**: Do 4–5 loops produce more complex dominance patterns?
- **Add speak()**: Read out speech when the social loop is dominant
- **Noise sensitivity**: How does NOISE magnitude relate to spontaneous switching frequency?
- **Integration with v1 Node/Edge structure**: Place loop routes as physical nodes in 2D space
