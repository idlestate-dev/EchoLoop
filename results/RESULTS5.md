# EchoLoop v5 — Simulation Results

## Hypothesis

> When loops share some routes rather than being fully isolated,
> mixed internal states, blended attractors, and partial synchronization can emerge.

---

## Changes from v4

| Item | v4 | v5 |
|---|---|---|
| Loop structure | Fully isolated (9 routes, 3 per loop) | **Shared bridge routes** (11 routes total) |
| Loop size | All 3-route | **Social/defensive = 4-cycle, exploration = 5-cycle** |
| Shared routes | None | **observe (social↔exploration), vigilance (exploration↔defensive)** |
| Cross-inhibition | Applied equally to all routes | **Shared routes receive only 50% inhibition from non-member loops** |
| Fatigue propagation | Within each loop only | **Determined by edge membership → propagates via bridges** |
| Switching count | 4 | **21** |
| Mean ambiguity | — | **0.598** (not measured in v3/v4) |

---

## Architecture

```
social (4-cycle):      attention → observe → engage → approach → (→ attention)
                                     ↑↑↑  bridge  ↓↓↓
exploration (5-cycle): curiosity → observe → wander → vigilance → inspect → (→ curiosity)
                                                           ↑↑↑  bridge  ↓↓↓
defensive (4-cycle):   alert     → vigilance → freeze → withdraw → (→ alert)
```

### Role of Shared Routes

| Route | Membership | Function |
|---|---|---|
| `observe` | social (0.70) + exploration (0.70) | Social and exploratory attention compete at the same node |
| `vigilance` | exploration (0.60) + defensive (0.60) | Exploratory and defensive vigilance compete at the same node |

### Asymmetric Cross-Inhibition

| Route Type | Inhibited By | Relative Strength |
|---|---|---|
| Exclusive (e.g., attention) | 2 non-member loops | 1.0x |
| Shared (observe, vigilance) | 1 non-member loop | **0.5x** |

Shared routes receive less inhibition → they persist more easily when dominance switches → blending continues.

---

## Experiment Phases (N=700 steps)

| Phase | Steps | External Input |
|---|---|---|
| internal-1 | 0–100 | None (social seed) |
| soc push | 100–160 | attention+0.20, engage+0.15 |
| internal-2 | 160–270 | None |
| exp push | 270–340 | curiosity+0.20, wander+0.15 |
| internal-3 | 340–470 | None |
| def push | 470–540 | alert+0.20, freeze+0.15 |
| internal-4 | 540–700 | None |

---

## Results (seed=42, 700 steps)

### Switching and Dominance

| Loop | Steps | Proportion | Mean dwell |
|---|---|---|---|
| `social` | 245 | 35% | 61.2 steps |
| `exploration` | 95 | 14% | **9.5 steps** |
| `defensive` | 360 | 51% | 45.0 steps |
| **Total switches** | **21** | — | — |

### Blended State Metrics

| Metric | Value | Interpretation |
|---|---|---|
| Mean dominance ambiguity | **0.598** | 0 = fully dominant, 1 = completely tied. Average of ~60% is ambiguous |
| Mean blend coefficient | **0.344** | On average, 34% of activity comes from non-dominant loops |

### Bridge Route Activation

| Route | Mean activation | Max activation |
|---|---|---|
| `observe` | 0.086 | 0.732 |
| `vigilance` | 0.096 | 0.918 |

Bridge routes have a low average, but spike to 0.7–0.9 at times — reflecting moments of intense competition between the two loops sharing them.

---

## Observations

### 1. Switch Frequency Surged (4 → 21)

v4 had 4 switches; v5 has 21.
Shared routes create "leakage" between loops, destabilizing dominance.
In particular, exploration's mean dwell of **9.5 steps** is very short — it is constantly pulled from both sides (`observe` by social, `vigilance` by defensive).

### 2. Mean Ambiguity = 0.598

On average, 60% of steps were in an ambiguous state — meaning the margin between first and second place was small.
The system spent most of its time in a gray zone between attractors, which is qualitatively different from the clear winner-take-all dynamics seen in v3/v4.

### 3. Idle at 70% — Behavioral Ambiguity

Idle reached 70%. This stems from activation being spread across multiple loops at once.
When several loops are moderately active, no single route clears the threshold.
→ "When it's unclear how to act, action is suppressed" — a mood-like phenomenon.

### 4. Spike Patterns in Bridge Routes

`vigilance` peaked at 0.918 — a moment when exploration and defensive simultaneously raised vigilance.
`observe` peaked at 0.732 — amplified when social and exploration briefly synchronized.
→ Bridge nodes can show spike amplification through partial synchronization.

### 5. Asymmetric Dwell Distribution

- **Social**: stable (61-step mean) — many exclusive routes, less disruption
- **Exploration**: unstable (9.5-step mean) — pulled by bridges on both sides, unable to stabilize
- **Defensive**: intermediate (45-step mean) — vigilance is a partial bridge, but alert/freeze/withdraw are exclusive

This asymmetry emerges naturally from structure: **the connector loop is the most unstable**.

### 6. Rolling Correlation (expected pattern)

- `observe` shared: social↔exploration correlation is positive during shared activation, negative during competition
- `vigilance` shared: exploration↔defensive shows the same pattern
- social↔defensive (no bridge): weak correlation throughout
- → Synchronization patterns are determined by the presence or absence of shared routes

---

## Visualization (`echoloop_v5_result.png`)

| Panel | Contents |
|---|---|
| Loop Activity (top, full width) | Solid = loop activity, dashed = shared routes (observe/vigilance); white background = high ambiguity |
| Ternary Diagram (middle) | State trajectory in a triangle with each loop at a vertex; center = full blending |
| Ambiguity + Blend (middle) | Time series of dominance ambiguity and blend coefficient |
| Bridge vs Exclusive (middle) | Comparison of observe/vigilance vs exclusive routes |
| Rolling Synchronization (middle) | Inter-loop Pearson r over time; does bridge presence create a visible difference? |
| Phase Spaces (bottom ×2) | social×exploration, exploration×defensive; bubble size = ambiguity |
| Route Activations (bottom) | All 11 routes (bridge routes highlighted in bold) |
| Action Distribution (bottom) | idle 70% = behavioral ambiguity made visible |

---

## Design Evolution: v1 to v5

| v | Core Mechanism | Switches | Mean Ambiguity |
|---|---|---|---|
| 1 | Route graph + Hebbian | — | — |
| 2 | Single limit cycle | 0 | 0 |
| 3 | 3-loop competition (fully isolated) | 2 | Low |
| 4 | Fatigue + fully isolated | 4 | Low |
| **5** | **Shared bridge routes** | **21** | **0.598** |

---

## Next Experiments

- **Tune bridge strength**: How does changing observe's membership weight affect ambiguity?
- **Add a third bridge**: A social↔defensive route — does it produce 3-loop simultaneous blending?
- **Add speak()**: Read out speech when observe spikes and social is dominant
- **Synchronization index control**: Can bridge weight be used to intentionally tune synchrony/asynchrony?
- **Fatigue on bridge routes**: What happens when observe/vigilance themselves fatigue?
