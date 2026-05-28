# EchoLoop v4 — Simulation Results

## Hypothesis

> State history itself changes the dynamics.
> The dominant loop fatigues and gradually loses the ability to sustain itself.
> Non-dominant loops slowly recover and become candidates for the next round.
> Something resembling mood-like dynamics emerges from internal state alone.

---

## Changes from v3

| Item | v3 | v4 |
|---|---|---|
| Attractor | Static structure | **Changes based on usage history** |
| Loop stability | No fatigue (fixed eigenvalues throughout) | **Fatigue dynamically changes the effective gain** |
| Switch trigger | External input only | **Fatigue + external input + noise** |
| Spontaneous switching | None (no switch without external input) | **Present (internal fatigue causes spontaneous collapse)** |

---

## Fatigue Mechanism

```python
# Each step:
fatigue[l] += FATIGUE_RATE * loop_activity[l]  # accumulates in proportion to activity
fatigue[l] *= FATIGUE_DECAY                     # all loops recover gradually

# Applied to forward edges:
eff_w = W_FWD * max(MIN_GAIN, 1.0 - fatigue[l])

# Effect on rotating eigenvalue:
|λ_rot(f)| = sqrt(DECAY^2 + (eff_w + W_BWD)^2)
```

### Fatigue vs. Eigenvalue

| fatigue | eff_W_fwd | \|λ_rot\| | State |
|---|---|---|---|
| 0.0 | 0.550 | 1.238 | Healthy, self-sustaining |
| 0.4 | 0.330 | 1.089 | Mildly fatigued |
| 0.6 | 0.220 | 1.023 | Heavily fatigued |
| **0.8** | 0.110 | **0.967** | **Collapse threshold** |
| 1.0 | 0.028 | 0.930 | Unable to self-sustain |

When fatigue exceeds 0.80, `|λ_rot| < 1` → the loop collapses autonomously and loses dominance.

---

## Experiment Phases (N=800 steps)

| Phase | Steps | External Input | Intent |
|---|---|---|---|
| internal-1 | 0–120 | None | Social start |
| exp trigger | 120–160 | exploration routes += 0.22 | Push toward exploration |
| internal-2 | 160–450 | None | **Long pure-internal wandering** |
| def trigger | 450–490 | defensive routes += 0.22 | Push toward defensive |
| internal-3 | 490–700 | None | Observe stuck state / recovery |
| mixed | 700–800 | Weak mixed input | Metastable behavior |

---

## Results (seed=42, 800 steps)

### Switching History

| Period | Dominant Loop | Length | Cause |
|---|---|---|---|
| 0–119 | **social** | 120 steps | Initial seed |
| 120–204 | **exploration** | 85 steps | exp trigger (120–160) seeding |
| 205–404 | **social** | 199 steps | exploration fatigued → social recovered and reclaimed dominance |
| 405–495 | **exploration** | 91 steps | social fatigued after 199-step stretch → collapsed |
| 496–800 | **defensive** | 305 steps | def trigger (450–490) seeding → **stuck state** |

### Loop Dominance Distribution

| Loop | Steps | Proportion |
|---|---|---|
| `social` | 319 | 40% |
| `exploration` | 176 | 22% |
| `defensive` | 305 | **38%** |

### Dwell Times

| Loop | Count | Mean | Min | Max |
|---|---|---|---|---|
| social | 2 | **159.5** steps | 120 | 199 |
| exploration | 2 | 88.0 steps | 85 | 91 |
| defensive | 1 | **305.0** steps | — | — |

### Final Fatigue State

| Loop | Fatigue | Interpretation |
|---|---|---|
| social | 0.213 | Recovering from exhaustion |
| exploration | 0.404 | Moderate fatigue (residual from the 91-step dominance period) |
| defensive | 0.154 | Still dominant (accumulated fatigue < 0.80) |

---

## Observations

### 1. Spontaneous Switching (confirmed)

**Step 205**: the exploration loop collapsed spontaneously, allowing social to re-emerge.
External input had ended at step 160, so this switch was driven entirely by exploration's accumulated fatigue — not any external event.
→ Spontaneous switching can occur without external input.

**Step 405**: the social loop collapsed after 199 steps of dominance due to fatigue.
This also happened in the pure-internal phase (internal-2), with no external input.

### 2. Stuck State / Hysteresis (confirmed)

The defensive loop held continuous dominance from step 496 to 800 — **305 steps**.
During this period, social and exploration had accumulated enough fatigue that recovery was slow.
Even during the mixed phase (700–800) with weak social/exploration input, the defensive loop held.
→ A specific loop can become "stuck" in dominance.

### 3. Asymmetric Dwell Times

social mean dwell (159.5 steps) >> exploration mean dwell (88.0 steps).
Social starts with high activation from the initial seed, giving it more resilience against fatigue.
Exploration always begins from a freshly seeded state, reaching the fatigue threshold more quickly.
→ Initial conditions and fatigue tolerance together shape the asymmetry in dwell time.

### 4. Idle at 65% — Low-Energy Transitional States

`idle 65%` in the action distribution reflects the many steps when routes fall below the activation threshold (0.38) — during transitions and periods of fatigue.
In v3, idle was under 1%; in v4, fatigue reduces peak activation values, causing frequent threshold misses.
This is part of the mood-like dynamics: a fatigued state leads to reduced action output.

### 5. Dynamically Changing Effective Gain

The dominant loop's effective rotating gain (`|λ| − 1`) gradually decreases.
When it drops below zero, the loop can no longer sustain itself → a spontaneous switch occurs.
At that moment, competing loops are still recovering and have relatively higher gain → a new dominant emerges.
→ A "tired attractor gives way to a fresh one" pattern appears from internal state alone.

---

## Visualization (`echoloop_v4_result.png`)

| Panel | Contents |
|---|---|
| Loop Activity + Fatigue (top, full width) | Solid = loop activity, dashed = fatigue level (right axis), dotted = fatigue threshold 0.80 |
| Effective Gain (middle) | \|λ_rot\|−1 over time; values below 0 indicate collapse; aligned with switch events |
| Dominance Raster (middle) | Dominant loop per step; switch points marked with white lines |
| Dwell Distribution (middle) | Histogram of dominance durations per loop |
| Entropy (middle) | Uncertainty over time; spikes at switches, drops during stuck states |
| Phase Space social×exploration (bottom) | Bubble size = fatigue of dominant loop; ★ = theoretical attractor |
| Phase Space exploration×defensive (bottom) | Same |
| Route Activations (bottom) | All 9 routes over time |
| Action Distribution (bottom) | High idle rate = many low-activity periods due to fatigue |

---

## Design Evolution: v1 to v4

| Version | Core Mechanism | Key Observation |
|---|---|---|
| v1 | Route graph + Hebbian reinforcement | Attention gains dominance through reward |
| v2 | Single closed loop + rotating wave | Limit cycle self-sustains; 100% persistence even during blackout |
| v3 | 3-loop competition + cross-inhibition | Attractor switching, hysteresis |
| **v4** | **Fatigue-driven dynamic landscape** | **Spontaneous switching, stuck states, mood-like oscillation** |

---

## Next Experiments

- **Asymmetric fatigue recovery**: Social recovers quickly (sociability bounces back); defensive recovers slowly
- **Variable loop length**: 2-node loops (faster cycling) vs. 5-node loops (slower)
- **Add speak()**: Read out speech when social loop is dominant and fatigue < 0.4
- **Fatigue contagion**: Dominant loop fatigue propagates to neighboring loops via shared routes
- **Input intensity effects**: Very strong external input interferes with fatigue recovery
