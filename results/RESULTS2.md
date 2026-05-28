# EchoLoop v2 — Simulation Results

## Hypothesis

> Intelligence is continuous state circulation. External input merely distorts the loop.
> Behavior emerges as a readout of route state.

---

## Changes from v1

| Item | v1 | v2 |
|---|---|---|
| Loop structure | Forward excitation only | **Forward excitation + backward inhibition** |
| Primary driver | External input + reward | **Internal circulation dynamics** |
| Effect of reward | Affects route strength → action selection | **Only fine-tunes forward edge weights by ±0.002/step** |
| Action selection | Picks route with highest `act × strength` | Reads out from route with highest `act` |
| Action distribution | look_at_user 78% (attention-dominated) | **4 actions distributed roughly evenly** |

---

## How the Rotating Wave Works

```
Forward excitation (+0.55):
  attention → observe → curiosity → approach → (→ attention)

Backward inhibition (−0.35):
  observe → attention
  curiosity → observe
  approach → curiosity
  attention → approach
```

When a route activates, it inhibits the preceding route, passing the activation baton to the next one in the cycle.

**Eigenvalue analysis (DECAY=0.85, w_f=0.55, w_b=0.35):**

| Mode | Eigenvalue | Meaning |
|---|---|---|
| Rotating wave (k=1) | √(0.85² + 0.90²) ≈ **1.24** | Rotation grows |
| Sync mode (k=0) | 0.85 + 0.55 − 0.35 = **1.05** | All routes activate simultaneously |

Rotating eigenvalue > sync eigenvalue → **a limit cycle emerges naturally**.

---

## Experiment Design

| Phase | Steps | Contents |
|---|---|---|
| Phase 1 | 0–120 | Normal external input (biased toward attention) |
| Phase 2 (blackout) | 120–220 | **Zero external input** — checking whether internal circulation is self-sustaining |
| Phase 3 | 220–400 | Context switch to curiosity-biased input |

---

## Results (seed=42, 400 steps)

### Loop Persistence Rate

| Phase | Persistence | Interpretation |
|---|---|---|
| Phase 1 (normal) | 99% | Continuously active via external input |
| Phase 2 (blackout) | **100%** | Loop sustains itself even with zero external input |
| Phase 3 (context switch) | 100% | Attractor maintained while content shifts |

### Action Distribution

| Action | Count | Proportion |
|---|---|---|
| `look_at_user` | 130 | **32%** |
| `point` | 100 | 25% |
| `look_at_object` | 89 | 22% |
| `move_toward` | 80 | 20% |
| `idle` | 1 | <1% |

**All 4 actions appeared at roughly equal rates.** Since the loop cycles through each route in turn, the circuit's structure — rather than a learned policy — is what produces behavioral variety.

### Dominant Route Distribution

| Route | Steps | Proportion |
|---|---|---|
| `attention` | 130 | 32% |
| `curiosity` | 100 | 25% |
| `observe` | 90 | 22% |
| `approach` | 80 | 20% |

### Final Loop Edge Weights

| Edge | Initial | Final | Change |
|---|---|---|---|
| attention → observe | +0.550 | **+0.605** | ↑ Reinforced by reward |
| observe → curiosity | +0.550 | **+0.590** | ↑ Slight increase |
| curiosity → approach | +0.550 | **+0.595** | ↑ Slight increase |
| approach → attention | +0.550 | **+0.605** | ↑ Reinforced by reward |
| Backward inhibition (4 edges) | −0.350 | −0.350 | → No change (not subject to reward) |

Backward inhibition does not respond to reward. This is intentional — reward adjusts how easily forward flow occurs, but **the rotation structure itself stays fixed**.

---

## Observations

### 1. Internal Circulation Persists Without External Input

During the blackout period (steps 120–220), even with zero external input, loop persistence was 100%.
This is because the rotating wave grows and sustains itself at eigenvalue 1.24.
The result is consistent with the idea that behavior originates from internal circulation rather than external stimuli.

### 2. The Attractor Holds Through Context Switches

In Phase 3, the external input shifted to favor curiosity, but persistence remained at 100%.
The loop structure appears robust — external input influences which routes are slightly more active, but does not disrupt the circulation rhythm itself.

### 3. Role Differentiation Emerged Naturally

All 4 routes held roughly equal "airtime" (32/25/22/20%).
This role distribution emerged from the circuit's geometry, without any direct policy learning via reward.

### 4. Reward Is Only a Fine-Tuning Signal

- v1: reward → `strength` → multiplied into action score (dominant effect)
- v2: reward → forward edge weight ±0.002/step (slightly adjusts forward excitation only)

Forward edges grew from 0.550 to 0.605, but the rotation structure (backward inhibition) was unchanged.
The design intent — "learning nudges the circulation toward more stable patterns rather than directly selecting actions" — appears to be working.

---

## Visualization (`echoloop_v2_result.png`)

| Panel | Contents |
|---|---|
| Route Activations (top, full width) | Route activation over 400 steps, 3-phase shading |
| Phase portraits (middle left ×3) | Trajectories in attention×observe, curiosity×approach, observe×curiosity space (plasma color = time) |
| Flow Matrix (middle right) | Mean inter-route flow (red = positive, blue = negative, signed heatmap) |
| Loop Persistence (bottom left) | Proportion of steps above threshold |
| Edge Weights (bottom middle-left) | Weight evolution of the 4 forward loop edges |
| World (bottom middle-right) | 2D trajectory + action type dots |
| Action Distribution (bottom right) | Horizontal bar chart |

---

## Next Experiments

- **Vary loop length**: Does rhythm change with 3-node or 6-node loops?
- **Phase reset via external stimulus**: Can a strong input reset the loop's phase?
- **Competing loops**: Can two separate closed loops coexist? Which one dominates?
- **Integration with node graph**: Combine with v1's Node/Edge graph (Hebbian structural learning)
- **Add speak()**: Implement language output as a readout from a specific route in the internal circulation
