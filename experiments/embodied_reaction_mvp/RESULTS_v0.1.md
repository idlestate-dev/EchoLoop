# EchoLoop Embodied Reaction MVP — v0.1 Results

Checkpoint date: 2026-05-29  
Branch: `embodied-simulation-mvp`  
Simulation settings: fixed 60 Hz, scenario durations 25–30 s, seed 42

---

## What this experiment explores

EchoLoop Embodied Reaction MVP is a small exploratory model for studying how simple path dynamics can generate nontrivial interaction timing without explicit emotional labels or scripted state transitions.

It is not a model of emotion, affect, consciousness, or biological behavior.  It does not claim life-like or human-like reaction.

The intended distinction from LLM-based systems:

- **LLMs** handle semantic reasoning and dialogue content.
- **EchoLoop** explores a lower-level reaction timing layer: orienting, settling, turn-taking pressure, freeze, recovery, and delayed release.  These operate on quantities — pressure, inhibition, delay, overflow, readiness, suppression, settling — not on emotional labels.

The system is headless (no rendering), deterministic given a seed, and produces output events that emerge purely from path dynamics.  No output event is hard-coded to a scenario name.

---

## What changed from the initial MVP (v0.0 → v0.1)

The initial MVP had a structural timing problem: `response_ready` behaved like a metronome.  Once `ttp` (turn-taking pressure) saturated at 1.0, the 4-second refire gap became the sole pacing factor.  Path dynamics stopped contributing to interval timing.

v0.1 addresses this with three coordinated changes to `echoloop.py`:

**1. State-dependent discharge**  
Previously, `response_ready` discharged `ttp` by a fixed fraction (`ttp × 0.25`), regardless of how saturated `ttp` was.  Now, excess pressure above the response threshold drains additionally:

```
excess  = max(0, ttp − threshold)
new_ttp = max(0, ttp × 0.25 − excess × 0.45)
```

At `ttp = 1.0`, this drops `ttp` to ~0.09.  At `ttp = 0.65` (threshold), it drops to ~0.16.  Discharge depth is now a continuous linear function of firing state.

**2. Dynamic refire gap**  
The fixed 4.0 s refire gap is replaced by a state-computed gap:

```
gap = 1.8 + recovery × 2.5 − min(1, silence_duration / 15) × 0.50
gap = max(1.0, gap)
```

When recent event activity has left elevated backpressure, the gap extends.  Sustained silence compresses it slightly.  This means no two firings necessarily produce the same wait time.

**3. Recovery suppresses `ttp` rebuild**  
`ttp` now rebuilds more slowly when backpressure is high:

```
rebuild_rate = ttp_gain × silence_factor × (1 − recovery × 0.25)
```

This adds a third source of interval variation, independent of the discharge amount and the timer.

---

## Key findings

### ttp saturation

**ttp saturation is 0% across all five scenarios.**  `ttp` no longer pins at 1.0.  This is the most direct confirmation that the metronome mechanism has been mitigated: when `ttp` never reaches the ceiling, it always has room to vary, and path dynamics always contribute to the next firing interval.

### State-dependent discharge sanity check

The correlation between `ttp` at firing and discharge depth is `r = +1.00` in every scenario.  This is not a surprising emergent result — it is mathematically guaranteed because the discharge formula is linear in `ttp`.  It functions as a sanity check confirming the rule is applied correctly.

### Inter-event interval results

| Scenario | n firings | IEI mean | IEI std | IEI min | IEI max | Clock-like? |
|---|---|---|---|---|---|---|
| `speech_then_silence` | 7 | 1.98 s | 0.16 s | 1.82 s | 2.18 s | No |
| `repeated_speech_silence` | 5 | 5.07 s | 1.88 s | 2.30 s | 6.37 s | No |
| `sudden_approach_while_speaking` | 5 | 1.97 s | 0.05 s | 1.92 s | 2.02 s | **Flagged** |
| `random_ambient_noise` | 8 | 3.08 s | 0.74 s | 2.05 s | 3.92 s | No |
| `long_listening` | 12 | 2.47 s | 1.35 s | 1.83 s | 6.50 s | No |

The 6.50 s outlier in `long_listening` corresponds to the speech block at t = 18–21 s discharging `ttp` from ~0.45 to ~0.19.  This is genuine dynamic behavior — speech interrupting the silence build-up — not a timer artifact.

### Recovery modulation

| Scenario | r(recovery → next IEI) | Interpretation |
|---|---|---|
| `speech_then_silence` | +0.65 | Moderate — recovery is a meaningful predictor of next interval |
| `repeated_speech_silence` | −0.97 | Strong but likely confounded — high recovery at the 4th firing coincides with end of speech-heavy phase; 4 data points make this unstable |
| `sudden_approach_while_speaking` | +0.95 | Strong — within the narrow range, recovery modulates the interval; the narrow range itself is the structural problem |
| `random_ambient_noise` | +0.12 | Negligible — silence gap length between bursts is the dominant driver here |
| `long_listening` | −0.13 | Negligible — the speech block (6.50 s outlier) dominates the variance |

### What `response_ready` behavior looks like now

The original problem:

```
ttp saturates at 1.0
→ response_ready is paced by fixed refire gap (4.0 s)
→ intervals: 4.02, 4.02, 4.02, ...
```

Current v0.1 behavior:

```
response_ready fires
→ ttp discharges in a state-dependent way (deeper when more saturated)
→ recovery slows ttp rebuild
→ silence structure and speech blocks modulate the next firing interval
→ ttp never saturates
→ response_ready interval now reflects path state, not just a timer
```

This is accurately described as **metronome issue mitigated**, not fully solved forever.  The system can still produce clustered or nearly-uniform intervals in specific structural conditions (see: freeze synchronization below).

---

## Remaining known limitations

### Freeze-specific global synchronization

`sudden_approach_while_speaking` still shows low IEI variance (std = 0.05 s) and is flagged as synchronized.  This is **not** the same as the original `ttp` metronome problem.

**Cause:** the `freeze` event injects a fixed-size recovery pulse (0.40) that suppresses all paths simultaneously.  When the suppression window clears, every path resumes from a similar initial state.  The subsequent `ttp` rebuild and `response_ready` firing dynamics therefore converge each cycle, producing near-uniform intervals.

This should be called **freeze-specific global synchronization**, not a `response_ready` metronome failure.

**Potential future fixes (not implemented in v0.1):**

1. Make the freeze recovery pulse proportional to `freeze_val` at firing (graded, not fixed)
2. Affect different paths differently during freeze: strongly suppress `gaze` / `fidget`, moderately dampen `ttp`, barely touch `settle`
3. Discharge each path by a different fraction at freeze time to desynchronize their rebuild trajectories

### Other known limitations

- `gaze_shift` strength is nearly fixed: every firing has strength ≈ `orient_gain × decay_factor`, regardless of speech energy at onset.
- `fidget_suppression` is modeled as a sparse stochastic event; at the current rate (0.10 × fidget_inhibit × dt), it fires 0–2 times per scenario.  It may be better represented as a continuous `fidget_level` output.
- No lateral inhibition between paths beyond the shared `recovery_backpressure` signal.
- No sensory noise model.
- No persistent state between scenario runs.
- Signal shapes are synthetic.
- Parameters were tuned by inspection; no optimization or fitting has been done.
- No baseline comparison (random generator, simple threshold model, FSM) and no ablation study yet.

---

## Full scenario results

### speech_then_silence

Speech t = 2–10 s, 15 s silence.  7 response_ready events.

`ttp` at firing: 0.668–0.801 (mean 0.744, std 0.048).  
Post-discharge `ttp`: 0.137–0.163 (mean 0.148).  
Recovery at firing: 0.076–0.240 (mean 0.174).  
IEI std 0.16 s — recovery is the primary modulator (r = +0.65).

### repeated_speech_silence

Four speech blocks (10.5 s total).  5 response_ready events.

Long intervals (5–6 s) correspond to windows where speech blocks are discharging `ttp` mid-silence.  Final interval (2.30 s) is shorter because the last silence phase is uninterrupted.  IEI std 1.88 s — speech structure is the dominant source of variation.

### sudden_approach_while_speaking

Speech t = 2–13 s, approach pulse (amplitude 2.5) at t ≈ 13.8 s.  5 response_ready events.

First firing after freeze suppression window (t ≈ 16.1 s): `ttp = 0.931`, discharge to 0.111.  Subsequent firings: `ttp` 0.678–0.781.  IEI std 0.05 s — flagged.  Freeze synchronization is the structural cause.

Recovery correlates strongly with interval (r = +0.95), and silence duration also correlates (r = +0.93), but the modulation acts over a narrow range bounded by the synchronized release window.

### random_ambient_noise

Five short speech bursts (0.4–0.7 s each).  8 response_ready events.

Varying silence gaps between bursts produce genuine interval variation (IEI std 0.74 s, range 2.05–3.92 s).  Recovery correlation is weak (r = +0.12) — the primary driver is silence gap length between bursts rather than backpressure state.

### long_listening

18 s silence, speech t = 18–21 s, silence again.  12 response_ready events.

Six seconds of intervals cluster around 1.9–2.2 s during the long initial silence; the 6.50 s gap corresponds to the speech block discharging `ttp`.  IEI std 1.35 s — speech block is the dominant source of variance.  Post-speech recovery (final 9 s) produces a new cluster at 2.1–2.2 s.

---

## Future roadmap

### v0.2 — Modulate gaze_shift strength

- Scale `orient_gain` by `speech_energy` at onset frame.
- Optionally discount by current `recovery` level.
- Goal: gaze shift intensity becomes informative about onset conditions, not a fixed spike amplitude.

### v0.3 — Replace fidget_suppression with continuous fidget_level

- Deprecate the sparse stochastic `fidget_suppression` event.
- Log `fidget_inhibit` directly as a continuous output channel `fidget_level`.
- Goal: suppression texture is visible as a signal, not a count.

### v0.4 — Graded freeze

- Replace fixed freeze recovery pulse with a pulse proportional to `freeze_val` at firing.
- Suppress `gaze` / `fidget` fully, `ttp` partially (multiplier ~0.5), `settle` minimally.
- Goal: break freeze-specific global synchronization; paths resume from different states.

### v0.5 — Baseline comparisons

Add comparison runs against:

- Random event generator (uniform rate)
- Simple threshold model (single accumulator, no backpressure)
- Markov / FSM model (state machine with fixed transition probabilities)
- EchoLoop full model (all six paths)

Goal: establish that EchoLoop timing is statistically distinguishable from simpler generators, and quantify how much complexity each mechanism contributes.

### v0.6 — Ablation studies

Run the simulation with individual mechanisms disabled:

- Without `recovery_backpressure` (set all recovery pulse sizes to 0)
- Without dynamic discharge (revert to fixed `ttp × 0.25`)
- Without `freeze_path` (zero out approach velocity)
- Without cross-path coupling (when added in future versions)

Goal: isolate each mechanism's contribution to timing variance.
