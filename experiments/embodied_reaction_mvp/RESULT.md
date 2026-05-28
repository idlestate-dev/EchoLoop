# EchoLoop Embodied Reaction MVP — Simulation Results

Run date: 2026-05-29  
Simulation settings: fixed 60 Hz timestep, scenario length 25–30 seconds

---

## Overview

A total of 101 events fired across five scenarios. All events were generated autonomously from path dynamics (accumulation, leakage, threshold overflow, and backpressure); no events were hard-coded to scenario names or triggered by scripts.

| Scenario | Steps | Total events | Types |
|---|---|---|---|
| speech_then_silence | 1,500 | 16 | gaze/settle/nod/response |
| repeated_speech_silence | 1,680 | 22 | gaze×4/settle/nod/response/fidget |
| sudden_approach_while_speaking | 1,500 | 15 | gaze/settle/nod/response/fidget/**freeze** |
| random_ambient_noise | 1,500 | 22 | gaze×5/settle/nod/response |
| long_listening | 1,800 | 22 | gaze/settle/nod/response×7/fidget×2 |

---

## Per-scenario observations

### 1. speech_then_silence

Speech t=2–10 s, followed by 15 seconds of silence.

**Event timeline (excerpt)**

```
t= 0.92  micro_nod_ready    ttp=0.282  (initial accumulation before speech)
t= 2.00  gaze_shift         orient=7.20 (immediate reaction to speech_onset)
t= 3.43  micro_nod_ready    ttp=0.317  (residual recovery keeps threshold mildly elevated)
t= 4.72  posture_settle     settle=0.516
t= 9.73  posture_settle     settle=0.765 (settle approaching saturation after sustained speech)
t=10.90  micro_nod_ready    ttp=0.294  (silence resumes)
t=12.13  response_ready     ttp=0.673
t=16.15  response_ready     ttp=1.000
t=20.17  response_ready     ttp=1.000
t=24.18  response_ready     ttp=1.000
```

**Observations**

- `gaze_shift` fires within one frame of `speech_onset` (orient spike exceeds threshold). `orient` is then reset to 0.2× its value, and the refractory flag prevents a double-fire on the same onset.
- The `settle` path rises from 0 to 0.52 in approximately 2.7 seconds of speech (consistent with the theoretical value). After speech ends, `settle` leaks slowly enough to produce a third `posture_settle` at t=14.75.
- `response_ready` fires at nearly equal 4.02-second intervals — this shows that `ttp_response_refire_gap` (4.0 s) is the dominant pacing factor. Once `ttp` saturates at 1.0, each firing discharges `ttp` to 25% but it recovers to 1.0 before the next firing.

---

### 2. repeated_speech_silence

Four speech blocks (10.5 seconds total) with intermittent silence gaps.

**Notable behaviors**

- `gaze_shift` fires four times, once per speech onset. `orient` spikes to 7.2 on each onset and the refractory state resets during decay, allowing re-firing at the next onset.
- `response_ready` intervals are longer than in `speech_then_silence` (6.1 s, 5.47 s, 6.23 s). Each speech block partially releases `ttp`, lengthening the time needed to rebuild to threshold.
- Near t=19.17, `micro_nod_ready`, `posture_settle`, and `gaze_shift` all fire within 0.35 seconds of each other. The preceding silence had raised both `ttp` and `settle` simultaneously, and the fourth speech onset then pushed `orient` over its threshold at the same moment.
- `fidget_suppression` fires once at t=20.33 (fidget_inhibit=0.929, during speech). The low frequency is expected given the stochastic rate (0.10 × inhibit × dt).

---

### 3. sudden_approach_while_speaking

Speech t=2–13 s, approach_velocity pulse (amplitude 2.5) at t≈13.8 s.

**freeze path values around the approach pulse**

```
t=13.00  freeze_val=0.002  recovery=0.049  ttp=0.009
t=13.40  freeze_val=0.124  recovery=0.036  ttp=0.130
t=13.47  freeze           str=0.204  (threshold 0.20 exceeded → fires)
t=13.53  freeze_val=0.314  recovery=0.406  (0.40 recovery injected)
t=13.80  freeze_val=1.000  (near peak of Gaussian pulse)
...
t=16.12  freeze_val=0.384  recovery=0.051  ← three events fire simultaneously
          posture_settle / micro_nod_ready / response_ready
```

**Observations**

- `freeze` crosses its threshold approximately 0.47 seconds after the approach pulse begins. Immediately after firing, recovery is injected at 0.406, raising the effective thresholds of all other paths.
- For 2.65 seconds after `freeze`, neither `micro_nod_ready` nor `response_ready` fires — freeze is functioning as a suppressor.
- At t=16.117, `posture_settle`, `micro_nod_ready`, and `response_ready` fire within one frame of each other. The simultaneous expiry of `freeze_val` decaying below threshold and the refire timers produces a concentrated release of accumulated pressure. The `response_ready` fires at a notably high strength of ttp=0.969.

---

### 4. random_ambient_noise

Five short speech bursts (0.4–0.7 s each) with continuous low-level energy.

**Observations**

- `gaze_shift` fires five times, once per burst onset. Because each burst is short, `settle` barely accumulates, and `posture_settle` fires only once over the full 25 seconds (at t=22.5).
- The short, frequent speech blocks leave `ttp` only partially discharged, resulting in dense nod/response firing during silence gaps (nod×10, response×6).
- Ambient approach_velocity fluctuations (amplitude 0.12) raise `freeze_val` to a maximum of ~0.10, well below the threshold of 0.20. `freeze` never fires, confirming that the threshold provides a useful buffer against noise.
- `fidget_suppression` does not fire: the speech bursts are too short for `fidget_inhibit` to accumulate before they end.

---

### 5. long_listening

18 seconds of silence (t=0–18 s), brief speech (t=18–21 s), then 9 more seconds of silence.

**ttp trajectory**

```
t= 1.00  ttp=0.308  (nod threshold 0.28 crossed within 1 second from rest)
t= 2.00  ttp=0.615
t= 2.15  response_ready (first, ttp=0.662)
t= 6.17  response_ready (second, ttp=1.000)
t=10.00  ttp=1.000  (saturation reached; stays near 1.0 from here)
t=18.00  ttp=0.993  gaze_shift (speech onset)
t=18.22  response_ready (ttp=0.900 — pressure still high at onset)
t=21.50  ttp=0.220  (rapid release after speech ends)
t=29.00  ttp=0.915  (rebuilt during resumed silence)
```

**Observations**

- During 18 seconds of silence, `response_ready` fires 5 times and `micro_nod_ready` 7 times. `ttp` saturates around t≈10 s; from then on the fire–discharge–rebuild cycle is paced almost entirely by `ttp_response_refire_gap` (4.0 s).
- The `gaze_shift` at t=18.0 followed immediately by `response_ready` at t=18.2 is the most interesting moment: `orient` spikes on speech onset while accumulated `ttp` (0.90) still exceeds the response threshold. Speech onset appears to release a queued response impulse rather than suppress it.
- `fidget_suppression` fires twice during speech (t≈19–20 s). The longer speech block (3 s vs. 8 s in `speech_then_silence`) drove `fidget_inhibit` deeper (≈0.98–0.99), raising the stochastic firing rate enough to produce two events where the shorter scenario produced none.
- The gentle `silence_duration` gain boost (up to 1.2× after 12 s of silence) slightly accelerates `ttp` re-accumulation in the later portion of the long silence.

---

## Cross-scenario patterns

### recovery_backpressure behavior

Mean and maximum recovery value at the moment of each event firing:

| Scenario | mean recovery at fire | max recovery at fire |
|---|---|---|
| speech_then_silence | 0.107 | 0.296 |
| repeated_speech_silence | 0.179 | **0.527** |
| sudden_approach_while_speaking | 0.069 | 0.248 |
| random_ambient_noise | 0.200 | 0.472 |
| long_listening | 0.138 | 0.339 |

- `sudden_approach` has the lowest mean (0.069): the long suppression window after `freeze` allows recovery to decay fully before subsequent events fire.
- `repeated_speech_silence` has the highest max (0.527): rapid cycling across multiple paths leads to events firing while backpressure is still relatively high. This is not a threshold bypass — it reflects the natural overlap when multiple paths converge closely in time.

### Inter-event interval (IEI) statistics

| Scenario | mean IEI | median IEI | min IEI | max IEI |
|---|---|---|---|---|
| speech_then_silence | 1.55 s | 1.28 s | 0.22 s | 5.02 s |
| repeated_speech_silence | 1.28 s | 1.23 s | 0.03 s | 2.65 s |
| sudden_approach_while_speaking | 1.66 s | 1.36 s | 0.00 s | 5.02 s |
| random_ambient_noise | 1.09 s | 0.98 s | 0.07 s | 2.45 s |
| long_listening | 1.35 s | 1.23 s | 0.20 s | 2.52 s |

- The max IEI of 5.02 s in `sudden_approach` corresponds to the suppression gap after `freeze`. The min IEI ≈ 0 corresponds to the three simultaneous events at t=16.12.
- `random_ambient_noise` has the shortest mean IEI (1.09 s): frequent short speech bursts keep `ttp` partially charged, distributing nod events densely across silence gaps.

---

## Verification against design intent

| Intended behavior | Status | Notes |
|---|---|---|
| Speech onset → gaze_shift (immediate) | ✓ | Fires within one frame of orient spike |
| Sustained speech → fidget suppression | ✓ | Mean fidget_inhibit 0.96; fidget_suppression fires at low probability |
| Sustained speech → posture_settle (delayed) | ✓ | First fires ~2.7 s after speech begins |
| Silence → turn-taking pressure build | ✓ | ttp accumulates at 0.30/s; staged nod → response firing |
| Sudden approach → freeze → suppress others | ✓ | nod/response silent for 2.65 s after freeze |
| Post-event → backpressure | ✓ | recovery raises effective thresholds; repeated firing suppressed |
| Fidget not fully eliminated | △ | Stochastic firing works as designed; does not fire during short speech blocks |

---

## Observed limitations and open questions

**Parameter dominance**  
`response_ready` intervals are almost entirely determined by `ttp_response_refire_gap` (4.0 s) once `ttp` saturates. After saturation, it is the parameter rather than path dynamics that sets the pace. Producing more organic interval variation would require revisiting how much `ttp` is discharged per firing and how backpressure interacts with the refire gap.

**Path independence**  
Currently, inter-path coupling is limited to the shared `recovery_backpressure` signal. Cross-path inhibition — for example, a high `orient` temporarily slowing `ttp` accumulation at speech onset — is not implemented.

**`fidget_suppression` density**  
With a stochastic rate of (0.10 × fidget_inhibit × dt), only 0–2 events fire per scenario. The design intent of "not fully eliminated" is technically satisfied, but the density may be too low to meaningfully visualize the texture of suppression.

**Underuse of `silence_duration`**  
The current role of `silence_duration` is a mild gain boost (up to 1.2× at 12 s). The effect is difficult to isolate statistically. A stronger nonlinear role — such as unlocking a separate threshold after a defined duration — would make its contribution more observable.

---

## Candidate next steps

1. Make `ttp_response_refire_gap` dynamic (scaling with the strength of the preceding `ttp`) to introduce more variation in response timing.
2. Add cross-path inhibition between `orient` and `ttp` to briefly hold back turn-taking pressure at the moment speech begins.
3. Feed `speech_energy` into the dynamics (currently logged as a signal but unused by any path).
4. Run statistical comparisons across multiple seeds to assess IEI distribution stability.
5. Qualitative comparison against real conversation timing data.
