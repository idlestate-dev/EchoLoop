# Diagnostic Report: `sudden_approach_while_speaking`

_Simulation output: 1,500 steps · 25.0 s · 15 events total_


---

## 1. What this scenario represents

Speech from t=2 to t=13 s, followed immediately by a brief high-amplitude approach-velocity pulse (Gaussian, peak at t=13.8 s, amplitude 2.5). The scenario tests whether the freeze path correctly interrupts the normal silence-pressure build-up and whether shared backpressure suppresses other events during the freeze window.

**Run stats:** 1,500 steps · 25.0 s total · speaking 44% of the time · longest silence 12.0 s · 1 speech onset(s).

---

## 2. Which paths were primarily activated

| Path | overall mean | mean during speech | mean during silence | peak |
|---|---|---|---|---|
| `orient` | 0.010 | 0.022 | 0.000 | 1.440 | ◀
| `settle` | 0.530 | 0.616 | 0.462 | 0.797 | ◀
| `fidget_inhibit` | 0.467 | 0.971 | 0.070 | 1.000 | ◀
| `ttp` | 0.403 | 0.121 | 0.625 | 1.000 | ◀
| `recovery` | 0.148 | 0.097 | 0.188 | 0.621 | ◀
| `freeze_val` | 0.074 | 0.000 | 0.132 | 1.000 | ◀

- `fidget_inhibit` averaged 0.97 during speech — the inhibition path is charging close to saturation while the user speaks, as designed.
- `settle` averaged 0.62 during speech, indicating sustained listening periods long enough for the slow path to accumulate meaningfully.
- `ttp` averaged 0.62 during silence — high enough to indicate that turn-taking pressure reached or approached saturation.
- `freeze_val` peaked at 1.000 — sufficient to cross the freeze threshold (0.20) and trigger a suppression window.
- `recovery` maintained a mean of 0.148 overall, meaning events were firing frequently enough to keep backpressure elevated for significant periods.

---

## 3. Which output events fired and why

### `gaze_shift` — fired 1 time

_Usually caused by `orienting_fast_path` crossing threshold after `speech_onset`. The path spikes instantly (orient += onset × gain) and decays fast (6/s); the event fires within one frame and the refractory flag prevents double-firing on the same onset._

**Times:** t=2.00  
**Mean strength:** 7.200  

One event per onset (1 onset(s) → 1 firing(s)). The refractory mechanism is working: orient decayed below 15% of threshold between speech blocks, allowing the flag to reset before the next onset.

### `posture_settle` — fired 3 times

_Caused by `listening_settle_slow_path` accumulating during sustained speech. The path saturates toward ~0.81 with a time constant of ~2.7 s; the threshold (0.50) is only crossed after several seconds of continuous speaking._

**Times:** t=4.72, t=9.73, t=16.12  
**Mean strength:** 0.640  

`settle` values at firing: 0.516, 0.765, 0.640. 
1 of 3 `posture_settle` event(s) fired during silence — the slow decay rate (0.07/s) allowed `settle` to persist above threshold well after speech ended.

### `fidget_suppression` — fired 1 time

_Caused by `fidget_inhibition_path` during speaking; fires stochastically at a rate proportional to inhibition strength (0.10 × fidget_inhibit × dt). As a discrete event, density is low (0–2 per scenario) and may be too sparse to meaningfully represent the texture of suppression._

**Times:** t=12.38  
**Mean strength:** 1.000  

`fidget_inhibit` at firing: 1.000. Fired during high-inhibition speech phases as expected, but the stochastic mechanism means this could just as easily have been zero.

### `micro_nod_ready` — fired 6 times

_Caused by moderate turn-taking pressure (`ttp`) building during silence. Fires at the lower threshold (0.28), often as a precursor to `response_ready`. The 2.5 s refire gap spaces out repeated nods._

**Times:** t=0.92, t=3.43, t=16.12, t=18.63, t=21.15, t=23.67  
**Mean strength:** 0.694  

Intervals between nods: 2.52, 12.68, 2.52, 2.52, 2.52 s. Mean 4.55 s, std 4.07 s. 
Higher variance suggests `ttp` fluctuations (from speech or recovery) are actively modulating nod timing.

### `response_ready` — fired 3 times

_Caused by high turn-taking pressure (`ttp`) crossing the upper threshold (0.65). Once `ttp` saturates, this event is largely paced by the 4.0 s refire gap rather than by path dynamics — an identified limitation._

**Times:** t=16.12, t=20.13, t=24.15  
**Mean strength:** 0.990  

`ttp` at firing: 0.969, 1.000, 1.000.  
`recovery` at firing: 0.051, 0.064, 0.130.  

Intervals: 4.02, 4.02 s. Mean 4.02 s, std 0.00 s. 
Near-zero variance: **`ttp_response_refire_gap` (4.0 s) is dominating.** The path dynamics are no longer contributing to timing once `ttp` saturates.

### `freeze` — fired 1 time

_Caused by `freeze_path` crossing threshold (0.20) after a sudden `approach_velocity` spike. Immediately injects a large recovery pulse (0.40), suppressing all other events until backpressure decays._

**Times:** t=13.47  
**Mean strength:** 0.204  

`freeze_val` at firing: 0.204 (threshold 0.20). Recovery injected: ~0.40, creating a suppression window for all downstream events. 
Next non-freeze event fired 2.65 s later.


---

## 4. Expected vs. surprising events

**Expected:**

- `gaze_shift` × 1 — speech onset(s) were present, so an immediate orienting response is expected.
- `posture_settle` — longest speech run was 11.0 s, long enough for `settle` to cross threshold (~2.7 s).
- `micro_nod_ready` — silence exceeded 1 s (max 12.0 s), sufficient for `ttp` to reach the nod threshold (0.28 at 0.30/s).
- `response_ready` — silence exceeded 2.5 s, sufficient for `ttp` to reach the response threshold (0.65).
- `freeze` — approach_velocity peaked at 2.50, large enough to push `freeze_val` above the threshold of 0.20.

**Surprising or noteworthy:**

- **Multi-event burst at t≈16.12 s** — `micro_nod_ready`, `posture_settle`, `response_ready` fired within 0–150 ms of each other. This is a compression-release artifact: multiple paths had accumulated pressure during a suppression window and discharged simultaneously when that window closed.
- **`posture_settle` at t=16.12 s — 3.1 s into silence** (settle=0.640). The slow leak rate (0.07/s) allows `settle` to persist above threshold long after speech stops, producing a gradual 'settling' signal that bridges active listening and rest.
- **Post-freeze release burst** — 2.65 s of silence after `freeze`, then `posture_settle`, `micro_nod_ready`, `response_ready` fired within 0.2 s of each other at t≈16.12 s. Multiple paths accumulated pressure behind the shared backpressure wall and released together once `freeze_val` and recovery both decayed. This was not scripted; it is a direct consequence of shared backpressure.


---

## 5. Path dynamics vs. fixed parameters

**Evidence of path-dynamics-driven behavior:**

- `settle` varied at `posture_settle` firings (0.516–0.765). This reflects genuine variation in how deeply the slow path charged across different speech phases — the settle path is contributing real dynamic information.
- Some events fired with recovery > 0 (max 0.248). Backpressure was actively raising effective thresholds at those moments, confirming that the shared recovery mechanism is influencing event timing beyond the bare refire gaps.
- The `freeze` event and its downstream suppression window involve multiple interacting paths: `freeze_val` accumulates from velocity input, fires and injects recovery, which then holds back `ttp`-driven events until it decays. This is a genuine multi-path interaction — not a scripted pause.

**Evidence of parameter-dominated behavior:**

- `response_ready` intervals: mean 4.02 s, std 0.00 s. The near-zero variance means `ttp_response_refire_gap` (4.0 s) is the sole pacemaker once `ttp` saturates. Path dynamics have effectively stopped contributing to timing at that point.


---

## 6. Which parameter seems most dominant

The most dominant parameter in this scenario is **`ttp_response_refire_gap` (4.0 s)**.

`response_ready` intervals: 4.02, 4.02 s (std 0.00 s). Once `ttp` saturates, this timer alone determines when `response_ready` fires. It is the single parameter with the most direct, measurable influence on output timing in this scenario. Changing it would immediately change the event rhythm.

**Other strong parameter influences:**

- **`orient_gain` (8.0)** — Every `gaze_shift` fired at strength ≈7.20 (= orient_gain × one-frame decay factor). There is no dynamic modulation of gaze intensity; the parameter completely determines event strength.
- **`settle_refire_gap` (5.0 s)** — No two `posture_settle` events fired within 5.02 s, confirming the refire gap is the minimum interval once `settle` remains above threshold.


---

## 7. What should be tuned next

**Introduce variation in `response_ready` timing.** The fixed `ttp_response_refire_gap` (4.0 s) is the only thing pacing this event once `ttp` saturates. Two options: (1) reduce `ttp_response_discharge` from 0.25 to ~0.08 so `ttp` drops further after each firing and takes noticeably longer to rebuild — making early refires impossible and later ones faster; (2) scale the refire gap dynamically with firing strength — a weak firing gets a shorter gap, a strong one gets a longer gap.

**Increase `fidget_base_rate` or reconsider `fidget_suppression` as an event.** With mean inhibition 0.97 during speech, the current rate (0.10/s × inhibit) produced only 1 discrete event(s). Raising `fidget_base_rate` to 0.30–0.50/s would produce a more continuous suppression signal. Alternatively, consider logging `fidget_inhibit` as a continuous output channel rather than a sparse event — it may be better represented as a value than a count.

**Modulate `gaze_shift` strength by `speech_energy` at onset.** Currently every gaze shift fires at the same strength (~7.2), regardless of whether the onset was a confident utterance or a quiet murmur. Scaling orient_gain by `speech_energy` at the onset frame would make gaze shift intensity informative.

**Consider whether `settle_decay` (0.07/s) is too slow.** `posture_settle` fired 1 time(s) during extended silence because `settle` persists above threshold for many seconds after speech ends. If the intended meaning is 'actively settling into listening', the signal should clear more quickly when speaking stops. Try raising `settle_decay` to 0.15–0.20/s.

**Try a graded freeze effect.** Currently `freeze` injects a flat 0.40 recovery pulse, blocking all events equally for ~2.6 s. A more nuanced approach: suppress fast-path events (gaze, fidget) fully but only partially dampen slow-path events (ttp-driven nod/response), so that turn-taking pressure is dampened but not completely frozen. Reduce `recovery_on_freeze` to 0.20 and add a direct `ttp` suppression multiplier.

**Add at least one cross-path interaction.** Currently paths only interact through shared `recovery_backpressure`. A small, targeted coupling — such as `orient` briefly suppressing `ttp` accumulation at speech onset — would make the system more responsive to context and could produce more naturalistic turn-taking timing around speech onset moments.


---

## Main diagnosis

### Working as intended

- **Orienting response** — `gaze_shift` fired exactly once per speech onset (1 → 1). The refractory mechanism prevented double-firing.
- **Settle accumulation** — `settle` built to mean 0.616 during speech, and `posture_settle` fired, confirming the slow-path accumulation mechanism.
- **Fidget inhibition** — `fidget_inhibit` averaged 0.97 during speech, confirming the inhibition path charges correctly while the user is speaking.
- **Turn-taking pressure staging** — `micro_nod_ready` (lower threshold) and `response_ready` (higher threshold) both fired, demonstrating the two-level pressure architecture.
- **Freeze suppression** — after `freeze`, the next event was delayed 2.65 s, confirming that the recovery pulse and `freeze_val` jointly held back downstream events.

### Interesting emergent-looking behavior

- **Simultaneous burst at t≈16.12 s** (`micro_nod_ready`, `posture_settle`, `response_ready`) — multiple paths discharged together after a shared suppression window. This compression-then-release pattern is a genuine emergent consequence of shared backpressure, not a scripted co-occurrence.
- **`posture_settle` at t=16.12 s — 3.1 s into silence** — the slow `settle` decay allows the path to linger above threshold long after speech ends, creating an unscripted gradual transition from active listening to rest.
- **Post-freeze release burst** — 2.65 s of suppression, then 3 events in a tight cluster. The freeze wall held back pressure across multiple paths simultaneously; the burst when it lifted was not scripted.

### Too parameter-dominated / needs tuning

- **`response_ready` timing is a clock** — intervals 4.02, 4.02 s, std 0.00 s. Once `ttp` saturates, the 4.0 s refire gap is the only thing varying the output. This event has stopped behaving like an emergent signal and is functioning as a metronome.
