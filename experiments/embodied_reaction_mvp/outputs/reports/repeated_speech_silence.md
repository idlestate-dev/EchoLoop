# Diagnostic Report: `repeated_speech_silence`

_Simulation output: 1,680 steps · 28.0 s · 22 events total_


---

## 1. What this scenario represents

Four speech blocks (t=1.5–4.5, 7–10.5, 13–16, 19.5–22.5 s) separated by silence gaps of roughly 2.5–3 s. The alternating pattern tests whether orienting and settle reset and re-activate correctly across cycles, and whether turn-taking pressure can be partially discharged by speech and rebuild again across repeated gaps.

**Run stats:** 1,680 steps · 28.0 s total · speaking 45% of the time · longest silence 5.5 s · 4 speech onset(s).

---

## 2. Which paths were primarily activated

| Path | overall mean | mean during speech | mean during silence | peak |
|---|---|---|---|---|
| `orient` | 0.034 | 0.077 | 0.000 | 1.440 | ◀
| `settle` | 0.561 | 0.574 | 0.551 | 0.741 | ◀
| `fidget_inhibit` | 0.537 | 0.903 | 0.241 | 1.000 | ◀
| `ttp` | 0.357 | 0.245 | 0.447 | 1.000 | ◀
| `recovery` | 0.225 | 0.251 | 0.203 | 0.697 | ◀
| `freeze_val` | 0.000 | 0.000 | 0.000 | 0.000 |

- `fidget_inhibit` averaged 0.90 during speech — the inhibition path is charging close to saturation while the user speaks, as designed.
- `settle` averaged 0.57 during speech, indicating sustained listening periods long enough for the slow path to accumulate meaningfully.
- `ttp` averaged 0.45 during silence, showing meaningful pressure build-up but without saturating.
- `recovery` maintained a mean of 0.225 overall, meaning events were firing frequently enough to keep backpressure elevated for significant periods.

---

## 3. Which output events fired and why

### `gaze_shift` — fired 4 times

_Usually caused by `orienting_fast_path` crossing threshold after `speech_onset`. The path spikes instantly (orient += onset × gain) and decays fast (6/s); the event fires within one frame and the refractory flag prevents double-firing on the same onset._

**Times:** t=1.50, t=7.00, t=13.00, t=19.50  
**Mean strength:** 7.200  

One event per onset (4 onset(s) → 4 firing(s)). The refractory mechanism is working: orient decayed below 15% of threshold between speech blocks, allowing the flag to reset before the next onset.

### `posture_settle` — fired 5 times

_Caused by `listening_settle_slow_path` accumulating during sustained speech. The path saturates toward ~0.81 with a time constant of ~2.7 s; the threshold (0.50) is only crossed after several seconds of continuous speaking._

**Times:** t=4.15, t=9.17, t=14.18, t=19.20, t=24.22  
**Mean strength:** 0.616  

`settle` values at firing: 0.509, 0.653, 0.675, 0.592, 0.651. 
2 of 5 `posture_settle` event(s) fired during silence — the slow decay rate (0.07/s) allowed `settle` to persist above threshold well after speech ended.

### `fidget_suppression` — fired 1 time

_Caused by `fidget_inhibition_path` during speaking; fires stochastically at a rate proportional to inhibition strength (0.10 × fidget_inhibit × dt). As a discrete event, density is low (0–2 per scenario) and may be too sparse to meaningfully represent the texture of suppression._

**Times:** t=20.33  
**Mean strength:** 0.929  

`fidget_inhibit` at firing: 0.929. Fired during high-inhibition speech phases as expected, but the stochastic mechanism means this could just as easily have been zero.

### `micro_nod_ready` — fired 8 times

_Caused by moderate turn-taking pressure (`ttp`) building during silence. Fires at the lower threshold (0.28), often as a precursor to `response_ready`. The 2.5 s refire gap spaces out repeated nods._

**Times:** t=0.92, t=5.08, t=11.18, t=16.65, t=19.17, t=22.87, t=25.38, t=27.90  
**Mean strength:** 0.447  

Intervals between nods: 4.17, 6.10, 5.47, 2.52, 3.70, 2.52, 2.52 s. Mean 3.85 s, std 1.37 s. 
Higher variance suggests `ttp` fluctuations (from speech or recovery) are actively modulating nod timing.

### `response_ready` — fired 4 times

_Caused by high turn-taking pressure (`ttp`) crossing the upper threshold (0.65). Once `ttp` saturates, this event is largely paced by the 4.0 s refire gap rather than by path dynamics — an identified limitation._

**Times:** t=6.32, t=12.42, t=17.88, t=24.12  
**Mean strength:** 0.670  

`ttp` at firing: 0.676, 0.667, 0.668, 0.669.  
`recovery` at firing: 0.091, 0.073, 0.074, 0.071.  

Intervals: 6.10, 5.47, 6.23 s. Mean 5.93 s, std 0.33 s. 
Variance present: speech, recovery, and `ttp` discharge are all contributing to interval variability — dynamics are active.

### `freeze` — did not fire

_Caused by `freeze_path` crossing threshold (0.20) after a sudden `approach_velocity` spike. Immediately injects a large recovery pulse (0.40), suppressing all other events until backpressure decays._

No significant approach-velocity input was present. `freeze_val` peaked at only 0.000, well below the threshold of 0.20.


---

## 4. Expected vs. surprising events

**Expected:**

- `gaze_shift` × 4 — speech onset(s) were present, so an immediate orienting response is expected.
- `posture_settle` — longest speech run was 3.5 s, long enough for `settle` to cross threshold (~2.7 s).
- `micro_nod_ready` — silence exceeded 1 s (max 5.5 s), sufficient for `ttp` to reach the nod threshold (0.28 at 0.30/s).
- `response_ready` — silence exceeded 2.5 s, sufficient for `ttp` to reach the response threshold (0.65).

**Surprising or noteworthy:**

- **Multi-event burst at t≈19.17 s** — `micro_nod_ready`, `posture_settle` fired within 33–150 ms of each other. This is a compression-release artifact: multiple paths had accumulated pressure during a suppression window and discharged simultaneously when that window closed.
- **`posture_settle` at t=19.20 s — 3.2 s into silence** (settle=0.592). The slow leak rate (0.07/s) allows `settle` to persist above threshold long after speech stops, producing a gradual 'settling' signal that bridges active listening and rest.


---

## 5. Path dynamics vs. fixed parameters

**Evidence of path-dynamics-driven behavior:**

- `response_ready` intervals: mean 5.93 s, std 0.33 s. The variance shows that `ttp` did not remain saturated throughout — speech discharge or recovery cycles were actively pulling it back down, creating genuine dynamic modulation of response timing.
- `settle` varied at `posture_settle` firings (0.509–0.675). This reflects genuine variation in how deeply the slow path charged across different speech phases — the settle path is contributing real dynamic information.
- Some events fired with recovery > 0 (max 0.527). Backpressure was actively raising effective thresholds at those moments, confirming that the shared recovery mechanism is influencing event timing beyond the bare refire gaps.
- `ttp` was not fully saturated at some `response_ready` firings (min 0.667). Speech blocks had discharged pressure enough to keep `ttp` below ceiling, meaning the threshold crossing reflects genuine path state rather than a guaranteed saturation.

**Evidence of parameter-dominated behavior:**

- `gaze_shift` strength range: 7.20–7.20. All orientings fired at nearly identical intensity, set entirely by `orient_gain` (8.0). There is no dynamic modulation; a loud onset and a quiet one look identical.


---

## 6. Which parameter seems most dominant

The most dominant parameter in this scenario is **`orient_gain` (8.0)**.

Every `gaze_shift` fired at strength ≈7.20 (= orient_gain × one-frame decay factor). There is no dynamic modulation of gaze intensity; the parameter completely determines event strength.

**Other strong parameter influences:**

- **`settle_refire_gap` (5.0 s)** — No two `posture_settle` events fired within 5.02 s, confirming the refire gap is the minimum interval once `settle` remains above threshold.


---

## 7. What should be tuned next

**Increase `fidget_base_rate` or reconsider `fidget_suppression` as an event.** With mean inhibition 0.90 during speech, the current rate (0.10/s × inhibit) produced only 1 discrete event(s). Raising `fidget_base_rate` to 0.30–0.50/s would produce a more continuous suppression signal. Alternatively, consider logging `fidget_inhibit` as a continuous output channel rather than a sparse event — it may be better represented as a value than a count.

**Modulate `gaze_shift` strength by `speech_energy` at onset.** Currently every gaze shift fires at the same strength (~7.2), regardless of whether the onset was a confident utterance or a quiet murmur. Scaling orient_gain by `speech_energy` at the onset frame would make gaze shift intensity informative.

**Consider whether `settle_decay` (0.07/s) is too slow.** `posture_settle` fired 1 time(s) during extended silence because `settle` persists above threshold for many seconds after speech ends. If the intended meaning is 'actively settling into listening', the signal should clear more quickly when speaking stops. Try raising `settle_decay` to 0.15–0.20/s.

**Add at least one cross-path interaction.** Currently paths only interact through shared `recovery_backpressure`. A small, targeted coupling — such as `orient` briefly suppressing `ttp` accumulation at speech onset — would make the system more responsive to context and could produce more naturalistic turn-taking timing around speech onset moments.


---

## Main diagnosis

### Working as intended

- **Orienting response** — `gaze_shift` fired exactly once per speech onset (4 → 4). The refractory mechanism prevented double-firing.
- **Settle accumulation** — `settle` built to mean 0.574 during speech, and `posture_settle` fired, confirming the slow-path accumulation mechanism.
- **Fidget inhibition** — `fidget_inhibit` averaged 0.90 during speech, confirming the inhibition path charges correctly while the user is speaking.
- **Turn-taking pressure staging** — `micro_nod_ready` (lower threshold) and `response_ready` (higher threshold) both fired, demonstrating the two-level pressure architecture.

### Interesting emergent-looking behavior

- **Simultaneous burst at t≈19.17 s** (`micro_nod_ready`, `posture_settle`) — multiple paths discharged together after a shared suppression window. This compression-then-release pattern is a genuine emergent consequence of shared backpressure, not a scripted co-occurrence.
- **`posture_settle` at t=19.20 s — 3.2 s into silence** — the slow `settle` decay allows the path to linger above threshold long after speech ends, creating an unscripted gradual transition from active listening to rest.

### Too parameter-dominated / needs tuning

- **`gaze_shift` strength is fixed** — all firings at 7.20. `orient_gain` entirely determines event intensity; speech context has no effect.
