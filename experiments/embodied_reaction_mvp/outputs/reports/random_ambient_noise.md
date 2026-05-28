# Diagnostic Report: `random_ambient_noise`

_Simulation output: 1,500 steps · 25.0 s · 22 events total_


---

## 1. What this scenario represents

Five very short speech bursts (0.4–0.7 s each) spread across 25 s, with continuous low-level speech energy and small ambient approach fluctuations (amplitude 0.12, intentionally below the freeze threshold). The bursts are too brief for settle or fidget-inhibition to accumulate deeply. This scenario probes what emerges when no single path has time to fully charge.

**Run stats:** 1,500 steps · 25.0 s total · speaking 21% of the time · longest silence 4.4 s · 5 speech onset(s).

---

## 2. Which paths were primarily activated

| Path | overall mean | mean during speech | mean during silence | peak |
|---|---|---|---|---|
| `orient` | 0.048 | 0.231 | 0.000 | 1.440 | ◀
| `settle` | 0.313 | 0.353 | 0.303 | 0.547 | ◀
| `fidget_inhibit` | 0.328 | 0.717 | 0.226 | 0.986 | ◀
| `ttp` | 0.581 | 0.672 | 0.557 | 1.000 | ◀
| `recovery` | 0.289 | 0.349 | 0.273 | 0.716 | ◀
| `freeze_val` | 0.024 | 0.015 | 0.026 | 0.122 |

- `fidget_inhibit` averaged 0.72 during speech — the inhibition path is charging close to saturation while the user speaks, as designed.
- `settle` averaged only 0.35 during speech — speech blocks were likely too short or sparse for the slow path to charge above threshold.
- `ttp` averaged 0.56 during silence, showing meaningful pressure build-up but without saturating.
- `recovery` maintained a mean of 0.289 overall, meaning events were firing frequently enough to keep backpressure elevated for significant periods.

---

## 3. Which output events fired and why

### `gaze_shift` — fired 5 times

_Usually caused by `orienting_fast_path` crossing threshold after `speech_onset`. The path spikes instantly (orient += onset × gain) and decays fast (6/s); the event fires within one frame and the refractory flag prevents double-firing on the same onset._

**Times:** t=3.50, t=8.10, t=13.32, t=17.55, t=21.45  
**Mean strength:** 7.200  

One event per onset (5 onset(s) → 5 firing(s)). The refractory mechanism is working: orient decayed below 15% of threshold between speech blocks, allowing the flag to reset before the next onset.

### `posture_settle` — fired 1 time

_Caused by `listening_settle_slow_path` accumulating during sustained speech. The path saturates toward ~0.81 with a time constant of ~2.7 s; the threshold (0.50) is only crossed after several seconds of continuous speaking._

**Times:** t=22.47  
**Mean strength:** 0.541  

`settle` values at firing: 0.541. 
All firings occurred during active speech phases.

### `fidget_suppression` — did not fire

_Caused by `fidget_inhibition_path` during speaking; fires stochastically at a rate proportional to inhibition strength (0.10 × fidget_inhibit × dt). As a discrete event, density is low (0–2 per scenario) and may be too sparse to meaningfully represent the texture of suppression._

`fidget_inhibit` averaged 0.717 during speech. At a base rate of 0.10/s × 0.72 inhibition, the expected count over the speaking window was low enough that zero firings is unsurprising given the stochastic mechanism.

### `micro_nod_ready` — fired 10 times

_Caused by moderate turn-taking pressure (`ttp`) building during silence. Fires at the lower threshold (0.28), often as a precursor to `response_ready`. The 2.5 s refire gap spaces out repeated nods._

**Times:** t=0.92, t=3.43, t=5.95, t=8.47, t=10.98, t=13.50, t=16.02, t=18.53, t=21.05, t=23.73  
**Mean strength:** 0.595  

Intervals between nods: 2.52, 2.52, 2.52, 2.52, 2.52, 2.52, 2.52, 2.52, 2.68 s. Mean 2.54 s, std 0.05 s. 
Low variance suggests the 2.5 s refire gap is the binding constraint.

### `response_ready` — fired 6 times

_Caused by high turn-taking pressure (`ttp`) crossing the upper threshold (0.65). Once `ttp` saturates, this event is largely paced by the 4.0 s refire gap rather than by path dynamics — an identified limitation._

**Times:** t=2.15, t=6.17, t=10.18, t=15.17, t=19.18, t=23.20  
**Mean strength:** 0.795  

`ttp` at firing: 0.662, 0.884, 0.967, 0.681, 0.826, 0.750.  
`recovery` at firing: 0.052, 0.190, 0.118, 0.120, 0.199, 0.225.  

Intervals: 4.02, 4.02, 4.98, 4.02, 4.02 s. Mean 4.21 s, std 0.39 s. 
Variance present: speech, recovery, and `ttp` discharge are all contributing to interval variability — dynamics are active.

### `freeze` — did not fire

_Caused by `freeze_path` crossing threshold (0.20) after a sudden `approach_velocity` spike. Immediately injects a large recovery pulse (0.40), suppressing all other events until backpressure decays._

No significant approach-velocity input was present. `freeze_val` peaked at only 0.122, well below the threshold of 0.20.


---

## 4. Expected vs. surprising events

**Expected:**

- `gaze_shift` × 5 — speech onset(s) were present, so an immediate orienting response is expected.
- `micro_nod_ready` — silence exceeded 1 s (max 4.4 s), sufficient for `ttp` to reach the nod threshold (0.28 at 0.30/s).
- `response_ready` — silence exceeded 2.5 s, sufficient for `ttp` to reach the response threshold (0.65).

**Surprising or noteworthy:**

- **Multi-event burst at t≈3.43 s** — `micro_nod_ready`, `gaze_shift` fired within 67–150 ms of each other. This is a compression-release artifact: multiple paths had accumulated pressure during a suppression window and discharged simultaneously when that window closed.
- **`fidget_suppression` did not fire** despite `fidget_inhibit` averaging 0.72 during speech. This is a known limitation: the stochastic base rate (0.10/s) is low enough that zero firings is plausible even with strong inhibition. The suppression is happening internally but is invisible in the discrete event stream.


---

## 5. Path dynamics vs. fixed parameters

**Evidence of path-dynamics-driven behavior:**

- `response_ready` intervals: mean 4.21 s, std 0.39 s. The variance shows that `ttp` did not remain saturated throughout — speech discharge or recovery cycles were actively pulling it back down, creating genuine dynamic modulation of response timing.
- Some events fired with recovery > 0 (max 0.472). Backpressure was actively raising effective thresholds at those moments, confirming that the shared recovery mechanism is influencing event timing beyond the bare refire gaps.
- `ttp` was not fully saturated at some `response_ready` firings (min 0.662). Speech blocks had discharged pressure enough to keep `ttp` below ceiling, meaning the threshold crossing reflects genuine path state rather than a guaranteed saturation.

**Evidence of parameter-dominated behavior:**

- `gaze_shift` strength range: 7.20–7.20. All orientings fired at nearly identical intensity, set entirely by `orient_gain` (8.0). There is no dynamic modulation; a loud onset and a quiet one look identical.


---

## 6. Which parameter seems most dominant

The most dominant parameter in this scenario is **`orient_gain` (8.0)**.

Every `gaze_shift` fired at strength ≈7.20 (= orient_gain × one-frame decay factor). There is no dynamic modulation of gaze intensity; the parameter completely determines event strength.


---

## 7. What should be tuned next

**Increase `fidget_base_rate` or reconsider `fidget_suppression` as an event.** With mean inhibition 0.72 during speech, the current rate (0.10/s × inhibit) produced only 0 discrete event(s). Raising `fidget_base_rate` to 0.30–0.50/s would produce a more continuous suppression signal. Alternatively, consider logging `fidget_inhibit` as a continuous output channel rather than a sparse event — it may be better represented as a value than a count.

**Modulate `gaze_shift` strength by `speech_energy` at onset.** Currently every gaze shift fires at the same strength (~7.2), regardless of whether the onset was a confident utterance or a quiet murmur. Scaling orient_gain by `speech_energy` at the onset frame would make gaze shift intensity informative.

**Add at least one cross-path interaction.** Currently paths only interact through shared `recovery_backpressure`. A small, targeted coupling — such as `orient` briefly suppressing `ttp` accumulation at speech onset — would make the system more responsive to context and could produce more naturalistic turn-taking timing around speech onset moments.


---

## Main diagnosis

### Working as intended

- **Orienting response** — `gaze_shift` fired exactly once per speech onset (5 → 5). The refractory mechanism prevented double-firing.
- **Settle accumulation** — `settle` built to mean 0.353 during speech, and `posture_settle` fired, confirming the slow-path accumulation mechanism.
- **Fidget inhibition** — `fidget_inhibit` averaged 0.72 during speech, confirming the inhibition path charges correctly while the user is speaking.
- **Turn-taking pressure staging** — `micro_nod_ready` (lower threshold) and `response_ready` (higher threshold) both fired, demonstrating the two-level pressure architecture.

### Interesting emergent-looking behavior

- **Simultaneous burst at t≈3.43 s** (`micro_nod_ready`, `gaze_shift`) — multiple paths discharged together after a shared suppression window. This compression-then-release pattern is a genuine emergent consequence of shared backpressure, not a scripted co-occurrence.

### Too parameter-dominated / needs tuning

- **`gaze_shift` strength is fixed** — all firings at 7.20. `orient_gain` entirely determines event intensity; speech context has no effect.
