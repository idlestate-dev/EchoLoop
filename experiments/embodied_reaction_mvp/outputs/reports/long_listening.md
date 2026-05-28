# Diagnostic Report: `long_listening`

_Simulation output: 1,800 steps · 30.0 s · 22 events total_


---

## 1. What this scenario represents

An extended 18-second silence (t=0–18 s), a short speech block (t=18–21 s), then 9 more seconds of silence. The long pre-speech silence allows turn-taking pressure to saturate and tests a specific question: does speech onset release accumulated pressure or interrupt it? The post-speech silence then tests recovery and re-accumulation dynamics.

**Run stats:** 1,800 steps · 30.0 s total · speaking 10% of the time · longest silence 18.0 s · 1 speech onset(s).

---

## 2. Which paths were primarily activated

| Path | overall mean | mean during speech | mean during silence | peak |
|---|---|---|---|---|
| `orient` | 0.008 | 0.080 | 0.000 | 1.440 | ◀
| `settle` | 0.153 | 0.323 | 0.134 | 0.544 | ◀
| `fidget_inhibit` | 0.122 | 0.894 | 0.036 | 1.000 | ◀
| `ttp` | 0.646 | 0.188 | 0.696 | 1.000 | ◀
| `recovery` | 0.226 | 0.355 | 0.212 | 0.789 | ◀
| `freeze_val` | 0.000 | 0.000 | 0.000 | 0.000 |

- `fidget_inhibit` averaged 0.89 during speech — the inhibition path is charging close to saturation while the user speaks, as designed.
- `settle` averaged only 0.32 during speech — speech blocks were likely too short or sparse for the slow path to charge above threshold.
- `ttp` averaged 0.70 during silence — high enough to indicate that turn-taking pressure reached or approached saturation.
- `recovery` maintained a mean of 0.226 overall, meaning events were firing frequently enough to keep backpressure elevated for significant periods.

---

## 3. Which output events fired and why

### `gaze_shift` — fired 1 time

_Usually caused by `orienting_fast_path` crossing threshold after `speech_onset`. The path spikes instantly (orient += onset × gain) and decays fast (6/s); the event fires within one frame and the refractory flag prevents double-firing on the same onset._

**Times:** t=18.00  
**Mean strength:** 7.200  

One event per onset (1 onset(s) → 1 firing(s)). The refractory mechanism is working: orient decayed below 15% of threshold between speech blocks, allowing the flag to reset before the next onset.

### `posture_settle` — fired 1 time

_Caused by `listening_settle_slow_path` accumulating during sustained speech. The path saturates toward ~0.81 with a time constant of ~2.7 s; the threshold (0.50) is only crossed after several seconds of continuous speaking._

**Times:** t=20.83  
**Mean strength:** 0.529  

`settle` values at firing: 0.529. 
All firings occurred during active speech phases.

### `fidget_suppression` — fired 2 times

_Caused by `fidget_inhibition_path` during speaking; fires stochastically at a rate proportional to inhibition strength (0.10 × fidget_inhibit × dt). As a discrete event, density is low (0–2 per scenario) and may be too sparse to meaningfully represent the texture of suppression._

**Times:** t=19.28, t=20.33  
**Mean strength:** 0.990  

`fidget_inhibit` at firing: 0.982, 0.999. Fired during high-inhibition speech phases as expected, but the stochastic mechanism means this could just as easily have been zero.

### `micro_nod_ready` — fired 11 times

_Caused by moderate turn-taking pressure (`ttp`) building during silence. Fires at the lower threshold (0.28), often as a precursor to `response_ready`. The 2.5 s refire gap spaces out repeated nods._

**Times:** t=0.92, t=3.43, t=5.95, t=8.47, t=10.98, t=13.50, t=16.02, t=21.78, t=24.30, t=26.82, t=29.33  
**Mean strength:** 0.743  

Intervals between nods: 2.52, 2.52, 2.52, 2.52, 2.52, 2.52, 5.77, 2.52, 2.52, 2.52 s. Mean 2.84 s, std 0.97 s. 
Higher variance suggests `ttp` fluctuations (from speech or recovery) are actively modulating nod timing.

### `response_ready` — fired 7 times

_Caused by high turn-taking pressure (`ttp`) crossing the upper threshold (0.65). Once `ttp` saturates, this event is largely paced by the 4.0 s refire gap rather than by path dynamics — an identified limitation._

**Times:** t=2.15, t=6.17, t=10.18, t=14.20, t=18.22, t=23.00, t=27.02  
**Mean strength:** 0.892  

`ttp` at firing: 0.662, 1.000, 1.000, 1.000, 0.900, 0.680, 1.000.  
`recovery` at firing: 0.052, 0.155, 0.061, 0.112, 0.300, 0.111, 0.159.  

Intervals: 4.02, 4.02, 4.02, 4.02, 4.78, 4.02 s. Mean 4.14 s, std 0.29 s. 
Variance present: speech, recovery, and `ttp` discharge are all contributing to interval variability — dynamics are active.

### `freeze` — did not fire

_Caused by `freeze_path` crossing threshold (0.20) after a sudden `approach_velocity` spike. Immediately injects a large recovery pulse (0.40), suppressing all other events until backpressure decays._

No significant approach-velocity input was present. `freeze_val` peaked at only 0.000, well below the threshold of 0.20.


---

## 4. Expected vs. surprising events

**Expected:**

- `gaze_shift` × 1 — speech onset(s) were present, so an immediate orienting response is expected.
- `posture_settle` — longest speech run was 3.0 s, long enough for `settle` to cross threshold (~2.7 s).
- `micro_nod_ready` — silence exceeded 1 s (max 18.0 s), sufficient for `ttp` to reach the nod threshold (0.28 at 0.30/s).
- `response_ready` — silence exceeded 2.5 s, sufficient for `ttp` to reach the response threshold (0.65).

**Surprising or noteworthy:**

- **`response_ready` at t=18.22 s — during active speech** (ttp=0.900). Turn-taking pressure built up during a preceding silence and was still above threshold when speaking began. The system was simultaneously orienting toward the speaker and signaling a response impulse — an emergent ambiguity.


---

## 5. Path dynamics vs. fixed parameters

**Evidence of path-dynamics-driven behavior:**

- `response_ready` intervals: mean 4.14 s, std 0.29 s. The variance shows that `ttp` did not remain saturated throughout — speech discharge or recovery cycles were actively pulling it back down, creating genuine dynamic modulation of response timing.
- Some events fired with recovery > 0 (max 0.339). Backpressure was actively raising effective thresholds at those moments, confirming that the shared recovery mechanism is influencing event timing beyond the bare refire gaps.
- `ttp` was not fully saturated at some `response_ready` firings (min 0.662). Speech blocks had discharged pressure enough to keep `ttp` below ceiling, meaning the threshold crossing reflects genuine path state rather than a guaranteed saturation.


---

## 6. Which parameter seems most dominant

The most dominant parameter in this scenario is **`orient_gain` (8.0)**.

Every `gaze_shift` fired at strength ≈7.20 (= orient_gain × one-frame decay factor). There is no dynamic modulation of gaze intensity; the parameter completely determines event strength.


---

## 7. What should be tuned next

**Modulate `gaze_shift` strength by `speech_energy` at onset.** Currently every gaze shift fires at the same strength (~7.2), regardless of whether the onset was a confident utterance or a quiet murmur. Scaling orient_gain by `speech_energy` at the onset frame would make gaze shift intensity informative.

**Strengthen the role of `silence_duration`.** The current 1.2× gain boost after 12 s is barely visible in the output. Consider adding a second response threshold — say 0.80 — that only unlocks after `silence_duration` exceeds 15 s, giving the system a distinct 'urgency' mode for very long pauses that differs qualitatively from the standard turn-taking cycle.

**Add at least one cross-path interaction.** Currently paths only interact through shared `recovery_backpressure`. A small, targeted coupling — such as `orient` briefly suppressing `ttp` accumulation at speech onset — would make the system more responsive to context and could produce more naturalistic turn-taking timing around speech onset moments.


---

## Main diagnosis

### Working as intended

- **Orienting response** — `gaze_shift` fired exactly once per speech onset (1 → 1). The refractory mechanism prevented double-firing.
- **Settle accumulation** — `settle` built to mean 0.323 during speech, and `posture_settle` fired, confirming the slow-path accumulation mechanism.
- **Fidget inhibition** — `fidget_inhibit` averaged 0.89 during speech, confirming the inhibition path charges correctly while the user is speaking.
- **Turn-taking pressure staging** — `micro_nod_ready` (lower threshold) and `response_ready` (higher threshold) both fired, demonstrating the two-level pressure architecture.

### Interesting emergent-looking behavior

- **`response_ready` at t=18.22 s — pressure carried into speech** — turn-taking pressure accumulated during silence was still above threshold when speech began. The system was simultaneously orienting and primed to respond. This cross-phase bleed-through was not scripted.

### Too parameter-dominated / needs tuning

- *(no strongly parameter-dominated patterns detected — event timing shows sufficient variance)*
