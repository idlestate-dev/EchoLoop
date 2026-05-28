# EchoLoop Embodied Reaction MVP

A small headless simulation for studying pre-linguistic reaction timing.

It asks one question: can simple path dynamics — delay, leakage, inhibition, overflow, and recovery/backpressure — generate nontrivial timing patterns such as orienting, settling, turn-taking readiness, freeze, and delayed release, without scripting those patterns explicitly?

Output events are generated through path dynamics rather than scenario-specific event scripts.

---

## What this is not

- Not a biological model
- Not an emotion or affect system
- Not a consciousness claim
- Not a production avatar or dialogue system
- Not evidence of life-like behavior

Quantities used: *pressure, inhibition, delay, recovery, overflow, readiness, suppression, settling.*  
No emotional labels.

---

## Relation to the original EchoLoop

| | Original EchoLoop | Embodied MVP |
|---|---|---|
| What it models | Abstract route/path dynamics | Interaction timing using low-level signals |
| Inputs | Abstract path loads | Speech onset, speaking state, silence duration, approach velocity |
| Outputs | Abstract flow events | Timing cues: gaze shift, nod, response readiness, freeze |
| Goal | Explore path dynamics as a toy model | Ask whether the same dynamics apply to interaction timing |

Both LLMs and EchoLoop operate on different layers. LLMs handle semantic reasoning and dialogue content. EchoLoop explores a lower-level reaction timing layer — orienting, settling, turn-taking pressure, freeze, recovery — without reference to meaning.

---

## Inputs and outputs

**Inputs** (continuous signals, sampled at 60 Hz)

| Signal | Role |
|---|---|
| `speech_onset` | Impulse at the start of each utterance |
| `user_speaking` | Binary flag, 1 while the user speaks |
| `speech_energy` | Envelope of speech amplitude |
| `silence_duration` | Seconds elapsed since last utterance |
| `approach_velocity` | Rate of physical approach (used for freeze) |

**Internal paths**

| Path | Dynamics | Triggers |
|---|---|---|
| `orienting_fast_path` | Spikes on `speech_onset`, decays fast (τ ≈ 0.17 s) | `gaze_shift` |
| `listening_settle_slow_path` | Saturates slowly during speech (τ ≈ 2.7 s) | `posture_settle` |
| `fidget_inhibition_path` | Rises while speaking, falls in silence | `fidget_suppression` (stochastic) |
| `turn_taking_pressure_path` | Builds during silence; two-level threshold | `micro_nod_ready`, `response_ready` |
| `freeze_path` | Integrates approach velocity | `freeze` |
| `recovery_backpressure_path` | Activated by any output event; raises all thresholds | (modulates all paths) |

**Output events**

`gaze_shift` · `posture_settle` · `micro_nod_ready` · `response_ready` · `freeze` · `fidget_suppression`

All event logic lives in `echoloop.py`. Scenario files contain only input signal generators.

---

## Key v0.1 finding — metronome issue mitigated

In the initial implementation (v0.0), `response_ready` behaved like a fixed-interval timer:

```
ttp saturated at 1.0
→ fixed 4.0 s refire gap became the sole pacing factor
→ response_ready intervals: 4.02, 4.02, 4.02, ...
```

v0.1 addresses this with three changes:

**1. State-dependent discharge.** Previously `ttp` was multiplied by a fixed fraction (0.25) on each firing. Now, pressure above the response threshold drains additionally:

```
new_ttp = ttp × 0.25 − max(0, ttp − threshold) × 0.45
```

At full saturation (`ttp = 1.0`), `ttp` drops to ~0.09, not ~0.25, forcing a genuine rebuild phase.

**2. Dynamic refire gap.** The gap now depends on path state at firing time:

```
gap = 1.8 + recovery × 2.5 − min(1, silence_duration / 15) × 0.50
```

Active backpressure extends the gap; sustained silence compresses it slightly.

**3. Recovery suppresses rebuild rate.** High backpressure slows `ttp` accumulation, adding a third source of variation independent of the discharge and the timer.

**Result across all five scenarios:**

| Scenario | IEI std before | IEI std after | ttp @ 1.0 |
|---|---|---|---|
| `speech_then_silence` | 0.00 s | 0.16 s | 0% |
| `sudden_approach_while_speaking` | 0.00 s | 0.05 s | 0% |
| `random_ambient_noise` | ~0.00 s | 0.74 s | 0% |
| `long_listening` | 0.00 s | 1.35 s | 0% |
| `repeated_speech_silence` | — | 1.88 s | 0% |

`ttp` saturation is 0% across all scenarios. `response_ready` timing now depends on pressure rebuilding, recovery state, and scenario structure.

This is accurately described as **metronome issue mitigated**, not solved permanently. Specific structural conditions can still produce near-uniform intervals (see below).

---

## Interesting behavior — post-freeze delayed release

In the `sudden_approach_while_speaking` scenario, a sudden high-velocity approach triggers `freeze`, which injects a large backpressure pulse and suppresses downstream paths for ~2.6 seconds. When suppression clears, multiple paths that accumulated pressure behind the backpressure wall can fire in a tight cluster.

This compression-then-release pattern is not scripted. It is a consequence of the shared `recovery_backpressure` path holding back multiple accumulators simultaneously, then releasing them together.

---

## Known limitations

**Freeze-specific global synchronization.** `sudden_approach_while_speaking` still shows low IEI variance (std = 0.05 s) even after v0.1 fixes. The cause is separate from the `ttp` metronome: `freeze` injects a fixed global recovery pulse, suppressing all paths together. When suppression clears, paths resume from similar states, so post-freeze `response_ready` timing converges each cycle. This is a structural property of the current freeze mechanism, not a timing bug.

**Other known limitations:**

- `gaze_shift` strength is nearly fixed regardless of speech energy at onset
- `fidget_suppression` is a sparse stochastic event; at current rates, it fires 0–2 times per scenario and may be better represented as a continuous level
- Paths interact only through shared backpressure; no lateral inhibition between paths
- No sensory noise model
- Synthetic input signals only — not derived from real interaction data
- Parameters tuned by inspection, no fitting or optimization
- No baseline comparison or ablation study yet

---

## Scenarios

| Scenario | Description |
|---|---|
| `speech_then_silence` | Speech t = 2–10 s, then 15 s silence |
| `repeated_speech_silence` | Four speech blocks with gaps of varying length |
| `sudden_approach_while_speaking` | Speech then a brief high-velocity approach pulse at t ≈ 13.8 s |
| `random_ambient_noise` | Five short speech bursts with low-level continuous energy |
| `long_listening` | 18 s silence, short speech block, silence again |

---

## How to run

```bash
cd experiments/embodied_reaction_mvp

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# single scenario
python run_simulation.py --scenario speech_then_silence

# all scenarios
python run_simulation.py --scenario all
```

Outputs are written to:

```
outputs/
  logs/
    <scenario>.csv              per-timestep: time, inputs, path values, events
    <scenario>_events.json      event log with timestamps and strengths
  plots/
    <scenario>_inputs.png       input signals over time
    <scenario>_internals.png    internal path energy over time
    <scenario>_raster.png       output event raster with ttp / speaking context
    <scenario>_iei.png          inter-event interval histograms
  reports/
    <scenario>.md               per-scenario diagnostic report
    response_dynamics.md        cross-scenario timing dynamics analysis
```

To run the response dynamics diagnostic separately (reads existing logs, no re-simulation):

```bash
python check_response_dynamics.py
```

---

## Tuning

All path parameters — gain, decay rate, threshold, refire gap, recovery pulse size — live in the `PathConfig` dataclass in `echoloop.py`. Scenario files contain no event or threshold logic, so parameter changes apply uniformly across all scenarios.

---

## Future roadmap

| Version | Goal |
|---|---|
| v0.2 | Modulate `gaze_shift` strength by `speech_energy` and recovery at onset |
| v0.3 | Replace sparse `fidget_suppression` events with a continuous `fidget_level` output |
| v0.4 | Graded freeze: suppress paths at different depths to break global synchronization |
| v0.5 | Baseline comparisons: random generator, simple threshold model, Markov / FSM, EchoLoop |
| v0.6 | Ablation studies: without recovery, without dynamic discharge, without freeze |

---

## Files

```
echoloop.py             core simulation — paths, thresholds, event detection
scenarios.py            input signal generators (no event logic)
plotting.py             matplotlib visualizations
run_simulation.py       CLI runner — logs, plots, reports
diagnostics.py          per-scenario diagnostic report generator
check_response_dynamics.py  response_ready timing analysis across scenarios
requirements.txt        numpy, matplotlib
CHANGELOG.md            version history
RESULTS_v0.1.md         full v0.1 results, analysis, and roadmap
```
