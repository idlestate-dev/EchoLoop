# EchoLoop Embodied Reaction MVP

EchoLoop Embodied Reaction MVP is a small exploratory model for studying how simple path dynamics can generate nontrivial interaction timing without explicit emotional labels or scripted state transitions.

## What this is

A headless Python simulation running at 60 Hz.  Six internal paths receive continuous input signals and accumulate, leak, or saturate over time.  Output events fire when path values cross thresholds—subject to inhibition, refractory gaps, and shared recovery/backpressure.  No output event is hard-coded to a scenario; all timing emerges from the dynamics.

### Internal paths

| Path | Role | Can trigger |
|---|---|---|
| `orienting_fast_path` | Spikes on speech onset, decays fast | `gaze_shift` |
| `listening_settle_slow_path` | Saturates slowly during speech | `posture_settle` |
| `fidget_inhibition_path` | Suppresses micro-motion during speech | `fidget_suppression` (stochastic) |
| `turn_taking_pressure_path` | Builds during silence; accelerates after 12 s | `micro_nod_ready`, `response_ready` |
| `recovery_backpressure_path` | Activated by any output event; raises all thresholds | (modulates all paths) |
| `freeze_path` | Integrates sudden approach velocity | `freeze` |

### Framing

Quantities used: *pressure, inhibition, delay, recovery, overflow, readiness, suppression, settling.*  
No emotional labels (happy, shy, nervous, etc.).

## What this is not

- Not a biologically accurate model
- Not an emotion or affect system
- Not a production avatar or dialogue system
- Not a speech recognition or NLP pipeline
- Not fit for any clinical or user-facing purpose

## How to run

```bash
pip install -r requirements.txt

# single scenario
python run_simulation.py --scenario speech_then_silence

# all scenarios
python run_simulation.py --scenario all
```

## Scenarios

| Scenario | Description |
|---|---|
| `speech_then_silence` | Speech t = 2–10 s, then silence |
| `repeated_speech_silence` | Four speech blocks with gaps of varying length |
| `sudden_approach_while_speaking` | Speech then a brief high-velocity approach pulse |
| `random_ambient_noise` | Sporadic short bursts plus continuous low-level energy |
| `long_listening` | 18 s silence, short speech, then silence again |

Scenario functions generate input signals only.  They contain no event or threshold logic.

## Outputs

```
outputs/
  logs/
    <scenario>.csv              per-timestep: time, inputs, path values, emitted events
    <scenario>_events.json      event log with timestamps and strengths
  plots/
    <scenario>_inputs.png       input signals over time
    <scenario>_internals.png    internal path energy over time
    <scenario>_raster.png       output event raster with ttp/speaking context
    <scenario>_iei.png          inter-event interval histograms (when ≥ 2 events exist)
```

## Tuning parameters

All path parameters (gain, decay rate, threshold, refractory gap, recovery pulse size) are in the `PathConfig` dataclass in `echoloop.py`.  Changing them requires no edits to scenarios or event-detection logic.

## Current limitations

- No sensory noise model beyond the stochastic fidget-suppression rate.
- Paths interact only through the shared `recovery_backpressure` signal; there is no lateral inhibition between paths.
- No persistent state between scenario runs; each run starts from rest.
- Signal shapes in scenarios are synthetic, not derived from real interaction data.
- Thresholds and gains were tuned by inspection; no fitting or optimization has been done.
- The `silence_duration` signal currently adds only a mild gain boost to `ttp`; its role could be expanded.
