# Changelog — EchoLoop Embodied Reaction MVP

---

## [v0.1] — 2026-05-29

### Motivation

In the initial MVP (`v0.0`), `response_ready` behaved like a fixed 4-second timer once `ttp` saturated at 1.0.  The refire gap was the sole pacing factor; path dynamics stopped contributing to timing.  v0.1 addresses this without adding scenario-specific rules or hard-coded randomness.

### Changed — `echoloop.py`

**State-dependent discharge on `response_ready`**

Previously: `ttp *= 0.25` (constant fraction, regardless of how saturated `ttp` was).

Now: excess pressure above the response threshold is drained additionally.

```
excess  = max(0, ttp − ttp_threshold_response)
new_ttp = max(0, ttp × 0.25 − excess × 0.45)
```

When `ttp` fires at saturation (`ttp ≈ 1.0`), it falls to ~0.09 instead of ~0.25.  At threshold (`ttp ≈ 0.65`) it falls to ~0.16 as before.  The discharge depth is now a continuous function of firing state, not a fixed fraction.

New parameter: `ttp_response_excess_discharge = 0.45`

**Dynamic refire gap**

Previously: `_response_timer = 4.0` (fixed).

Now: the gap is computed at each firing from current path state.

```
gap = ttp_response_refire_gap
    + recovery × ttp_response_gap_recovery_scale
    − min(1, silence_duration / 15) × ttp_response_gap_silence_discount
gap = max(1.0, gap)
```

High recovery (recent burst of events) extends the gap; sustained silence compresses it slightly.  The base gap is reduced from 4.0 s to 1.8 s.

New parameters: `ttp_response_refire_gap = 1.8`, `ttp_response_gap_recovery_scale = 2.5`, `ttp_response_gap_silence_discount = 0.50`

**Recovery suppresses `ttp` accumulation**

Previously: `ttp` rebuilt at a fixed rate during silence.

Now: active backpressure slows rebuilding.

```
rebuild_rate = ttp_gain × silence_factor × (1 − recovery × ttp_accumulation_recovery_suppression)
```

New parameter: `ttp_accumulation_recovery_suppression = 0.25`

### Added — `check_response_dynamics.py`

Standalone diagnostic script.  Reads existing logs; does not run the simulation.  For each scenario, reports:

- `response_ready` interval list, mean, std, min/max
- `ttp` saturation percentage and max continuous saturation duration
- `ttp` at firing and immediately after discharge
- Discharge depth (pre-fire minus post-fire)
- Recovery at firing
- Pearson correlations: recovery-at-fire → next interval; silence-at-fire → next interval; ttp-at-fire → discharge depth
- Short per-scenario interpretation (clock-like assessment, dynamic rebuild check, recovery modulation check)

Output: `outputs/reports/response_dynamics.md`

### Results summary

| Scenario | IEI std before | IEI std after | ttp@1.0 before | ttp@1.0 after |
|---|---|---|---|---|
| `speech_then_silence` | 0.00 s | 0.16 s | ~35% | 0% |
| `sudden_approach_while_speaking` | 0.00 s | 0.05 s | ~60% | 0% |
| `random_ambient_noise` | ~0.00 s | 0.74 s | ~20% | 0% |
| `long_listening` | 0.00 s | 1.35 s | ~75% | 0% |
| `repeated_speech_silence` | (variable) | 1.88 s | ~10% | 0% |

`response_ready` is no longer paced purely by a fixed timer in four of five scenarios.  `sudden_approach_while_speaking` still shows low IEI variance (std = 0.05 s) due to freeze-specific global synchronization — a separate structural cause documented in `RESULTS_v0.1.md`.

### Known remaining limitation

`sudden_approach_while_speaking`: the `freeze` event injects a fixed-size global recovery pulse, suppressing all paths together.  When suppression clears, paths resume from similar states each time, producing synchronized post-freeze `response_ready` timing.  This is distinct from the original ttp metronome and is not addressed in v0.1.

---

## [v0.0] — 2026-05-29 (initial commit)

Initial MVP implementation inside `experiments/embodied_reaction_mvp/`.

- Six internal paths: orienting, settle, fidget inhibition, turn-taking pressure, recovery/backpressure, freeze
- Five scenarios with synthetic input signals
- Fixed 60 Hz simulation, CSV + JSON logging, four plot types
- Diagnostic report layer (`diagnostics.py`) generating per-scenario Markdown reports
- Simulation results documented in `RESULT.md`

Known issue at v0.0: `response_ready` behaved as a fixed 4-second timer once `ttp` saturated.
