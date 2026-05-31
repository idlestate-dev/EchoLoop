# EchoLoop v0.1.0 — Research Snapshot

## What this is

A minimal toy model for exploring how removing a static readout layer affects behavioral diversity and network structure in a survival-only neuroevolution setting.

This is a research snapshot, not a finished system.
Experiments are preliminary. Claims should be interpreted cautiously.

## Key observation (10-seed reproducibility check)

When a readout layer is removed from a survival-only embodied agent:

| Criterion | Result |
|---|---|
| New arch path length > Old arch × 3 | 9/10 seeds |
| Survivor ablation < Non-survivor ablation | 8/10 seeds |
| Old arch clustering > New arch clustering | 9/10 seeds |

These observations are reproducible across seeds but have not been independently verified.

## What this release includes

- Session 10–14 experiment scripts
- 10-seed reproducibility data (results_s14_raw.csv)
- Key result figures (images/session_13/, images/session_14/)
- Abstract draft (see README)

## What this is NOT

- A theory of intelligence or consciousness
- A claim that reward-free value emergence has been proven
- A production-ready system
- A comparison against established baselines (NEAT etc.)

## Known limitations

- N=20 only (scaling untested)
- Hyperparameter sensitivity not fully characterized
- No independent replication yet
- Theoretical explanation of recurrence emergence unknown

## Planned next steps

- Hyperparameter sensitivity analysis
- Hidden state reset ablation
- Scaling to N=50, N=100
- ALIFE 2026 Late Breaking Abstract submission

## License

CC BY-NC-ND 4.0
