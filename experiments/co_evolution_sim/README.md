# Co-Evolution Sim

A simulation of self-organizing dynamics through the mutual co-evolution of node activity and network topology.

## What This Is

A directed network of 20 nodes where two processes interact in a feedback loop:

- **Activity dynamics** — each node's activation is updated by `tanh`-weighted propagation from its inputs
- **Topology dynamics** — edge weights are reinforced by Hebbian learning and decay exponentially toward zero

These two processes run on different timescales (controlled by `K`, the topology update interval), producing qualitatively different emergent structures depending on their ratio.

The experiment series progressively adds complexity: input pattern encoding, STDP, E/I switching, and finally tests whether oscillation requires co-evolution or arises from E/I rules alone.

## Structure

```
co_evolution_sim/
├── world_base.py               # Base simulation class
├── session_3_ei_evo.py         # Exp 3: topology fingerprint encoding
├── session_4_context.py        # Exp 4: STDP
├── session_5_sweep.py          # Exp 5: topology delta tracking
├── session_6_ei_static.py      # Exp 6: long-term convergence
├── session_7_context_activation.py  # Exp 7: cycle detection
├── session_8_world_test.py     # Exp 8: rate-based E/I (failed design)
├── session_9_topology_sculpting.py  # Exp 9: E/I silencing
├── session_10_embodied_output.py    # Exp 10: E/I isolation
├── session_11_noise_escape.py       # Exp 11: input removal test
├── session_12_sleep_consolidation.py # Exp 12: fixed topology control
├── run_all.py                  # Run all sessions
├── RESULTS.md                  # Results summary (Japanese)
└── RESULTS_en.md               # Results summary (English)
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `K` | varies | Topology update interval (steps). K=1: fast co-evolution; K=50: slow. |
| `N` | 20 | Number of nodes |
| `T` | varies | Simulation steps |
| `ei_threshold` | 0.7–0.9 | Activity level above which a node switches to inhibitory |
| `decay` | 0.99 | Per-step edge weight decay rate |

## Main Results

| Finding | Experiment |
|---|---|
| K determines the structural class of the final topology | Exp 1, 6 |
| Topology encodes input history (distinguishable by probe) | Exp 3 |
| Rate-based inhibition silences itself via half-wave rectification | Exp 8 |
| Full E/I disconnection is required for inhibition to function | Exp 10 |
| Oscillation does not require topology co-evolution | Exp 12 |
| Removing input *increases* activity (inh→exc recovery wave) | Exp 11 |

## Running

```bash
# Run a specific session
python session_12_sleep_consolidation.py

# Run all sessions sequentially
python run_all.py
```

Output figures are written to `images/session_<N>/`.

## Related Work

- **EchoLoop v6** (`feature/v6-social-echo`): multi-agent acoustic proto-signal simulation — agents externalize internal loop state as continuous acoustic parameters and observe whether this protects a group from a danger zone.
