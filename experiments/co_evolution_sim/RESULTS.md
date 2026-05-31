# Co-Evolution Sim — Experiment Series

**Script:** `echo_sim.py`
**Branch:** `feature/co-evolution-sim`
**Theme:** Self-organizing dynamics through co-evolution of activity and topology

---

## Overview

A directed network of 20 nodes in which node activity (`tanh` propagation) and topology (Hebbian reinforcement + exponential decay) mutually influence each other. Each experiment was motivated by an open question raised by the previous result, forming a chain of inquiry.

---

## Exp 1 — K Comparison: Ratio of Activity Update Rate to Topology Change Rate

**Question:** How does varying K (topology update interval) affect the final topology?

**Result:** Smaller K (activity and topology time constants closer together) yields denser networks with higher clustering coefficients. K=1 produces dense networks; K=50 produces sparse ones.

**Figure:** `images/results_comparison_fine.png`

---

## Exp 2 — Input Pattern Comparison: constant / alternating / random

**Question:** Starting from the same initial state, do different input patterns produce different final topologies?

**Result:** Constant and alternating inputs produce distinct topologies. The input pattern leaves a fingerprint on the topology.

**Figures:** `images/results_input_comparison.png`, `images/results_topology_fingerprint.png`

---

## Exp 3 — Encoding of Input History: Topology Fingerprint

**Question:** Does the post-training topology encode input history? Can it be distinguished by probe responses after a silence period?

**Result:** The Euclidean distance between probe trajectories of networks trained on A (constant) and B (alternating) exceeded 1.0, confirming that the topology retains input history.

**Figure:** `images/results_encoding.png`

---

## Exp 4 — STDP: Spike-Timing-Dependent Plasticity

**Question:** What changes when the rate-based reinforcement rule is replaced with STDP (spike-timing-dependent)?

**Result:** Encoding survives under STDP, but all edges collapse during the silence period (full −1 suppression → topology loss). Edge collapse recorded as a finding.

**Figures:** `images/results_stdp_comparison.png`, `images/results_stdp_encoding.png`

---

## Exp 5 — Tracking Topology Change Magnitude

**Question:** How does the per-step topology change magnitude evolve over time? Does it differ by K?

**Result:** K=1 shows large early changes and rapid convergence. K=50 shows slower, more sustained changes.

**Figure:** `images/results_delta_comparison.png`

---

## Exp 6 — Long-Term Convergence Test

**Question:** Does K=50 eventually converge to the same structure as K=1?

**Result:** No convergence. The K difference produces qualitatively distinct final states. K determines the structural class of the final topology.

**Figure:** `images/results_convergence.png`

---

## Exp 7 — Cycle Detection

**Question:** Does co-evolution spontaneously generate directed cycles (loops)?

**Result:** At K=1, >10,000 cycles emerge by T=100. SCCs are large from the start (19–20 nodes). K=50 yields fewer cycles and sparser structure.

**Figure:** `images/results_cycle_analysis.png`

---

## Exp 8 — E/I Dynamics (Rate-Based Inhibition)

**Question:** If over-active nodes switch to inhibitory, can runaway activity be prevented?

**Result:** Half-wave rectification (`tanh(max(0, influence))`) zeroes out the inhibitory effect — inhibition_strength has no effect regardless of value. Recorded as a finding; moved to next experiment.

**Figures:** `images/results_ei_comparison.png`, `images/results_ei_inhibition.png`, `images/results_ei_threshold.png`

---

## Exp 9 — E/I Silencing: Freezing Instead of Suppression

**Question:** What happens if inhibitory nodes freeze associated edges rather than attenuating activity?

**Design change:** Edges connected to inhibitory nodes are excluded from Hebbian update, decay, and pruning — weights fixed at current values. Activity propagation maintained on all edges (no type_strength scaling).

**Result:** Inhibitory node activity stalls at 0.95–0.99. No inh→exc recovery occurs. The majority of edges become permanently silenced.

**Figure:** `images/results_ei_silencing.png`

---

## Exp 10 — E/I Isolation: Full Disconnection

**Question:** What happens when silenced edges are also cut from activity propagation, and inhibitory nodes decay exponentially?

**Design change:** Activity propagation via exc–exc edges only. Inhibitory nodes: `a(t+1) = a(t) × 0.9`.

**Result:** Inhibitory node activity converges to zero (0.001–0.004). Zeroing satisfies `recent_mean < threshold×0.5`, triggering frequent inh→exc recovery (~500 switches / 5000 steps). Only ei_threshold=0.9 settles into a qualitatively distinct regime (active edges 218, clustering 0.92, variance 0.18).

**Figure:** `images/results_ei_isolation.png`

---

## Exp 11 — Input Removal Test

**Question:** After training with input, is activity maintained when input is removed?

**Protocol:** Training (T=3000, input=0.5) → No input (T=3000, input=0.0) → Restored (T=2000, input=0.5)

**Result (ei_threshold=0.9):**
- End of training: variance=0.117, active_edges=22, excitatory=9
- End of no-input phase: variance=0.157, active_edges=249, excitatory=20 → **activity_maintained: True**
- End of restored phase: variance=0.024, active_edges=0, excitatory=4

**Paradoxical finding:** Activity *increases* after input removal. Setting input=0 causes all inhibitory nodes to recover at once (inh→exc), dramatically expanding the excitatory network. Activity collapses again when input is restored.

**Figure:** `images/results_input_removal.png`

---

## Exp 12 — Fixed Topology Control

**Question:** Does oscillation require topology co-evolution? Can E/I switching rules alone generate oscillation?

**Protocol:** Train for T=3000 → use trained topology as starting point; run A (dynamic topology) and B (fixed topology) for T=5000 (no input).

**Result:**
| Condition | variance mean (last 1000) | oscillation |
|---|---|---|
| Dynamic topology | 0.1050 | True |
| Fixed topology | 0.1070 | True |

**Conclusion:** Oscillation does not require topology co-evolution. E/I switching rules alone sustain autonomous oscillation. The trained network structure provides the substrate that makes oscillation possible.

**Figure:** `images/results_fixed_topology_control.png`

---

## Key Findings Across the Series

1. **K determines the structural class of the topology:** K=1 (dense, high-clustering, many cycles) and K=50 (sparse, low-clustering) do not converge even in the long run.
2. **Topology encodes input history:** Post-training weight matrices produce distinguishable probe responses.
3. **Inhibitory design is non-trivial:** Rate-based inhibition via half-wave rectification silences inhibitory effects entirely. Full disconnection (isolation) is required for inhibition to work — but then introduces excessive recovery cycling.
4. **Oscillation originates from E/I switching:** Topology dynamics are not required. Given a suitable initial structure, E/I switching rules alone sustain autonomous oscillation.
5. **Paradox: removing input increases activity:** Setting input=0 triggers a simultaneous inh→exc recovery wave, activating the excitatory network rather than silencing it.
