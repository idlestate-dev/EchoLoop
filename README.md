# EchoLoop

### A tiny sandbox for route-based internal dynamics, metastability, and attractor competition

EchoLoop is an experimental toy model exploring a simple question:

> What if intelligence is less about static weights, and more about recurrent flows through evolving routes?

This project is **not** an AGI architecture.
It is not optimized for benchmarks, reasoning tasks, or next-token prediction.

Instead, EchoLoop is a lightweight dynamical sandbox for observing:

* recurrent loop competition
* metastable internal states
* fatigue-driven switching
* blended attractors
* ambiguity-induced suppression
* route-based behavioral dynamics

The project emerged from long conversations with LLMs while thinking about the limitations of purely static weight-based models.

---

# Core Idea

Modern LLMs are extraordinarily capable, but their interaction pattern often feels fundamentally stateless.

Biological systems do not simply react to stimuli.
Their internal state continuously drifts:

* attention wanders
* vigilance rises and falls
* fatigue accumulates
* curiosity competes against fear
* behavior depends heavily on previous internal momentum

This project explores a different perspective:

> Perhaps cognition does not primarily live inside static weights, but inside recurrent flows through a continuously evolving state landscape.

In EchoLoop:

* nodes are intersections
* weights are terrain/topology
* activation flow behaves more like water through riverbeds
* loops compete for persistence
* behavior is treated as a readout of internal dynamics, not the optimization target itself

---

# Architecture

Current versions of EchoLoop use several recurrent loops:

## Social Loop

attention → observe → engage → approach

## Exploration Loop

curiosity → observe → wander → vigilance → inspect

## Defensive Loop

alert → vigilance → freeze → withdraw

Some nodes are intentionally shared:

* `observe`
  bridges Social ↔ Exploration

* `vigilance`
  bridges Exploration ↔ Defensive

These shared routes create:

* interference
* blended states
* metastability
* dominance ambiguity

---

# Key Mechanisms

## Fatigue / Adaptation

Frequently active loops accumulate fatigue.

As fatigue rises:

* attractors weaken
* dominance collapses
* spontaneous switching becomes more likely

This creates:

* persistence
* recovery
* oscillation
* wandering dynamics

---

## No Global Objective

EchoLoop intentionally avoids:

* global reward optimization
* argmax action selection
* explicit task solving

The goal is not to maximize performance.

The goal is to observe what kinds of internal dynamics emerge from:

* recurrent routes
* shared bottlenecks
* fatigue
* inhibition
* attractor competition

---

# Interesting Behaviors Observed

Depending on topology and parameters, the system exhibits:

* spontaneous attractor switching
* hysteresis
* metastability
* ambiguity-driven freezing
* blended internal states
* dominance collapse
* partial synchronization
* long idle phases caused by unresolved competition

One particularly interesting result:

The Exploration loop tends to become unstable because it is sandwiched between two shared bridge routes (`observe` and `vigilance`), causing constant interruption by social or defensive dynamics.

This was not explicitly programmed as a behavioral rule.
It emerged from the topology itself.

---

# Why This Exists

This repository is intentionally exploratory and incomplete.

It is closer to:

* Artificial Life
* dynamical systems playgrounds
* cognitive toy models
* metastable route simulations

than to production AI systems.

The purpose is not to propose a complete theory of intelligence.

The purpose is to explore whether:

* routes
* bottlenecks
* recurrent flows
* competing internal states

can produce interesting dynamical behavior in extremely lightweight systems.

---

# Current Status

The project currently explores:

* recurrent loop dynamics
* shared-route interference
* fatigue-driven landscape modulation
* metastability
* blended attractors

Potential future directions:

* multi-timescale dynamics
* slow/fast hierarchical states
* asymmetric recovery
* memory traces
* spontaneous recovery
* route growth/pruning
* topology evolution

---

# Running

```bash
python echoloop5.py
```

Generated outputs typically include:

* loop activity plots
* ternary dominance diagrams
* shared-route activations
* action distributions
* switching statistics

---

# Disclaimer

This is not a neuroscience model.
This is not a claim of AGI.
This is not a replacement for LLMs.

It is a small experimental sandbox for observing route-based internal dynamics.

If you are interested in:

* dynamical systems
* ALife
* metastability
* recurrent cognition
* strange attractors
* lightweight internal-state simulations

feel free to fork it, break it, and experiment with the topology.
