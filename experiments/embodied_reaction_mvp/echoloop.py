"""
EchoLoop core simulation paths.

Six internal paths with readable gain/decay/threshold parameters.
Output events emerge from accumulation, leakage, inhibition, threshold
overflow, and recovery/backpressure.  No event logic is tied to scenario names.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PathConfig:
    """Tunable parameters for all internal paths."""

    # orienting_fast_path
    orient_gain: float = 8.0        # impulse added per unit speech_onset
    orient_decay: float = 6.0       # exponential decay rate (per second)
    orient_threshold: float = 0.45  # level that triggers gaze_shift

    # listening_settle_slow_path
    settle_gain: float = 0.30       # max accumulation rate while speaking (per second)
    settle_decay: float = 0.07      # leak rate (per second)
    settle_threshold: float = 0.50  # level that triggers posture_settle
    settle_refire_gap: float = 5.0  # minimum seconds between posture_settle events

    # fidget_inhibition_path
    fidget_inhibit_rise: float = 3.0   # rise rate when user speaks (per second)
    fidget_inhibit_fall: float = 1.0   # fall rate when user silent (per second)
    fidget_base_rate: float = 0.10     # suppression events per second at full inhibition

    # turn_taking_pressure_path
    ttp_gain: float = 0.30              # pressure buildup per second of silence
    ttp_release_rate: float = 0.45      # release multiplier when user speaks (per second)
    ttp_threshold_nod: float = 0.28     # level that triggers micro_nod_ready
    ttp_threshold_response: float = 0.65  # level that triggers response_ready
    ttp_nod_refire_gap: float = 2.5     # minimum seconds between micro_nod_ready
    ttp_response_refire_gap: float = 4.0  # minimum seconds between response_ready
    ttp_response_discharge: float = 0.25  # fraction of ttp remaining after response_ready fires
    ttp_silence_acceleration: float = 0.20  # extra gain factor at 12 s of silence

    # recovery_backpressure_path
    recovery_decay: float = 0.80     # decay rate (per second)

    # freeze_path
    freeze_gain: float = 1.5         # gain applied to approach_velocity
    freeze_decay: float = 0.90       # decay rate (per second)
    freeze_threshold: float = 0.20   # level that triggers freeze
    freeze_refire_gap: float = 3.0   # minimum seconds between freeze events

    # Recovery pulse sizes injected on each event type
    recovery_on_gaze: float = 0.30
    recovery_on_settle: float = 0.18
    recovery_on_fidget: float = 0.06
    recovery_on_nod: float = 0.14
    recovery_on_response: float = 0.50
    recovery_on_freeze: float = 0.40


class EchoLoopState:
    """
    Stateful simulation of six internal paths.
    Call step() once per timestep; collect returned events.
    """

    def __init__(
        self,
        config: Optional[PathConfig] = None,
        dt: float = 1.0 / 60.0,
        seed: int = 42,
    ) -> None:
        self.cfg = config if config is not None else PathConfig()
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Path accumulators
        self.orient: float = 0.0
        self.settle: float = 0.0
        self.fidget_inhibit: float = 0.0
        self.ttp: float = 0.0
        self.recovery: float = 0.0
        self.freeze_val: float = 0.0

        # Refractory state
        self._orient_refractory: bool = False
        self._settle_timer: float = 0.0
        self._nod_timer: float = 0.0
        self._response_timer: float = 0.0
        self._freeze_timer: float = 0.0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def step(
        self,
        speech_onset: float,
        user_speaking: float,
        speech_energy: float,
        silence_duration: float,
        approach_velocity: float,
    ) -> List[Dict[str, Any]]:
        """
        Advance the simulation by one timestep.

        Returns a (possibly empty) list of event dicts that fired this step.
        """
        dt = self.dt
        cfg = self.cfg
        events: List[Dict[str, Any]] = []

        # ── orienting_fast_path ──────────────────────────────────────
        self.orient += speech_onset * cfg.orient_gain
        self.orient -= self.orient * cfg.orient_decay * dt
        self.orient = max(0.0, self.orient)
        if self.orient < cfg.orient_threshold * 0.15:
            self._orient_refractory = False

        # ── listening_settle_slow_path ───────────────────────────────
        if user_speaking > 0.5:
            self.settle += cfg.settle_gain * (1.0 - self.settle) * dt
        self.settle -= self.settle * cfg.settle_decay * dt
        self.settle = float(np.clip(self.settle, 0.0, 1.0))

        # ── fidget_inhibition_path ───────────────────────────────────
        if user_speaking > 0.5:
            self.fidget_inhibit += (1.0 - self.fidget_inhibit) * cfg.fidget_inhibit_rise * dt
        else:
            self.fidget_inhibit -= self.fidget_inhibit * cfg.fidget_inhibit_fall * dt
        self.fidget_inhibit = float(np.clip(self.fidget_inhibit, 0.0, 1.0))

        # ── turn_taking_pressure_path ────────────────────────────────
        if user_speaking < 0.5:
            # silence_duration provides a gentle late-pressure boost
            silence_factor = 1.0 + cfg.ttp_silence_acceleration * min(silence_duration / 12.0, 1.0)
            self.ttp = min(1.0, self.ttp + cfg.ttp_gain * silence_factor * dt)
        else:
            self.ttp -= self.ttp * cfg.ttp_release_rate * dt
        self.ttp = max(0.0, self.ttp)

        # ── freeze_path ──────────────────────────────────────────────
        v_in = max(0.0, approach_velocity)
        self.freeze_val += v_in * cfg.freeze_gain * dt
        self.freeze_val -= self.freeze_val * cfg.freeze_decay * dt
        self.freeze_val = float(np.clip(self.freeze_val, 0.0, 1.0))

        # Decay refractory timers
        self._settle_timer = max(0.0, self._settle_timer - dt)
        self._nod_timer = max(0.0, self._nod_timer - dt)
        self._response_timer = max(0.0, self._response_timer - dt)
        self._freeze_timer = max(0.0, self._freeze_timer - dt)

        r = self.recovery  # recovery raises effective thresholds

        # ── gaze_shift ───────────────────────────────────────────────
        if self.orient >= cfg.orient_threshold * (1.0 + r) and not self._orient_refractory:
            events.append(_ev("gaze_shift", self.orient,
                              orient=self.orient, recovery=r))
            self._orient_refractory = True
            self.orient *= 0.20
            self.recovery = min(1.0, r + cfg.recovery_on_gaze)

        # ── posture_settle ───────────────────────────────────────────
        if (self.settle >= cfg.settle_threshold * (1.0 + r * 0.35)
                and self._settle_timer <= 0.0
                and self.freeze_val < cfg.freeze_threshold):
            events.append(_ev("posture_settle", self.settle,
                              settle=self.settle, recovery=r))
            self._settle_timer = cfg.settle_refire_gap
            self.recovery = min(1.0, r + cfg.recovery_on_settle)

        # ── fidget_suppression ───────────────────────────────────────
        # Stochastic: rate proportional to active inhibition
        suppress_rate = cfg.fidget_base_rate * self.fidget_inhibit * (1.0 - r * 0.4)
        if self.rng.random() < suppress_rate * dt:
            events.append(_ev("fidget_suppression", self.fidget_inhibit,
                              fidget_inhibit=self.fidget_inhibit, recovery=r))
            self.recovery = min(1.0, r + cfg.recovery_on_fidget)

        # ── micro_nod_ready ──────────────────────────────────────────
        freeze_active = self.freeze_val >= cfg.freeze_threshold
        if (self.ttp >= cfg.ttp_threshold_nod * (1.0 + r * 0.50)
                and self._nod_timer <= 0.0
                and not freeze_active
                and r < 0.55):
            events.append(_ev("micro_nod_ready", self.ttp,
                               ttp=self.ttp, recovery=r))
            self._nod_timer = cfg.ttp_nod_refire_gap
            self.recovery = min(1.0, r + cfg.recovery_on_nod)

        # ── response_ready ───────────────────────────────────────────
        if (self.ttp >= cfg.ttp_threshold_response * (1.0 + r * 0.35)
                and self._response_timer <= 0.0
                and not freeze_active
                and r < 0.42):
            events.append(_ev("response_ready", self.ttp,
                               ttp=self.ttp, recovery=r))
            self.ttp *= cfg.ttp_response_discharge
            self._response_timer = cfg.ttp_response_refire_gap
            self.recovery = min(1.0, r + cfg.recovery_on_response)

        # ── freeze ───────────────────────────────────────────────────
        if self.freeze_val >= cfg.freeze_threshold and self._freeze_timer <= 0.0:
            events.append(_ev("freeze", self.freeze_val,
                              freeze_val=self.freeze_val, recovery=r))
            self._freeze_timer = cfg.freeze_refire_gap
            self.recovery = min(1.0, r + cfg.recovery_on_freeze)

        # ── recovery_backpressure_path decay (end of step) ───────────
        self.recovery -= self.recovery * cfg.recovery_decay * dt
        self.recovery = max(0.0, self.recovery)

        return events

    def state_dict(self) -> Dict[str, float]:
        return {
            "orient": self.orient,
            "settle": self.settle,
            "fidget_inhibit": self.fidget_inhibit,
            "ttp": self.ttp,
            "recovery": self.recovery,
            "freeze_val": self.freeze_val,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ev(name: str, strength: float, **kwargs: float) -> Dict[str, Any]:
    return {"name": name, "strength": float(strength),
            **{k: float(v) for k, v in kwargs.items()}}


def run_simulation(
    scenario_signals: Dict[str, np.ndarray],
    config: Optional[PathConfig] = None,
    dt: float = 1.0 / 60.0,
    seed: int = 42,
) -> tuple:
    """
    Step through pre-generated scenario signals.

    Returns:
        records   – list of per-timestep dicts (inputs + path values + event names)
        event_log – list of event dicts, each including a 'time' key
    """
    state = EchoLoopState(config=config, dt=dt, seed=seed)

    input_keys = ("speech_onset", "user_speaking", "speech_energy", "approach_velocity")
    n = len(scenario_signals["user_speaking"])
    t = scenario_signals.get("time", np.arange(n) * dt)

    silence_duration = 0.0
    records: List[Dict[str, Any]] = []
    event_log: List[Dict[str, Any]] = []

    for i in range(n):
        us = float(scenario_signals["user_speaking"][i])
        silence_duration = 0.0 if us > 0.5 else silence_duration + dt

        inputs = {k: float(scenario_signals[k][i]) for k in input_keys}
        inputs["silence_duration"] = silence_duration

        step_events = state.step(**inputs)

        record = {
            "time": float(t[i]),
            **inputs,
            **state.state_dict(),
            "events": ";".join(e["name"] for e in step_events),
        }
        records.append(record)

        for ev in step_events:
            event_log.append({"time": float(t[i]), **ev})

    return records, event_log
