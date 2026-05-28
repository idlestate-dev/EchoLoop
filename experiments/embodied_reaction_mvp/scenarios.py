"""
Scenario signal generators for embodied reaction experiments.

Each function returns a dict of numpy arrays (input signals only).
No event logic lives here.  Output events emerge from EchoLoop path dynamics.
"""

import numpy as np
from typing import Dict, Optional

DT = 1.0 / 60.0


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _time_array(duration: float, dt: float = DT) -> np.ndarray:
    return np.arange(0.0, duration, dt)


def _gaussian_pulse(
    t: np.ndarray, center: float, width: float, amplitude: float
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _speech_block(
    t: np.ndarray,
    start: float,
    end: float,
    energy_mean: float = 0.65,
    energy_noise: float = 0.12,
    rng: Optional[np.random.Generator] = None,
):
    """Return (onset_arr, speaking_arr, energy_arr) for one contiguous speech block."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(t)
    onset = np.zeros(n)
    speaking = np.zeros(n)
    energy = np.zeros(n)

    mask = (t >= start) & (t < end)
    speaking[mask] = 1.0

    idx = np.where(mask)[0]
    if len(idx):
        onset[idx[0]] = 1.0

    dur = end - start
    t_local = t[mask] - start
    taper = np.clip(np.minimum(t_local, dur - t_local) / 0.30, 0.0, 1.0)
    raw = energy_mean + energy_noise * rng.standard_normal(mask.sum())
    raw += 0.10 * np.sin(2.0 * np.pi * 2.5 * t[mask])
    energy[mask] = np.clip(raw * taper, 0.0, 1.0)

    return onset, speaking, energy


def _empty(t: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(t)
    return {
        "time": t,
        "speech_onset": np.zeros(n),
        "user_speaking": np.zeros(n),
        "speech_energy": np.zeros(n),
        "approach_velocity": np.zeros(n),
    }



# ------------------------------------------------------------------
# Scenario 1 — user speaks, then extended silence
# ------------------------------------------------------------------

def speech_then_silence(
    duration: float = 25.0, dt: float = DT, seed: int = 1
) -> Dict[str, np.ndarray]:
    """Single speech block (t=2–10 s), then uninterrupted silence."""
    t = _time_array(duration, dt)
    rng = np.random.default_rng(seed)
    s = _empty(t)

    onset, speaking, energy = _speech_block(t, 2.0, 10.0, rng=rng)
    s["speech_onset"] += onset
    s["user_speaking"] += speaking
    s["speech_energy"] += energy
    return s


# ------------------------------------------------------------------
# Scenario 2 — alternating speech / silence cycles
# ------------------------------------------------------------------

def repeated_speech_silence(
    duration: float = 28.0, dt: float = DT, seed: int = 2
) -> Dict[str, np.ndarray]:
    """Four speech blocks separated by silence gaps of varying length."""
    t = _time_array(duration, dt)
    rng = np.random.default_rng(seed)
    s = _empty(t)

    for start, end in [(1.5, 4.5), (7.0, 10.5), (13.0, 16.0), (19.5, 22.5)]:
        onset, speaking, energy = _speech_block(t, start, end, rng=rng)
        s["speech_onset"] += onset
        s["user_speaking"] += speaking
        s["speech_energy"] += energy
    return s


# ------------------------------------------------------------------
# Scenario 3 — sudden approach just after speech ends
# ------------------------------------------------------------------

def sudden_approach_while_speaking(
    duration: float = 25.0, dt: float = DT, seed: int = 3
) -> Dict[str, np.ndarray]:
    """Speech t=2–13 s, then a brief high-velocity approach at t≈13.8 s."""
    t = _time_array(duration, dt)
    rng = np.random.default_rng(seed)
    s = _empty(t)

    onset, speaking, energy = _speech_block(t, 2.0, 13.0, rng=rng)
    s["speech_onset"] += onset
    s["user_speaking"] += speaking
    s["speech_energy"] += energy

    # Sharp approach pulse; amplitude chosen to reliably cross freeze threshold
    s["approach_velocity"] += _gaussian_pulse(t, center=13.8, width=0.25, amplitude=2.5)
    return s


# ------------------------------------------------------------------
# Scenario 4 — random ambient noise with sporadic short bursts
# ------------------------------------------------------------------

def random_ambient_noise(
    duration: float = 25.0, dt: float = DT, seed: int = 4
) -> Dict[str, np.ndarray]:
    """Continuous low-level energy with short, irregular speech bursts.
    Approach velocity pulses are kept small (below freeze threshold)."""
    t = _time_array(duration, dt)
    rng = np.random.default_rng(seed)
    s = _empty(t)

    # Continuous low-level energy
    base = 0.15 + 0.08 * rng.standard_normal(len(t))
    s["speech_energy"] = np.clip(base, 0.0, 0.35)

    # Short sporadic speech bursts
    burst_params = [(4.0, 0.5, 0.55), (8.5, 0.4, 0.50),
                    (14.0, 0.7, 0.60), (18.0, 0.45, 0.50), (22.0, 0.55, 0.55)]
    for center, half_width, amplitude in burst_params:
        start, end = center - half_width, center + half_width
        onset, speaking, energy = _speech_block(
            t, start, end, energy_mean=amplitude, rng=rng
        )
        s["speech_onset"] += onset
        s["user_speaking"] = np.clip(s["user_speaking"] + speaking, 0.0, 1.0)
        s["speech_energy"] += energy

    s["speech_energy"] = np.clip(s["speech_energy"], 0.0, 1.0)

    # Subtle ambient approach fluctuations — amplitude 0.12, well below freeze threshold
    for center in [6.0, 16.0]:
        s["approach_velocity"] += _gaussian_pulse(t, center, 0.6, 0.12)

    return s


# ------------------------------------------------------------------
# Scenario 5 — long silence, brief speech, then silence again
# ------------------------------------------------------------------

def long_listening(
    duration: float = 30.0, dt: float = DT, seed: int = 5
) -> Dict[str, np.ndarray]:
    """18 s of silence, a short speech burst at t=18–21 s, then 9 s of silence."""
    t = _time_array(duration, dt)
    rng = np.random.default_rng(seed)
    s = _empty(t)

    onset, speaking, energy = _speech_block(t, 18.0, 21.0, rng=rng)
    s["speech_onset"] += onset
    s["user_speaking"] += speaking
    s["speech_energy"] += energy
    return s


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

SCENARIOS = {
    "speech_then_silence": speech_then_silence,
    "repeated_speech_silence": repeated_speech_silence,
    "sudden_approach_while_speaking": sudden_approach_while_speaking,
    "random_ambient_noise": random_ambient_noise,
    "long_listening": long_listening,
}


def get_scenario(name: str, **kwargs) -> Dict[str, np.ndarray]:
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {name!r}.  Available: {list(SCENARIOS)}"
        )
    return SCENARIOS[name](**kwargs)
