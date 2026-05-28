"""
Visualization for EchoLoop embodied reaction simulations.
Uses matplotlib only (no seaborn).
"""

import os
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

EVENT_NAMES = [
    "gaze_shift",
    "posture_settle",
    "fidget_suppression",
    "micro_nod_ready",
    "response_ready",
    "freeze",
]

EVENT_COLORS = {
    "gaze_shift":        "#1E88E5",
    "posture_settle":    "#43A047",
    "fidget_suppression": "#FB8C00",
    "micro_nod_ready":   "#8E24AA",
    "response_ready":    "#E53935",
    "freeze":            "#546E7A",
}

INPUT_COLORS = {
    "speech_onset":      "#1565C0",
    "user_speaking":     "#0288D1",
    "speech_energy":     "#00ACC1",
    "silence_duration":  "#388E3C",
    "approach_velocity": "#BF360C",
}

PATH_COLORS = {
    "orient":         "#1E88E5",
    "settle":         "#43A047",
    "fidget_inhibit": "#FB8C00",
    "ttp":            "#8E24AA",
    "recovery":       "#E53935",
    "freeze_val":     "#546E7A",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _extract(records: List[Dict], key: str) -> List[float]:
    return [r[key] for r in records]


# ------------------------------------------------------------------
# Plot 1 — input signals
# ------------------------------------------------------------------

def plot_inputs(
    records: List[Dict[str, Any]],
    scenario_name: str,
    output_dir: str,
) -> None:
    signals = [
        "speech_onset",
        "user_speaking",
        "speech_energy",
        "silence_duration",
        "approach_velocity",
    ]
    t = _extract(records, "time")
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Input Signals  —  {scenario_name}", fontsize=12)

    for ax, sig in zip(axes, signals):
        y = _extract(records, sig)
        ax.plot(t, y, color=INPUT_COLORS.get(sig, "#444"), linewidth=0.85)
        ax.set_ylabel(sig, fontsize=8, labelpad=2)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=min(-0.05, float(min(y)) - 0.02))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"{scenario_name}_inputs.png"))


# ------------------------------------------------------------------
# Plot 2 — internal path energy
# ------------------------------------------------------------------

def plot_internals(
    records: List[Dict[str, Any]],
    scenario_name: str,
    output_dir: str,
) -> None:
    paths = ["orient", "settle", "fidget_inhibit", "ttp", "recovery", "freeze_val"]
    t = _extract(records, "time")
    fig, axes = plt.subplots(len(paths), 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Internal Path Energy  —  {scenario_name}", fontsize=12)

    for ax, path in zip(axes, paths):
        y = _extract(records, path)
        color = PATH_COLORS.get(path, "#444")
        ax.fill_between(t, y, alpha=0.15, color=color)
        ax.plot(t, y, color=color, linewidth=0.85)
        ax.set_ylabel(path, fontsize=8, labelpad=2)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.02, max(1.05, float(max(y)) + 0.05))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"{scenario_name}_internals.png"))


# ------------------------------------------------------------------
# Plot 3 — output event raster
# ------------------------------------------------------------------

def plot_event_raster(
    records: List[Dict[str, Any]],
    event_log: List[Dict[str, Any]],
    scenario_name: str,
    output_dir: str,
) -> None:
    t = _extract(records, "time")
    t_max = t[-1]

    fig, (ax_raster, ax_ctx) = plt.subplots(
        2, 1, figsize=(12, 5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(f"Output Event Raster  —  {scenario_name}", fontsize=12)

    # Raster rows
    for y_pos, ename in enumerate(EVENT_NAMES):
        ev_times = [e["time"] for e in event_log if e["name"] == ename]
        ev_strengths = [e["strength"] for e in event_log if e["name"] == ename]
        for tx, s in zip(ev_times, ev_strengths):
            ax_raster.vlines(
                tx, y_pos + 0.08, y_pos + 0.92,
                colors=EVENT_COLORS[ename],
                linewidth=max(1.2, s * 3.0),
            )

    ax_raster.set_yticks(np.arange(len(EVENT_NAMES)) + 0.5)
    ax_raster.set_yticklabels(EVENT_NAMES, fontsize=9)
    ax_raster.set_xlim(0.0, t_max)
    ax_raster.set_ylim(-0.1, len(EVENT_NAMES) + 0.1)
    ax_raster.grid(True, axis="x", alpha=0.25)
    ax_raster.set_ylabel("Event type", fontsize=9)

    # Context: ttp + user_speaking shading
    ttp = _extract(records, "ttp")
    speaking = _extract(records, "user_speaking")
    ax_ctx.fill_between(t, ttp, alpha=0.28, color=PATH_COLORS["ttp"], label="ttp")
    ax_ctx.plot(t, ttp, color=PATH_COLORS["ttp"], linewidth=0.8)
    ax_ctx.fill_between(t, speaking, alpha=0.12, color=INPUT_COLORS["user_speaking"], label="speaking")
    ax_ctx.set_ylabel("ttp / speaking", fontsize=8)
    ax_ctx.set_ylim(0.0, 1.08)
    ax_ctx.tick_params(labelsize=7)
    ax_ctx.legend(fontsize=7, loc="upper right")
    ax_ctx.grid(True, alpha=0.25)
    ax_ctx.set_xlabel("Time (s)", fontsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"{scenario_name}_raster.png"))


# ------------------------------------------------------------------
# Plot 4 — inter-event intervals
# ------------------------------------------------------------------

def plot_iei(
    event_log: List[Dict[str, Any]],
    scenario_name: str,
    output_dir: str,
) -> None:
    event_types = [
        e for e in EVENT_NAMES
        if sum(1 for ev in event_log if ev["name"] == e) >= 2
    ]
    if not event_types:
        return

    n_cols = len(event_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3), squeeze=False)
    fig.suptitle(f"Inter-Event Intervals  —  {scenario_name}", fontsize=11)

    for ax, ename in zip(axes[0], event_types):
        times = sorted(e["time"] for e in event_log if e["name"] == ename)
        intervals = np.diff(times)
        color = EVENT_COLORS[ename]
        n_bins = max(3, len(intervals) // 2 + 1)
        iei_range = float(intervals.max() - intervals.min())
        if iei_range < 1e-6:
            # All intervals identical — plot as a single bar
            ax.bar([float(intervals[0])], [len(intervals)],
                   width=0.1, color=color, edgecolor="white", alpha=0.85)
        else:
            ax.hist(
                intervals,
                bins=n_bins,
                range=(float(intervals.min()), float(intervals.max())),
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
        ax.set_title(ename, fontsize=9)
        ax.set_xlabel("IEI (s)", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"{scenario_name}_iei.png"))
