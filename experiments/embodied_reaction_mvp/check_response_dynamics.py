"""
Diagnostic analysis: verify that response_ready timing is driven by internal
dynamics rather than a fixed timer.

Reads outputs/logs/<scenario>.csv and outputs/logs/<scenario>_events.json.
Writes outputs/reports/response_dynamics.md (and prints to stdout).

Usage:
    python3 check_response_dynamics.py
"""

import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
SCENARIOS = [
    "speech_then_silence",
    "repeated_speech_silence",
    "sudden_approach_while_speaking",
    "random_ambient_noise",
    "long_listening",
]
LOGS_DIR = os.path.join("outputs", "logs")
REPORTS_DIR = os.path.join("outputs", "reports")


# ---------------------------------------------------------------------------
# Low-level helpers (no numpy — keep the script self-contained)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson r; returns None when either series has zero variance."""
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    xs, ys = xs[:n], ys[:n]
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-9 or sy < 1e-9:
        return None
    return num / (sx * sy)


def _fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def _interpret_r(r: Optional[float]) -> str:
    if r is None:
        return "n/a (insufficient data)"
    a = abs(r)
    direction = "positive" if r > 0 else "negative"
    if a < 0.15:
        return f"{r:+.2f} — negligible"
    if a < 0.40:
        return f"{r:+.2f} — weak {direction}"
    if a < 0.65:
        return f"{r:+.2f} — moderate {direction}"
    return f"{r:+.2f} — strong {direction}"


# ---------------------------------------------------------------------------
# Per-scenario data extraction
# ---------------------------------------------------------------------------

def _load(scenario: str) -> Tuple[List[dict], List[dict]]:
    csv_path = os.path.join(LOGS_DIR, f"{scenario}.csv")
    json_path = os.path.join(LOGS_DIR, f"{scenario}_events.json")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    with open(json_path) as f:
        events = json.load(f)
    return rows, events


def _nearest_row(rows: List[dict], t: float) -> dict:
    """Return the row whose time is closest to t."""
    return min(rows, key=lambda r: abs(float(r["time"]) - t))


def _row_after(rows: List[dict], t: float, dt: float = 1.0 / 60) -> Optional[dict]:
    """Return the row whose time is closest to t + dt (i.e. one step after fire)."""
    target = t + dt
    r = min(rows, key=lambda r: abs(float(r["time"]) - target))
    return r if abs(float(r["time"]) - target) < dt * 3 else None


def _ttp_saturation(rows: List[dict]) -> Tuple[float, float]:
    """
    Returns (pct_at_saturation, max_continuous_sat_duration).
    Saturation threshold: ttp >= 0.99.
    """
    sat_count = 0
    max_run = 0.0
    cur_run = 0.0
    times = [float(r["time"]) for r in rows]
    dt_vals = [times[i + 1] - times[i] for i in range(len(times) - 1)] + [1 / 60]
    total_time = times[-1] - times[0] if len(times) > 1 else 0.0

    for r, dt in zip(rows, dt_vals):
        if float(r["ttp"]) >= 0.99:
            sat_count += 1
            cur_run += dt
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0.0

    pct = (sat_count / len(rows)) * 100.0 if rows else 0.0
    return pct, max_run


def analyze(scenario: str) -> dict:
    rows, events = _load(scenario)
    rr_events = [e for e in events if e["name"] == "response_ready"]

    # ── 1. intervals ────────────────────────────────────────────────
    times = sorted(e["time"] for e in rr_events)
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    # ── 2. ttp saturation ───────────────────────────────────────────
    pct_sat, max_sat_dur = _ttp_saturation(rows)

    # ── 3. ttp at firing ────────────────────────────────────────────
    ttp_at_fire = [e["ttp"] for e in rr_events]

    # ── 4. ttp immediately after discharge ──────────────────────────
    ttp_post = []
    for e in rr_events:
        r = _row_after(rows, e["time"])
        if r is not None:
            ttp_post.append(float(r["ttp"]))

    # ── 5. recovery at firing ───────────────────────────────────────
    recovery_at_fire = [e["recovery"] for e in rr_events]

    # ── 6. silence_duration at firing ───────────────────────────────
    silence_at_fire = []
    for e in rr_events:
        r = _nearest_row(rows, e["time"])
        silence_at_fire.append(float(r["silence_duration"]))

    # ── 6. correlations ─────────────────────────────────────────────
    # recovery at firing i  →  interval i (the interval that follows firing i)
    r_recovery_interval = _pearson(recovery_at_fire[:-1], intervals)
    # silence_duration at firing i  →  interval i
    r_silence_interval = _pearson(silence_at_fire[:-1], intervals)
    # ttp at firing i  →  discharge depth (ttp_at - ttp_post)
    discharge_depth = [a - b for a, b in zip(ttp_at_fire, ttp_post)]
    r_ttp_discharge = _pearson(ttp_at_fire, discharge_depth)

    return {
        "n_firings": len(rr_events),
        "intervals": intervals,
        "pct_sat": pct_sat,
        "max_sat_dur": max_sat_dur,
        "ttp_at_fire": ttp_at_fire,
        "ttp_post": ttp_post,
        "discharge_depth": discharge_depth,
        "recovery_at_fire": recovery_at_fire,
        "silence_at_fire": silence_at_fire,
        "r_recovery_interval": r_recovery_interval,
        "r_silence_interval": r_silence_interval,
        "r_ttp_discharge": r_ttp_discharge,
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

_CLOCK_THRESHOLD_STD = 0.08   # std below this = still clock-like (s)
_SAT_THRESHOLD_PCT   = 5.0    # ttp@1.0 above this = not dynamically rebuilding
_MODULATION_R_MIN    = 0.30   # |r| above this = recovery is modulating interval


def _interpret(name: str, d: dict) -> str:
    lines = []

    # Clock-like?
    ieis = d["intervals"]
    std = _std(ieis)
    if std is None:
        lines.append("- **Clock-like?** Cannot assess — fewer than 2 intervals.")
    elif std < _CLOCK_THRESHOLD_STD:
        lines.append(
            f"- **Clock-like?** YES — std={std:.3f} s is below {_CLOCK_THRESHOLD_STD} s. "
            "Intervals are still nearly uniform."
        )
    else:
        lines.append(
            f"- **Clock-like?** No — std={std:.3f} s indicates genuine interval variation."
        )

    # Dynamic rebuild?
    if d["pct_sat"] > _SAT_THRESHOLD_PCT:
        lines.append(
            f"- **ttp rebuilding dynamically?** No — ttp spent {d['pct_sat']:.1f}% of the run "
            "at saturation (≥0.99). Path dynamics cannot contribute while ttp is pinned at 1.0."
        )
    elif len(d["ttp_post"]) >= 2:
        post_std = _std(d["ttp_post"])
        lines.append(
            f"- **ttp rebuilding dynamically?** Yes — ttp never saturates "
            f"({d['pct_sat']:.1f}% at ≥0.99). Post-discharge values vary "
            f"({_fmt(min(d['ttp_post']))}-{_fmt(max(d['ttp_post']))}, std={_fmt(post_std)}), "
            "so rebuild length differs each cycle."
        )
    else:
        lines.append(f"- **ttp rebuilding dynamically?** ttp@1.0: {d['pct_sat']:.1f}%.")

    # Recovery modulating interval?
    r = d["r_recovery_interval"]
    if r is None:
        lines.append("- **Recovery modulating interval?** Cannot assess — insufficient data.")
    elif abs(r) >= _MODULATION_R_MIN:
        direction = "longer" if r > 0 else "shorter"
        lines.append(
            f"- **Recovery modulating interval?** Yes — r={r:+.2f} between recovery-at-fire "
            f"and next interval. Higher recovery → {direction} next gap, as designed."
        )
    else:
        lines.append(
            f"- **Recovery modulating interval?** Weak — r={r:+.2f}. Recovery influence on "
            "next interval is not strong enough to be the dominant source of variation."
        )

    # Synchronized?
    std_val = _std(ieis) if ieis else None
    if std_val is not None and std_val < _CLOCK_THRESHOLD_STD:
        lines.append(
            f"- **Synchronized?** This scenario still looks too synchronized (std={std_val:.3f} s). "
            "Consider checking whether the freeze path or recovery pulse is resetting all paths "
            "to a common state, or whether the base refire gap is the binding constraint."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _row_sep(widths: List[int]) -> str:
    return "|" + "|".join("-" * (w + 2) for w in widths) + "|"


def _table_row(cells: List[str], widths: List[int]) -> str:
    return "|" + "|".join(f" {c:<{w}} " for c, w in zip(cells, widths)) + "|"


def _section_scenario(name: str, d: dict) -> List[str]:
    lines: List[str] = []
    lines.append(f"## {name}")
    lines.append("")

    n = d["n_firings"]
    lines.append(f"**{n} firing(s) total.**")
    lines.append("")

    # 1. Intervals
    lines.append("### 1. response_ready intervals")
    if not d["intervals"]:
        lines.append("_Fewer than 2 firings — no intervals to report._")
    else:
        ieis = d["intervals"]
        lines.append(f"- List: {', '.join(f'{x:.2f}' for x in ieis)} s")
        lines.append(f"- Mean: {_fmt(_mean(ieis))} s")
        lines.append(f"- Std:  {_fmt(_std(ieis))} s")
        lines.append(f"- Min:  {_fmt(min(ieis))} s   Max: {_fmt(max(ieis))} s")
    lines.append("")

    # 2. ttp saturation
    lines.append("### 2. ttp saturation")
    lines.append(f"- Steps with ttp ≥ 0.99: **{d['pct_sat']:.2f}%**")
    lines.append(f"- Max continuous saturation: **{d['max_sat_dur']:.2f} s**")
    lines.append("")

    # 3. ttp at firing
    lines.append("### 3. ttp at response_ready firing")
    af = d["ttp_at_fire"]
    if af:
        lines.append(f"- Values: {', '.join(f'{x:.3f}' for x in af)}")
        lines.append(f"- Mean: {_fmt(_mean(af))}   Std: {_fmt(_std(af))}")
        lines.append(f"- Min:  {_fmt(min(af))}   Max: {_fmt(max(af))}")
    lines.append("")

    # 4. ttp after discharge
    lines.append("### 4. ttp immediately after discharge")
    po = d["ttp_post"]
    dd = d["discharge_depth"]
    if po:
        lines.append(f"- Post-fire values: {', '.join(f'{x:.3f}' for x in po)}")
        lines.append(f"- Mean: {_fmt(_mean(po))}   Std: {_fmt(_std(po))}")
        lines.append(f"- Min:  {_fmt(min(po))}   Max: {_fmt(max(po))}")
        if dd:
            lines.append(f"- Discharge depth (ttp_before − ttp_after): "
                         f"{', '.join(f'{x:.3f}' for x in dd)}")
            lines.append(f"  Mean: {_fmt(_mean(dd))}   Std: {_fmt(_std(dd))}")
    lines.append("")

    # 5. recovery at firing
    lines.append("### 5. recovery at response_ready firing")
    rf = d["recovery_at_fire"]
    if rf:
        lines.append(f"- Values: {', '.join(f'{x:.3f}' for x in rf)}")
        lines.append(f"- Mean: {_fmt(_mean(rf))}   Std: {_fmt(_std(rf))}")
    lines.append("")

    # 6. correlations
    lines.append("### 6. Correlation checks")
    lines.append(f"- recovery at firing → next interval:      {_interpret_r(d['r_recovery_interval'])}")
    lines.append(f"- silence_duration at firing → next interval: {_interpret_r(d['r_silence_interval'])}")
    lines.append(f"- ttp at firing → discharge depth:         {_interpret_r(d['r_ttp_discharge'])}")
    lines.append("")

    # 7. interpretation
    lines.append("### 7. Interpretation")
    lines.append(_interpret(name, d))
    lines.append("")
    lines.append("---")
    lines.append("")

    return lines


def build_report(results: Dict[str, dict]) -> str:
    out: List[str] = []
    out.append("# response_ready Dynamics Diagnostic")
    out.append("")
    out.append("Verifies that `response_ready` timing is driven by internal path dynamics")
    out.append("rather than a fixed timer. All values derived from simulation logs.")
    out.append("")

    # Cross-scenario summary table
    out.append("## Cross-scenario summary")
    out.append("")
    headers = ["Scenario", "n", "mean IEI", "std IEI", "ttp@≥0.99", "r(rec→IEI)", "r(sil→IEI)", "r(ttp→depth)"]
    col_w = [max(len(h), 38) if i == 0 else max(len(h), 10) for i, h in enumerate(headers)]
    col_w[0] = 38

    out.append(_table_row(headers, col_w))
    out.append(_row_sep(col_w))
    for name, d in results.items():
        ieis = d["intervals"]
        row = [
            name,
            str(d["n_firings"]),
            _fmt(_mean(ieis), 2) if ieis else "n/a",
            _fmt(_std(ieis), 2) if ieis else "n/a",
            f"{d['pct_sat']:.1f}%",
            _interpret_r(d["r_recovery_interval"]).split("—")[0].strip(),
            _interpret_r(d["r_silence_interval"]).split("—")[0].strip(),
            _interpret_r(d["r_ttp_discharge"]).split("—")[0].strip(),
        ]
        out.append(_table_row(row, col_w))
    out.append("")
    out.append("---")
    out.append("")

    for name, d in results.items():
        out.extend(_section_scenario(name, d))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results = {}
    for sc in SCENARIOS:
        csv_path = os.path.join(LOGS_DIR, f"{sc}.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {sc} — logs not found (run run_simulation.py first)")
            continue
        results[sc] = analyze(sc)

    report = build_report(results)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "response_dynamics.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n→ saved to {out_path}")


if __name__ == "__main__":
    main()
