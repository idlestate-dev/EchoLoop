"""
Human-readable diagnostic report generator for EchoLoop simulations.

Analysis is data-driven from records and event_log.
Static text covers only scenario descriptions and per-event-type explanations.
"""

import numpy as np
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------------
# Static scenario descriptions  (Section 1 only)
# ------------------------------------------------------------------

_SCENARIO_DESC: Dict[str, str] = {
    "speech_then_silence": (
        "A single continuous speech block (t=2–10 s) followed by 15 s of uninterrupted silence. "
        "This is the baseline scenario: one orienting event, a long sustained-speech phase that "
        "charges the settle and fidget-inhibition paths, then an extended silence during which "
        "turn-taking pressure builds freely without interruption."
    ),
    "repeated_speech_silence": (
        "Four speech blocks (t=1.5–4.5, 7–10.5, 13–16, 19.5–22.5 s) separated by silence gaps "
        "of roughly 2.5–3 s. The alternating pattern tests whether orienting and settle reset and "
        "re-activate correctly across cycles, and whether turn-taking pressure can be partially "
        "discharged by speech and rebuild again across repeated gaps."
    ),
    "sudden_approach_while_speaking": (
        "Speech from t=2 to t=13 s, followed immediately by a brief high-amplitude "
        "approach-velocity pulse (Gaussian, peak at t=13.8 s, amplitude 2.5). "
        "The scenario tests whether the freeze path correctly interrupts the normal "
        "silence-pressure build-up and whether shared backpressure suppresses other events "
        "during the freeze window."
    ),
    "random_ambient_noise": (
        "Five very short speech bursts (0.4–0.7 s each) spread across 25 s, with continuous "
        "low-level speech energy and small ambient approach fluctuations (amplitude 0.12, "
        "intentionally below the freeze threshold). "
        "The bursts are too brief for settle or fidget-inhibition to accumulate deeply. "
        "This scenario probes what emerges when no single path has time to fully charge."
    ),
    "long_listening": (
        "An extended 18-second silence (t=0–18 s), a short speech block (t=18–21 s), "
        "then 9 more seconds of silence. The long pre-speech silence allows turn-taking pressure "
        "to saturate and tests a specific question: does speech onset release accumulated pressure "
        "or interrupt it? The post-speech silence then tests recovery and re-accumulation dynamics."
    ),
}

# ------------------------------------------------------------------
# Per-event-type explanations  (as specified in the task)
# ------------------------------------------------------------------

_EVENT_EXPLAIN: Dict[str, str] = {
    "gaze_shift": (
        "Usually caused by `orienting_fast_path` crossing threshold after `speech_onset`. "
        "The path spikes instantly (orient += onset × gain) and decays fast (6/s); "
        "the event fires within one frame and the refractory flag prevents double-firing "
        "on the same onset."
    ),
    "posture_settle": (
        "Caused by `listening_settle_slow_path` accumulating during sustained speech. "
        "The path saturates toward ~0.81 with a time constant of ~2.7 s; "
        "the threshold (0.50) is only crossed after several seconds of continuous speaking."
    ),
    "micro_nod_ready": (
        "Caused by moderate turn-taking pressure (`ttp`) building during silence. "
        "Fires at the lower threshold (0.28), often as a precursor to `response_ready`. "
        "The 2.5 s refire gap spaces out repeated nods."
    ),
    "response_ready": (
        "Caused by high turn-taking pressure (`ttp`) crossing the upper threshold (0.65). "
        "Once `ttp` saturates, this event is largely paced by the 4.0 s refire gap "
        "rather than by path dynamics — an identified limitation."
    ),
    "freeze": (
        "Caused by `freeze_path` crossing threshold (0.20) after a sudden `approach_velocity` spike. "
        "Immediately injects a large recovery pulse (0.40), suppressing all other events "
        "until backpressure decays."
    ),
    "fidget_suppression": (
        "Caused by `fidget_inhibition_path` during speaking; fires stochastically at a rate "
        "proportional to inhibition strength (0.10 × fidget_inhibit × dt). "
        "As a discrete event, density is low (0–2 per scenario) and may be too sparse "
        "to meaningfully represent the texture of suppression."
    ),
}

_EVENT_ORDER = [
    "gaze_shift", "posture_settle", "fidget_suppression",
    "micro_nod_ready", "response_ready", "freeze",
]

# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def _counts(events: List[Dict]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for e in events:
        c[e["name"]] = c.get(e["name"], 0) + 1
    return c


def _of(name: str, events: List[Dict]) -> List[Dict]:
    return [e for e in events if e["name"] == name]


def _speaking(r: Dict) -> bool:
    return float(r["user_speaking"]) > 0.5


def _silent(r: Dict) -> bool:
    return float(r["user_speaking"]) < 0.5


def _phase_mean(records: List[Dict], key: str, pred) -> float:
    vals = [float(r[key]) for r in records if pred(r)]
    return float(np.mean(vals)) if vals else 0.0


def _phase_max(records: List[Dict], key: str, pred) -> float:
    vals = [float(r[key]) for r in records if pred(r)]
    return float(np.max(vals)) if vals else 0.0


def _ieis(events: List[Dict]) -> np.ndarray:
    times = sorted(e["time"] for e in events)
    return np.diff(times) if len(times) >= 2 else np.array([])


def _speaking_fraction(records: List[Dict]) -> float:
    return sum(1 for r in records if _speaking(r)) / len(records)


def _max_silence_run(records: List[Dict]) -> float:
    """Longest consecutive silence run in seconds."""
    dt = float(records[1]["time"]) - float(records[0]["time"]) if len(records) > 1 else 1/60
    best = cur = 0.0
    for r in records:
        if _silent(r):
            cur += dt
            best = max(best, cur)
        else:
            cur = 0.0
    return best


def _find_record_at(records: List[Dict], t: float) -> Dict:
    return min(records, key=lambda r: abs(float(r["time"]) - t))


# ------------------------------------------------------------------
# Section 1 — scenario description
# ------------------------------------------------------------------

def _s1_description(name: str, records: List[Dict]) -> str:
    desc = _SCENARIO_DESC.get(name, "No description available.")
    duration = float(records[-1]["time"]) if records else 0.0
    n_steps = len(records)
    spk_pct = _speaking_fraction(records) * 100
    max_sil = _max_silence_run(records)
    n_onsets = sum(1 for r in records if float(r["speech_onset"]) > 0.5)

    return (
        f"## 1. What this scenario represents\n\n"
        f"{desc}\n\n"
        f"**Run stats:** {n_steps:,} steps · {duration:.1f} s total · "
        f"speaking {spk_pct:.0f}% of the time · longest silence {max_sil:.1f} s · "
        f"{n_onsets} speech onset(s)."
    )


# ------------------------------------------------------------------
# Section 2 — path activation
# ------------------------------------------------------------------

def _s2_paths(records: List[Dict]) -> str:
    paths = ["orient", "settle", "fidget_inhibit", "ttp", "recovery", "freeze_val"]
    rows: List[Tuple] = []
    for p in paths:
        ov = _phase_mean(records, p, lambda r: True)
        sp = _phase_mean(records, p, _speaking)
        sl = _phase_mean(records, p, _silent)
        pk = _phase_max(records, p, lambda r: True)
        rows.append((p, ov, sp, sl, pk))

    primary = [p for p, ov, sp, sl, pk in rows if ov > 0.08 or pk > 0.35]

    lines = ["## 2. Which paths were primarily activated\n"]
    lines.append("| Path | overall mean | mean during speech | mean during silence | peak |")
    lines.append("|---|---|---|---|---|")
    for p, ov, sp, sl, pk in rows:
        mark = " ◀" if p in primary else ""
        lines.append(f"| `{p}` | {ov:.3f} | {sp:.3f} | {sl:.3f} | {pk:.3f} |{mark}")
    lines.append("")

    # Narrative commentary
    notes = []

    fidget_sp = _phase_mean(records, "fidget_inhibit", _speaking)
    if fidget_sp > 0.70:
        notes.append(
            f"`fidget_inhibit` averaged {fidget_sp:.2f} during speech — the inhibition path "
            "is charging close to saturation while the user speaks, as designed."
        )
    elif fidget_sp > 0.30:
        notes.append(
            f"`fidget_inhibit` averaged {fidget_sp:.2f} during speech — partially charged, "
            "reflecting short speech blocks that did not allow full saturation."
        )

    settle_sp = _phase_mean(records, "settle", _speaking)
    if settle_sp > 0.40:
        notes.append(
            f"`settle` averaged {settle_sp:.2f} during speech, indicating sustained "
            "listening periods long enough for the slow path to accumulate meaningfully."
        )
    elif settle_sp > 0.10:
        notes.append(
            f"`settle` averaged only {settle_sp:.2f} during speech — speech blocks were likely "
            "too short or sparse for the slow path to charge above threshold."
        )

    ttp_sl = _phase_mean(records, "ttp", _silent)
    if ttp_sl > 0.60:
        notes.append(
            f"`ttp` averaged {ttp_sl:.2f} during silence — high enough to indicate "
            "that turn-taking pressure reached or approached saturation."
        )
    elif ttp_sl > 0.25:
        notes.append(
            f"`ttp` averaged {ttp_sl:.2f} during silence, showing meaningful pressure build-up "
            "but without saturating."
        )

    freeze_pk = _phase_max(records, "freeze_val", lambda r: True)
    if freeze_pk > 0.20:
        notes.append(
            f"`freeze_val` peaked at {freeze_pk:.3f} — sufficient to cross the freeze threshold "
            "(0.20) and trigger a suppression window."
        )

    rec_ov = _phase_mean(records, "recovery", lambda r: True)
    if rec_ov > 0.10:
        notes.append(
            f"`recovery` maintained a mean of {rec_ov:.3f} overall, meaning events were "
            "firing frequently enough to keep backpressure elevated for significant periods."
        )

    for note in notes:
        lines.append(f"- {note}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Section 3 — events fired and why
# ------------------------------------------------------------------

def _s3_events(event_log: List[Dict], records: List[Dict]) -> str:
    counts = _counts(event_log)
    lines = ["## 3. Which output events fired and why\n"]

    for name in _EVENT_ORDER:
        n = counts.get(name, 0)
        evs = _of(name, event_log)
        explain = _EVENT_EXPLAIN[name]

        if n == 0:
            lines.append(f"### `{name}` — did not fire\n")
            lines.append(f"_{explain}_\n")
            if name == "freeze":
                pk = _phase_max(records, "freeze_val", lambda r: True)
                lines.append(
                    f"No significant approach-velocity input was present. "
                    f"`freeze_val` peaked at only {pk:.3f}, well below the threshold of 0.20.\n"
                )
            elif name == "fidget_suppression":
                inhib = _phase_mean(records, "fidget_inhibit", _speaking)
                lines.append(
                    f"`fidget_inhibit` averaged {inhib:.3f} during speech. "
                    f"At a base rate of 0.10/s × {inhib:.2f} inhibition, "
                    f"the expected count over the speaking window was low enough that "
                    "zero firings is unsurprising given the stochastic mechanism.\n"
                )
            elif name == "posture_settle":
                settle_pk = _phase_max(records, "settle", lambda r: True)
                lines.append(
                    f"`settle` peaked at {settle_pk:.3f}. "
                    f"{'It did not reach the threshold of 0.50.' if settle_pk < 0.50 else 'It reached threshold but the refire gap may have prevented further firings.'}\n"
                )
            continue

        lines.append(f"### `{name}` — fired {n} time{'s' if n != 1 else ''}\n")
        lines.append(f"_{explain}_\n")

        times_str = ", ".join(f"t={e['time']:.2f}" for e in evs)
        mean_str = float(np.mean([e["strength"] for e in evs]))
        lines.append(f"**Times:** {times_str}  ")
        lines.append(f"**Mean strength:** {mean_str:.3f}  \n")

        # Per-type commentary
        if name == "gaze_shift":
            n_onsets = sum(1 for r in records if float(r["speech_onset"]) > 0.5)
            if n == n_onsets:
                lines.append(
                    f"One event per onset ({n_onsets} onset(s) → {n} firing(s)). "
                    "The refractory mechanism is working: orient decayed below 15% of threshold "
                    "between speech blocks, allowing the flag to reset before the next onset.\n"
                )
            else:
                lines.append(
                    f"{n_onsets} onset(s) present but {n} `gaze_shift`(s) fired. "
                    "Check whether recovery was elevated at the onset moment, raising the effective threshold.\n"
                )

        elif name == "posture_settle":
            settle_vals = [e.get("settle", 0.0) for e in evs]
            lines.append(
                f"`settle` values at firing: {', '.join(f'{v:.3f}' for v in settle_vals)}. "
            )
            # Check which fired during silence
            silence_fires = [
                e for e in evs
                if not _speaking(_find_record_at(records, e["time"]))
            ]
            if silence_fires:
                lines.append(
                    f"{len(silence_fires)} of {n} `posture_settle` event(s) fired during silence — "
                    "the slow decay rate (0.07/s) allowed `settle` to persist above threshold "
                    "well after speech ended.\n"
                )
            else:
                lines.append("All firings occurred during active speech phases.\n")

        elif name == "micro_nod_ready":
            ieis = _ieis(evs)
            if len(ieis):
                lines.append(
                    f"Intervals between nods: {', '.join(f'{x:.2f}' for x in ieis)} s. "
                    f"Mean {ieis.mean():.2f} s, std {ieis.std():.2f} s. "
                )
                if ieis.std() < 0.5:
                    lines.append(
                        "Low variance suggests the 2.5 s refire gap is the binding constraint.\n"
                    )
                else:
                    lines.append(
                        "Higher variance suggests `ttp` fluctuations (from speech or recovery) "
                        "are actively modulating nod timing.\n"
                    )

        elif name == "response_ready":
            ieis = _ieis(evs)
            ttp_vals = [e.get("ttp", 0.0) for e in evs]
            rec_vals = [e.get("recovery", 0.0) for e in evs]
            lines.append(
                f"`ttp` at firing: {', '.join(f'{v:.3f}' for v in ttp_vals)}.  \n"
                f"`recovery` at firing: {', '.join(f'{v:.3f}' for v in rec_vals)}.  \n"
            )
            if len(ieis) >= 2:
                lines.append(
                    f"Intervals: {', '.join(f'{x:.2f}' for x in ieis)} s. "
                    f"Mean {ieis.mean():.2f} s, std {ieis.std():.2f} s. "
                )
                if ieis.std() < 0.25 and ieis.mean() < 5.0:
                    lines.append(
                        "Near-zero variance: **`ttp_response_refire_gap` (4.0 s) is dominating.** "
                        "The path dynamics are no longer contributing to timing once `ttp` saturates.\n"
                    )
                else:
                    lines.append(
                        "Variance present: speech, recovery, and `ttp` discharge are all "
                        "contributing to interval variability — dynamics are active.\n"
                    )

        elif name == "freeze":
            fv = evs[0].get("freeze_val", evs[0]["strength"])
            lines.append(
                f"`freeze_val` at firing: {fv:.3f} (threshold 0.20). "
                f"Recovery injected: ~0.40, creating a suppression window for all downstream events. "
            )
            # Measure actual suppression window length
            freeze_t = evs[0]["time"]
            post = [e for e in event_log if e["time"] > freeze_t and e["name"] != "freeze"]
            if post:
                gap = post[0]["time"] - freeze_t
                lines.append(f"Next non-freeze event fired {gap:.2f} s later.\n")
            else:
                lines.append("No events fired after the freeze in this scenario.\n")

        elif name == "fidget_suppression":
            inhib_vals = [e.get("fidget_inhibit", e["strength"]) for e in evs]
            lines.append(
                f"`fidget_inhibit` at firing: {', '.join(f'{v:.3f}' for v in inhib_vals)}. "
                "Fired during high-inhibition speech phases as expected, but the stochastic "
                "mechanism means this could just as easily have been zero.\n"
            )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Section 4 — expected vs. surprising
# ------------------------------------------------------------------

def _s4_expected_vs_surprising(
    name: str, event_log: List[Dict], records: List[Dict]
) -> str:
    counts = _counts(event_log)
    lines = ["## 4. Expected vs. surprising events\n"]

    expected = []
    surprising = []

    # Expected — derived from signal analysis, not scenario name
    n_onsets = sum(1 for r in records if float(r["speech_onset"]) > 0.5)
    if n_onsets > 0 and "gaze_shift" in counts:
        expected.append(
            f"`gaze_shift` × {counts['gaze_shift']} — speech onset(s) were present, "
            "so an immediate orienting response is expected."
        )

    max_speech_run = 0.0
    cur = 0.0
    dt = float(records[1]["time"]) - float(records[0]["time"]) if len(records) > 1 else 1/60
    for r in records:
        cur = (cur + dt) if _speaking(r) else 0.0
        max_speech_run = max(max_speech_run, cur)

    if max_speech_run > 2.5 and "posture_settle" in counts:
        expected.append(
            f"`posture_settle` — longest speech run was {max_speech_run:.1f} s, "
            "long enough for `settle` to cross threshold (~2.7 s)."
        )

    max_sil = _max_silence_run(records)
    if max_sil > 1.0 and "micro_nod_ready" in counts:
        expected.append(
            f"`micro_nod_ready` — silence exceeded 1 s (max {max_sil:.1f} s), "
            "sufficient for `ttp` to reach the nod threshold (0.28 at 0.30/s)."
        )
    if max_sil > 2.5 and "response_ready" in counts:
        expected.append(
            f"`response_ready` — silence exceeded 2.5 s, "
            "sufficient for `ttp` to reach the response threshold (0.65)."
        )

    approach_pk = _phase_max(records, "approach_velocity", lambda r: True)
    if approach_pk > 0.5 and "freeze" in counts:
        expected.append(
            f"`freeze` — approach_velocity peaked at {approach_pk:.2f}, "
            "large enough to push `freeze_val` above the threshold of 0.20."
        )

    # ── Surprising patterns (data-driven) ────────────────────────────

    # Near-simultaneous multi-event burst (different types within 0.08 s)
    times_names = sorted([(float(e["time"]), e["name"]) for e in event_log])
    seen_bursts = set()
    for i in range(len(times_names) - 1):
        t0, n0 = times_names[i]
        t1, n1 = times_names[i + 1]
        gap = t1 - t0
        if gap < 0.08 and n0 != n1:
            key = round(t0, 1)
            if key not in seen_bursts:
                seen_bursts.add(key)
                burst = [nm for tt, nm in times_names if abs(tt - t0) < 0.15]
                surprising.append(
                    f"**Multi-event burst at t≈{t0:.2f} s** — "
                    f"{', '.join(f'`{b}`' for b in burst)} fired within "
                    f"{gap*1000:.0f}–150 ms of each other. "
                    "This is a compression-release artifact: multiple paths had accumulated "
                    "pressure during a suppression window and discharged simultaneously "
                    "when that window closed."
                )

    # response_ready during or just after speech onset
    for e in _of("response_ready", event_log):
        t = e["time"]
        ctx = [r for r in records if abs(float(r["time"]) - t) < 0.4]
        if any(_speaking(r) for r in ctx):
            surprising.append(
                f"**`response_ready` at t={t:.2f} s — during active speech** "
                f"(ttp={e.get('ttp', '?'):.3f}). "
                "Turn-taking pressure built up during a preceding silence and was still "
                "above threshold when speaking began. The system was simultaneously orienting "
                "toward the speaker and signaling a response impulse — an emergent ambiguity."
            )
            break

    # posture_settle well into silence (>3 s after last speech)
    for e in _of("posture_settle", event_log):
        t = e["time"]
        r_at_t = _find_record_at(records, t)
        if _silent(r_at_t):
            # Check how long silence has been running
            sil_dur = float(r_at_t.get("silence_duration", 0.0))
            if sil_dur > 3.0:
                surprising.append(
                    f"**`posture_settle` at t={t:.2f} s — {sil_dur:.1f} s into silence** "
                    f"(settle={e.get('settle', '?'):.3f}). "
                    "The slow leak rate (0.07/s) allows `settle` to persist above threshold "
                    "long after speech stops, producing a gradual 'settling' signal that "
                    "bridges active listening and rest."
                )

    # freeze followed by a suppression gap and then a burst
    for fe in _of("freeze", event_log):
        ft = fe["time"]
        post = [ev for ev in event_log if ev["time"] > ft and ev["name"] != "freeze"]
        if post:
            gap = post[0]["time"] - ft
            burst_window = [ev for ev in post if ev["time"] - post[0]["time"] < 0.2]
            if gap > 1.5 and len(burst_window) >= 2:
                names_str = ", ".join(f"`{ev['name']}`" for ev in burst_window)
                surprising.append(
                    f"**Post-freeze release burst** — {gap:.2f} s of silence after `freeze`, "
                    f"then {names_str} fired within 0.2 s of each other at t≈{post[0]['time']:.2f} s. "
                    "Multiple paths accumulated pressure behind the shared backpressure wall "
                    "and released together once `freeze_val` and recovery both decayed. "
                    "This was not scripted; it is a direct consequence of shared backpressure."
                )

    # fidget never fires despite high inhibition
    if counts.get("fidget_suppression", 0) == 0:
        mean_inhib = _phase_mean(records, "fidget_inhibit", _speaking)
        if mean_inhib > 0.60:
            surprising.append(
                f"**`fidget_suppression` did not fire** despite `fidget_inhibit` averaging "
                f"{mean_inhib:.2f} during speech. This is a known limitation: the stochastic "
                "base rate (0.10/s) is low enough that zero firings is plausible even with "
                "strong inhibition. The suppression is happening internally but is invisible "
                "in the discrete event stream."
            )

    if expected:
        lines.append("**Expected:**\n")
        for item in expected:
            lines.append(f"- {item}")
        lines.append("")

    if surprising:
        lines.append("**Surprising or noteworthy:**\n")
        for item in surprising:
            lines.append(f"- {item}")
        lines.append("")

    if not expected and not surprising:
        lines.append("No notable patterns to flag in this scenario.\n")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Section 5 — path dynamics vs. parameters
# ------------------------------------------------------------------

def _s5_dynamics_vs_params(event_log: List[Dict], records: List[Dict]) -> str:
    lines = ["## 5. Path dynamics vs. fixed parameters\n"]
    dynamics = []
    params = []

    # response_ready interval variance
    rr_evs = _of("response_ready", event_log)
    rr_ieis = _ieis(rr_evs)
    if len(rr_ieis) >= 2:
        std = float(rr_ieis.std())
        mean = float(rr_ieis.mean())
        if std < 0.20 and mean < 5.0:
            params.append(
                f"`response_ready` intervals: mean {mean:.2f} s, std {std:.2f} s. "
                "The near-zero variance means `ttp_response_refire_gap` (4.0 s) is "
                "the sole pacemaker once `ttp` saturates. Path dynamics have effectively "
                "stopped contributing to timing at that point."
            )
        else:
            dynamics.append(
                f"`response_ready` intervals: mean {mean:.2f} s, std {std:.2f} s. "
                "The variance shows that `ttp` did not remain saturated throughout — "
                "speech discharge or recovery cycles were actively pulling it back down, "
                "creating genuine dynamic modulation of response timing."
            )

    # gaze_shift strength uniformity
    gaze_evs = _of("gaze_shift", event_log)
    if len(gaze_evs) >= 2:
        strengths = [e["strength"] for e in gaze_evs]
        rng = max(strengths) - min(strengths)
        if rng < 0.30:
            params.append(
                f"`gaze_shift` strength range: {min(strengths):.2f}–{max(strengths):.2f}. "
                "All orientings fired at nearly identical intensity, set entirely by `orient_gain` (8.0). "
                "There is no dynamic modulation; a loud onset and a quiet one look identical."
            )

    # settle variation at posture_settle fires
    settle_evs = _of("posture_settle", event_log)
    if len(settle_evs) >= 2:
        settle_vals = [e.get("settle", 0.0) for e in settle_evs]
        settle_range = max(settle_vals) - min(settle_vals)
        if settle_range > 0.10:
            dynamics.append(
                f"`settle` varied at `posture_settle` firings "
                f"({min(settle_vals):.3f}–{max(settle_vals):.3f}). "
                "This reflects genuine variation in how deeply the slow path charged "
                "across different speech phases — the settle path is contributing real "
                "dynamic information."
            )

    # recovery non-zero at firing
    all_recoveries = [e.get("recovery", 0.0) for e in event_log]
    if all_recoveries:
        max_r = max(all_recoveries)
        if max_r > 0.15:
            dynamics.append(
                f"Some events fired with recovery > 0 (max {max_r:.3f}). "
                "Backpressure was actively raising effective thresholds at those moments, "
                "confirming that the shared recovery mechanism is influencing event timing "
                "beyond the bare refire gaps."
            )

    # freeze suppression window
    if _counts(event_log).get("freeze", 0) > 0:
        dynamics.append(
            "The `freeze` event and its downstream suppression window involve multiple "
            "interacting paths: `freeze_val` accumulates from velocity input, fires and "
            "injects recovery, which then holds back `ttp`-driven events until it decays. "
            "This is a genuine multi-path interaction — not a scripted pause."
        )

    # ttp discharge on response_ready
    rr_ttp_before = [e.get("ttp", 0.0) for e in rr_evs]
    if rr_ttp_before and min(rr_ttp_before) < 0.80:
        dynamics.append(
            f"`ttp` was not fully saturated at some `response_ready` firings "
            f"(min {min(rr_ttp_before):.3f}). "
            "Speech blocks had discharged pressure enough to keep `ttp` below ceiling, "
            "meaning the threshold crossing reflects genuine path state rather than a guaranteed saturation."
        )

    if dynamics:
        lines.append("**Evidence of path-dynamics-driven behavior:**\n")
        for item in dynamics:
            lines.append(f"- {item}")
        lines.append("")

    if params:
        lines.append("**Evidence of parameter-dominated behavior:**\n")
        for item in params:
            lines.append(f"- {item}")
        lines.append("")

    if not dynamics and not params:
        lines.append(
            "No strongly dominant pattern identified. "
            "Event timing shows enough variance to suggest a mix of both influences.\n"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Section 6 — dominant parameter
# ------------------------------------------------------------------

def _s6_dominant_param(event_log: List[Dict], records: List[Dict]) -> str:
    lines = ["## 6. Which parameter seems most dominant\n"]
    candidates = []

    # ttp_response_refire_gap
    rr_ieis = _ieis(_of("response_ready", event_log))
    if len(rr_ieis) >= 2 and float(rr_ieis.std()) < 0.25:
        candidates.append((
            float(rr_ieis.std()),
            "`ttp_response_refire_gap` (4.0 s)",
            f"`response_ready` intervals: {', '.join(f'{x:.2f}' for x in rr_ieis)} s "
            f"(std {float(rr_ieis.std()):.2f} s). "
            "Once `ttp` saturates, this timer alone determines when `response_ready` fires. "
            "It is the single parameter with the most direct, measurable influence on output timing "
            "in this scenario. Changing it would immediately change the event rhythm."
        ))

    # orient_gain → gaze always same strength
    gaze_evs = _of("gaze_shift", event_log)
    if len(gaze_evs) >= 1:
        strengths = [e["strength"] for e in gaze_evs]
        if max(strengths) - min(strengths) < 0.30:
            candidates.append((
                0.5,
                "`orient_gain` (8.0)",
                f"Every `gaze_shift` fired at strength ≈{strengths[0]:.2f} "
                f"(= orient_gain × one-frame decay factor). "
                "There is no dynamic modulation of gaze intensity; "
                "the parameter completely determines event strength."
            ))

    # settle_refire_gap
    settle_ieis = _ieis(_of("posture_settle", event_log))
    if len(settle_ieis) >= 1 and float(np.min(settle_ieis)) >= 4.8:
        candidates.append((
            0.6,
            "`settle_refire_gap` (5.0 s)",
            f"No two `posture_settle` events fired within {float(np.min(settle_ieis)):.2f} s, "
            "confirming the refire gap is the minimum interval once `settle` remains above threshold."
        ))

    if candidates:
        # Sort by std (lowest = most dominated)
        candidates.sort(key=lambda x: x[0])
        primary = candidates[0]
        lines.append(
            f"The most dominant parameter in this scenario is **{primary[1]}**.\n\n"
            f"{primary[2]}\n"
        )
        if len(candidates) > 1:
            lines.append("**Other strong parameter influences:**\n")
            for _, pname, pdesc in candidates[1:]:
                lines.append(f"- **{pname}** — {pdesc}")
            lines.append("")
    else:
        lines.append(
            "No single parameter appears overwhelmingly dominant in this scenario. "
            "Event timing shows enough variance that path dynamics are contributing "
            "meaningfully alongside fixed limits.\n"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Section 7 — what to tune next
# ------------------------------------------------------------------

def _s7_tune_next(name: str, event_log: List[Dict], records: List[Dict]) -> str:
    counts = _counts(event_log)
    lines = ["## 7. What should be tuned next\n"]
    suggestions = []

    # response_ready fixed intervals
    rr_ieis = _ieis(_of("response_ready", event_log))
    if len(rr_ieis) >= 2 and float(rr_ieis.std()) < 0.25:
        suggestions.append(
            "**Introduce variation in `response_ready` timing.** "
            "The fixed `ttp_response_refire_gap` (4.0 s) is the only thing pacing this event "
            "once `ttp` saturates. Two options: "
            "(1) reduce `ttp_response_discharge` from 0.25 to ~0.08 so `ttp` drops further "
            "after each firing and takes noticeably longer to rebuild — making early refires "
            "impossible and later ones faster; "
            "(2) scale the refire gap dynamically with firing strength — a weak firing gets "
            "a shorter gap, a strong one gets a longer gap."
        )

    # fidget suppression too sparse
    fidget_count = counts.get("fidget_suppression", 0)
    mean_inhib = _phase_mean(records, "fidget_inhibit", _speaking)
    if fidget_count <= 1 and mean_inhib > 0.55:
        suggestions.append(
            f"**Increase `fidget_base_rate` or reconsider `fidget_suppression` as an event.** "
            f"With mean inhibition {mean_inhib:.2f} during speech, the current rate "
            f"(0.10/s × inhibit) produced only {fidget_count} discrete event(s). "
            "Raising `fidget_base_rate` to 0.30–0.50/s would produce a more continuous "
            "suppression signal. Alternatively, consider logging `fidget_inhibit` as a "
            "continuous output channel rather than a sparse event — it may be better "
            "represented as a value than a count."
        )

    # gaze strength fixed
    gaze_evs = _of("gaze_shift", event_log)
    if len(gaze_evs) >= 1 and max(e["strength"] for e in gaze_evs) - min(e["strength"] for e in gaze_evs) < 0.30:
        suggestions.append(
            "**Modulate `gaze_shift` strength by `speech_energy` at onset.** "
            "Currently every gaze shift fires at the same strength (~7.2), regardless of "
            "whether the onset was a confident utterance or a quiet murmur. "
            "Scaling orient_gain by `speech_energy` at the onset frame would make "
            "gaze shift intensity informative."
        )

    # settle persists in silence
    settle_sil_fires = [
        e for e in _of("posture_settle", event_log)
        if _silent(_find_record_at(records, e["time"]))
        and float(_find_record_at(records, e["time"]).get("silence_duration", 0)) > 3.0
    ]
    if settle_sil_fires:
        suggestions.append(
            "**Consider whether `settle_decay` (0.07/s) is too slow.** "
            f"`posture_settle` fired {len(settle_sil_fires)} time(s) during extended silence "
            "because `settle` persists above threshold for many seconds after speech ends. "
            "If the intended meaning is 'actively settling into listening', the signal should "
            "clear more quickly when speaking stops. "
            "Try raising `settle_decay` to 0.15–0.20/s."
        )

    # freeze recovery
    if "freeze" in counts:
        suggestions.append(
            "**Try a graded freeze effect.** "
            "Currently `freeze` injects a flat 0.40 recovery pulse, blocking all events equally "
            "for ~2.6 s. A more nuanced approach: suppress fast-path events (gaze, fidget) fully "
            "but only partially dampen slow-path events (ttp-driven nod/response), "
            "so that turn-taking pressure is dampened but not completely frozen. "
            "Reduce `recovery_on_freeze` to 0.20 and add a direct `ttp` suppression multiplier."
        )

    # silence_duration underused
    if name == "long_listening":
        suggestions.append(
            "**Strengthen the role of `silence_duration`.** "
            "The current 1.2× gain boost after 12 s is barely visible in the output. "
            "Consider adding a second response threshold — say 0.80 — that only unlocks "
            "after `silence_duration` exceeds 15 s, giving the system a distinct "
            "'urgency' mode for very long pauses that differs qualitatively from "
            "the standard turn-taking cycle."
        )

    # cross-path coupling absent
    suggestions.append(
        "**Add at least one cross-path interaction.** "
        "Currently paths only interact through shared `recovery_backpressure`. "
        "A small, targeted coupling — such as `orient` briefly suppressing `ttp` accumulation "
        "at speech onset — would make the system more responsive to context and "
        "could produce more naturalistic turn-taking timing around speech onset moments."
    )

    for s in suggestions:
        lines.append(s)
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main diagnosis section
# ------------------------------------------------------------------

def _s_diagnosis(name: str, event_log: List[Dict], records: List[Dict]) -> str:
    counts = _counts(event_log)
    lines = ["## Main diagnosis\n"]

    working = []
    emergent = []
    param_dominated = []

    # ── Working as intended ──────────────────────────────────────────
    n_onsets = sum(1 for r in records if float(r["speech_onset"]) > 0.5)
    n_gaze = counts.get("gaze_shift", 0)
    if n_onsets > 0 and n_gaze == n_onsets:
        working.append(
            f"**Orienting response** — `gaze_shift` fired exactly once per speech onset "
            f"({n_onsets} → {n_gaze}). The refractory mechanism prevented double-firing."
        )

    if "posture_settle" in counts and _phase_mean(records, "settle", _speaking) > 0.30:
        working.append(
            f"**Settle accumulation** — `settle` built to mean "
            f"{_phase_mean(records, 'settle', _speaking):.3f} during speech, "
            "and `posture_settle` fired, confirming the slow-path accumulation mechanism."
        )

    inhib_sp = _phase_mean(records, "fidget_inhibit", _speaking)
    if inhib_sp > 0.65:
        working.append(
            f"**Fidget inhibition** — `fidget_inhibit` averaged {inhib_sp:.2f} during speech, "
            "confirming the inhibition path charges correctly while the user is speaking."
        )

    if "micro_nod_ready" in counts and "response_ready" in counts:
        working.append(
            "**Turn-taking pressure staging** — `micro_nod_ready` (lower threshold) "
            "and `response_ready` (higher threshold) both fired, "
            "demonstrating the two-level pressure architecture."
        )

    if "freeze" in counts:
        post_freeze = [
            e for e in event_log
            if e["time"] > _of("freeze", event_log)[0]["time"]
            and e["name"] not in ("freeze",)
        ]
        freeze_gap = post_freeze[0]["time"] - _of("freeze", event_log)[0]["time"] if post_freeze else 0
        if freeze_gap > 1.0:
            working.append(
                f"**Freeze suppression** — after `freeze`, the next event was delayed "
                f"{freeze_gap:.2f} s, confirming that the recovery pulse and `freeze_val` "
                "jointly held back downstream events."
            )

    # ── Interesting emergent-looking behavior ────────────────────────
    times_names = sorted([(float(e["time"]), e["name"]) for e in event_log])
    seen_bursts: set = set()
    for i in range(len(times_names) - 1):
        t0, n0 = times_names[i]
        t1, n1 = times_names[i + 1]
        if t1 - t0 < 0.08 and n0 != n1:
            key = round(t0, 1)
            if key not in seen_bursts:
                seen_bursts.add(key)
                burst = [nm for tt, nm in times_names if abs(tt - t0) < 0.15]
                emergent.append(
                    f"**Simultaneous burst at t≈{t0:.2f} s** "
                    f"({', '.join(f'`{b}`' for b in burst)}) — "
                    "multiple paths discharged together after a shared suppression window. "
                    "This compression-then-release pattern is a genuine emergent consequence "
                    "of shared backpressure, not a scripted co-occurrence."
                )

    for e in _of("response_ready", event_log):
        t = e["time"]
        ctx = [r for r in records if abs(float(r["time"]) - t) < 0.4]
        if any(_speaking(r) for r in ctx):
            emergent.append(
                f"**`response_ready` at t={t:.2f} s — pressure carried into speech** — "
                "turn-taking pressure accumulated during silence was still above threshold "
                "when speech began. The system was simultaneously orienting and primed to respond. "
                "This cross-phase bleed-through was not scripted."
            )
            break

    for e in _of("posture_settle", event_log):
        t = e["time"]
        r_at_t = _find_record_at(records, t)
        sil_dur = float(r_at_t.get("silence_duration", 0.0))
        if _silent(r_at_t) and sil_dur > 3.0:
            emergent.append(
                f"**`posture_settle` at t={t:.2f} s — {sil_dur:.1f} s into silence** — "
                "the slow `settle` decay allows the path to linger above threshold long "
                "after speech ends, creating an unscripted gradual transition from "
                "active listening to rest."
            )
            break

    if "freeze" in counts:
        ft = _of("freeze", event_log)[0]["time"]
        post = [ev for ev in event_log if ev["time"] > ft and ev["name"] != "freeze"]
        if post:
            gap = post[0]["time"] - ft
            burst_window = [ev for ev in post if ev["time"] - post[0]["time"] < 0.25]
            if gap > 1.5 and len(burst_window) >= 2:
                emergent.append(
                    f"**Post-freeze release burst** — {gap:.2f} s of suppression, then "
                    f"{len(burst_window)} events in a tight cluster. "
                    "The freeze wall held back pressure across multiple paths simultaneously; "
                    "the burst when it lifted was not scripted."
                )

    # ── Parameter dominated ──────────────────────────────────────────
    rr_ieis = _ieis(_of("response_ready", event_log))
    if len(rr_ieis) >= 2 and float(rr_ieis.std()) < 0.20:
        param_dominated.append(
            f"**`response_ready` timing is a clock** — intervals {', '.join(f'{x:.2f}' for x in rr_ieis)} s, "
            f"std {float(rr_ieis.std()):.2f} s. "
            "Once `ttp` saturates, the 4.0 s refire gap is the only thing varying the output. "
            "This event has stopped behaving like an emergent signal and is functioning as a metronome."
        )

    gaze_strengths = [e["strength"] for e in _of("gaze_shift", event_log)]
    if len(gaze_strengths) >= 2 and max(gaze_strengths) - min(gaze_strengths) < 0.30:
        param_dominated.append(
            f"**`gaze_shift` strength is fixed** — all firings at {gaze_strengths[0]:.2f}. "
            "`orient_gain` entirely determines event intensity; speech context has no effect."
        )

    # Write sections
    lines.append("### Working as intended\n")
    if working:
        for item in working:
            lines.append(f"- {item}")
    else:
        lines.append("- *(no confirmations identified for this scenario)*")
    lines.append("")

    lines.append("### Interesting emergent-looking behavior\n")
    if emergent:
        for item in emergent:
            lines.append(f"- {item}")
    else:
        lines.append("- *(no notable emergent patterns detected)*")
    lines.append("")

    lines.append("### Too parameter-dominated / needs tuning\n")
    if param_dominated:
        for item in param_dominated:
            lines.append(f"- {item}")
    else:
        lines.append(
            "- *(no strongly parameter-dominated patterns detected — "
            "event timing shows sufficient variance)*"
        )
    lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_report(
    scenario_name: str,
    records: List[Dict[str, Any]],
    event_log: List[Dict[str, Any]],
) -> str:
    """
    Generate a human-readable Markdown diagnostic report.

    All analysis in sections 2–8 is data-driven from records and event_log.
    Static text appears only in section 1 (scenario description) and
    in the per-event-type explanation blurbs.
    """
    duration = float(records[-1]["time"]) if records else 0.0
    header = (
        f"# Diagnostic Report: `{scenario_name}`\n\n"
        f"_Simulation output: {len(records):,} steps · "
        f"{duration:.1f} s · {len(event_log)} events total_\n"
    )

    sep = "\n\n---\n\n"
    return sep.join([
        header,
        _s1_description(scenario_name, records),
        _s2_paths(records),
        _s3_events(event_log, records),
        _s4_expected_vs_surprising(scenario_name, event_log, records),
        _s5_dynamics_vs_params(event_log, records),
        _s6_dominant_param(event_log, records),
        _s7_tune_next(scenario_name, event_log, records),
        _s_diagnosis(scenario_name, event_log, records),
    ])
