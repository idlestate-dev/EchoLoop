#!/usr/bin/env python3
"""
EchoLoop Embodied Reaction MVP – Closed-Loop World Interaction Demo

Connects EchoLoop to a minimal 2D world:

    observation → EchoLoop → action → world update → new observation → …

Separate from viewer_2d.py (which plays back pre-recorded logs).
Does not modify EchoLoop model dynamics.

Usage:
    python world_loop_demo.py --scenario sudden_approach_while_speaking_closed_loop
    python world_loop_demo.py --scenario sudden_approach_while_speaking_closed_loop --speed 2.0
    python world_loop_demo.py --scenario sudden_approach_while_speaking_closed_loop --record

Keys: SPACE pause/resume   R restart   Q/ESC quit
"""

import argparse
import csv
import math
import os
import random
import sys
import time as _wall
from dataclasses import dataclass

# ── pygame guard ──────────────────────────────────────────────────────────────
try:
    import pygame
except ImportError:
    print(
        "\npygame is not installed.\n"
        "Install it with:\n"
        "    pip install pygame\n"
        "or:\n"
        "    pip install pygame-ce\n"
    )
    sys.exit(1)

# ── EchoLoop import ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from echoloop import EchoLoopState

# ── Layout ────────────────────────────────────────────────────────────────────
W, H     = 920, 640
SCENE_W  = 620
SCENE_H  = 510
BAR_X    = 640
BAR_W    = 260
TL_Y     = 520
TL_H     = H - TL_Y

SIM_DT = 1.0 / 60.0

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = (18,  20,  28)
SCENE_BG    = (24,  26,  36)
PANEL_BG    = (20,  22,  32)
TL_BG       = (16,  18,  26)
NPC_COL     = (180, 200, 255)
PLAYER_COL  = (255, 220, 140)
GAZE_COL    = (100, 220, 255)
DIST_COL    = (45,  55,  85)
TEXT_DIM    = (115, 125, 150)
TEXT_BRIGHT = (210, 220, 240)

EVENT_COLS = {
    "gaze_shift":         (100, 220, 255),
    "posture_settle":     (100, 220, 110),
    "fidget_suppression": (220, 195,  75),
    "micro_nod_ready":    (160, 155, 255),
    "response_ready":     (255, 255, 120),
    "freeze":             (255,  90,  70),
}
ACTION_COLS = {
    "gaze_at_player": (100, 220, 255),
    "freeze":         (255,  90,  70),
    "step_back":      (220, 155,  90),
    "micro_nod":      (160, 155, 255),
    "response":       (255, 255, 120),
}
DEFAULT_EV_COL = (180, 180, 180)

BAR_KEYS = ["orient", "settle", "fidget_inhibit", "ttp", "recovery", "freeze_val"]
BAR_COLS = {
    "orient":         (100, 220, 255),
    "settle":         (100, 220, 110),
    "fidget_inhibit": (220, 195,  75),
    "ttp":            (150, 150, 255),
    "recovery":       (255, 160,  70),
    "freeze_val":     (255,  90,  70),
}
BAR_DISPLAY = {
    "orient":         "orient",
    "settle":         "settle",
    "fidget_inhibit": "fidget inh",
    "ttp":            "ttp",
    "recovery":       "recovery",
    "freeze_val":     "freeze",
}

LABEL_LIFE       = 2.5
GAZE_SNAP_LIFE   = 0.55
SETTLE_FLASH_LIFE= 0.70
RESP_FLASH_LIFE  = 0.50
NOD_FLASH_LIFE   = 0.40

STEP_BACK_GATE     = 0.30   # recovery must be below this to execute step_back
STEP_BACK_DURATION = 0.30   # s of active NPC retreat per step_back event (0.2–0.4 range)
STEP_BACK_COOLDOWN = 2.0    # s before another step_back can be triggered
STEP_BACK_RATE     = 55.0   # px/s during active phase → ~16.5 px total retreat

FREEZE_REFRACTORY  = 6.0    # s: freeze event is suppressed after it fires once

COMFORT_DISTANCE  = 120.0   # px: interpersonal spacing; slow player drift stops here
COMFORT_RATE      = 8.0     # px/s: mild NPC outward correction when inside comfort zone

# Observation normalisation for approach_velocity.
# EchoLoop expects a normalised signal (0–2.5 range, not raw px/s).
# Dead zone: gradual walking does not register as a freeze-relevant approach.
AV_DEAD_ZONE     = 28.0   # px/s; below this → approach_velocity_norm = 0
AV_SCALE         = 14.0   # px/s per unit; 65 px/s → (65-28)/14 ≈ 2.6


# ── World state ───────────────────────────────────────────────────────────────

@dataclass
class World:
    # Positions in scene pixel coordinates
    player_x: float = 120.0
    player_y: float = 255.0
    npc_x: float    = 420.0
    npc_y: float    = 255.0
    npc_home_x: float = 420.0

    # Player signals
    player_speaking: bool  = False
    prev_speaking: bool    = False
    speech_energy: float   = 0.0
    silence_duration: float= 0.0

    # Derived observation
    distance: float          = 300.0
    approach_velocity: float = 0.0   # positive = player moving toward NPC

    # NPC action state
    npc_frozen: bool             = False
    npc_freeze_timer: float      = 0.0
    npc_freeze_refractory: float = 0.0  # s remaining; freeze suppressed while > 0
    npc_step_back_active: float  = 0.0  # s of movement remaining in current step_back
    npc_step_back_cooldown: float= 0.0  # s until next step_back can trigger

    # Visual timers (seconds remaining)
    gaze_snap: float    = 0.0
    settle_flash: float = 0.0
    resp_flash: float   = 0.0
    nod_flash: float    = 0.0

    # NPC jitter (driven by fidget_inhibit)
    jitter_x: float = 0.0
    jitter_y: float = 0.0
    jitter_tx: float = 0.0
    jitter_ty: float = 0.0
    jitter_timer: float = 0.0

    sim_time: float = 0.0


# ── Observation adapter ───────────────────────────────────────────────────────

def observe(w: World) -> dict:
    """
    Convert world state to EchoLoop step inputs.

    approach_velocity is normalised so that the EchoLoop freeze path
    receives the same signal scale it was designed for (~0–2.5 range).
    Slow walking stays below AV_DEAD_ZONE and maps to 0; a sudden
    rush at ~65 px/s maps to ≈2.6, matching the original scenario pulse.
    """
    raw_av  = max(0.0, w.approach_velocity)
    av_norm = max(0.0, (raw_av - AV_DEAD_ZONE) / AV_SCALE)
    return {
        "speech_onset":     1.0 if (w.player_speaking and not w.prev_speaking) else 0.0,
        "user_speaking":    1.0 if w.player_speaking else 0.0,
        "speech_energy":    w.speech_energy if w.player_speaking else 0.0,
        "silence_duration": w.silence_duration,
        "approach_velocity": av_norm,
    }


# ── Player scripts ────────────────────────────────────────────────────────────

class PlayerScript_SuddenApproach:
    """
    Reactive player for sudden_approach_while_speaking_closed_loop.

    Phase 1 is time-seeded (idle → speak → approach) to establish the scenario.
    Everything after the first freeze is driven by NPC-visible signals only:
      npc_frozen, npc_stepped_back, npc_responded, npc_gaze_active.
    No access to EchoLoop internals.

    Two guaranteed interaction cycles; the loop stays open after that via
    watchdog transitions so the demo does not collapse into passive waiting.

    Cycle:
      approach  → [freeze] → hesitate → [step_back] → back_off  → speak2 → approach2
      approach2 → [freeze] → hesitate2→ [step_back] → back_off2 → speak3 → approach3
      approach3 → [freeze] → hesitate  (re-enters cycle)
    """

    APPROACH_SPEED  = 65.0   # px/s → av_norm ≈ 2.6, triggers freeze fast (~0.1s)
    APPROACH2_SPEED = 30.0   # px/s → av_norm ≈ 0.14, slow build (~1s before freeze)
    RETREAT_SPEED   = 20.0   # px/s player backs away after NPC step_back
    RETREAT_PX      = 14.0   # total retreat distance per back_off phase
    HESITATE_DUR    = 2.2    # s frozen-response hesitation before timeout
    WATCHDOG_DUR    = 5.0    # s: advance waiting phase if NPC does nothing
    SPEAK2_DUR      = 3.5    # s for second utterance
    SPEAK3_DUR      = 4.0    # s for third utterance
    MIN_DISTANCE    = 65.0   # px: hard collision guard; prevents visual pass-through

    def __init__(self, seed: int = 7):
        self.phase         = "idle"
        self._phase_t      = 0.0
        self._watchdog     = 0.0
        self._retreat_done = 0.0
        self._rng          = random.Random(seed)

    def _enter(self, new_phase: str):
        self.phase         = new_phase
        self._phase_t      = 0.0
        self._watchdog     = 0.0
        self._retreat_done = 0.0

    def update(self, w: World, npc_frozen: bool,
               npc_responded: bool, npc_stepped_back: bool,
               npc_gaze_active: bool, dt: float, *, npc_nodded: bool = False):
        """
        Returns (speaking: bool, vx: float, energy: float).
        vx > 0 = moving toward NPC (right); vx < 0 = retreating (left).
        npc_nodded is accepted for interface compatibility; baseline ignores it.
        """
        self._phase_t += dt

        # Watchdog resets whenever the NPC does something visible.
        if npc_frozen or npc_stepped_back or npc_responded or npc_gaze_active:
            self._watchdog = 0.0
        else:
            self._watchdog += dt

        # ── Phase transitions ──────────────────────────────────────────────────

        if self.phase == "idle":
            if w.sim_time >= 2.0:
                self._enter("speak")

        elif self.phase == "speak":
            # Time-based trigger for the initial rush; all later transitions are event-driven.
            if w.sim_time >= 10.0:
                self._enter("approach")

        elif self.phase == "approach":
            if npc_frozen:
                self._enter("hesitate")
            elif w.distance < COMFORT_DISTANCE:
                self._enter("linger")

        elif self.phase == "hesitate":
            if npc_stepped_back:
                # NPC backed off — player mirrors with a brief retreat.
                # This increases distance → next observe() sees av_norm drop.
                self._enter("back_off")
            elif npc_responded:
                # NPC responded → player re-engages with speech immediately.
                self._enter("speak2")
            elif self._phase_t >= self.HESITATE_DUR:
                self._enter("resume")

        elif self.phase == "back_off":
            self._retreat_done += self.RETREAT_SPEED * dt
            if npc_responded:
                # NPC responded while player is backing off → stop retreating, re-engage.
                self._enter("speak2")
            elif self._retreat_done >= self.RETREAT_PX or self._phase_t >= 2.0:
                self._enter("speak2")

        elif self.phase == "resume":
            # Slow silent approach; waiting for NPC to respond before re-engaging.
            if npc_responded or self._watchdog >= self.WATCHDOG_DUR:
                self._enter("speak2")

        elif self.phase == "speak2":
            if self._phase_t >= self.SPEAK2_DUR:
                self._enter("approach2")

        elif self.phase == "approach2":
            # Primary exit: freeze. If freeze is suppressed by refractory the player
            # stops at COMFORT_DISTANCE (after at least 1s of approach) rather than
            # pressing in and triggering NPC comfort correction accumulation.
            if npc_frozen:
                self._enter("hesitate2")
            elif w.distance < COMFORT_DISTANCE and self._phase_t >= 1.0:
                self._enter("linger")
            elif self._phase_t >= 8.0:
                self._enter("linger")

        elif self.phase == "hesitate2":
            if npc_stepped_back:
                self._enter("back_off2")
            elif npc_responded:
                self._enter("speak3")
            elif self._phase_t >= self.HESITATE_DUR:
                self._enter("speak3")

        elif self.phase == "back_off2":
            self._retreat_done += self.RETREAT_SPEED * dt
            if npc_responded:
                self._enter("speak3")
            elif self._retreat_done >= self.RETREAT_PX or self._phase_t >= 2.0:
                self._enter("speak3")

        elif self.phase == "speak3":
            if self._phase_t >= self.SPEAK3_DUR or npc_responded:
                self._enter("approach3")

        elif self.phase == "approach3":
            # Same logic as approach2; re-uses hesitate so the cycle can repeat.
            if npc_frozen:
                self._enter("hesitate")
            elif w.distance < COMFORT_DISTANCE and self._phase_t >= 1.0:
                self._enter("linger")
            elif self._phase_t >= 8.0:
                self._enter("linger")

        elif self.phase == "linger":
            # React to NPC turn signals so the interaction doesn't stall here.
            if npc_frozen:
                self._enter("hesitate")
            elif npc_responded:
                self._enter("speak2")
            elif self._watchdog >= self.WATCHDOG_DUR:
                self._enter("speak2")

        # ── Outputs ────────────────────────────────────────────────────────────

        # Speak during initial utterance, the rushing approach, and re-engagement phases.
        # approach2/approach3 are deliberately silent — silence builds ttp pressure.
        speaking = self.phase in ("speak", "approach", "speak2", "speak3")

        _vx = {
            "idle":       0.0,
            "speak":      0.0,
            "approach":   self.APPROACH_SPEED,
            "hesitate":   0.0,
            "back_off":  -self.RETREAT_SPEED,
            "resume":     14.0,
            "speak2":    -self.RETREAT_SPEED,  # retreat to speaking range before next approach
            "approach2":  self.APPROACH2_SPEED,
            "hesitate2":  0.0,
            "back_off2": -self.RETREAT_SPEED,
            "speak3":     0.0,   # hold position while speaking
            "approach3":  self.APPROACH2_SPEED,
            "linger":     0.0,   # stationary at comfort distance
        }
        vx = _vx.get(self.phase, 0.0)
        # speak2 retreats until COMFORT_DISTANCE + 30px; stops retreating once far enough.
        if self.phase == "speak2" and w.distance >= COMFORT_DISTANCE + 30:
            vx = 0.0
        # Hard collision guard for active approaches.
        elif vx > 0 and w.distance <= self.MIN_DISTANCE:
            vx = 0.0

        energy = 0.0
        if speaking:
            t      = w.sim_time
            energy = 0.55 + 0.20 * math.sin(t * 5.5)
            energy += 0.08 * self._rng.gauss(0, 1)
            energy = max(0.0, min(1.0, energy))

        return speaking, vx, energy


class PlayerScript_SuddenApproachLoose:
    """
    Loose variant of PlayerScript_SuddenApproach (--variant loose).

    Same phase graph; stochastic timing variation breaks the fixed-period feel.

    Differences vs baseline:
    - speak2/speak3 duration varies ±random per utterance
    - retreat distance varies per back_off
    - speak2 retreat target varies (sets starting distance for each re-approach)
    - response_ready can briefly pause speaking instead of always advancing phase
    - linger: micro_nod or response_ready can enter comfortable_idle (silent hold)
    - comfortable_idle: 3–7 s stationary silence at comfort distance before re-engaging

    Deterministic with --seed.
    """

    APPROACH_SPEED  = 65.0
    APPROACH2_SPEED = 30.0
    RETREAT_SPEED   = 20.0
    RETREAT_PX      = 14.0
    HESITATE_DUR    = 2.2
    WATCHDOG_DUR    = 5.0
    SPEAK2_DUR      = 3.5
    SPEAK3_DUR      = 4.0
    MIN_DISTANCE    = 65.0

    def __init__(self, seed: int = 7):
        self.phase           = "idle"
        self._phase_t        = 0.0
        self._watchdog       = 0.0
        self._retreat_done   = 0.0
        self._rng            = random.Random(seed)
        # Per-phase stochastic state (set in _enter)
        self._speak_dur      = self.SPEAK2_DUR
        self._speak_range    = COMFORT_DISTANCE + 30.0
        self._retreat_target = self.RETREAT_PX
        self._idle_dur       = 0.0
        self._speaking_pause = 0.0   # s remaining; suppresses speaking output

    def _enter(self, new_phase: str):
        self.phase         = new_phase
        self._phase_t      = 0.0
        self._watchdog     = 0.0
        self._retreat_done = 0.0
        if new_phase == "speak2":
            self._speak_dur   = self.SPEAK2_DUR  + self._rng.uniform(-0.5, 2.0)
            self._speak_range = COMFORT_DISTANCE + self._rng.uniform(20.0, 50.0)
        elif new_phase == "speak3":
            self._speak_dur   = self.SPEAK3_DUR  + self._rng.uniform(-0.5, 2.0)
        elif new_phase in ("back_off", "back_off2"):
            self._retreat_target = self.RETREAT_PX * self._rng.uniform(0.8, 1.6)
        elif new_phase == "comfortable_idle":
            self._idle_dur = self._rng.uniform(3.0, 7.0)

    def update(self, w: World, npc_frozen: bool,
               npc_responded: bool, npc_stepped_back: bool,
               npc_gaze_active: bool, dt: float, *, npc_nodded: bool = False):
        """Returns (speaking: bool, vx: float, energy: float)."""
        self._phase_t        += dt
        self._speaking_pause  = max(0.0, self._speaking_pause - dt)

        if npc_frozen or npc_stepped_back or npc_responded or npc_gaze_active:
            self._watchdog = 0.0
        else:
            self._watchdog += dt

        # ── Phase transitions ──────────────────────────────────────────────────

        if self.phase == "idle":
            if w.sim_time >= 2.0:
                self._enter("speak")

        elif self.phase == "speak":
            if w.sim_time >= 10.0:
                self._enter("approach")

        elif self.phase == "approach":
            if npc_frozen:
                self._enter("hesitate")
            elif w.distance < COMFORT_DISTANCE:
                self._enter("linger")

        elif self.phase == "hesitate":
            if npc_stepped_back:
                self._enter("back_off")
            elif npc_responded:
                self._enter("speak2")
            elif self._phase_t >= self.HESITATE_DUR:
                self._enter("resume")

        elif self.phase == "back_off":
            self._retreat_done += self.RETREAT_SPEED * dt
            if npc_responded:
                self._enter("speak2")
            elif self._retreat_done >= self._retreat_target or self._phase_t >= 2.0:
                self._enter("speak2")

        elif self.phase == "resume":
            if npc_responded or self._watchdog >= self.WATCHDOG_DUR:
                self._enter("speak2")

        elif self.phase == "speak2":
            if npc_responded and self._speaking_pause <= 0 and self._rng.random() < 0.35:
                self._speaking_pause = self._rng.uniform(0.5, 1.5)
            if self._phase_t >= self._speak_dur:
                self._enter("approach2")

        elif self.phase == "approach2":
            if npc_frozen:
                self._enter("hesitate2")
            elif w.distance < COMFORT_DISTANCE and self._phase_t >= 1.0:
                self._enter("linger")
            elif self._phase_t >= 8.0:
                self._enter("linger")

        elif self.phase == "hesitate2":
            if npc_stepped_back:
                self._enter("back_off2")
            elif npc_responded:
                self._enter("speak3")
            elif self._phase_t >= self.HESITATE_DUR:
                self._enter("speak3")

        elif self.phase == "back_off2":
            self._retreat_done += self.RETREAT_SPEED * dt
            if npc_responded:
                self._enter("speak3")
            elif self._retreat_done >= self._retreat_target or self._phase_t >= 2.0:
                self._enter("speak3")

        elif self.phase == "speak3":
            if npc_responded and self._speaking_pause <= 0 and self._rng.random() < 0.35:
                self._speaking_pause = self._rng.uniform(0.5, 1.5)
            if self._phase_t >= self._speak_dur:
                self._enter("approach3")

        elif self.phase == "approach3":
            if npc_frozen:
                self._enter("hesitate")
            elif w.distance < COMFORT_DISTANCE and self._phase_t >= 1.0:
                self._enter("linger")
            elif self._phase_t >= 8.0:
                self._enter("linger")

        elif self.phase == "linger":
            if npc_frozen:
                self._enter("hesitate")
            elif npc_nodded and self._rng.random() < 0.45:
                self._enter("comfortable_idle")
            elif npc_responded:
                if self._rng.random() < 0.25:
                    self._enter("comfortable_idle")
                else:
                    self._enter("speak2")
            elif self._watchdog >= self.WATCHDOG_DUR:
                self._enter("speak2")

        elif self.phase == "comfortable_idle":
            self._idle_dur -= dt
            if npc_frozen:
                self._enter("hesitate")
            elif npc_responded:
                self._enter("speak2")
            elif self._idle_dur <= 0.0 or self._watchdog >= self.WATCHDOG_DUR:
                self._enter("speak2")

        # ── Outputs ────────────────────────────────────────────────────────────

        speaking = self.phase in ("speak", "approach", "speak2", "speak3")
        if self._speaking_pause > 0:
            speaking = False

        _vx = {
            "idle":             0.0,
            "speak":            0.0,
            "approach":         self.APPROACH_SPEED,
            "hesitate":         0.0,
            "back_off":        -self.RETREAT_SPEED,
            "resume":           14.0,
            "speak2":          -self.RETREAT_SPEED,
            "approach2":        self.APPROACH2_SPEED,
            "hesitate2":        0.0,
            "back_off2":       -self.RETREAT_SPEED,
            "speak3":           0.0,
            "approach3":        self.APPROACH2_SPEED,
            "linger":           0.0,
            "comfortable_idle": 0.0,
        }
        vx = _vx.get(self.phase, 0.0)
        if self.phase == "speak2" and w.distance >= self._speak_range:
            vx = 0.0
        elif vx > 0 and w.distance <= self.MIN_DISTANCE:
            vx = 0.0

        energy = 0.0
        if speaking:
            t      = w.sim_time
            energy = 0.55 + 0.20 * math.sin(t * 5.5)
            energy += 0.08 * self._rng.gauss(0, 1)
            energy = max(0.0, min(1.0, energy))

        return speaking, vx, energy


PLAYER_SCRIPTS = {
    "sudden_approach_while_speaking_closed_loop": {
        "baseline": PlayerScript_SuddenApproach,
        "loose":    PlayerScript_SuddenApproachLoose,
    },
}


# ── EchoLoop event → NPC action mapping ──────────────────────────────────────

def apply_events(events: list, w: World) -> list:
    """
    Map EchoLoop output events to NPC actions.
    Mutates w (visual timers, freeze state, step_back queue).
    Returns list of action name strings that fired.
    """
    actions = []
    for ev in events:
        name = ev["name"]
        if name == "gaze_shift":
            w.gaze_snap = GAZE_SNAP_LIFE
            actions.append("gaze_at_player")
        elif name == "posture_settle":
            w.settle_flash = SETTLE_FLASH_LIFE
        elif name == "freeze":
            if w.npc_freeze_refractory > 0:
                pass   # refractory active — suppress this freeze event
            else:
                w.npc_frozen = True
                w.npc_freeze_timer = max(w.npc_freeze_timer, 0.5)
                if w.npc_step_back_cooldown <= 0:
                    w.npc_step_back_active = STEP_BACK_DURATION
                    w.npc_step_back_cooldown = STEP_BACK_COOLDOWN
                w.npc_freeze_refractory = FREEZE_REFRACTORY
                actions.append("freeze")
        elif name == "micro_nod_ready":
            w.nod_flash = NOD_FLASH_LIFE
            actions.append("micro_nod")
        elif name == "response_ready":
            w.resp_flash = RESP_FLASH_LIFE
            actions.append("response")
    return actions


def update_npc(w: World, state: dict, dt: float) -> bool:
    """
    Apply NPC movement.

    step_back is discrete: STEP_BACK_DURATION seconds of active movement at
    STEP_BACK_RATE px/s, gated by recovery < STEP_BACK_GATE (timer pauses while
    gated so total retreat distance is preserved). Followed by STEP_BACK_COOLDOWN
    seconds during which a new step_back cannot trigger.

    freeze has a FREEZE_REFRACTORY period enforced in apply_events; update_npc
    only needs to tick the freeze timer.

    Returns True if step_back movement occurred this step.
    """
    recovery = state.get("recovery", 0.0)

    # Tick refractory and cooldown regardless of other state
    w.npc_freeze_refractory  = max(0.0, w.npc_freeze_refractory  - dt)
    w.npc_step_back_cooldown = max(0.0, w.npc_step_back_cooldown - dt)

    # Tick freeze timer
    if w.npc_frozen:
        w.npc_freeze_timer = max(0.0, w.npc_freeze_timer - dt)
        if w.npc_freeze_timer <= 0.0:
            w.npc_frozen = False

    # step_back: active timer only ticks when recovery gate allows movement.
    # This preserves the full retreat distance even if recovery delays the start.
    stepped = False
    if w.npc_step_back_active > 0.0 and recovery < STEP_BACK_GATE:
        w.npc_x += STEP_BACK_RATE * dt   # NPC moves right (away from player)
        w.npc_step_back_active -= dt
        stepped = True

    # Clamp NPC within scene
    w.npc_x = max(200.0, min(SCENE_W - 30.0, w.npc_x))

    # Drift logic: home pull when outside comfort zone; mild outward correction inside it.
    # Keeps NPC from drifting toward a player who is already too close.
    if not w.npc_frozen and w.npc_step_back_active <= 0.0:
        if w.distance < COMFORT_DISTANCE:
            encroachment = (COMFORT_DISTANCE - w.distance) / COMFORT_DISTANCE
            w.npc_x += COMFORT_RATE * encroachment * dt
        else:
            w.npc_x += (w.npc_home_x - w.npc_x) * 0.003

    return stepped


# ── Closed-loop step ──────────────────────────────────────────────────────────

def loop_step(w: World, script, echoloop: EchoLoopState,
              rng: random.Random) -> tuple:
    """
    One full observation → EchoLoop → action → world update cycle.
    Returns (events, state_dict, action_names).
    """
    prev_dist = w.distance

    # ── Observe ───────────────────────────────────────────────────────
    obs = observe(w)

    # ── EchoLoop step ─────────────────────────────────────────────────
    events = echoloop.step(**obs)
    state  = echoloop.state_dict()

    # ── NPC actions from events ───────────────────────────────────────
    actions = apply_events(events, w)

    # ── NPC position update runs first so player can observe step_back ─
    stepped = update_npc(w, state, SIM_DT)
    if stepped:
        actions.append("step_back")

    # ── Player update (reacts to NPC state including this frame's move) ─
    npc_responded    = any(a == "response"   for a in actions)
    npc_stepped_back = any(a == "step_back"  for a in actions)
    npc_gaze_active  = w.gaze_snap > 0
    npc_nodded       = any(a == "micro_nod"  for a in actions)

    speaking, vx, energy = script.update(
        w, w.npc_frozen, npc_responded, npc_stepped_back, npc_gaze_active, SIM_DT,
        npc_nodded=npc_nodded,
    )

    w.player_x = max(20.0, min(SCENE_W - 20.0, w.player_x + vx * SIM_DT))
    w.prev_speaking  = w.player_speaking
    w.player_speaking = speaking
    w.speech_energy   = energy

    # ── Silence tracking ──────────────────────────────────────────────
    if speaking:
        w.silence_duration = 0.0
    else:
        w.silence_duration += SIM_DT

    # ── Jitter (low fidget_inhibit = more NPC body movement) ──────────
    fi = state.get("fidget_inhibit", 0.0)
    w.jitter_timer -= SIM_DT
    if w.jitter_timer <= 0.0:
        mag = (1.0 - fi) * 5.0
        w.jitter_tx = rng.uniform(-mag, mag)
        w.jitter_ty = rng.uniform(-mag, mag)
        w.jitter_timer = 0.10
    lerp     = min(1.0, SIM_DT * 12.0)
    w.jitter_x += (w.jitter_tx - w.jitter_x) * lerp
    w.jitter_y += (w.jitter_ty - w.jitter_y) * lerp

    # ── Derived observations for next step ────────────────────────────
    new_dist           = abs(w.player_x - w.npc_x)
    w.approach_velocity = (prev_dist - new_dist) / SIM_DT   # positive = approaching
    w.distance          = new_dist

    # ── Decay visual timers ───────────────────────────────────────────
    w.gaze_snap    = max(0.0, w.gaze_snap    - SIM_DT)
    w.settle_flash = max(0.0, w.settle_flash - SIM_DT)
    w.resp_flash   = max(0.0, w.resp_flash   - SIM_DT)
    w.nod_flash    = max(0.0, w.nod_flash    - SIM_DT)

    w.sim_time += SIM_DT
    return events, state, actions


# ── CSV log ───────────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    "step", "time",
    "player_x", "player_y", "npc_x", "npc_y",
    "distance", "player_speaking", "approach_velocity",
    *BAR_KEYS,
    "player_phase", "events", "actions",
]


def open_log(scenario: str, variant: str = "baseline"):
    log_dir = os.path.join(_HERE, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    suffix = f"_{variant}" if variant != "baseline" else ""
    path = os.path.join(log_dir, f"{scenario}{suffix}_world_loop.csv")
    f    = open(path, "w", newline="")
    wtr  = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
    wtr.writeheader()
    return f, wtr, path


def write_log(wtr, step: int, w: World, state: dict,
              events: list, actions: list, player_phase: str = ""):
    wtr.writerow({
        "step":              step,
        "time":              f"{w.sim_time:.4f}",
        "player_x":          f"{w.player_x:.2f}",
        "player_y":          f"{w.player_y:.2f}",
        "npc_x":             f"{w.npc_x:.2f}",
        "npc_y":             f"{w.npc_y:.2f}",
        "distance":          f"{w.distance:.2f}",
        "player_speaking":   int(w.player_speaking),
        "approach_velocity": f"{w.approach_velocity:.4f}",
        **{k: f"{state.get(k, 0.0):.4f}" for k in BAR_KEYS},
        "player_phase":      player_phase,
        "events":            ";".join(e["name"] for e in events),
        "actions":           ";".join(actions),
    })


# ── Label / timeline tracker ──────────────────────────────────────────────────

class LabelTracker:
    def __init__(self):
        self.labels    = []   # [{text, color, born}]
        self.tl_events = []   # [{time, name, color}] for timeline strip

    def push(self, events: list, actions: list, sim_time: float):
        for ev in events:
            col = EVENT_COLS.get(ev["name"], DEFAULT_EV_COL)
            self.labels.append({"text": ev["name"].replace("_", " "),
                                 "color": col, "born": sim_time})
            self.tl_events.append({"time": sim_time, "name": ev["name"], "color": col})
        for name in actions:
            if name not in ("gaze_at_player", "micro_nod"):  # shown via arrow / nod flash
                col = ACTION_COLS.get(name, DEFAULT_EV_COL)
                self.tl_events.append({"time": sim_time, "name": name, "color": col})

    def prune(self, sim_time: float):
        self.labels    = [l for l in self.labels
                           if sim_time - l["born"] < LABEL_LIFE]
        self.tl_events = [e for e in self.tl_events
                           if sim_time - e["time"] < 10.0]


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_scene(screen, fonts, w: World, state: dict, labels: list, phase: str):
    surf = pygame.Surface((SCENE_W, SCENE_H))
    surf.fill(SCENE_BG)

    nx = int(w.npc_x)
    ny = int(w.npc_y)
    px = int(w.player_x)
    py = int(w.player_y)

    # Distance line
    pygame.draw.line(surf, DIST_COL, (px, py), (nx, ny), 1)

    # NPC home marker (thin tick)
    hx = int(w.npc_home_x)
    pygame.draw.line(surf, (42, 52, 72), (hx, ny - 10), (hx, ny + 10), 1)

    # Freeze aura
    freeze_val = state.get("freeze_val", 0.0)
    _draw_freeze_aura(surf, nx, ny, w.npc_frozen, freeze_val)

    # Recovery ring
    recovery = state.get("recovery", 0.0)
    settle   = state.get("settle",   0.0)
    npc_r    = int(26 + settle * 5 + (w.settle_flash / SETTLE_FLASH_LIFE) * 6
                   if w.settle_flash > 0 else 26 + settle * 5)
    ndx = nx + int(w.jitter_x)
    ndy = ny + int(w.jitter_y)

    if recovery > 0.06:
        rec_alpha = int(recovery * 165)
        rec_r     = npc_r + 10
        rs = pygame.Surface((rec_r * 2 + 4, rec_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(rs, (255, 160, 70, rec_alpha),
                           (rec_r + 2, rec_r + 2), rec_r, 3)
        surf.blit(rs, (ndx - rec_r - 2, ndy - rec_r - 2))

    # NPC body
    pygame.draw.circle(surf, NPC_COL, (ndx, ndy), npc_r)
    pygame.draw.circle(surf, (75, 95, 140), (ndx, ndy), npc_r, 2)
    surf.blit(fonts["sm"].render("NPC", True, TEXT_DIM),
              (ndx - 12, ndy + npc_r + 6))

    # Gaze arrow
    _draw_gaze(surf, ndx, ndy, px, py, npc_r, w.gaze_snap)

    # Player
    _draw_player(surf, fonts, px, py, w.player_speaking, w.speech_energy, w.sim_time)

    # Event labels (upper-left)
    _draw_labels(surf, fonts, labels, w.sim_time)

    # Player phase + distance overlay (bottom-left)
    ph_txt = fonts["sm"].render(f"player: {phase}   dist: {w.distance:.0f}px",
                                 True, (60, 72, 95))
    surf.blit(ph_txt, (8, SCENE_H - 20))

    # Closed-loop caption
    cl_txt = fonts["sm"].render("closed loop", True, (42, 52, 72))
    surf.blit(cl_txt, (SCENE_W - cl_txt.get_width() - 8, 8))

    screen.blit(surf, (0, 0))


def _draw_freeze_aura(surf, cx, cy, frozen: bool, freeze_val: float):
    if frozen:
        rs = pygame.Surface((112, 112), pygame.SRCALPHA)
        pygame.draw.circle(rs, (255, 90, 70, 80), (56, 56), 52, 3)
        surf.blit(rs, (cx - 56, cy - 56))

    if freeze_val > 0.02:
        radius = int(34 + (1.0 - freeze_val) * 40)
        alpha  = int(200 * freeze_val)
        rs = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(rs, (255, 90, 70, alpha),
                           (radius + 3, radius + 3), radius, 3)
        surf.blit(rs, (cx - radius - 3, cy - radius - 3))


def _draw_gaze(surf, nx, ny, px, py, npc_r, snap):
    dx, dy = px - nx, py - ny
    dist   = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy     = dx / dist, dy / dist
    snap_t     = snap / GAZE_SNAP_LIFE if snap > 0 else 0.0
    arrow_len  = 38 + snap_t * 24
    width      = 2 + int(snap_t * 2)
    brightness = 0.35 + 0.65 * snap_t
    col        = tuple(int(c * brightness) for c in GAZE_COL)
    origin     = (int(nx + ux * (npc_r + 4)), int(ny + uy * (npc_r + 4)))
    tip        = (int(nx + ux * (npc_r + 4 + arrow_len)),
                  int(ny + uy * (npc_r + 4 + arrow_len)))
    pygame.draw.line(surf, col, origin, tip, width)
    angle = math.atan2(uy, ux)
    head  = 8 + int(snap_t * 4)
    for da in (0.48, -0.48):
        ax = tip[0] - head * math.cos(angle + da)
        ay = tip[1] - head * math.sin(angle + da)
        pygame.draw.line(surf, col, tip, (int(ax), int(ay)), width)


def _draw_player(surf, fonts, px, py, speaking, energy, sim_time):
    r = 18
    pygame.draw.circle(surf, PLAYER_COL, (px, py), r)
    pygame.draw.circle(surf, (150, 115, 52), (px, py), r, 2)
    surf.blit(fonts["sm"].render("PLAYER", True, TEXT_DIM), (px - 24, py + r + 6))
    if speaking:
        n_bars, bar_w, gap = 7, 4, 5
        total_w = n_bars * (bar_w + gap) - gap
        x0 = px - total_w // 2
        for i in range(n_bars):
            phase = sim_time * 11.0 + i * 0.78
            h     = int((0.25 + 0.75 * energy) * 20 * abs(math.sin(phase)))
            bx    = x0 + i * (bar_w + gap)
            pygame.draw.rect(surf, PLAYER_COL, (bx, py - r - 8 - h, bar_w, max(2, h)))


def _draw_labels(surf, fonts, labels, sim_time):
    visible = sorted(labels, key=lambda l: -l["born"])[:6]
    x0 = 12
    for slot, lbl in enumerate(visible):
        age  = sim_time - lbl["born"]
        fade = max(0.0, 1.0 - (age / LABEL_LIFE) ** 1.5)
        col  = tuple(int(c * fade) for c in lbl["color"])
        y    = 32 + slot * 20
        pygame.draw.circle(surf, col, (x0 + 5, y + 5), 4)
        surf.blit(fonts["sm"].render(lbl["text"], True, col), (x0 + 14, y))


def _draw_bars(screen, fonts, state: dict, w: World):
    panel = pygame.Surface((BAR_W, SCENE_H))
    panel.fill(PANEL_BG)
    panel.blit(fonts["med"].render("Internal State", True, TEXT_BRIGHT), (8, 8))

    bar_max_w = BAR_W - 68
    row_h     = (SCENE_H - 110) // len(BAR_KEYS)
    bar_h     = 18

    for i, key in enumerate(BAR_KEYS):
        val  = state.get(key, 0.0)
        col  = BAR_COLS.get(key, (180, 180, 180))
        y    = 40 + i * row_h
        disp = BAR_DISPLAY.get(key, key)
        panel.blit(fonts["sm"].render(disp, True, TEXT_DIM), (5, y))
        pygame.draw.rect(panel, (38, 42, 58),
                         (5, y + 15, bar_max_w, bar_h), border_radius=3)
        fill = int(val * bar_max_w)
        if fill > 0:
            draw_col = col
            if key == "ttp" and w.resp_flash > 0:
                t_f = w.resp_flash / RESP_FLASH_LIFE
                draw_col = tuple(min(255, int(c + (255 - c) * t_f * 0.85)) for c in col)
            pygame.draw.rect(panel, draw_col, (5, y + 15, fill, bar_h), border_radius=3)
        panel.blit(fonts["sm"].render(f"{val:.2f}", True, TEXT_DIM),
                   (bar_max_w + 10, y + 15))

    # Distance meter
    dy = 40 + len(BAR_KEYS) * row_h + 8
    panel.blit(fonts["sm"].render("distance", True, TEXT_DIM), (5, dy))
    pygame.draw.rect(panel, (38, 42, 58), (5, dy + 15, bar_max_w, bar_h), border_radius=3)
    dist_fill = int(min(1.0, w.distance / 350.0) * bar_max_w)
    pygame.draw.rect(panel, (120, 135, 155), (5, dy + 15, dist_fill, bar_h), border_radius=3)
    panel.blit(fonts["sm"].render(f"{w.distance:.0f}px", True, TEXT_DIM),
               (bar_max_w + 10, dy + 15))

    # Approach velocity meter
    dy2 = dy + row_h
    panel.blit(fonts["sm"].render("appr vel", True, TEXT_DIM), (5, dy2))
    pygame.draw.rect(panel, (38, 42, 58), (5, dy2 + 15, bar_max_w, bar_h), border_radius=3)
    av   = max(0.0, w.approach_velocity)
    av_f = int(min(1.0, av / 120.0) * bar_max_w)
    if av_f > 0:
        pygame.draw.rect(panel, (200, 130, 80), (5, dy2 + 15, av_f, bar_h), border_radius=3)
    panel.blit(fonts["sm"].render(f"{av:.1f}", True, TEXT_DIM),
               (bar_max_w + 10, dy2 + 15))

    screen.blit(panel, (BAR_X, 0))


def _draw_timeline(screen, fonts, tl_events: list, sim_time: float):
    tl = pygame.Surface((W, TL_H))
    tl.fill(TL_BG)

    window     = 10.0   # seconds of history to show
    pad_l, pad_r = 18, 18
    tl_w       = W - pad_l - pad_r
    t_start    = sim_time - window

    def tx(t):
        frac = (t - t_start) / window
        return pad_l + int(frac * tl_w)

    # Time ticks
    for t_tick in range(max(0, int(t_start)), int(sim_time) + 1):
        x = tx(float(t_tick))
        if pad_l <= x <= W - pad_r:
            pygame.draw.line(tl, (52, 62, 82), (x, 18), (x, 26))
            tl.blit(fonts["sm"].render(f"{t_tick}s", True, (72, 82, 105)), (x - 10, 26))

    # Event / action ticks
    for ev in tl_events:
        x = tx(ev["time"])
        if pad_l <= x <= W - pad_r:
            pygame.draw.line(tl, ev["color"], (x, 35), (x, 57), 2)

    # Current-time cursor (right edge of visible window)
    cx = W - pad_r
    pygame.draw.line(tl, (200, 210, 255), (cx, 8), (cx, TL_H - 4), 2)

    # Legend: events
    lx = 10
    for name, col in EVENT_COLS.items():
        if lx > W - 140:
            break
        pygame.draw.rect(tl, col, (lx, 63, 10, 4))
        txt = fonts["sm"].render(name.replace("_", " "), True, TEXT_DIM)
        tl.blit(txt, (lx + 13, 58))
        lx += txt.get_width() + 18

    # Legend: actions (step_back, response)
    for name in ("step_back", "response"):
        col = ACTION_COLS[name]
        pygame.draw.rect(tl, col, (lx, 63, 10, 4))
        txt = fonts["sm"].render(f"→{name.replace('_', ' ')}", True, TEXT_DIM)
        tl.blit(txt, (lx + 13, 58))
        lx += txt.get_width() + 18

    screen.blit(tl, (0, TL_Y))


def _draw_hud(screen, fonts, w: World, speed: float,
              paused: bool, frame_n: int, record: bool):
    status = (f"t = {w.sim_time:6.2f}s   "
              f"dist = {w.distance:.0f}px   "
              f"appr = {max(0, w.approach_velocity):.1f}px/s   "
              f"×{speed}")
    if record:
        status += f"  [REC  frame {frame_n}]"
    elif paused:
        status += "  [PAUSED]"
    screen.blit(fonts["sm"].render(status, True, TEXT_DIM), (8, SCENE_H - 18))
    if not record:
        screen.blit(
            fonts["sm"].render("SPACE pause  R restart  Q quit", True, (52, 62, 82)),
            (8, SCENE_H - 4),
        )


# ── Main demo ─────────────────────────────────────────────────────────────────

def run(scenario: str, speed: float, record: bool,
        variant: str = "baseline", seed: int = 7):
    variant_map = PLAYER_SCRIPTS.get(scenario)
    if variant_map is None:
        print(f"Unknown scenario: {scenario!r}")
        print(f"Available: {list(PLAYER_SCRIPTS)}")
        sys.exit(1)
    script_cls = variant_map.get(variant)
    if script_cls is None:
        print(f"Unknown variant: {variant!r}  (available: {list(variant_map)})")
        sys.exit(1)

    rec_dir = None
    if record:
        rec_dir = os.path.join(_HERE, "outputs", "recordings", "world_loop_demo")
        os.makedirs(rec_dir, exist_ok=True)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    variant_label = f"  [{variant}  seed={seed}]" if variant != "baseline" else ""
    pygame.display.set_caption(
        f"EchoLoop v0.1 – Closed-Loop Demo  ·  {scenario}  ·  ×{speed}{variant_label}"
    )
    clock = pygame.time.Clock()
    fonts = {
        "sm":  pygame.font.SysFont("monospace", 12),
        "med": pygame.font.SysFont("monospace", 14),
    }

    # ── Per-run state factory ─────────────────────────────────────────
    def fresh():
        w = World()
        w.distance = abs(w.player_x - w.npc_x)
        return (w, script_cls(seed=seed), EchoLoopState(seed=42),
                LabelTracker(), random.Random(99))

    w, script, echoloop, tracker, rng = fresh()
    log_f, log_wtr, log_path = open_log(scenario, variant)
    print(f"[demo] Logging to {log_path}")

    paused   = False
    frame_n  = 0
    step     = 0
    wall_ref = _wall.monotonic()
    last_wall= wall_ref
    # Accumulator for fractional-speed recording
    step_acc = 0.0

    running = True
    while running:
        now      = _wall.monotonic()
        dt_wall  = now - last_wall
        last_wall = now

        # ── Input ─────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE and not record:
                    paused = not paused
                    if not paused:
                        wall_ref = now - w.sim_time / speed
                elif ev.key == pygame.K_r and not record:
                    log_f.close()
                    w, script, echoloop, tracker, rng = fresh()
                    log_f, log_wtr, log_path = open_log(scenario)
                    step = 0
                    wall_ref = now
                    paused   = False

        # ── Simulation steps ──────────────────────────────────────────
        if record:
            step_acc += speed
            while step_acc >= 1.0:
                events, state, actions = loop_step(w, script, echoloop, rng)
                tracker.push(events, actions, w.sim_time)
                write_log(log_wtr, step, w, state, events, actions, script.phase)
                step    += 1
                step_acc -= 1.0
        elif not paused:
            target = (now - wall_ref) * speed
            while w.sim_time < target:
                events, state, actions = loop_step(w, script, echoloop, rng)
                tracker.push(events, actions, w.sim_time)
                write_log(log_wtr, step, w, state, events, actions, script.phase)
                step += 1

        tracker.prune(w.sim_time)
        state = echoloop.state_dict()
        phase = getattr(script, "phase", "?")

        # ── Draw ──────────────────────────────────────────────────────
        screen.fill(BG)
        _draw_scene(screen, fonts, w, state, tracker.labels, phase)
        _draw_bars(screen, fonts, state, w)
        _draw_timeline(screen, fonts, tracker.tl_events, w.sim_time)
        _draw_hud(screen, fonts, w, speed, paused, frame_n, record)
        pygame.display.flip()

        if record:
            path = os.path.join(rec_dir, f"frame_{frame_n:05d}.png")
            pygame.image.save(screen, path)
            frame_n += 1

        if record and w.sim_time >= 28.0:
            print(f"[demo] {frame_n} frames saved to {rec_dir}")
            running = False

        if not record:
            clock.tick(60)

    log_f.close()
    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="EchoLoop v0.1 – Closed-Loop World Interaction Demo"
    )
    ap.add_argument(
        "--scenario",
        default="sudden_approach_while_speaking_closed_loop",
        help="Scenario name (default: sudden_approach_while_speaking_closed_loop)",
    )
    ap.add_argument(
        "--speed", type=float, default=1.0,
        help="Simulation speed multiplier (default: 1.0)",
    )
    ap.add_argument(
        "--record", action="store_true",
        help="Save PNG frames to outputs/recordings/world_loop_demo/",
    )
    ap.add_argument(
        "--variant", default="baseline", choices=["baseline", "loose"],
        help="Player script variant: baseline (default) or loose",
    )
    ap.add_argument(
        "--seed", type=int, default=7,
        help="RNG seed for the player script (default: 7; most visible in --variant loose)",
    )
    args = ap.parse_args()
    run(args.scenario, args.speed, args.record, args.variant, args.seed)


if __name__ == "__main__":
    main()
