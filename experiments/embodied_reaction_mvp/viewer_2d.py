#!/usr/bin/env python3
"""
EchoLoop Embodied Reaction MVP – 2D Playback Viewer (v0.1)

Reads per-timestep logs from outputs/logs/ and replays them as a 2D animation.
Does not modify model dynamics.  If logs are missing, runs the simulation first.

Usage:
    python viewer_2d.py --scenario sudden_approach_while_speaking
    python viewer_2d.py --scenario speech_then_silence --speed 2.0
    python viewer_2d.py --scenario sudden_approach_while_speaking --record

Keys during playback:
    SPACE   pause / resume
    R       restart from t=0
    Q/ESC   quit
"""

import argparse
import bisect
import csv
import json
import math
import os
import random
import sys
import time as _wall

# ── pygame import guard ───────────────────────────────────────────────────────
try:
    import pygame
except ImportError:
    print(
        "\npygame is not installed.\n"
        "Install it with one of:\n"
        "    pip install pygame\n"
        "    pip install pygame-ce\n"
        "\nThen re-run the viewer.\n"
    )
    sys.exit(1)

# ── Layout ────────────────────────────────────────────────────────────────────
W, H        = 920, 640
SCENE_W     = 620
SCENE_H     = 510
BAR_X       = 640       # left edge of the internal-state bars panel
BAR_W       = 260
TL_Y        = 520       # timeline top edge
TL_H        = H - TL_Y  # 120 px

NPC_BASE_R      = 26    # NPC circle radius at settle=0
PLAYER_BASE_DIST = 250  # default NPC→Player distance (pixels)
PLAYER_MIN_DIST  = 70

NPC_POS = (190, 255)    # NPC centre in scene coordinates

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
    "fidget_suppression": (220, 195, 75),
    "micro_nod_ready":    (160, 155, 255),
    "response_ready":     (255, 255, 120),
    "freeze":             (255,  90, 70),
}
DEFAULT_EV_COL = (180, 180, 180)

BAR_KEYS = ["orient", "settle", "fidget_inhibit", "ttp", "recovery", "freeze_val"]
BAR_COLS = {
    "orient":         (100, 220, 255),
    "settle":         (100, 220, 110),
    "fidget_inhibit": (220, 195, 75),
    "ttp":            (150, 150, 255),
    "recovery":       (255, 160, 70),
    "freeze_val":     (255,  90, 70),
}
BAR_DISPLAY = {
    "orient":         "orient",
    "settle":         "settle",
    "fidget_inhibit": "fidget inh",
    "ttp":            "ttp",
    "recovery":       "recovery",
    "freeze_val":     "freeze",
}

# Visual effect durations (seconds)
LABEL_LIFE        = 2.5
FREEZE_RING_LIFE  = 0.90
GAZE_SNAP_LIFE    = 0.55
SETTLE_FLASH_LIFE = 0.70
RESP_FLASH_LIFE   = 0.50


# ── Log helpers ───────────────────────────────────────────────────────────────

def _log_paths(scenario):
    here    = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(here, "outputs", "logs")
    return (
        os.path.join(log_dir, f"{scenario}.csv"),
        os.path.join(log_dir, f"{scenario}_events.json"),
    )


def _load_logs(scenario):
    csv_path, ev_path = _log_paths(scenario)
    if not os.path.isfile(csv_path) or not os.path.isfile(ev_path):
        return None, None

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k != "events" else v) for k, v in r.items()})

    with open(ev_path) as f:
        events = json.load(f)

    return rows, events


def _run_and_save(scenario):
    """Fall-back: run the simulation, save logs, then return the loaded data."""
    import pathlib
    here = pathlib.Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from scenarios import get_scenario
    from echoloop import run_simulation

    print(f"[viewer] Logs not found — running simulation for '{scenario}' …")
    signals  = get_scenario(scenario)
    records, event_log = run_simulation(signals)

    log_dir = here / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / f"{scenario}.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wtr.writeheader()
        wtr.writerows(records)

    ev_path = log_dir / f"{scenario}_events.json"
    with open(ev_path, "w") as f:
        json.dump(event_log, f, indent=2)

    print(f"[viewer] Saved logs to {log_dir}")
    return _load_logs(scenario)


# ── Viewer ─────────────────────────────────────────────────────────────────────

class Viewer:
    def __init__(self, rows, events, scenario, speed, record):
        self.rows     = rows
        self.events   = events
        self.scenario = scenario
        self.speed    = speed
        self.record   = record
        self.t_max    = rows[-1]["time"]

        self._times = [r["time"] for r in rows]

        # Map rounded sim_time → list of event names that fire at that step
        self._ev_at: dict = {}
        for ev in events:
            key = round(ev["time"], 4)
            self._ev_at.setdefault(key, []).append(ev["name"])

        # Integrate approach_velocity into a smooth player-distance signal
        self._player_dists = self._compute_player_dists()

        # Playback state
        self.sim_time  = 0.0
        self._wall_ref = None
        self._frame_n  = 0   # recording frame counter

        # Visual effect state
        self._labels         = []          # [{text, color, born}]
        self._freeze_rings   = []          # [{born}]
        self._gaze_snap      = 0.0
        self._settle_flash   = 0.0
        self._resp_flash     = 0.0
        self._jitter_target  = (0.0, 0.0)
        self._jitter_cur     = (0.0, 0.0)
        self._jitter_timer   = 0.0
        self._rng            = random.Random(42)

        # Recording directory
        self._rec_dir = None
        if record:
            here = os.path.dirname(os.path.abspath(__file__))
            self._rec_dir = os.path.join(here, "outputs", "recordings", scenario)
            os.makedirs(self._rec_dir, exist_ok=True)

    # ── Pre-computation ───────────────────────────────────────────────────────

    def _compute_player_dists(self):
        """
        Integrate approach_velocity into per-frame player distance values.
        Player retreats back to base distance when approach_velocity is low.
        This makes the sudden approach pulse visible as the player dot moving closer.
        """
        dist    = float(PLAYER_BASE_DIST)
        dists   = []
        retreat = 0.38
        scale   = 88.0
        dt      = 1.0 / 60.0
        for r in self.rows:
            v     = r.get("approach_velocity", 0.0)
            dist -= v * scale * dt
            dist += (PLAYER_BASE_DIST - dist) * retreat * dt
            dist  = max(PLAYER_MIN_DIST, min(PLAYER_BASE_DIST, dist))
            dists.append(dist)
        return dists

    def _row_at(self, t):
        idx = min(bisect.bisect_left(self._times, t), len(self.rows) - 1)
        return self.rows[idx], idx

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(
            f"EchoLoop v0.1  ·  {self.scenario}  ·  ×{self.speed}"
        )
        clock = pygame.time.Clock()
        self._fonts = {
            "sm":  pygame.font.SysFont("monospace", 12),
            "med": pygame.font.SysFont("monospace", 14),
        }

        paused   = False
        prev_idx = -1
        now      = _wall.monotonic()
        self._wall_ref = now
        last_wall      = now

        running = True
        while running:
            now      = _wall.monotonic()
            dt_wall  = now - last_wall
            last_wall = now

            # ── Input ─────────────────────────────────────────────────
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_SPACE and not self.record:
                        paused = not paused
                        if not paused:
                            self._wall_ref = now - self.sim_time / self.speed
                    elif ev.key == pygame.K_r and not self.record:
                        self._reset(now)
                        prev_idx = -1

            # ── Advance simulation time ───────────────────────────────
            if self.record:
                self.sim_time += self.speed / 60.0
            elif not paused:
                self.sim_time = (now - self._wall_ref) * self.speed

            self.sim_time = min(self.sim_time, self.t_max)

            # ── Detect new events (rows prev_idx+1 .. cur_idx) ────────
            row, cur_idx = self._row_at(self.sim_time)
            new_evs = []
            for i in range(max(prev_idx + 1, 0), cur_idx + 1):
                key = round(self._times[i], 4)
                if key in self._ev_at:
                    new_evs.extend(self._ev_at[key])
            prev_idx = cur_idx

            # ── Update visual state ───────────────────────────────────
            eff_dt = (self.speed / 60.0) if self.record else dt_wall
            self._update_visual(row, new_evs, eff_dt)

            # ── Draw ──────────────────────────────────────────────────
            screen.fill(BG)
            self._draw_scene(screen, row, cur_idx)
            self._draw_bars(screen, row)
            self._draw_timeline(screen, cur_idx)
            self._draw_hud(screen, paused)
            pygame.display.flip()

            # ── Recording ─────────────────────────────────────────────
            if self.record:
                path = os.path.join(self._rec_dir, f"frame_{self._frame_n:05d}.png")
                pygame.image.save(screen, path)
                self._frame_n += 1

            # ── Loop / finish ─────────────────────────────────────────
            if self.sim_time >= self.t_max:
                if self.record:
                    print(f"[viewer] {self._frame_n} frames saved to {self._rec_dir}")
                    running = False
                else:
                    self._reset(now)
                    prev_idx = -1

            if not self.record:
                clock.tick(60)

        pygame.quit()

    def _reset(self, now):
        self.sim_time        = 0.0
        self._wall_ref       = now
        self._labels         = []
        self._freeze_rings   = []
        self._gaze_snap      = 0.0
        self._settle_flash   = 0.0
        self._resp_flash     = 0.0
        self._jitter_cur     = (0.0, 0.0)
        self._jitter_target  = (0.0, 0.0)
        self._jitter_timer   = 0.0
        self._rng            = random.Random(42)

    # ── Visual state update ───────────────────────────────────────────────────

    def _update_visual(self, row, new_evs, dt):
        fi = row.get("fidget_inhibit", 0.0)

        # Decay timers
        self._gaze_snap    = max(0.0, self._gaze_snap    - dt)
        self._settle_flash = max(0.0, self._settle_flash - dt)
        self._resp_flash   = max(0.0, self._resp_flash   - dt)

        # NPC jitter: magnitude driven by (1 − fidget_inhibit)
        self._jitter_timer -= dt
        if self._jitter_timer <= 0.0:
            mag = (1.0 - fi) * 5.0
            self._jitter_target = (
                self._rng.uniform(-mag, mag),
                self._rng.uniform(-mag, mag),
            )
            self._jitter_timer = 0.10
        lerp = min(1.0, dt * 12.0)
        jx   = self._jitter_cur[0] + (self._jitter_target[0] - self._jitter_cur[0]) * lerp
        jy   = self._jitter_cur[1] + (self._jitter_target[1] - self._jitter_cur[1]) * lerp
        self._jitter_cur = (jx, jy)

        # Prune aged effects
        self._labels       = [l for l in self._labels
                               if self.sim_time - l["born"] < LABEL_LIFE]
        self._freeze_rings = [r for r in self._freeze_rings
                               if self.sim_time - r["born"] < FREEZE_RING_LIFE]

        # Map events to visual triggers
        for name in new_evs:
            if name == "gaze_shift":
                self._gaze_snap = GAZE_SNAP_LIFE
            elif name == "posture_settle":
                self._settle_flash = SETTLE_FLASH_LIFE
            elif name == "response_ready":
                self._resp_flash = RESP_FLASH_LIFE
            elif name == "freeze":
                self._freeze_rings.append({"born": self.sim_time})

            col = EVENT_COLS.get(name, DEFAULT_EV_COL)
            self._labels.append({
                "text":  name.replace("_", " "),
                "color": col,
                "born":  self.sim_time,
            })

    # ── Scene ─────────────────────────────────────────────────────────────────

    def _draw_scene(self, screen, row, cur_idx):
        surf = pygame.Surface((SCENE_W, SCENE_H))
        surf.fill(SCENE_BG)

        nx, ny = NPC_POS
        pdist  = self._player_dists[cur_idx]
        px, py = int(nx + pdist), ny

        # Faint line showing NPC–Player distance
        pygame.draw.line(surf, DIST_COL, (nx, ny), (px, py), 1)

        # Freeze rings (expand outward and fade)
        for ring in self._freeze_rings:
            age = self.sim_time - ring["born"]
            t   = age / FREEZE_RING_LIFE
            if t >= 1.0:
                continue
            radius = int(34 + t * 54)
            alpha  = int(255 * (1.0 - t) ** 1.6)
            rs = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
            pygame.draw.circle(rs, (255, 90, 70, alpha),
                               (radius + 3, radius + 3), radius, 3)
            surf.blit(rs, (nx - radius - 3, ny - radius - 3))

        # Recovery aura: orange ring proportional to backpressure
        recovery = row.get("recovery", 0.0)
        settle   = row.get("settle", 0.0)
        npc_r    = int(NPC_BASE_R + settle * 5)
        if self._settle_flash > 0:
            npc_r += int(self._settle_flash / SETTLE_FLASH_LIFE * 6)

        jx, jy  = int(self._jitter_cur[0]), int(self._jitter_cur[1])
        ndx, ndy = nx + jx, ny + jy

        if recovery > 0.06:
            rec_alpha = int(recovery * 165)
            rec_r     = npc_r + 10
            rs = pygame.Surface((rec_r * 2 + 4, rec_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(rs, (255, 160, 70, rec_alpha),
                               (rec_r + 2, rec_r + 2), rec_r, 3)
            surf.blit(rs, (ndx - rec_r - 2, ndy - rec_r - 2))

        # NPC body (jitter magnitude reflects low fidget_inhibit)
        pygame.draw.circle(surf, NPC_COL, (ndx, ndy), npc_r)
        pygame.draw.circle(surf, (75, 95, 140), (ndx, ndy), npc_r, 2)
        lbl = self._fonts["sm"].render("NPC", True, TEXT_DIM)
        surf.blit(lbl, (ndx - 12, ndy + npc_r + 6))

        # Gaze arrow
        self._draw_gaze(surf, nx, ny, px, py, npc_r)

        # Player dot and optional speech waveform
        speaking = row.get("user_speaking", 0.0)
        energy   = row.get("speech_energy", 0.0)
        self._draw_player(surf, px, py, speaking, energy)

        # Event labels (upper-left of scene)
        self._draw_labels(surf)

        # Scenario name (dim, top-right)
        sc = self._fonts["sm"].render(self.scenario.replace("_", " "), True,
                                      (55, 65, 90))
        surf.blit(sc, (SCENE_W - sc.get_width() - 8, 8))

        screen.blit(surf, (0, 0))

    def _draw_gaze(self, surf, nx, ny, px, py, npc_r):
        """Arrow from NPC toward Player; pulses brighter on gaze_shift."""
        dx, dy = px - nx, py - ny
        dist   = math.hypot(dx, dy)
        if dist < 1:
            return
        ux, uy = dx / dist, dy / dist

        snap_t     = self._gaze_snap / GAZE_SNAP_LIFE if self._gaze_snap > 0 else 0.0
        arrow_len  = 38 + snap_t * 24
        width      = 2 + int(snap_t * 2)
        brightness = 0.35 + 0.65 * snap_t
        col = tuple(int(c * brightness) for c in GAZE_COL)

        origin = (int(nx + ux * (npc_r + 4)), int(ny + uy * (npc_r + 4)))
        tip    = (int(nx + ux * (npc_r + 4 + arrow_len)),
                  int(ny + uy * (npc_r + 4 + arrow_len)))
        pygame.draw.line(surf, col, origin, tip, width)

        angle = math.atan2(uy, ux)
        head  = 8 + int(snap_t * 4)
        for da in (0.48, -0.48):
            ax = tip[0] - head * math.cos(angle + da)
            ay = tip[1] - head * math.sin(angle + da)
            pygame.draw.line(surf, col, tip, (int(ax), int(ay)), width)

    def _draw_player(self, surf, px, py, speaking, energy):
        r = 18
        pygame.draw.circle(surf, PLAYER_COL, (px, py), r)
        pygame.draw.circle(surf, (150, 115, 52), (px, py), r, 2)
        lbl = self._fonts["sm"].render("PLAYER", True, TEXT_DIM)
        surf.blit(lbl, (px - 24, py + r + 6))

        # Speech waveform above player when user_speaking > 0.5
        if speaking > 0.5:
            n_bars  = 7
            bar_w   = 4
            gap     = 5
            total_w = n_bars * (bar_w + gap) - gap
            x0      = px - total_w // 2
            for i in range(n_bars):
                phase = self.sim_time * 11.0 + i * 0.78
                h     = int((0.25 + 0.75 * energy) * 20 * abs(math.sin(phase)))
                bx    = x0 + i * (bar_w + gap)
                pygame.draw.rect(surf, PLAYER_COL,
                                 (bx, py - r - 8 - h, bar_w, max(2, h)))

    def _draw_labels(self, surf):
        """Show the 6 most recent event labels, fading out over LABEL_LIFE seconds."""
        visible = sorted(self._labels, key=lambda l: -l["born"])[:6]
        x0 = 12
        for slot, lbl in enumerate(visible):
            age  = self.sim_time - lbl["born"]
            fade = max(0.0, 1.0 - (age / LABEL_LIFE) ** 1.5)
            col  = tuple(int(c * fade) for c in lbl["color"])
            y    = 32 + slot * 20
            pygame.draw.circle(surf, col, (x0 + 5, y + 5), 4)
            txt = self._fonts["sm"].render(lbl["text"], True, col)
            surf.blit(txt, (x0 + 14, y))

    # ── Bars panel ────────────────────────────────────────────────────────────

    def _draw_bars(self, screen, row):
        panel = pygame.Surface((BAR_W, SCENE_H))
        panel.fill(PANEL_BG)

        title = self._fonts["med"].render("Internal State", True, TEXT_BRIGHT)
        panel.blit(title, (8, 8))

        bar_max_w = BAR_W - 68
        row_h     = (SCENE_H - 55) // len(BAR_KEYS)
        bar_h     = 18

        for i, key in enumerate(BAR_KEYS):
            val  = row.get(key, 0.0)
            col  = BAR_COLS.get(key, (180, 180, 180))
            y    = 40 + i * row_h
            disp = BAR_DISPLAY.get(key, key)

            lbl = self._fonts["sm"].render(disp, True, TEXT_DIM)
            panel.blit(lbl, (5, y))

            # Track background
            pygame.draw.rect(panel, (38, 42, 58),
                             (5, y + 15, bar_max_w, bar_h), border_radius=3)

            fill = int(val * bar_max_w)
            if fill > 0:
                draw_col = col
                # Flash the ttp bar white when response_ready fires
                if key == "ttp" and self._resp_flash > 0:
                    t_f      = self._resp_flash / RESP_FLASH_LIFE
                    draw_col = tuple(min(255, int(c + (255 - c) * t_f * 0.85))
                                     for c in col)
                pygame.draw.rect(panel, draw_col,
                                (5, y + 15, fill, bar_h), border_radius=3)

            val_txt = self._fonts["sm"].render(f"{val:.2f}", True, TEXT_DIM)
            panel.blit(val_txt, (bar_max_w + 10, y + 15))

        screen.blit(panel, (BAR_X, 0))

    # ── Timeline ──────────────────────────────────────────────────────────────

    def _draw_timeline(self, screen, cur_idx):
        tl = pygame.Surface((W, TL_H))
        tl.fill(TL_BG)

        pad_l, pad_r = 18, 18
        tl_w = W - pad_l - pad_r

        def tx(t):
            return pad_l + int(t / self.t_max * tl_w)

        # ── Time-axis ticks ──
        for t in range(0, int(self.t_max) + 1, 5):
            x = tx(t)
            pygame.draw.line(tl, (52, 62, 82), (x, 18), (x, 26))
            txt = self._fonts["sm"].render(str(t), True, (72, 82, 105))
            tl.blit(txt, (x - 7, 26))

        # ── ttp and recovery signal lines (sampled from CSV) ──
        step     = max(1, len(self.rows) // tl_w)
        ttp_pts  = []
        rec_pts  = []
        for i in range(0, len(self.rows), step):
            r   = self.rows[i]
            x   = tx(r["time"])
            ttp_pts.append((x, 88 - int(r["ttp"]      * 28)))
            rec_pts.append((x, 88 - int(r["recovery"] * 28)))
        if len(ttp_pts) > 1:
            pygame.draw.lines(tl, (90, 90, 200), False, ttp_pts, 1)
        if len(rec_pts) > 1:
            pygame.draw.lines(tl, (190, 120, 52), False, rec_pts, 1)

        # ── Event ticks ──
        for ev in self.events:
            x   = tx(ev["time"])
            col = EVENT_COLS.get(ev["name"], DEFAULT_EV_COL)
            pygame.draw.line(tl, col, (x, 35), (x, 57), 2)

        # ── Current-time cursor ──
        cx = tx(self.sim_time)
        pygame.draw.line(tl, (200, 210, 255), (cx, 8), (cx, TL_H - 4), 2)

        # ── Event legend ──
        lx = 10
        for name, col in EVENT_COLS.items():
            if lx > W - 130:
                break
            pygame.draw.rect(tl, col, (lx, 63, 10, 4))
            txt = self._fonts["sm"].render(name.replace("_", " "), True, TEXT_DIM)
            tl.blit(txt, (lx + 13, 58))
            lx += txt.get_width() + 18

        # ── Signal legend ──
        pygame.draw.line(tl, (90, 90, 200),   (10, 97), (28, 97), 2)
        tl.blit(self._fonts["sm"].render("ttp",      True, TEXT_DIM), (30,  92))
        pygame.draw.line(tl, (190, 120, 52),  (65, 97), (83, 97), 2)
        tl.blit(self._fonts["sm"].render("recovery", True, TEXT_DIM), (85,  92))

        screen.blit(tl, (0, TL_Y))

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, screen, paused):
        status = f"t = {self.sim_time:6.2f}s / {self.t_max:.1f}s   ×{self.speed}"
        if self.record:
            status += f"  [REC  frame {self._frame_n}]"
        elif paused:
            status += "  [PAUSED]"

        screen.blit(self._fonts["sm"].render(status, True, TEXT_DIM),
                    (8, SCENE_H - 18))
        if not self.record:
            screen.blit(
                self._fonts["sm"].render("SPACE pause  R restart  Q quit",
                                         True, (52, 62, 82)),
                (8, SCENE_H - 4),
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EchoLoop v0.1 – 2D Playback Viewer")
    ap.add_argument(
        "--scenario",
        default="sudden_approach_while_speaking",
        help=(
            "Scenario to visualize. Available: sudden_approach_while_speaking, "
            "speech_then_silence, repeated_speech_silence, "
            "random_ambient_noise, long_listening. "
            "(default: sudden_approach_while_speaking)"
        ),
    )
    ap.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    ap.add_argument(
        "--record", action="store_true",
        help="Save one PNG frame per display frame to outputs/recordings/<scenario>/",
    )
    args = ap.parse_args()

    rows, events = _load_logs(args.scenario)
    if rows is None:
        rows, events = _run_and_save(args.scenario)
        if rows is None:
            print(f"[viewer] Could not load or generate logs for '{args.scenario}'.")
            sys.exit(1)

    Viewer(rows, events, args.scenario, args.speed, args.record).run()


if __name__ == "__main__":
    main()
