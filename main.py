"""
main.py — State machine loop for eye-tracker cursor control.

States:  IDLE → ACTIVATING → ACTIVE ⇄ PAUSED
                   ↑                       │
                   └───────────────────────┘  (deactivate via hotkey)

Global hotkeys (via pynput):
  Ctrl+Shift+E  — toggle IDLE ↔ ACTIVATING, or deactivate from any state
  Ctrl+Shift+P  — toggle ACTIVE ↔ PAUSED
"""

from __future__ import annotations

import config  # must be first — sets env vars to suppress TF/MediaPipe warnings

import enum
import math
import sys
import time

import cv2
import numpy as np
import pyautogui
from pynput import keyboard

from eye_tracker import EyeTracker
from gaze_model import GazeModel
from hud import Dashboard, HUD

pyautogui.FAILSAFE = False          # allow moving to corners
pyautogui.PAUSE    = 0              # no artificial delay


# ──────────────────────────────────────────────────────────────────────────────
# Application states
# ──────────────────────────────────────────────────────────────────────────────
class State(enum.Enum):
    IDLE       = "IDLE"
    ACTIVATING = "ACTIVATING"
    ACTIVE     = "ACTIVE"
    PAUSED     = "PAUSED"


# ──────────────────────────────────────────────────────────────────────────────
# Main controller
# ──────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self) -> None:
        self.state = State.IDLE

        # Sub-systems
        self.tracker   = EyeTracker()
        self.model     = GazeModel()
        self.hud       = HUD()
        self.dashboard = Dashboard()
        self._status_text = "IDLE"

        # Smoothing / cursor state
        self._smooth_x: float = config.SCREEN_W / 2
        self._smooth_y: float = config.SCREEN_H / 2
        self._raw_x: float = self._smooth_x
        self._raw_y: float = self._smooth_y
        self._fps: float = 0.0
        self._last_frame_time: float = 0.0

        # Activating state tracking
        self._center_start: float = 0.0
        self._center_locked: bool = False
        self._awaiting_double_blink: bool = False
        self._activation_remaining: float | None = None
        self._activation_progress: float = 0.0

        # Auto-pause tracking
        self._still_anchor_x: float = 0.0
        self._still_anchor_y: float = 0.0
        self._still_start: float = 0.0
        self._still_remaining: float | None = None
        self._active_since: float = 0.0

        # Hotkey flags (set from listener thread)
        self._toggle_flag: bool = False
        self._pause_flag:  bool = False

    # ── hotkey callbacks ──────────────────────────────────────────────────
    def _on_toggle(self) -> None:
        self._toggle_flag = True

    def _on_pause(self) -> None:
        self._pause_flag = True

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.hud.set_text(text)

    def _is_gaze_centered(self, eye) -> bool:
        if eye.ear_avg <= config.EAR_THRESHOLD + 0.03:
            return False

        lox, loy = eye.iris_offset_left
        rox, roy = eye.iris_offset_right
        avg_x = (lox + rox) / 2.0
        avg_y = (loy + roy) / 2.0

        return (
            abs(avg_x) <= config.GAZE_CENTER_TOLERANCE
            and abs(avg_y) <= config.GAZE_CENTER_TOLERANCE
            and max(abs(lox), abs(rox), abs(loy), abs(roy))
            <= config.GAZE_CENTER_MAX_EYE_OFFSET
        )

    # ── state transitions ─────────────────────────────────────────────────
    def _handle_hotkeys(self) -> None:
        if self._toggle_flag:
            self._toggle_flag = False
            if self.state == State.IDLE:
                self._enter_activating()
            else:
                self._enter_idle()

        if self._pause_flag:
            self._pause_flag = False
            if self.state == State.ACTIVE:
                self._enter_paused()
            elif self.state == State.PAUSED:
                self._enter_active()

    def _enter_idle(self) -> None:
        self.state = State.IDLE
        self._activation_remaining = None
        self._activation_progress = 0.0
        self._still_remaining = None
        self._set_status("IDLE")

    def _enter_activating(self) -> None:
        self.state = State.ACTIVATING
        self._center_start = 0.0
        self._center_locked = False
        self._awaiting_double_blink = False
        self._activation_remaining = config.GAZE_CENTER_HOLD_SECS
        self._activation_progress = 0.0
        self._still_remaining = None
        self.tracker.blink_detector = type(self.tracker.blink_detector)()
        self._set_status("ACTIVATING")

    def _enter_active(self) -> None:
        self.state = State.ACTIVE
        self._activation_remaining = None
        self._activation_progress = 0.0
        now = time.monotonic()
        self._active_since = now
        self._still_start = now
        self._still_anchor_x = self._smooth_x
        self._still_anchor_y = self._smooth_y
        self._still_remaining = config.STILL_DURATION_SECS
        self._set_status("ACTIVE")

    def _enter_paused(self) -> None:
        self.state = State.PAUSED
        self._activation_remaining = None
        self._activation_progress = 0.0
        self._still_remaining = None
        self._set_status("Paused - reading mode")

    # ── per-state logic ───────────────────────────────────────────────────
    def _tick_activating(self, eye) -> None:
        now = time.monotonic()

        if not eye.face_detected:
            self._center_start = 0.0
            self._center_locked = False
            self._awaiting_double_blink = False
            self._activation_remaining = config.GAZE_CENTER_HOLD_SECS
            self._activation_progress = 0.0
            self._set_status("ACTIVATING - no face")
            return

        # ── Phase 1: centered gaze for 5 s ───────────────────────────────
        if not self._awaiting_double_blink:
            if self._is_gaze_centered(eye):
                if self._center_start == 0.0:
                    self._center_start = now
                elapsed = now - self._center_start
                remaining = max(0, config.GAZE_CENTER_HOLD_SECS - elapsed)
                self._activation_remaining = remaining
                self._activation_progress = min(1.0, elapsed / config.GAZE_CENTER_HOLD_SECS)
                self._set_status(f"Look at camera - {remaining:.1f}s")
                if elapsed >= config.GAZE_CENTER_HOLD_SECS:
                    self._awaiting_double_blink = True
                    self._activation_remaining = 0.0
                    self._activation_progress = 1.0
                    self._set_status("Double-blink to confirm")
            else:
                self._center_start = 0.0
                self._activation_remaining = config.GAZE_CENTER_HOLD_SECS
                self._activation_progress = 0.0
                self._set_status("Center gaze - timer reset")
        else:
            # ── Phase 2: double blink to confirm ─────────────────────────
            self.tracker.blink_detector.update(eye.ear_avg, now)
            if self.tracker.blink_detector.is_double_blink():
                # check if model is calibrated
                if not self.model.is_ready:
                    self._set_status("Calibrating")
                    ok = self.model.calibrate(self.tracker)
                    if not ok:
                        self._set_status("Calibration failed - retry")
                        self._awaiting_double_blink = False
                        self._center_start = 0.0
                        self._activation_remaining = config.GAZE_CENTER_HOLD_SECS
                        self._activation_progress = 0.0
                        return
                self._enter_active()

    def _tick_active(self, eye) -> None:
        now = time.monotonic()

        if not self.model.is_ready:
            self._set_status("Model not calibrated")
            return

        if not eye.face_detected or eye.gaze_features is None:
            self._still_start = now
            self._still_remaining = config.STILL_DURATION_SECS
            return

        # Predict raw gaze → screen
        raw_x, raw_y = self.model.predict(eye.gaze_features)
        self._raw_x = raw_x
        self._raw_y = raw_y

        # EMA smoothing
        self._smooth_x += config.EMA_ALPHA * (raw_x - self._smooth_x)
        self._smooth_y += config.EMA_ALPHA * (raw_y - self._smooth_y)

        sx = int(self._smooth_x)
        sy = int(self._smooth_y)

        pyautogui.moveTo(sx, sy, _pause=False)

        # ── blink → click ────────────────────────────────────────────────
        evt = self.tracker.blink_detector.update(eye.ear_avg, now)
        if evt and self.tracker.blink_detector.last_blink_is_click():
            pyautogui.click(_pause=False)

        # ── auto-pause (reading mode) ────────────────────────────────────
        dx = sx - self._still_anchor_x
        dy = sy - self._still_anchor_y
        dist = math.hypot(dx, dy)

        active_elapsed = now - self._active_since
        if active_elapsed < config.ACTIVE_AUTOPAUSE_GRACE_SECS:
            self._still_anchor_x = sx
            self._still_anchor_y = sy
            self._still_start = now
            self._still_remaining = config.STILL_DURATION_SECS
        elif dist > config.STILL_RADIUS_PX:
            self._still_anchor_x = sx
            self._still_anchor_y = sy
            self._still_start = now
            self._still_remaining = config.STILL_DURATION_SECS
        elif now - self._still_start >= config.STILL_DURATION_SECS:
            self._enter_paused()
        else:
            self._still_remaining = max(0.0, config.STILL_DURATION_SECS - (now - self._still_start))

    def _tick_paused(self, eye) -> None:
        now = time.monotonic()

        if not eye.face_detected or eye.gaze_features is None:
            return

        # Predict (don't move cursor, just check for saccade)
        raw_x, raw_y = self.model.predict(eye.gaze_features)
        self._raw_x = raw_x
        self._raw_y = raw_y
        dx = raw_x - self._smooth_x
        dy = raw_y - self._smooth_y
        dist = math.hypot(dx, dy)

        if dist > config.SACCADE_THRESHOLD_PX:
            self._smooth_x = raw_x
            self._smooth_y = raw_y
            self._enter_active()

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self) -> None:
        print("[eye-tracker] Opening webcam …")
        if not self.tracker.open():
            print("[eye-tracker] ERROR: cannot open webcam.")
            sys.exit(1)

        # Try to load a previously saved calibration model
        if self.model.load():
            print("[eye-tracker] Loaded saved gaze model.")
        else:
            print("[eye-tracker] No saved model — calibration will run on first activation.")

        # Start HUD
        self.hud.start()
        self._set_status("IDLE")

        # Register global hotkeys
        hotkeys = keyboard.GlobalHotKeys({
            config.HOTKEY_TOGGLE: self._on_toggle,
            config.HOTKEY_PAUSE:  self._on_pause,
        })
        hotkeys.start()

        print(f"[eye-tracker] Ready.  Ctrl+Shift+E to activate.  Ctrl+Shift+P to pause.")

        try:
            while True:
                self._handle_hotkeys()

                frame = self.tracker.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.monotonic()
                if self._last_frame_time:
                    instant_fps = 1.0 / max(0.001, now - self._last_frame_time)
                    self._fps += 0.15 * (instant_fps - self._fps)
                self._last_frame_time = now

                eye = self.tracker.process(frame)

                if self.state == State.ACTIVATING:
                    self._tick_activating(eye)
                elif self.state == State.ACTIVE:
                    self._tick_active(eye)
                elif self.state == State.PAUSED:
                    self._tick_paused(eye)
                # IDLE — nothing to do

                dashboard = self.dashboard.render(
                    frame,
                    state=self.state.value,
                    status=self._status_text,
                    eye=eye,
                    activation_remaining=self._activation_remaining,
                    activation_progress=self._activation_progress,
                    cursor_xy=(self._smooth_x, self._smooth_y),
                    raw_xy=(self._raw_x, self._raw_y),
                    model_ready=self.model.is_ready,
                    fps=self._fps,
                    still_remaining=self._still_remaining,
                )
                self.dashboard.show(dashboard)

                # Small delay to limit CPU usage
                key = cv2.waitKey(config.LOOP_DELAY_MS) & 0xFF
                if key in (27, ord("q")):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            print("\n[eye-tracker] Shutting down …")
            hotkeys.stop()
            self.hud.stop()
            self.tracker.close()
            cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
