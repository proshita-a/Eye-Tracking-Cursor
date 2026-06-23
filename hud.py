"""
hud.py — Always-on-top tkinter HUD overlay.

Shows the current application state (IDLE / ACTIVATING / ACTIVE / PAUSED)
in a small, semi-transparent, non-interactive badge pinned to the top-right
corner of the screen.

The HUD runs on its own Tk mainloop in a dedicated thread so it never blocks
the main tracking loop.
"""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Optional

import cv2
import numpy as np

import config


class HUD:
    """Lightweight status badge rendered via tkinter."""

    def __init__(self) -> None:
        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._thread: Optional[threading.Thread] = None
        self._current_text: str = "IDLE"
        self._running = False

    # ── public API (called from main thread) ──────────────────────────────
    def start(self) -> None:
        """Spawn the HUD window in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def set_text(self, text: str) -> None:
        """Update the displayed status string (thread-safe)."""
        self._current_text = text

    # ── internals ─────────────────────────────────────────────────────────
    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("Eye Tracker HUD")
        self._root.overrideredirect(True)          # no title bar
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", config.HUD_ALPHA)

        # position: top-right corner
        x = config.SCREEN_W - config.HUD_WIDTH - config.HUD_MARGIN
        y = config.HUD_MARGIN
        self._root.geometry(f"{config.HUD_WIDTH}x{config.HUD_HEIGHT}+{x}+{y}")
        self._root.configure(bg=config.HUD_BG_COLOR)

        # make the window click-through on Windows
        try:
            # WS_EX_TRANSPARENT | WS_EX_LAYERED
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            # We rely on overrideredirect + topmost; full click-through
            # requires platform-specific hacks that may not be portable.
        except Exception:
            pass

        self._label = tk.Label(
            self._root,
            text=self._current_text,
            bg=config.HUD_BG_COLOR,
            fg=config.HUD_FG_COLOR,
            font=config.HUD_FONT,
            anchor="center",
        )
        self._label.pack(fill="both", expand=True)

        self._refresh()
        self._root.mainloop()

    def _refresh(self) -> None:
        """Periodic update — syncs displayed text with _current_text."""
        if not self._running:
            self._root.destroy()
            return
        if self._label:
            self._label.configure(text=self._current_text)
        self._root.after(config.HUD_UPDATE_MS, self._refresh)


class Dashboard:
    """OpenCV dashboard with live camera, tracker metrics, and timers."""

    _BG = (23, 24, 28)
    _PANEL = (34, 38, 45)
    _PANEL_2 = (42, 47, 55)
    _TEXT = (232, 236, 241)
    _MUTED = (146, 154, 166)
    _GREEN = (90, 214, 146)
    _BLUE = (107, 169, 255)
    _AMBER = (255, 190, 92)
    _RED = (255, 105, 112)
    _LINE = (69, 76, 88)

    def __init__(self) -> None:
        self._window_ready = False

    def show(self, image: np.ndarray) -> None:
        if not config.DASHBOARD_ENABLED:
            return
        if not self._window_ready:
            cv2.namedWindow(config.DASHBOARD_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                config.DASHBOARD_TITLE,
                config.DASHBOARD_WIDTH,
                config.DASHBOARD_HEIGHT,
            )
            self._window_ready = True
        cv2.imshow(config.DASHBOARD_TITLE, image)

    def render(
        self,
        frame: np.ndarray,
        *,
        state: str,
        status: str,
        eye,
        activation_remaining: Optional[float],
        activation_progress: float,
        cursor_xy: tuple[float, float],
        raw_xy: tuple[float, float],
        model_ready: bool,
        fps: float,
        still_remaining: Optional[float],
    ) -> np.ndarray:
        canvas = np.full(
            (config.DASHBOARD_HEIGHT, config.DASHBOARD_WIDTH, 3),
            self._BG,
            dtype=np.uint8,
        )

        self._draw_header(canvas, state, fps)
        self._draw_camera(canvas, frame, eye)
        self._draw_side_panel(
            canvas,
            state=state,
            status=status,
            eye=eye,
            activation_remaining=activation_remaining,
            activation_progress=activation_progress,
            cursor_xy=cursor_xy,
            raw_xy=raw_xy,
            model_ready=model_ready,
            still_remaining=still_remaining,
        )
        return canvas

    def _draw_header(self, canvas: np.ndarray, state: str, fps: float) -> None:
        self._text(canvas, "Eye Tracker Dashboard", (24, 36), 0.82, self._TEXT, 2)
        self._text(canvas, "live camera + gaze control telemetry", (25, 58), 0.46, self._MUTED, 1)
        self._pill(canvas, (934, 24), (1138, 54), state, self._state_color(state))
        self._text(canvas, f"{fps:04.1f} FPS", (782, 42), 0.5, self._MUTED, 1)

    def _draw_camera(self, canvas: np.ndarray, frame: np.ndarray, eye) -> None:
        x, y, w, h = 24, 78, 820, 616
        self._panel(canvas, (x, y), (x + w, y + h), radius=8)
        self._text(canvas, "Camera Feed", (x + 18, y + 30), 0.62, self._TEXT, 2)

        vx, vy, vw, vh = x + 18, y + 48, w - 36, h - 70
        fitted = self._fit_frame(frame, vw, vh)
        fh, fw = fitted.shape[:2]
        ox = vx + (vw - fw) // 2
        oy = vy + (vh - fh) // 2
        canvas[oy:oy + fh, ox:ox + fw] = fitted
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), self._LINE, 1)

        if eye.face_detected:
            self._pill(canvas, (vx + 16, vy + 16), (vx + 146, vy + 48), "FACE LOCK", self._GREEN)
            cv2.drawMarker(
                canvas,
                (vx + vw // 2, vy + vh // 2),
                self._BLUE,
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
            )
        else:
            self._pill(canvas, (vx + 16, vy + 16), (vx + 156, vy + 48), "NO FACE", self._RED)

        self._text(canvas, "Q or Esc closes the app", (x + 18, y + h - 22), 0.45, self._MUTED, 1)

    def _draw_side_panel(
        self,
        canvas: np.ndarray,
        *,
        state: str,
        status: str,
        eye,
        activation_remaining: Optional[float],
        activation_progress: float,
        cursor_xy: tuple[float, float],
        raw_xy: tuple[float, float],
        model_ready: bool,
        still_remaining: Optional[float],
    ) -> None:
        x, y, w = 872, 78, 284
        self._panel(canvas, (x, y), (x + w, 694), radius=8)

        status_color = self._state_color(state)
        self._text(canvas, "Status", (x + 18, y + 32), 0.58, self._MUTED, 1)
        self._text(canvas, self._clean_status(status), (x + 18, y + 62), 0.57, self._TEXT, 2)

        self._draw_timer(
            canvas,
            center=(x + w // 2, y + 164),
            state=state,
            activation_remaining=activation_remaining,
            activation_progress=activation_progress,
            color=status_color,
        )

        self._text(canvas, "Tracker", (x + 18, y + 262), 0.58, self._MUTED, 1)
        self._metric(canvas, x + 18, y + 292, "Face", "detected" if eye.face_detected else "missing",
                     self._GREEN if eye.face_detected else self._RED)
        self._metric(canvas, x + 18, y + 326, "Model", "calibrated" if model_ready else "needs calibration",
                     self._GREEN if model_ready else self._AMBER)
        self._metric(canvas, x + 18, y + 360, "Cursor", f"{int(cursor_xy[0])}, {int(cursor_xy[1])}", self._BLUE)
        self._metric(canvas, x + 18, y + 394, "Raw gaze", f"{int(raw_xy[0])}, {int(raw_xy[1])}", self._BLUE)

        if still_remaining is not None:
            self._metric(canvas, x + 18, y + 428, "Auto pause", f"{still_remaining:.1f}s", self._AMBER)
        else:
            self._metric(canvas, x + 18, y + 428, "Auto pause", "-", self._MUTED)

        self._text(canvas, "Eye Signals", (x + 18, y + 454), 0.58, self._MUTED, 1)
        self._bar(canvas, x + 18, y + 480, "EAR", eye.ear_avg, 0.0, 0.45, self._GREEN)
        self._bar(canvas, x + 18, y + 522, "Left X", eye.iris_offset_left[0], -0.5, 0.5, self._BLUE)
        self._bar(canvas, x + 18, y + 564, "Left Y", eye.iris_offset_left[1], -0.5, 0.5, self._BLUE)
        self._bar(canvas, x + 18, y + 606, "Right X", eye.iris_offset_right[0], -0.5, 0.5, self._AMBER)
        self._bar(canvas, x + 18, y + 648, "Right Y", eye.iris_offset_right[1], -0.5, 0.5, self._AMBER)

    def _draw_timer(
        self,
        canvas: np.ndarray,
        *,
        center: tuple[int, int],
        state: str,
        activation_remaining: Optional[float],
        activation_progress: float,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        radius = 66
        cv2.circle(canvas, center, radius, self._PANEL_2, 13, lineType=cv2.LINE_AA)

        if state == "ACTIVATING":
            remaining_fraction = max(0.0, min(1.0, 1.0 - activation_progress))
            end_angle = -90 + int(360 * remaining_fraction)
            if remaining_fraction > 0:
                cv2.ellipse(canvas, center, (radius, radius), 0, -90, end_angle, color, 13, cv2.LINE_AA)
            text = "READY" if activation_remaining == 0 else f"{activation_remaining or 0:.1f}s"
            label = "activation"
        else:
            cv2.ellipse(canvas, center, (radius, radius), 0, -90, 270, color, 13, cv2.LINE_AA)
            text = state
            label = "current mode"

        scale = 0.68 if len(text) <= 5 else 0.52
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        self._text(canvas, text, (cx - size[0] // 2, cy + 7), scale, self._TEXT, 2)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        self._text(canvas, label, (cx - label_size[0] // 2, cy + radius + 34), 0.42, self._MUTED, 1)

    def _bar(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        color: tuple[int, int, int],
    ) -> None:
        self._text(canvas, label, (x, y), 0.43, self._MUTED, 1)
        text_value = f"{value:+.3f}" if min_value < 0 else f"{value:.3f}"
        self._text(canvas, text_value, (x + 168, y), 0.43, self._TEXT, 1)
        bx, by, bw, bh = x, y + 10, 248, 8
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), self._PANEL_2, -1)
        t = 0.0 if max_value == min_value else (value - min_value) / (max_value - min_value)
        t = max(0.0, min(1.0, t))
        pos = bx + int(bw * t)
        if min_value < 0 < max_value:
            mid = bx + int(bw * ((0 - min_value) / (max_value - min_value)))
            cv2.line(canvas, (mid, by - 2), (mid, by + bh + 2), self._LINE, 1)
        cv2.rectangle(canvas, (bx, by), (pos, by + bh), color, -1)
        cv2.circle(canvas, (pos, by + bh // 2), 5, color, -1, lineType=cv2.LINE_AA)

    def _metric(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        label: str,
        value: str,
        color: tuple[int, int, int],
    ) -> None:
        self._text(canvas, label, (x, y), 0.46, self._MUTED, 1)
        self._text(canvas, value, (x + 92, y), 0.46, color, 1)

    def _panel(self, canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], radius: int = 8) -> None:
        x1, y1 = p1
        x2, y2 = p2
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), self._PANEL, -1)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), self._PANEL, -1)
        for cx, cy in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(canvas, (cx, cy), radius, self._PANEL, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, p1, p2, self._LINE, 1)

    def _pill(
        self,
        canvas: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        cv2.rectangle(canvas, p1, p2, self._PANEL_2, -1)
        cv2.rectangle(canvas, p1, p2, color, 1)
        cv2.circle(canvas, (x1 + 16, (y1 + y2) // 2), 5, color, -1, lineType=cv2.LINE_AA)
        self._text(canvas, text, (x1 + 30, y1 + 21), 0.45, self._TEXT, 1)

    def _fit_frame(self, frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        scale = min(max_w / src_w, max_h / src_h)
        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def _state_color(self, state: str) -> tuple[int, int, int]:
        if state == "ACTIVE":
            return self._GREEN
        if state == "ACTIVATING":
            return self._AMBER
        if state == "PAUSED":
            return self._BLUE
        return self._MUTED

    def _clean_status(self, status: str) -> str:
        encoded = status.encode("ascii", "ignore").decode("ascii").strip()
        if not encoded:
            return status.strip()[:30]
        return encoded[:30]

    def _text(
        self,
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
