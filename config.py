"""
config.py — Central configuration for eye-tracker cursor control.

All thresholds, hotkeys, timing constants, and calibration parameters
are defined here for easy tuning.
"""

import os

# Suppress noisy TensorFlow / MediaPipe warnings before any imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # ERROR-only
os.environ["GLOG_minloglevel"]       = "2"

import pyautogui

# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = pyautogui.size()

# ---------------------------------------------------------------------------
# Hotkeys  (pynput format — each combo is a *set* of Key / KeyCode objects)
# ---------------------------------------------------------------------------
HOTKEY_TOGGLE = "<ctrl>+<shift>+e"       # IDLE ↔ ACTIVATING / deactivate
HOTKEY_PAUSE  = "<ctrl>+<shift>+p"       # toggle PAUSED ↔ ACTIVE

# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) — blink detection
# ---------------------------------------------------------------------------
EAR_THRESHOLD          = 0.20            # below this → eye is closed
BLINK_MIN_DURATION     = 0.15            # seconds — min closed time for a blink
BLINK_MAX_DURATION     = 0.40            # seconds — max closed time (single blink)
DOUBLE_BLINK_WINDOW    = 1.5             # seconds — max elapsed between two blinks
CLICK_BLINK_MIN        = 0.25            # seconds — intentional "click" blink min
CLICK_BLINK_MAX        = 0.40            # seconds — intentional "click" blink max

# ---------------------------------------------------------------------------
# Iris / Gaze center offset — used during ACTIVATING to verify centered gaze
# ---------------------------------------------------------------------------
GAZE_CENTER_TOLERANCE  = 0.10            # normalized offset ±
GAZE_CENTER_MAX_EYE_OFFSET = 0.18        # per-eye guard for obvious off-center gaze
GAZE_CENTER_HOLD_SECS  = 5.0             # seconds of centered gaze required

# ---------------------------------------------------------------------------
# Cursor smoothing (Exponential Moving Average)
# ---------------------------------------------------------------------------
EMA_ALPHA              = 0.15            # lower = smoother, higher = responsive

# ---------------------------------------------------------------------------
# Auto-pause (reading mode)
# ---------------------------------------------------------------------------
STILL_RADIUS_PX        = 80              # pixels — cursor must stay within
STILL_DURATION_SECS    = 6.0             # seconds of stillness → auto-pause
ACTIVE_AUTOPAUSE_GRACE_SECS = 8.0        # ignore auto-pause just after activation

# ---------------------------------------------------------------------------
# Saccade detection (resume from PAUSED)
# ---------------------------------------------------------------------------
SACCADE_THRESHOLD_PX   = 120             # pixel jump to count as saccade

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_POINTS     = 9               # 3×3 grid (corners + edges + center)
CALIBRATION_HOLD_SECS  = 1.0             # seconds to record per dot
CALIBRATION_DOT_RADIUS = 18              # pixels — dot size on calibration screen
MODEL_SAVE_PATH        = "gaze_model.pkl"

# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker (Tasks API — v0.10.x+)
# ---------------------------------------------------------------------------
FACE_LANDMARKER_MODEL  = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
FACE_MESH_MAX_FACES    = 1
FACE_MESH_MIN_DETECT   = 0.5
FACE_MESH_MIN_TRACK    = 0.5

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------
CAMERA_INDEX           = 0
CAMERA_WIDTH           = 640
CAMERA_HEIGHT          = 480

# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
HUD_WIDTH              = 260
HUD_HEIGHT             = 60
HUD_MARGIN             = 20             # px from screen edge
HUD_ALPHA              = 0.85           # 0.0 – 1.0  (tkinter wm_attributes)
HUD_BG_COLOR           = "#1a1a2e"
HUD_FG_COLOR           = "#00ffaa"
HUD_FONT               = ("Consolas", 12, "bold")
HUD_UPDATE_MS          = 100            # millisecond refresh interval

# ---------------------------------------------------------------------------
# Dashboard window
# ---------------------------------------------------------------------------
DASHBOARD_ENABLED      = True
DASHBOARD_TITLE        = "Eye Tracker Dashboard"
DASHBOARD_WIDTH        = 1180
DASHBOARD_HEIGHT       = 720

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
LOOP_DELAY_MS          = 16              # ≈60 fps target
