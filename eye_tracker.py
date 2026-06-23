"""
eye_tracker.py — MediaPipe FaceLandmarker (Tasks API) wrapper.

Provides:
  • Webcam capture lifecycle (open / read / release)
  • Face landmark extraction via the Tasks-based FaceLandmarker (with iris)
  • Eye Aspect Ratio (EAR) computation
  • Iris-offset from eye center (normalized)
  • Feature vector for gaze regression model
  • Blink detector (single + double) via EAR duration tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

import config

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices  (478-point model with iris)
# ──────────────────────────────────────────────────────────────────────────────
# Right eye outline (user's right — image left)
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye outline (user's left — image right)
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# Iris centre landmarks (468 = right iris centre, 473 = left iris centre)
_RIGHT_IRIS = [468, 469, 470, 471, 472]   # 468 = centre
_LEFT_IRIS  = [473, 474, 475, 476, 477]   # 473 = centre
_FEATURE_SIZE = 24


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EyeData:
    """Per-frame eye measurements."""
    ear_left:        float = 0.0
    ear_right:       float = 0.0
    ear_avg:         float = 0.0
    iris_offset_left:  Tuple[float, float] = (0.0, 0.0)   # (dx, dy) normalised
    iris_offset_right: Tuple[float, float] = (0.0, 0.0)
    gaze_features:   Optional[np.ndarray] = None           # for regression model
    face_detected:   bool = False


@dataclass
class BlinkEvent:
    """Represents a single completed blink."""
    timestamp: float   # time.monotonic() when blink ended
    duration:  float   # how long eye was closed (seconds)


@dataclass
class BlinkDetector:
    """Tracks blink state over consecutive frames."""
    _closed:       bool  = False
    _close_start:  float = 0.0
    _history:      list  = field(default_factory=list)   # recent BlinkEvents

    def update(self, ear: float, now: float) -> Optional[BlinkEvent]:
        """Feed per-frame EAR; returns a BlinkEvent when a blink completes."""
        if ear < config.EAR_THRESHOLD:
            if not self._closed:
                self._closed = True
                self._close_start = now
            return None
        else:
            if self._closed:
                self._closed = False
                dur = now - self._close_start
                if config.BLINK_MIN_DURATION <= dur <= config.BLINK_MAX_DURATION:
                    evt = BlinkEvent(timestamp=now, duration=dur)
                    self._history.append(evt)
                    # prune old events
                    cutoff = now - config.DOUBLE_BLINK_WINDOW * 2
                    self._history = [e for e in self._history if e.timestamp > cutoff]
                    return evt
            return None

    def is_double_blink(self) -> bool:
        """True if last two blinks form a valid double-blink."""
        if len(self._history) < 2:
            return False
        b1, b2 = self._history[-2], self._history[-1]
        return (b2.timestamp - b1.timestamp) <= config.DOUBLE_BLINK_WINDOW

    def last_blink_is_click(self) -> bool:
        """True if the most recent blink duration qualifies as an intentional click."""
        if not self._history:
            return False
        dur = self._history[-1].duration
        return config.CLICK_BLINK_MIN <= dur <= config.CLICK_BLINK_MAX


# ──────────────────────────────────────────────────────────────────────────────
# Main Eye Tracker class
# ──────────────────────────────────────────────────────────────────────────────
class EyeTracker:
    """Wraps webcam + MediaPipe FaceLandmarker (Tasks API) to produce EyeData."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_timestamp_ms = 0

        # Build the FaceLandmarker via the Tasks API
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.FACE_LANDMARKER_MODEL),
            running_mode=RunningMode.VIDEO,
            num_faces=config.FACE_MESH_MAX_FACES,
            min_face_detection_confidence=config.FACE_MESH_MIN_DETECT,
            min_face_presence_confidence=config.FACE_MESH_MIN_DETECT,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACK,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self.blink_detector = BlinkDetector()

    @staticmethod
    def feature_size() -> int:
        return _FEATURE_SIZE

    # ── lifecycle ─────────────────────────────────────────────────────────
    def open(self) -> bool:
        """Open the webcam.  Returns True on success."""
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            return True
        return False

    def close(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._landmarker.close()

    # ── per-frame processing ──────────────────────────────────────────────
    def read_frame(self) -> Optional[np.ndarray]:
        """Grab a BGR frame from the webcam (or None)."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def process(self, frame: np.ndarray) -> EyeData:
        """Run FaceLandmarker on *frame* and return EyeData."""
        data = EyeData()
        h, w = frame.shape[:2]

        # Convert BGR → RGB and wrap in MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            return data

        lm = results.face_landmarks[0]   # list of NormalizedLandmark
        data.face_detected = True

        # EAR
        data.ear_left  = self._ear(lm, _LEFT_EYE, w, h)
        data.ear_right = self._ear(lm, _RIGHT_EYE, w, h)
        data.ear_avg   = (data.ear_left + data.ear_right) / 2.0

        # Iris offsets (normalised)
        data.iris_offset_left  = self._iris_offset(lm, _LEFT_IRIS, _LEFT_EYE, w, h)
        data.iris_offset_right = self._iris_offset(lm, _RIGHT_IRIS, _RIGHT_EYE, w, h)

        # Feature vector for gaze model
        data.gaze_features = self._gaze_features(lm, w, h)

        return data

    # ── internal helpers ──────────────────────────────────────────────────
    @staticmethod
    def _landmark_px(lm, idx: int, w: int, h: int) -> np.ndarray:
        """Return (x, y) pixel coords for a single landmark."""
        p = lm[idx]
        return np.array([p.x * w, p.y * h])

    def _ear(self, lm, indices: list, w: int, h: int) -> float:
        """Eye Aspect Ratio for the 6-point eye model (Soukupová & Čech)."""
        pts = [self._landmark_px(lm, i, w, h) for i in indices]
        # vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # horizontal distance
        hz = np.linalg.norm(pts[0] - pts[3])
        if hz == 0:
            return 0.0
        return (v1 + v2) / (2.0 * hz)

    def _iris_offset(
        self, lm, iris_idx: list, eye_idx: list, w: int, h: int
    ) -> Tuple[float, float]:
        """
        Normalised offset of the iris centre relative to the eye bounding-box.
        Returns (dx, dy) each in roughly [−0.5, 0.5].
        """
        iris_c = np.mean([self._landmark_px(lm, i, w, h) for i in iris_idx], axis=0)
        eye_pts = np.array([self._landmark_px(lm, i, w, h) for i in eye_idx])
        eye_min = eye_pts.min(axis=0)
        eye_max = eye_pts.max(axis=0)
        eye_range = eye_max - eye_min
        if eye_range[0] == 0 or eye_range[1] == 0:
            return (0.0, 0.0)
        # normalise to [0, 1], then centre → [−0.5, 0.5]
        norm = (iris_c - eye_min) / eye_range - 0.5
        return (float(norm[0]), float(norm[1]))

    def _gaze_features(self, lm, w: int, h: int) -> np.ndarray:
        """
        Build a feature vector for the gaze regression model.

        Features (24-D):
          – left iris (x, y) normalised to frame
          – right iris (x, y) normalised to frame
          – left iris offset dx, dy
          – right iris offset dx, dy
          – left / right eye-corner x/y normalised  (head-pose proxy)
        """
        li_px = np.mean([self._landmark_px(lm, i, w, h) for i in _LEFT_IRIS], axis=0)
        ri_px = np.mean([self._landmark_px(lm, i, w, h) for i in _RIGHT_IRIS], axis=0)
        left_eye_pts = np.array([self._landmark_px(lm, i, w, h) for i in _LEFT_EYE])
        right_eye_pts = np.array([self._landmark_px(lm, i, w, h) for i in _RIGHT_EYE])
        lo = self._iris_offset(lm, _LEFT_IRIS, _LEFT_EYE, w, h)
        ro = self._iris_offset(lm, _RIGHT_IRIS, _RIGHT_EYE, w, h)

        # outer corners — rough head pose proxy
        left_min = left_eye_pts.min(axis=0)
        left_max = left_eye_pts.max(axis=0)
        right_min = right_eye_pts.min(axis=0)
        right_max = right_eye_pts.max(axis=0)
        left_center = (left_min + left_max) / 2.0
        right_center = (right_min + right_max) / 2.0
        left_size = np.maximum(left_max - left_min, 1.0)
        right_size = np.maximum(right_max - right_min, 1.0)

        face_pts = np.array(
            [self._landmark_px(lm, i, w, h) for i in (10, 152, 234, 454, 1)]
        )
        face_min = face_pts[:4].min(axis=0)
        face_max = face_pts[:4].max(axis=0)
        face_center = (face_min + face_max) / 2.0
        face_size = np.maximum(face_max - face_min, 1.0)
        nose = face_pts[4]

        ear_left = self._ear(lm, _LEFT_EYE, w, h)
        ear_right = self._ear(lm, _RIGHT_EYE, w, h)

        return np.array([
            li_px[0] / w, li_px[1] / h,
            ri_px[0] / w, ri_px[1] / h,
            lo[0], lo[1],
            ro[0], ro[1],
            left_center[0] / w, left_center[1] / h,
            right_center[0] / w, right_center[1] / h,
            left_size[0] / w, left_size[1] / h,
            right_size[0] / w, right_size[1] / h,
            face_center[0] / w, face_center[1] / h,
            face_size[0] / w, face_size[1] / h,
            (nose[0] - face_center[0]) / face_size[0],
            (nose[1] - face_center[1]) / face_size[1],
            ear_left,
            ear_right,
        ], dtype=np.float64)
