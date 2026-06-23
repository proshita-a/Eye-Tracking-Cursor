"""
gaze_model.py — Calibration, training, prediction for gaze → screen mapping.

Uses scikit-learn Ridge regression trained on per-user iris features
collected during a 9-point calibration sequence.  The trained model is
persisted to disk via pickle so it survives across sessions.
"""

from __future__ import annotations

import os
import pickle
import time
import tkinter as tk
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from eye_tracker import EyeTracker

_MODEL_VERSION = 2
_MIN_VALID_POINTS = 5
_MIN_TOTAL_SAMPLES = 40
_MIN_POINT_SAMPLES = 6


class GazeModel:
    """Train & use a Ridge regression to map iris features → screen (x, y)."""

    def __init__(self) -> None:
        self._model_x: Optional[object] = None
        self._model_y: Optional[object] = None

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str = config.MODEL_SAVE_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "version": _MODEL_VERSION,
                    "feature_size": EyeTracker.feature_size(),
                    "x": self._model_x,
                    "y": self._model_y,
                },
                f,
            )

    def load(self, path: str = config.MODEL_SAVE_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False

        model_x = data.get("x")
        model_y = data.get("y")
        saved_feature_size = data.get("feature_size")
        if saved_feature_size is None and model_x is not None:
            saved_feature_size = getattr(model_x, "n_features_in_", None)

        if (
            model_x is None
            or model_y is None
            or saved_feature_size != EyeTracker.feature_size()
        ):
            self._model_x = None
            self._model_y = None
            return False

        self._model_x = model_x
        self._model_y = model_y
        return True

    @property
    def is_ready(self) -> bool:
        return self._model_x is not None and self._model_y is not None

    # ── prediction ────────────────────────────────────────────────────────
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Return predicted (screen_x, screen_y) for a feature vector."""
        if not self.is_ready:
            return (config.SCREEN_W / 2, config.SCREEN_H / 2)
        if features.size != EyeTracker.feature_size():
            return (config.SCREEN_W / 2, config.SCREEN_H / 2)
        f = features.reshape(1, -1)
        sx = float(self._model_x.predict(f)[0])
        sy = float(self._model_y.predict(f)[0])
        # clamp to screen
        sx = max(0, min(config.SCREEN_W - 1, sx))
        sy = max(0, min(config.SCREEN_H - 1, sy))
        return (sx, sy)

    # ── training ──────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, screen_coords: np.ndarray) -> None:
        """
        Train on collected calibration data.
        X:             (N, feature_dim)
        screen_coords: (N, 2)  — columns are (screen_x, screen_y)
        """
        if X.ndim != 2 or X.shape[1] != EyeTracker.feature_size():
            raise ValueError("Unexpected gaze feature shape")

        self._model_x = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        self._model_y = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        self._model_x.fit(X, screen_coords[:, 0])
        self._model_y.fit(X, screen_coords[:, 1])

    # ── calibration routine ───────────────────────────────────────────────
    def calibrate(self, tracker: EyeTracker) -> bool:
        """
        Run the full interactive 9-point calibration.

        Opens a fullscreen tkinter window, shows dots one at a time,
        collects iris features while the user fixates each dot, then
        trains the model and saves it.

        Returns True on success, False on cancel / failure.
        """
        points = self._calibration_points()
        all_features: List[np.ndarray] = []
        all_targets: List[Tuple[float, float]] = []
        valid_points = 0

        # ── fullscreen calibration window ─────────────────────────────────
        root = tk.Tk()
        root.title("Gaze Calibration")
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.configure(bg="black")
        root.update()

        canvas = tk.Canvas(root, bg="black", highlightthickness=0,
                           width=config.SCREEN_W, height=config.SCREEN_H)
        canvas.pack(fill="both", expand=True)

        label = canvas.create_text(
            config.SCREEN_W // 2, 40,
            text="Look at each dot until it turns green.",
            fill="#00ffaa", font=("Consolas", 16, "bold"),
        )

        cancelled = False

        def _on_escape(_event):
            nonlocal cancelled
            cancelled = True
            root.destroy()

        root.bind("<Escape>", _on_escape)

        for idx, (px, py) in enumerate(points):
            if cancelled:
                break

            # draw dot
            canvas.delete("dot")
            r = config.CALIBRATION_DOT_RADIUS
            canvas.create_oval(px - r, py - r, px + r, py + r,
                               fill="#ff5555", outline="", tags="dot")
            canvas.itemconfig(label,
                              text=f"Point {idx + 1}/{len(points)} — look at the red dot")
            root.update()
            time.sleep(0.5)  # brief settle time

            # collect features for CALIBRATION_HOLD_SECS
            start = time.monotonic()
            point_features: List[np.ndarray] = []
            while time.monotonic() - start < config.CALIBRATION_HOLD_SECS:
                frame = tracker.read_frame()
                if frame is None:
                    continue
                eye_data = tracker.process(frame)
                if self._is_valid_sample(eye_data):
                    point_features.append(eye_data.gaze_features)
                    canvas.itemconfig(
                        label,
                        text=(
                            f"Point {idx + 1}/{len(points)} - "
                            f"hold steady ({len(point_features)} samples)"
                        ),
                    )
                root.update()
                cv2.waitKey(1)

            if len(point_features) >= _MIN_POINT_SAMPLES:
                filtered_features = self._filter_point_features(point_features)
                for feat in filtered_features:
                    all_features.append(feat)
                    all_targets.append((px, py))
                valid_points += 1

            # mark dot green to indicate done
            canvas.delete("dot")
            canvas.create_oval(px - r, py - r, px + r, py + r,
                               fill="#00ff88", outline="", tags="dot")
            root.update()
            time.sleep(0.3)

        root.destroy()

        if (
            cancelled
            or valid_points < _MIN_VALID_POINTS
            or len(all_features) < _MIN_TOTAL_SAMPLES
        ):
            return False

        X = np.array(all_features)
        Y = np.array(all_targets)
        self.train(X, Y)
        self.save()
        return True

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _is_valid_sample(eye_data) -> bool:
        if not eye_data.face_detected or eye_data.gaze_features is None:
            return False
        if eye_data.gaze_features.size != EyeTracker.feature_size():
            return False
        if not np.all(np.isfinite(eye_data.gaze_features)):
            return False
        if eye_data.ear_avg <= config.EAR_THRESHOLD + 0.04:
            return False

        offsets = (
            *eye_data.iris_offset_left,
            *eye_data.iris_offset_right,
        )
        return max(abs(v) for v in offsets) <= 0.55

    @staticmethod
    def _filter_point_features(point_features: List[np.ndarray]) -> np.ndarray:
        arr = np.asarray(point_features, dtype=np.float64)
        if len(arr) < _MIN_POINT_SAMPLES * 2:
            return arr

        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        robust_scale = 1.4826 * mad + 1e-6
        scores = np.mean(np.minimum(np.abs((arr - med) / robust_scale), 6.0), axis=1)
        cutoff = np.percentile(scores, 85)
        filtered = arr[scores <= cutoff]
        return filtered if len(filtered) >= _MIN_POINT_SAMPLES else arr

    @staticmethod
    def _calibration_points() -> List[Tuple[int, int]]:
        """Return the 9 calibration screen positions (3×3 grid)."""
        margin = 80
        w, h = config.SCREEN_W, config.SCREEN_H
        xs = [margin, w // 2, w - margin]
        ys = [margin, h // 2, h - margin]
        return [(x, y) for y in ys for x in xs]
