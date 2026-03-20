from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ellipse2D:
    center_x: float
    center_y: float
    major: float
    minor: float
    angle: float
    valid: bool = True

    @classmethod
    def invalid(cls) -> "Ellipse2D":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, valid=False)


@dataclass(frozen=True)
class FrameSegmentation:
    frame_index: int
    left_iris: Ellipse2D
    right_iris: Ellipse2D
    left_pupil: Ellipse2D | None = None
    right_pupil: Ellipse2D | None = None
    timestamp_ms: float | None = None


@dataclass(frozen=True)
class GazeEstimate:
    frame_index: int
    valid: bool
    left_gaze_point_px: tuple[float, float] | None = None
    right_gaze_point_px: tuple[float, float] | None = None
    fused_gaze_point_px: tuple[float, float] | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class CalibrationFrame:
    width_px: int
    height_px: int


@dataclass(frozen=True)
class BatchFrameOverlay:
    frame_index: int
    left_frame_bgr: np.ndarray
    right_frame_bgr: np.ndarray
    left_iris: Ellipse2D
    right_iris: Ellipse2D
    gaze: GazeEstimate
    left_pupil: Ellipse2D | None = None
    right_pupil: Ellipse2D | None = None
