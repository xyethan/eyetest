from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eyetest.config import CalibrationConfig


@dataclass(frozen=True)
class ResolvedCalibration:
    screen_corners: np.ndarray
    screen_normal: np.ndarray
    screen_width_mm: float
    screen_height_mm: float


def resolve_calibration(calibration: CalibrationConfig) -> ResolvedCalibration:
    screen_corners = np.array(calibration.screen_corners, dtype=np.float64)
    screen_normal = np.cross(
        screen_corners[1] - screen_corners[0],
        screen_corners[2] - screen_corners[1],
    )
    screen_normal = screen_normal / np.linalg.norm(screen_normal)
    if screen_normal[2] < 0:
        screen_normal = -screen_normal
    return ResolvedCalibration(
        screen_corners=screen_corners,
        screen_normal=screen_normal,
        screen_width_mm=float(np.linalg.norm(screen_corners[1] - screen_corners[0])),
        screen_height_mm=float(np.linalg.norm(screen_corners[2] - screen_corners[1])),
    )
