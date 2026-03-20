from __future__ import annotations

import cv2
import numpy as np

from eyetest.models.types import GazeEstimate


def draw_gaze_points(canvas: np.ndarray, result: GazeEstimate) -> np.ndarray:
    if result.left_gaze_point_px is not None:
        cv2.circle(canvas, (int(result.left_gaze_point_px[0]), int(result.left_gaze_point_px[1])), 6, (255, 0, 0), -1)
    if result.right_gaze_point_px is not None:
        cv2.circle(canvas, (int(result.right_gaze_point_px[0]), int(result.right_gaze_point_px[1])), 6, (0, 0, 255), -1)
    if result.fused_gaze_point_px is not None:
        cv2.circle(canvas, (int(result.fused_gaze_point_px[0]), int(result.fused_gaze_point_px[1])), 6, (0, 255, 0), -1)
    return canvas


def draw_eye_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def blank_screen(width_px: int, height_px: int) -> np.ndarray:
    return np.zeros((height_px, width_px, 3), dtype=np.uint8)
