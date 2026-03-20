from __future__ import annotations

import math

import cv2
import numpy as np

from eyetest.models.types import BatchFrameOverlay, Ellipse2D, GazeEstimate


STATUS_BAR_HEIGHT = 36
INSET_MARGIN = 10
INSET_MIN_WIDTH = 96
INSET_MAX_WIDTH = 160


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


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def draw_ellipse(frame: np.ndarray, ellipse: Ellipse2D | None, color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    if ellipse is None or not ellipse.valid:
        return frame
    center = (int(round(ellipse.center_x)), int(round(ellipse.center_y)))
    axes = (max(int(round(ellipse.major)), 1), max(int(round(ellipse.minor)), 1))
    angle_deg = math.degrees(ellipse.angle)
    cv2.ellipse(frame, center, axes, angle_deg, 0.0, 360.0, color, thickness)
    return frame


def draw_eye_overlay(
    frame: np.ndarray,
    iris: Ellipse2D,
    pupil: Ellipse2D | None,
    label: str,
    label_color: tuple[int, int, int],
) -> np.ndarray:
    canvas = ensure_bgr(frame)
    draw_ellipse(canvas, iris, (0, 255, 0), thickness=2)
    draw_ellipse(canvas, pupil, (0, 215, 255), thickness=2)
    cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)
    return canvas


def resize_panel(frame: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    if target_size is None:
        return frame
    width_px, height_px = target_size
    if frame.shape[1] == width_px and frame.shape[0] == height_px:
        return frame
    return cv2.resize(frame, (width_px, height_px), interpolation=cv2.INTER_LINEAR)


def draw_gaze_inset(
    width_px: int,
    height_px: int,
    result: GazeEstimate,
    screen_width_px: int,
    screen_height_px: int,
) -> np.ndarray:
    inset = blank_screen(width_px, height_px)
    cv2.rectangle(inset, (0, 0), (width_px - 1, height_px - 1), (255, 255, 255), 1)
    cv2.putText(inset, "FUSED", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    point = result.fused_gaze_point_px
    if point is not None and screen_width_px > 0 and screen_height_px > 0:
        x = int(round(np.clip(point[0] / screen_width_px, 0.0, 1.0) * (width_px - 1)))
        y = int(round(np.clip(point[1] / screen_height_px, 0.0, 1.0) * (height_px - 1)))
        cv2.circle(inset, (x, y), 5, (0, 255, 0), -1)
    return inset


def _format_point(point: tuple[float, float] | None) -> str:
    if point is None:
        return "None"
    return f"({int(round(point[0]))},{int(round(point[1]))})"


def draw_status_bar(width_px: int, result: GazeEstimate) -> np.ndarray:
    bar = np.full((STATUS_BAR_HEIGHT, width_px, 3), 18, dtype=np.uint8)
    status = (
        f"frame={result.frame_index}  "
        f"valid={str(result.valid).lower()}  "
        f"L={_format_point(result.left_gaze_point_px)}  "
        f"R={_format_point(result.right_gaze_point_px)}  "
        f"F={_format_point(result.fused_gaze_point_px)}"
    )
    if result.error_message:
        status = f"{status}  error={result.error_message}"
    cv2.putText(bar, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def compose_side_by_side_overlay(
    frame_overlay: BatchFrameOverlay,
    screen_width_px: int,
    screen_height_px: int,
    left_panel_size: tuple[int, int] | None = None,
    right_panel_size: tuple[int, int] | None = None,
) -> np.ndarray:
    left_panel = draw_eye_overlay(
        frame_overlay.left_frame_bgr,
        frame_overlay.left_iris,
        frame_overlay.left_pupil,
        label="LEFT",
        label_color=(114, 227, 255),
    )
    left_panel = resize_panel(left_panel, left_panel_size)
    right_panel = draw_eye_overlay(
        frame_overlay.right_frame_bgr,
        frame_overlay.right_iris,
        frame_overlay.right_pupil,
        label="RIGHT",
        label_color=(255, 155, 210),
    )
    right_panel = resize_panel(right_panel, right_panel_size)

    content_height = max(left_panel.shape[0], right_panel.shape[0])
    content_width = left_panel.shape[1] + right_panel.shape[1]
    canvas = blank_screen(content_width, content_height + STATUS_BAR_HEIGHT)
    canvas[: left_panel.shape[0], : left_panel.shape[1]] = left_panel
    canvas[: right_panel.shape[0], left_panel.shape[1] : content_width] = right_panel

    inset_width = min(max(content_width // 3, INSET_MIN_WIDTH), INSET_MAX_WIDTH)
    inset_height = min(
        max(int(round(inset_width * max(screen_height_px, 1) / max(screen_width_px, 1))), 72),
        max(content_height - 2 * INSET_MARGIN, 72),
    )
    inset_height = min(inset_height, max(content_height - 2 * INSET_MARGIN, 1))
    inset = draw_gaze_inset(
        inset_width,
        inset_height,
        frame_overlay.gaze,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
    )
    x0 = content_width - inset_width - INSET_MARGIN
    y0 = INSET_MARGIN
    canvas[y0 : y0 + inset_height, x0 : x0 + inset_width] = inset
    canvas[content_height:, :] = draw_status_bar(content_width, frame_overlay.gaze)
    return canvas
