from __future__ import annotations

import math

import numpy as np

from eyetest.config import CalibrationConfig
from eyetest.gaze.calibration import resolve_calibration
from eyetest.gaze.geometry import (
    esti_normal_fun,
    get_rotation,
    get_vector_onto_plane,
    line_plane_intersection,
    trans_camera_to_screen,
    vector_norm,
)
from eyetest.models.types import CalibrationFrame, Ellipse2D, GazeEstimate


def _ellipse_to_center_normal(
    ellipse: Ellipse2D,
    calibration: CalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    xe = (ellipse.center_x - calibration.camera.principal_point_x) * calibration.camera.pixel_size_x
    ye = (ellipse.center_y - calibration.camera.principal_point_y) * calibration.camera.pixel_size_y
    amajor = ellipse.major / 2.0 * calibration.camera.pixel_size_x
    aminor = ellipse.minor / 2.0 * calibration.camera.pixel_size_y
    theta_deg = math.degrees(ellipse.angle)
    iris_1, iris_2, normal_1, normal_2, _ = esti_normal_fun(
        amajor,
        aminor,
        xe,
        ye,
        theta_deg,
        calibration.camera.focal_length,
    )
    if normal_1[1] * normal_1[2] < 0 and normal_2[1] * normal_2[2] > 0:
        return iris_1, normal_1
    return iris_2, normal_2


def _get_los(
    optical_axis: np.ndarray,
    kappa_alpha: float,
    kappa_beta: float,
    left_iris_center: np.ndarray,
    right_iris_center: np.ndarray,
) -> np.ndarray:
    z_axis = optical_axis / vector_norm(optical_axis)
    x_axis = get_vector_onto_plane(right_iris_center - left_iris_center, z_axis)
    x_axis = x_axis / vector_norm(x_axis)
    if x_axis[0] < 0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    rotation_world = get_rotation(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        x_axis,
        y_axis,
        z_axis,
    )
    rotation_local = np.array(
        [
            [math.cos(kappa_beta), 0.0, math.sin(kappa_beta)],
            [
                math.sin(kappa_alpha) * math.sin(kappa_beta),
                math.cos(kappa_alpha),
                -math.sin(kappa_alpha) * math.cos(kappa_beta),
            ],
            [
                -math.cos(kappa_alpha) * math.sin(kappa_beta),
                math.sin(kappa_alpha),
                math.cos(kappa_alpha) * math.cos(kappa_beta),
            ],
        ]
    )
    visual_axis = rotation_local @ np.array([0.0, 0.0, 1.0])
    visual_axis = rotation_world @ visual_axis
    return visual_axis / vector_norm(visual_axis)


def _to_screen_pixels(
    point_camera: np.ndarray,
    resolved,
    frame: CalibrationFrame,
) -> tuple[float, float]:
    rotation, translation = trans_camera_to_screen(resolved.screen_corners)
    point_screen = point_camera @ rotation.T + translation.reshape(1, 3)
    x_px = float(point_screen[0][0] / resolved.screen_width_mm * frame.width_px)
    y_px = float(point_screen[0][1] / resolved.screen_height_mm * frame.height_px)
    return x_px, y_px


def estimate_frame_gaze(
    frame_index: int,
    left_ellipse: Ellipse2D,
    right_ellipse: Ellipse2D,
    calibration: CalibrationConfig,
    frame: CalibrationFrame,
) -> GazeEstimate:
    if not left_ellipse.valid or not right_ellipse.valid:
        return GazeEstimate(
            frame_index=frame_index,
            valid=False,
            error_message="Both left and right iris ellipses must be valid.",
        )

    try:
        resolved = resolve_calibration(calibration)
        left_center_unit, left_normal = _ellipse_to_center_normal(left_ellipse, calibration)
        right_center_unit, right_normal = _ellipse_to_center_normal(right_ellipse, calibration)

        left_center = calibration.left_eye.radius * left_center_unit
        right_center = calibration.right_eye.radius * right_center_unit

        left_los = _get_los(
            left_normal,
            calibration.left_eye.kappa_alpha,
            calibration.left_eye.kappa_beta,
            left_center,
            right_center,
        )
        right_los = _get_los(
            right_normal,
            calibration.right_eye.kappa_alpha,
            calibration.right_eye.kappa_beta,
            left_center,
            right_center,
        )

        left_point = line_plane_intersection(
            left_center,
            left_los,
            resolved.screen_corners[0],
            resolved.screen_normal,
        )
        right_point = line_plane_intersection(
            right_center,
            right_los,
            resolved.screen_corners[0],
            resolved.screen_normal,
        )

        left_px = _to_screen_pixels(left_point.reshape(1, 3), resolved, frame)
        right_px = _to_screen_pixels(right_point.reshape(1, 3), resolved, frame)
        fused_px = ((left_px[0] + right_px[0]) / 2.0, (left_px[1] + right_px[1]) / 2.0)
        return GazeEstimate(
            frame_index=frame_index,
            valid=True,
            left_gaze_point_px=left_px,
            right_gaze_point_px=right_px,
            fused_gaze_point_px=fused_px,
        )
    except Exception as exc:
        return GazeEstimate(
            frame_index=frame_index,
            valid=False,
            error_message=str(exc),
        )
