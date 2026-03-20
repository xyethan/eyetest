from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2

from eyetest.models.types import BatchFrameOverlay, GazeEstimate
from eyetest.outputs.overlay import blank_screen, compose_side_by_side_overlay, draw_gaze_points

SIDE_BY_SIDE_PANEL_SIZE = (240, 240)


def _as_record(result: GazeEstimate) -> dict[str, object]:
    return {
        "frame_index": result.frame_index,
        "valid": result.valid,
        "left_gaze_point_px": list(result.left_gaze_point_px) if result.left_gaze_point_px else None,
        "right_gaze_point_px": list(result.right_gaze_point_px) if result.right_gaze_point_px else None,
        "fused_gaze_point_px": list(result.fused_gaze_point_px) if result.fused_gaze_point_px else None,
        "error_message": result.error_message,
    }


def write_gaze_results_json(path: str | Path, results: list[GazeEstimate]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_as_record(result) for result in results]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_gaze_overlay_video(
    path: str | Path,
    results: Sequence[GazeEstimate],
    width_px: int,
    height_px: int,
    fps: float = 25.0,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0.0 else 25.0,
        (width_px, height_px),
    )
    if not writer.isOpened():
        raise ValueError(f"Unable to open overlay video writer: {output_path}")
    try:
        frames = results if results else [GazeEstimate(frame_index=0, valid=False)]
        for result in frames:
            canvas = blank_screen(width_px, height_px)
            writer.write(draw_gaze_points(canvas, result))
    finally:
        writer.release()


def write_side_by_side_overlay_video(
    path: str | Path,
    frames: Sequence[BatchFrameOverlay],
    screen_width_px: int,
    screen_height_px: int,
    fps: float = 25.0,
) -> None:
    if not frames:
        raise ValueError("At least one frame is required to write the side-by-side overlay video.")

    left_panel_size = SIDE_BY_SIDE_PANEL_SIZE
    right_panel_size = SIDE_BY_SIDE_PANEL_SIZE
    first_frame = compose_side_by_side_overlay(
        frames[0],
        screen_width_px,
        screen_height_px,
        left_panel_size=left_panel_size,
        right_panel_size=right_panel_size,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0.0 else 25.0,
        (first_frame.shape[1], first_frame.shape[0]),
    )
    if not writer.isOpened():
        raise ValueError(f"Unable to open side-by-side overlay video writer: {output_path}")
    try:
        writer.write(first_frame)
        for frame_overlay in frames[1:]:
            writer.write(
                compose_side_by_side_overlay(
                    frame_overlay,
                    screen_width_px,
                    screen_height_px,
                    left_panel_size=left_panel_size,
                    right_panel_size=right_panel_size,
                )
            )
    finally:
        writer.release()
