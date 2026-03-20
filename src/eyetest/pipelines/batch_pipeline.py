from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eyetest.adapters.ellipse_adapter import adapt_ellseg_ellipse_pair
from eyetest.config import AppConfig, CalibrationConfig
from eyetest.gaze.estimator import estimate_frame_gaze
from eyetest.inputs.video import read_video_frames
from eyetest.models.types import CalibrationFrame, GazeEstimate
from eyetest.outputs.writers import write_gaze_results_json
from eyetest.segmentation.ellseg_pipeline import EllSegSegmenter


def run_batch_pipeline(
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    left_segmenter,
    right_segmenter,
    calibration: CalibrationConfig,
) -> list[GazeEstimate]:
    if len(left_frames) != len(right_frames):
        raise ValueError("Left and right frame sequences must have the same length.")

    results: list[GazeEstimate] = []
    for index, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames, strict=True)):
        left_output = left_segmenter.segment(left_frame)
        right_output = right_segmenter.segment(right_frame)
        left_ellipse, right_ellipse = adapt_ellseg_ellipse_pair(
            left_output["iris_ellipse"],
            right_output["iris_ellipse"],
        )
        frame_meta = CalibrationFrame(
            width_px=calibration.screen_width_px,
            height_px=calibration.screen_height_px,
        )
        results.append(
            estimate_frame_gaze(
                frame_index=index,
                left_ellipse=left_ellipse,
                right_ellipse=right_ellipse,
                calibration=calibration,
                frame=frame_meta,
            )
        )
    return results


def run_batch_from_eye_videos(
    left_video_path: str,
    right_video_path: str,
    app_config: AppConfig,
    calibration: CalibrationConfig,
    output_json_path: str | None = None,
    eval_on_cpu: bool = True,
) -> list[GazeEstimate]:
    left_frames = read_video_frames(left_video_path, grayscale=True)
    right_frames = read_video_frames(right_video_path, grayscale=True)
    left_segmenter = EllSegSegmenter(app_config.segmentation, eval_on_cpu=eval_on_cpu)
    right_segmenter = EllSegSegmenter(app_config.segmentation, eval_on_cpu=eval_on_cpu)
    results = run_batch_pipeline(
        left_frames=left_frames,
        right_frames=right_frames,
        left_segmenter=left_segmenter,
        right_segmenter=right_segmenter,
        calibration=calibration,
    )
    if output_json_path:
        write_gaze_results_json(output_json_path, results)
    return results
