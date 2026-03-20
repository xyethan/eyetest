from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eyetest.adapters.ellipse_adapter import adapt_ellseg_ellipse_pair
from eyetest.config import AppConfig, CalibrationConfig
from eyetest.gaze.estimator import estimate_frame_gaze
from eyetest.inputs.camera import EyePairExtractor, EyePairFrame, TrackedEyePairExtractor
from eyetest.inputs.video import read_video_fps, read_video_frames
from eyetest.models.types import BatchFrameOverlay, CalibrationFrame, Ellipse2D, GazeEstimate
from eyetest.outputs.writers import (
    write_gaze_overlay_video,
    write_gaze_results_json,
    write_side_by_side_overlay_video,
)
from eyetest.segmentation.ellseg_pipeline import EllSegSegmenter


def run_batch_pipeline(
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    left_segmenter,
    right_segmenter,
    calibration: CalibrationConfig,
) -> list[GazeEstimate]:
    return [frame.gaze for frame in run_batch_pipeline_with_details(left_frames, right_frames, left_segmenter, right_segmenter, calibration)]


def run_batch_pipeline_with_details(
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    left_segmenter,
    right_segmenter,
    calibration: CalibrationConfig,
) -> list[BatchFrameOverlay]:
    if len(left_frames) != len(right_frames):
        raise ValueError("Left and right frame sequences must have the same length.")

    frames: list[BatchFrameOverlay] = []
    for index, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames, strict=True)):
        left_output = left_segmenter.segment(left_frame)
        right_output = right_segmenter.segment(right_frame)
        left_ellipse, right_ellipse = adapt_ellseg_ellipse_pair(
            left_output["iris_ellipse"],
            right_output["iris_ellipse"],
        )
        left_pupil, right_pupil = adapt_ellseg_ellipse_pair(
            left_output["pupil_ellipse"],
            right_output["pupil_ellipse"],
        )
        frame_meta = CalibrationFrame(
            width_px=calibration.screen_width_px,
            height_px=calibration.screen_height_px,
        )
        gaze = estimate_frame_gaze(
            frame_index=index,
            left_ellipse=left_ellipse,
            right_ellipse=right_ellipse,
            calibration=calibration,
            frame=frame_meta,
        )
        frames.append(
            BatchFrameOverlay(
                frame_index=index,
                left_frame_bgr=left_frame,
                right_frame_bgr=right_frame,
                left_iris=left_ellipse,
                right_iris=right_ellipse,
                left_pupil=left_pupil if left_pupil.valid else None,
                right_pupil=right_pupil if right_pupil.valid else None,
                gaze=gaze,
            )
        )
    return frames


def _placeholder_eye_frame(face_frame: np.ndarray) -> np.ndarray:
    height_px = max(face_frame.shape[0] // 4, 80)
    width_px = max(face_frame.shape[1] // 4, 120)
    return np.zeros((height_px, width_px, 3), dtype=np.uint8)


def _invalid_overlay_from_face_frame(frame_index: int, face_frame: np.ndarray, error_message: str) -> BatchFrameOverlay:
    placeholder = _placeholder_eye_frame(face_frame)
    return BatchFrameOverlay(
        frame_index=frame_index,
        left_frame_bgr=placeholder.copy(),
        right_frame_bgr=placeholder.copy(),
        left_iris=Ellipse2D.invalid(),
        right_iris=Ellipse2D.invalid(),
        left_pupil=None,
        right_pupil=None,
        gaze=GazeEstimate(
            frame_index=frame_index,
            valid=False,
            error_message=error_message,
        ),
    )


def _overlay_from_eye_pair(
    frame_index: int,
    eye_pair: EyePairFrame,
    left_segmenter,
    right_segmenter,
    calibration: CalibrationConfig,
) -> BatchFrameOverlay:
    if (
        not eye_pair.valid
        or eye_pair.left_eye_bgr is None
        or eye_pair.right_eye_bgr is None
    ):
        return _invalid_overlay_from_face_frame(
            frame_index=frame_index,
            face_frame=eye_pair.frame_bgr,
            error_message=eye_pair.error_message or "Eye-pair detection failed.",
        )

    left_output = left_segmenter.segment(eye_pair.left_eye_bgr)
    right_output = right_segmenter.segment(eye_pair.right_eye_bgr)
    left_ellipse, right_ellipse = adapt_ellseg_ellipse_pair(
        left_output["iris_ellipse"],
        right_output["iris_ellipse"],
    )
    left_pupil, right_pupil = adapt_ellseg_ellipse_pair(
        left_output["pupil_ellipse"],
        right_output["pupil_ellipse"],
    )
    frame_meta = CalibrationFrame(
        width_px=calibration.screen_width_px,
        height_px=calibration.screen_height_px,
    )
    gaze = estimate_frame_gaze(
        frame_index=frame_index,
        left_ellipse=left_ellipse,
        right_ellipse=right_ellipse,
        calibration=calibration,
        frame=frame_meta,
    )
    return BatchFrameOverlay(
        frame_index=frame_index,
        left_frame_bgr=eye_pair.left_eye_bgr,
        right_frame_bgr=eye_pair.right_eye_bgr,
        left_iris=left_ellipse,
        right_iris=right_ellipse,
        left_pupil=left_pupil if left_pupil.valid else None,
        right_pupil=right_pupil if right_pupil.valid else None,
        gaze=gaze,
    )


def run_batch_pipeline_from_face_frames_with_details(
    face_frames: Sequence[np.ndarray],
    eye_pair_extractor,
    left_segmenter,
    right_segmenter,
    calibration: CalibrationConfig,
) -> list[BatchFrameOverlay]:
    frames: list[BatchFrameOverlay] = []
    for index, face_frame in enumerate(face_frames):
        eye_pair = eye_pair_extractor.extract(face_frame)
        frames.append(
            _overlay_from_eye_pair(
                frame_index=index,
                eye_pair=eye_pair,
                left_segmenter=left_segmenter,
                right_segmenter=right_segmenter,
                calibration=calibration,
            )
        )
    return frames


def run_batch_pipeline_from_face_frames(
    face_frames: Sequence[np.ndarray],
    eye_pair_extractor,
    left_segmenter,
    right_segmenter,
    calibration: CalibrationConfig,
) -> list[GazeEstimate]:
    return [
        frame.gaze
        for frame in run_batch_pipeline_from_face_frames_with_details(
            face_frames=face_frames,
            eye_pair_extractor=eye_pair_extractor,
            left_segmenter=left_segmenter,
            right_segmenter=right_segmenter,
            calibration=calibration,
        )
    ]


def run_batch_from_face_video(
    face_video_path: str,
    app_config: AppConfig,
    calibration: CalibrationConfig,
    output_json_path: str | None = None,
    output_video_path: str | None = None,
    output_overlay_video_path: str | None = None,
    eval_on_cpu: bool = True,
) -> list[GazeEstimate]:
    face_frames = read_video_frames(face_video_path, grayscale=False)
    fps = read_video_fps(face_video_path)
    eye_pair_extractor = TrackedEyePairExtractor(detector=EyePairExtractor())
    left_segmenter = EllSegSegmenter(app_config.segmentation, eval_on_cpu=eval_on_cpu)
    right_segmenter = EllSegSegmenter(app_config.segmentation, eval_on_cpu=eval_on_cpu)
    frame_overlays = run_batch_pipeline_from_face_frames_with_details(
        face_frames=face_frames,
        eye_pair_extractor=eye_pair_extractor,
        left_segmenter=left_segmenter,
        right_segmenter=right_segmenter,
        calibration=calibration,
    )
    results = [frame.gaze for frame in frame_overlays]
    if output_json_path:
        write_gaze_results_json(output_json_path, results)
    if output_video_path:
        write_gaze_overlay_video(
            path=output_video_path,
            results=results,
            width_px=calibration.screen_width_px,
            height_px=calibration.screen_height_px,
            fps=fps,
        )
    if output_overlay_video_path:
        write_side_by_side_overlay_video(
            path=output_overlay_video_path,
            frames=frame_overlays,
            screen_width_px=calibration.screen_width_px,
            screen_height_px=calibration.screen_height_px,
            fps=fps,
        )
    return results
