from __future__ import annotations

import cv2

from eyetest.config import AppConfig, CalibrationConfig
from eyetest.gaze.estimator import estimate_frame_gaze
from eyetest.inputs.camera import EyePairCamera, EyePairExtractor, TrackedEyePairExtractor
from eyetest.models.types import CalibrationFrame, GazeEstimate
from eyetest.outputs.overlay import blank_screen, draw_eye_boxes, draw_gaze_points
from eyetest.segmentation.ellseg_pipeline import EllSegSegmenter
from eyetest.adapters.ellipse_adapter import adapt_ellseg_ellipse_pair


def run_realtime_pipeline(
    app_config: AppConfig,
    calibration: CalibrationConfig,
    camera_index: int = 0,
    eval_on_cpu: bool = True,
) -> None:
    camera = EyePairCamera(
        camera_index=camera_index,
        extractor=TrackedEyePairExtractor(detector=EyePairExtractor()),
    )
    left_segmenter = EllSegSegmenter(app_config.segmentation, eval_on_cpu=eval_on_cpu)
    right_segmenter = EllSegSegmenter(app_config.segmentation, eval_on_cpu=eval_on_cpu)
    frame_meta = CalibrationFrame(
        width_px=calibration.screen_width_px,
        height_px=calibration.screen_height_px,
    )

    try:
        frame_index = 0
        while True:
            eye_pair = camera.read()
            if eye_pair is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if eye_pair.valid and eye_pair.left_eye_bgr is not None and eye_pair.right_eye_bgr is not None:
                left_output = left_segmenter.segment(eye_pair.left_eye_bgr)
                right_output = right_segmenter.segment(eye_pair.right_eye_bgr)
                left_ellipse, right_ellipse = adapt_ellseg_ellipse_pair(
                    left_output["iris_ellipse"],
                    right_output["iris_ellipse"],
                )
                result = estimate_frame_gaze(
                    frame_index=frame_index,
                    left_ellipse=left_ellipse,
                    right_ellipse=right_ellipse,
                    calibration=calibration,
                    frame=frame_meta,
                )
            else:
                result = GazeEstimate(
                    frame_index=frame_index,
                    valid=False,
                    error_message=eye_pair.error_message or "Eye-pair detection failed.",
                )

            frame_vis = draw_eye_boxes(eye_pair.frame_bgr.copy(), eye_pair.boxes)
            screen_vis = blank_screen(calibration.screen_width_px, calibration.screen_height_px)
            screen_vis = draw_gaze_points(screen_vis, result)

            cv2.imshow("eyetest-camera", frame_vis)
            cv2.imshow("eyetest-gaze", screen_vis)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
