from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.config import load_calibration_config  # noqa: E402
from eyetest.inputs.camera import EyePairFrame  # noqa: E402
from eyetest.pipelines.batch_pipeline import (  # noqa: E402
    run_batch_pipeline,
    run_batch_pipeline_from_face_frames_with_details,
)


class _FakeSegmenter:
    def __init__(self, outputs: list[list[float]]) -> None:
        self._outputs = outputs
        self._index = 0

    def segment(self, _frame: np.ndarray) -> dict[str, object]:
        ellipse = self._outputs[self._index]
        self._index += 1
        return {
            "seg_map": None,
            "pupil_ellipse": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "iris_ellipse": ellipse,
        }


class _FakeEyePairExtractor:
    def __init__(self, outputs: list[EyePairFrame]) -> None:
        self._outputs = outputs
        self._index = 0

    def extract(self, _frame: np.ndarray) -> EyePairFrame:
        output = self._outputs[self._index]
        self._index += 1
        return output


class BatchPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = load_calibration_config(
            PROJECT_ROOT / "configs" / "calibration.default.yaml"
        )
        self.left_frames = [np.zeros((1080, 1920), dtype=np.uint8) for _ in range(2)]
        self.right_frames = [np.zeros((1080, 1920), dtype=np.uint8) for _ in range(2)]
        self.left_segmenter = _FakeSegmenter(
            [
                [596.1666083455085, 1117.006573232015, 110.72297024601819, 116.52391586776352, math.radians(-4.434906005859375)],
                [598.523441529274, 1119.8511252085368, 115.66845185255353, 115.29578997570856, math.radians(7.288200378417969)],
            ]
        )
        self.right_segmenter = _FakeSegmenter(
            [
                [1298.118955039978, 1080.9469323794046, 146.45118647447984, 111.27496131621524, math.radians(81.47233581542969)],
                [1301.1791583895683, 1083.3823882102965, 112.66681525873844, 131.09218762890433, math.radians(26.97820281982422)],
            ]
        )
        self.face_frames = [np.zeros((360, 480, 3), dtype=np.uint8) for _ in range(3)]

    def test_runs_batch_pipeline_over_paired_eye_frames(self) -> None:
        results = run_batch_pipeline(
            left_frames=self.left_frames,
            right_frames=self.right_frames,
            left_segmenter=self.left_segmenter,
            right_segmenter=self.right_segmenter,
            calibration=self.calibration,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.valid for result in results))
        self.assertTrue(all(result.fused_gaze_point_px is not None for result in results))

    def test_rejects_mismatched_frame_counts(self) -> None:
        with self.assertRaises(ValueError):
            run_batch_pipeline(
                left_frames=self.left_frames,
                right_frames=self.right_frames[:1],
                left_segmenter=self.left_segmenter,
                right_segmenter=self.right_segmenter,
                calibration=self.calibration,
            )

    def test_continues_after_eye_detection_failure_in_face_video(self) -> None:
        extractor = _FakeEyePairExtractor(
            [
                EyePairFrame(
                    frame_bgr=self.face_frames[0],
                    left_eye_bgr=np.zeros((80, 120, 3), dtype=np.uint8),
                    right_eye_bgr=np.zeros((80, 120, 3), dtype=np.uint8),
                    boxes=[(10, 10, 120, 80), (200, 10, 120, 80)],
                    valid=True,
                ),
                EyePairFrame(
                    frame_bgr=self.face_frames[1],
                    left_eye_bgr=None,
                    right_eye_bgr=None,
                    boxes=[],
                    valid=False,
                    error_message="Eye-pair detection failed.",
                ),
                EyePairFrame(
                    frame_bgr=self.face_frames[2],
                    left_eye_bgr=np.zeros((80, 120, 3), dtype=np.uint8),
                    right_eye_bgr=np.zeros((80, 120, 3), dtype=np.uint8),
                    boxes=[(12, 12, 120, 80), (202, 12, 120, 80)],
                    valid=True,
                ),
            ]
        )

        details = run_batch_pipeline_from_face_frames_with_details(
            face_frames=self.face_frames,
            eye_pair_extractor=extractor,
            left_segmenter=self.left_segmenter,
            right_segmenter=self.right_segmenter,
            calibration=self.calibration,
        )

        self.assertEqual(len(details), 3)
        self.assertTrue(details[0].gaze.valid)
        self.assertFalse(details[1].gaze.valid)
        self.assertEqual(details[1].gaze.error_message, "Eye-pair detection failed.")
        self.assertTrue(details[2].gaze.valid)
        self.assertEqual(self.left_segmenter._index, 2)
        self.assertEqual(self.right_segmenter._index, 2)


if __name__ == "__main__":
    unittest.main()
