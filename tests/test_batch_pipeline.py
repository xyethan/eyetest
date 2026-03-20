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
from eyetest.pipelines.batch_pipeline import run_batch_pipeline  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
