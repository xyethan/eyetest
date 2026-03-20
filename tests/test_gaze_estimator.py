from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.adapters.ellipse_adapter import adapt_ellseg_ellipse_pair  # noqa: E402
from eyetest.config import load_calibration_config  # noqa: E402
from eyetest.gaze.estimator import estimate_frame_gaze  # noqa: E402
from eyetest.models.types import CalibrationFrame  # noqa: E402


class GazeEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = load_calibration_config(
            PROJECT_ROOT / "configs" / "calibration.default.yaml"
        )
        self.frame = CalibrationFrame(width_px=1920, height_px=1080)

    def test_estimates_gaze_from_valid_left_and_right_ellipses(self) -> None:
        left, right = adapt_ellseg_ellipse_pair(
            [596.1666083455085, 1117.006573232015, 110.72297024601819, 116.52391586776352, math.radians(-4.434906005859375)],
            [1298.118955039978, 1080.9469323794046, 146.45118647447984, 111.27496131621524, math.radians(81.47233581542969)],
        )
        result = estimate_frame_gaze(
            frame_index=0,
            left_ellipse=left,
            right_ellipse=right,
            calibration=self.calibration,
            frame=self.frame,
        )
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.left_gaze_point_px)
        self.assertIsNotNone(result.right_gaze_point_px)
        self.assertIsNotNone(result.fused_gaze_point_px)
        for point in (
            result.left_gaze_point_px,
            result.right_gaze_point_px,
            result.fused_gaze_point_px,
        ):
            assert point is not None
            self.assertTrue(math.isfinite(point[0]))
            self.assertTrue(math.isfinite(point[1]))

    def test_returns_invalid_when_any_eye_is_invalid(self) -> None:
        left, right = adapt_ellseg_ellipse_pair(
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1298.118955039978, 1080.9469323794046, 146.45118647447984, 111.27496131621524, math.radians(81.47233581542969)],
        )
        result = estimate_frame_gaze(
            frame_index=1,
            left_ellipse=left,
            right_ellipse=right,
            calibration=self.calibration,
            frame=self.frame,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.left_gaze_point_px)
        self.assertIsNone(result.fused_gaze_point_px)


if __name__ == "__main__":
    unittest.main()
