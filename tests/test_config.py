from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.config import load_app_config, load_calibration_config  # noqa: E402


class ConfigLoadingTests(unittest.TestCase):
    def test_loads_default_app_config(self) -> None:
        config = load_app_config(PROJECT_ROOT / "configs" / "default.yaml")
        self.assertEqual(config.modes.batch.command, "batch")
        self.assertEqual(config.segmentation.input_height, 240)
        self.assertEqual(config.segmentation.input_width, 320)

    def test_loads_default_calibration_config(self) -> None:
        calibration = load_calibration_config(
            PROJECT_ROOT / "configs" / "calibration.default.yaml"
        )
        self.assertEqual(len(calibration.screen_corners), 4)
        self.assertAlmostEqual(calibration.left_eye.radius, 5.372254417341973)
        self.assertAlmostEqual(calibration.right_eye.kappa_beta, -0.03691749084564522)

    def test_raises_for_missing_calibration_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibration.yaml"
            path.write_text("screen_corners: []\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration_config(path)


if __name__ == "__main__":
    unittest.main()
