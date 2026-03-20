from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.adapters.ellipse_adapter import adapt_ellseg_ellipse_pair  # noqa: E402


class EllipseAdapterTests(unittest.TestCase):
    def test_converts_valid_ellseg_output(self) -> None:
        left, right = adapt_ellseg_ellipse_pair(
            [100.0, 50.0, 30.0, 20.0, 0.25],
            [200.0, 70.0, 28.0, 18.0, -0.3],
        )
        self.assertTrue(left.valid)
        self.assertTrue(right.valid)
        self.assertAlmostEqual(left.center_x, 100.0)
        self.assertAlmostEqual(left.center_y, 50.0)
        self.assertAlmostEqual(left.major, 30.0)
        self.assertAlmostEqual(left.minor, 20.0)
        self.assertAlmostEqual(left.angle, 0.25)

    def test_marks_negative_sentinel_as_invalid(self) -> None:
        left, right = adapt_ellseg_ellipse_pair(
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [200.0, 70.0, 28.0, 18.0, -0.3],
        )
        self.assertFalse(left.valid)
        self.assertTrue(right.valid)

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            adapt_ellseg_ellipse_pair([1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0])


if __name__ == "__main__":
    unittest.main()
