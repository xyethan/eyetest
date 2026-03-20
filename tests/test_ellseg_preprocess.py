from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.segmentation.ellseg_preprocess import preprocess_frame  # noqa: E402


class EllSegPreprocessTests(unittest.TestCase):
    def test_align_width_hits_exact_target_width_for_small_square_crop(self) -> None:
        image = np.zeros((77, 77), dtype=np.uint8)
        tensor, _ = preprocess_frame(image, (240, 320), align_width=True)
        self.assertEqual(tuple(tensor.shape), (1, 240, 320))


if __name__ == "__main__":
    unittest.main()
