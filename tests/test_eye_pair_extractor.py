from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.inputs.camera import EyePairExtractor, TrackedEyePairExtractor  # noqa: E402


class _FakeCascade:
    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self._boxes = boxes

    def detectMultiScale(self, image, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)):
        return np.array(self._boxes, dtype=np.int32)


class EyePairExtractorTests(unittest.TestCase):
    def test_prefers_upper_aligned_eye_pair_over_lower_false_positives(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        extractor = EyePairExtractor(
            cascade=_FakeCascade(
                [
                    (935, 924, 77, 77),
                    (1146, 928, 70, 70),
                    (591, 360, 346, 346),
                    (1147, 325, 392, 392),
                    (1088, 898, 175, 175),
                ]
            )
        )

        eye_pair = extractor.extract(frame)

        self.assertTrue(eye_pair.valid)
        self.assertEqual(eye_pair.boxes, [(591, 360, 346, 346), (1147, 325, 392, 392)])
        self.assertEqual(eye_pair.left_eye_bgr.shape[:2], (346, 346))
        self.assertEqual(eye_pair.right_eye_bgr.shape[:2], (392, 392))

    def test_marks_frame_invalid_when_only_overlapping_candidates_exist(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        extractor = EyePairExtractor(
            cascade=_FakeCascade(
                [
                    (100, 100, 220, 220),
                    (130, 120, 200, 200),
                ]
            )
        )

        eye_pair = extractor.extract(frame)

        self.assertFalse(eye_pair.valid)
        self.assertEqual(eye_pair.error_message, "Eye-pair detection failed.")
        self.assertEqual(eye_pair.boxes, [])

    def test_tracked_extractor_reuses_previous_boxes_between_refreshes(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detector = EyePairExtractor(
            cascade=_FakeCascade(
                [
                    (600, 360, 300, 300),
                    (1200, 360, 300, 300),
                ]
            )
        )
        tracker = TrackedEyePairExtractor(detector=detector, refresh_interval=10, smoothing=0.5)

        first = tracker.extract(frame)
        second = tracker.extract(frame)

        self.assertTrue(first.valid)
        self.assertTrue(second.valid)
        self.assertEqual(first.boxes, second.boxes)

    def test_tracked_extractor_smooths_refresh_updates(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tracker = TrackedEyePairExtractor(
            detector=EyePairExtractor(
                cascade=_FakeCascade(
                    [
                        (600, 360, 300, 300),
                        (1200, 360, 300, 300),
                    ]
                )
            ),
            refresh_interval=1,
            smoothing=0.5,
        )

        first = tracker.extract(frame)
        tracker.detector = EyePairExtractor(
            cascade=_FakeCascade(
                [
                    (700, 360, 300, 300),
                    (1300, 360, 300, 300),
                ]
            )
        )
        second = tracker.extract(frame)

        self.assertTrue(first.valid)
        self.assertTrue(second.valid)
        self.assertEqual(second.boxes, [(650, 360, 300, 300), (1250, 360, 300, 300)])


if __name__ == "__main__":
    unittest.main()
