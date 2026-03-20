from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.models.types import BatchFrameOverlay, Ellipse2D, GazeEstimate  # noqa: E402
from eyetest.outputs.writers import write_gaze_overlay_video, write_side_by_side_overlay_video  # noqa: E402


class OutputWriterTests(unittest.TestCase):
    def _overlay_frames(self) -> list[BatchFrameOverlay]:
        left_frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        right_frame = np.full((120, 160, 3), 80, dtype=np.uint8)
        return [
            BatchFrameOverlay(
                frame_index=0,
                left_frame_bgr=left_frame,
                right_frame_bgr=right_frame,
                left_iris=Ellipse2D(center_x=80.0, center_y=60.0, major=30.0, minor=24.0, angle=0.2),
                right_iris=Ellipse2D(center_x=78.0, center_y=58.0, major=28.0, minor=22.0, angle=-0.15),
                left_pupil=Ellipse2D(center_x=80.0, center_y=60.0, major=11.0, minor=11.0, angle=0.0),
                right_pupil=Ellipse2D(center_x=78.0, center_y=58.0, major=10.0, minor=10.0, angle=0.0),
                gaze=GazeEstimate(
                    frame_index=0,
                    valid=True,
                    left_gaze_point_px=(100.0, 120.0),
                    right_gaze_point_px=(130.0, 150.0),
                    fused_gaze_point_px=(115.0, 135.0),
                ),
            )
        ]

    def test_writes_gaze_overlay_video(self) -> None:
        results = [
            GazeEstimate(
                frame_index=0,
                valid=True,
                left_gaze_point_px=(100.0, 120.0),
                right_gaze_point_px=(130.0, 150.0),
                fused_gaze_point_px=(115.0, 135.0),
            ),
            GazeEstimate(
                frame_index=1,
                valid=True,
                left_gaze_point_px=(110.0, 140.0),
                right_gaze_point_px=(150.0, 160.0),
                fused_gaze_point_px=(130.0, 150.0),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "gaze-overlay.mp4"
            write_gaze_overlay_video(
                path=output_path,
                results=results,
                width_px=640,
                height_px=480,
                fps=25.0,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_writes_side_by_side_overlay_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "gaze-side-by-side.mp4"
            write_side_by_side_overlay_video(
                path=output_path,
                frames=self._overlay_frames(),
                screen_width_px=640,
                screen_height_px=480,
                fps=25.0,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

            capture = cv2.VideoCapture(str(output_path))
            ok, frame = capture.read()
            capture.release()

            self.assertTrue(ok)
            self.assertEqual(frame.shape[1], 480)
            self.assertEqual(frame.shape[0], 276)


if __name__ == "__main__":
    unittest.main()
