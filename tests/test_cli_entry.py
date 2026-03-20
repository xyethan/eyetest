from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from eyetest.cli import main as cli_main  # noqa: E402


class CliEntryTests(unittest.TestCase):
    def test_main_help_runs_without_pythonpath(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("batch", result.stdout)
        self.assertIn("realtime", result.stdout)

    @patch("eyetest.cli.run_batch_from_face_video")
    def test_batch_main_passes_output_video_path(self, run_batch_from_face_video) -> None:
        exit_code = cli_main(
            [
                "--config",
                str(PROJECT_ROOT / "configs" / "default.yaml"),
                "batch",
                "--face-video",
                "video/near_1.mp4",
                "--output-json",
                "outputs/gaze-results.json",
                "--output-video",
                "outputs/gaze-overlay.mp4",
                "--eval-on-cpu",
            ]
        )
        self.assertEqual(exit_code, 0)
        _, kwargs = run_batch_from_face_video.call_args
        self.assertEqual(kwargs["face_video_path"], "video/near_1.mp4")
        self.assertEqual(kwargs["output_video_path"], "outputs/gaze-overlay.mp4")

    @patch("eyetest.cli.run_batch_from_face_video")
    def test_batch_main_passes_output_overlay_video_path(self, run_batch_from_face_video) -> None:
        exit_code = cli_main(
            [
                "--config",
                str(PROJECT_ROOT / "configs" / "default.yaml"),
                "batch",
                "--face-video",
                "video/near_1.mp4",
                "--output-overlay-video",
                "outputs/gaze-side-by-side.mp4",
            ]
        )
        self.assertEqual(exit_code, 0)
        _, kwargs = run_batch_from_face_video.call_args
        self.assertEqual(kwargs["output_overlay_video_path"], "outputs/gaze-side-by-side.mp4")


if __name__ == "__main__":
    unittest.main()
