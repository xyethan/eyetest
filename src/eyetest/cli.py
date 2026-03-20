from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_app_config, load_calibration_config
from .pipelines.batch_pipeline import run_batch_from_eye_videos
from .pipelines.realtime_pipeline import run_realtime_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eyetest")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to the eyetest app config.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    batch = subparsers.add_parser("batch")
    batch.add_argument("--left-video", type=Path, required=True, help="Path to the left-eye video.")
    batch.add_argument("--right-video", type=Path, required=True, help="Path to the right-eye video.")
    batch.add_argument("--output-json", type=Path, default=Path("outputs/gaze-results.json"), help="Path to write batch gaze results.")
    batch.add_argument("--eval-on-cpu", action="store_true", help="Run EllSeg on CPU.")

    realtime = subparsers.add_parser("realtime")
    realtime.add_argument("--camera-index", type=int, default=0, help="Camera index.")
    realtime.add_argument("--eval-on-cpu", action="store_true", help="Run EllSeg on CPU.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command is None:
        parser.print_help()
        return 1
    app_config = load_app_config(args.config)
    calibration_path = Path(app_config.calibration.path)
    if not calibration_path.is_absolute():
        calibration_path = (args.config.parent / calibration_path).resolve()
    calibration = load_calibration_config(calibration_path)
    if command == "batch":
        run_batch_from_eye_videos(
            left_video_path=str(args.left_video),
            right_video_path=str(args.right_video),
            app_config=app_config,
            calibration=calibration,
            output_json_path=str(args.output_json),
            eval_on_cpu=bool(args.eval_on_cpu),
        )
    elif command == "realtime":
        run_realtime_pipeline(
            app_config=app_config,
            calibration=calibration,
            camera_index=int(args.camera_index),
            eval_on_cpu=bool(args.eval_on_cpu),
        )
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
