from __future__ import annotations

import json
from pathlib import Path

from eyetest.models.types import GazeEstimate


def _as_record(result: GazeEstimate) -> dict[str, object]:
    return {
        "frame_index": result.frame_index,
        "valid": result.valid,
        "left_gaze_point_px": list(result.left_gaze_point_px) if result.left_gaze_point_px else None,
        "right_gaze_point_px": list(result.right_gaze_point_px) if result.right_gaze_point_px else None,
        "fused_gaze_point_px": list(result.fused_gaze_point_px) if result.fused_gaze_point_px else None,
        "error_message": result.error_message,
    }


def write_gaze_results_json(path: str | Path, results: list[GazeEstimate]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_as_record(result) for result in results]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
