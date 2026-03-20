from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CommandConfig:
    command: str


@dataclass(frozen=True)
class ModesConfig:
    batch: CommandConfig
    realtime: CommandConfig


@dataclass(frozen=True)
class SegmentationConfig:
    backend_root: str
    input_height: int
    input_width: int
    align_width: bool
    model_path: str
    use_regressed_ellipses: bool
    segment_iris: bool
    segment_pupil: bool
    skip_ransac: bool


@dataclass(frozen=True)
class CalibrationReference:
    path: str


@dataclass(frozen=True)
class OutputConfig:
    save_overlay: bool
    save_results: bool


@dataclass(frozen=True)
class AppConfig:
    modes: ModesConfig
    segmentation: SegmentationConfig
    calibration: CalibrationReference
    outputs: OutputConfig


@dataclass(frozen=True)
class EyeCalibration:
    radius: float
    kappa_alpha: float
    kappa_beta: float


@dataclass(frozen=True)
class CameraCalibration:
    principal_point_x: float
    principal_point_y: float
    fx: float
    fy: float
    focal_length: float
    pixel_size_x: float
    pixel_size_y: float


@dataclass(frozen=True)
class CalibrationConfig:
    screen_corners: list[tuple[float, float, float]]
    screen_width_px: int
    screen_height_px: int
    camera: CameraCalibration
    left_eye: EyeCalibration
    right_eye: EyeCalibration

    @property
    def camera_params(self) -> list[float]:
        return [
            self.camera.principal_point_x,
            self.camera.principal_point_y,
            self.camera.fx,
            self.camera.fy,
            self.camera.focal_length,
            self.camera.pixel_size_x,
            self.camera.pixel_size_y,
        ]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping: {key}")
    return value


def _require_float(mapping: dict[str, Any], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing required numeric key: {key}")
    return float(mapping[key])


def _require_bool(mapping: dict[str, Any], key: str) -> bool:
    if key not in mapping:
        raise ValueError(f"Missing required boolean key: {key}")
    return bool(mapping[key])


def _require_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise ValueError(f"Missing required integer key: {key}")
    return int(mapping[key])


def load_app_config(path: str | Path) -> AppConfig:
    raw = _read_yaml(Path(path))
    modes = _require_mapping(raw, "modes")
    batch = _require_mapping(modes, "batch")
    realtime = _require_mapping(modes, "realtime")
    segmentation = _require_mapping(raw, "segmentation")
    calibration = _require_mapping(raw, "calibration")
    outputs = _require_mapping(raw, "outputs")
    return AppConfig(
        modes=ModesConfig(
            batch=CommandConfig(command=str(batch["command"])),
            realtime=CommandConfig(command=str(realtime["command"])),
        ),
        segmentation=SegmentationConfig(
            backend_root=str(segmentation.get("backend_root", "/mnt/c/Project/EllSeg")),
            input_height=_require_int(segmentation, "input_height"),
            input_width=_require_int(segmentation, "input_width"),
            align_width=_require_bool(segmentation, "align_width"),
            model_path=str(segmentation["model_path"]),
            use_regressed_ellipses=_require_bool(segmentation, "use_regressed_ellipses"),
            segment_iris=_require_bool(segmentation, "segment_iris"),
            segment_pupil=_require_bool(segmentation, "segment_pupil"),
            skip_ransac=_require_bool(segmentation, "skip_ransac"),
        ),
        calibration=CalibrationReference(path=str(calibration["path"])),
        outputs=OutputConfig(
            save_overlay=_require_bool(outputs, "save_overlay"),
            save_results=_require_bool(outputs, "save_results"),
        ),
    )


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    raw = _read_yaml(Path(path))
    if "screen_corners" not in raw or not isinstance(raw["screen_corners"], list):
        raise ValueError("Missing required key: screen_corners")
    if len(raw["screen_corners"]) != 4:
        raise ValueError("screen_corners must contain exactly 4 points")

    screen_pixels = _require_mapping(raw, "screen_pixels")
    camera = _require_mapping(raw, "camera")
    eyes = _require_mapping(raw, "eyes")
    left = _require_mapping(eyes, "left")
    right = _require_mapping(eyes, "right")

    return CalibrationConfig(
        screen_corners=[tuple(float(v) for v in point) for point in raw["screen_corners"]],
        screen_width_px=_require_int(screen_pixels, "width"),
        screen_height_px=_require_int(screen_pixels, "height"),
        camera=CameraCalibration(
            principal_point_x=_require_float(camera, "principal_point_x"),
            principal_point_y=_require_float(camera, "principal_point_y"),
            fx=_require_float(camera, "fx"),
            fy=_require_float(camera, "fy"),
            focal_length=_require_float(camera, "focal_length"),
            pixel_size_x=_require_float(camera, "pixel_size_x"),
            pixel_size_y=_require_float(camera, "pixel_size_y"),
        ),
        left_eye=EyeCalibration(
            radius=_require_float(left, "radius"),
            kappa_alpha=_require_float(left, "kappa_alpha"),
            kappa_beta=_require_float(left, "kappa_beta"),
        ),
        right_eye=EyeCalibration(
            radius=_require_float(right, "radius"),
            kappa_alpha=_require_float(right, "kappa_alpha"),
            kappa_beta=_require_float(right, "kappa_beta"),
        ),
    )
