from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from eyetest.config import SegmentationConfig
from eyetest.segmentation import ellseg_compat_loss, ellseg_compat_utils


@dataclass(frozen=True)
class EllSegRuntime:
    model: object
    device: torch.device


def _ensure_backend_on_path(backend_root: Path) -> None:
    backend_root_str = str(backend_root)
    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)


def load_ellseg_runtime(
    segmentation: SegmentationConfig,
    eval_on_cpu: bool = False,
) -> EllSegRuntime:
    backend_root = Path(segmentation.backend_root)
    _ensure_backend_on_path(backend_root)
    importlib.invalidate_caches()
    sys.modules["utils"] = ellseg_compat_utils
    sys.modules["loss"] = ellseg_compat_loss
    model_summary_module = importlib.import_module("modelSummary")

    device = torch.device("cpu" if eval_on_cpu or not torch.cuda.is_available() else "cuda")
    state = torch.load(segmentation.model_path, map_location=device)
    model = model_summary_module.model_dict["ritnet_v3"]
    model.load_state_dict(state["state_dict"], strict=True)
    if device.type == "cuda":
        model.cuda()
    model.eval()
    return EllSegRuntime(
        model=model,
        device=device,
    )
