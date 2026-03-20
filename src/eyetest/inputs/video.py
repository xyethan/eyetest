from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def read_video_frames(path: str | Path, grayscale: bool = True) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    ok = True
    while ok:
        ok, frame = capture.read()
        if not ok:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    capture.release()
    return frames


def read_video_fps(path: str | Path, default_fps: float = 25.0) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if math.isfinite(fps) and fps > 0.0:
        return fps
    return default_fps
