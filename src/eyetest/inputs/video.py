from __future__ import annotations

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
