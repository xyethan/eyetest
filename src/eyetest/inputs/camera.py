from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EyePairFrame:
    frame_bgr: np.ndarray
    left_eye_gray: np.ndarray
    right_eye_gray: np.ndarray
    boxes: list[tuple[int, int, int, int]]


class EyePairCamera:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    def read(self) -> EyePairFrame | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
        if len(eyes) < 2:
            return None
        ordered = sorted(eyes, key=lambda item: item[0])[:2]
        left_box, right_box = ordered[0], ordered[1]
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        left_eye = gray[ly : ly + lh, lx : lx + lw]
        right_eye = gray[ry : ry + rh, rx : rx + rw]
        return EyePairFrame(
            frame_bgr=frame,
            left_eye_gray=left_eye,
            right_eye_gray=right_eye,
            boxes=[tuple(int(v) for v in left_box), tuple(int(v) for v in right_box)],
        )

    def close(self) -> None:
        self.capture.release()
