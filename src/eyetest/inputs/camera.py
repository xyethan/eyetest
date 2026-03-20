from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class EyePairFrame:
    frame_bgr: np.ndarray
    left_eye_bgr: np.ndarray | None
    right_eye_bgr: np.ndarray | None
    boxes: list[tuple[int, int, int, int]]
    valid: bool
    error_message: str | None = None


class EyeDetector(Protocol):
    def detectMultiScale(self, image, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)):
        ...


class EyePairExtractor:
    def __init__(self, cascade: EyeDetector | None = None) -> None:
        self.cascade = cascade or cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    @staticmethod
    def _intersection_area(left_box: tuple[int, int, int, int], right_box: tuple[int, int, int, int]) -> int:
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        x0 = max(lx, rx)
        y0 = max(ly, ry)
        x1 = min(lx + lw, rx + rw)
        y1 = min(ly + lh, ry + rh)
        return max(0, x1 - x0) * max(0, y1 - y0)

    @staticmethod
    def _pair_score(
        left_box: tuple[int, int, int, int],
        right_box: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
    ) -> float | None:
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        left_area = lw * lh
        right_area = rw * rh
        overlap = EyePairExtractor._intersection_area(left_box, right_box)
        if overlap > 0:
            overlap_ratio = overlap / max(min(left_area, right_area), 1)
            if overlap_ratio > 0.15:
                return None

        left_center_x = lx + (lw / 2.0)
        left_center_y = ly + (lh / 2.0)
        right_center_x = rx + (rw / 2.0)
        right_center_y = ry + (rh / 2.0)
        center_distance_x = abs(right_center_x - left_center_x)
        center_distance_y = abs(right_center_y - left_center_y)

        if center_distance_x < max(min(lw, rw) * 0.75, 1.0):
            return None

        mean_height = max((lh + rh) / 2.0, 1.0)
        width_penalty = abs(lw - rw) / max(lw, rw, 1)
        height_penalty = abs(lh - rh) / max(lh, rh, 1)
        vertical_penalty = center_distance_y / mean_height
        area_penalty = (left_area + right_area) / max(frame_shape[0] * frame_shape[1], 1)
        position_penalty = ((left_center_y + right_center_y) / 2.0) / max(frame_shape[0], 1)
        return vertical_penalty + width_penalty + height_penalty + area_penalty + (1.5 * position_penalty)

    def _select_eye_pair(
        self,
        eyes: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        candidates = [tuple(int(v) for v in box) for box in eyes]
        best_pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
        best_score: float | None = None
        for raw_left, raw_right in combinations(candidates, 2):
            left_box, right_box = sorted((raw_left, raw_right), key=lambda item: item[0])
            score = self._pair_score(left_box, right_box, frame_shape)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (left_box, right_box)
        return best_pair

    def extract(self, frame_bgr: np.ndarray) -> EyePairFrame:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        eyes = self.cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
        if len(eyes) < 2:
            return EyePairFrame(
                frame_bgr=frame_bgr,
                left_eye_bgr=None,
                right_eye_bgr=None,
                boxes=[],
                valid=False,
                error_message="Eye-pair detection failed.",
            )
        selected_pair = self._select_eye_pair(eyes, frame_bgr.shape)
        if selected_pair is None:
            return EyePairFrame(
                frame_bgr=frame_bgr,
                left_eye_bgr=None,
                right_eye_bgr=None,
                boxes=[],
                valid=False,
                error_message="Eye-pair detection failed.",
            )
        left_box, right_box = selected_pair
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        left_eye = frame_bgr[ly : ly + lh, lx : lx + lw].copy()
        right_eye = frame_bgr[ry : ry + rh, rx : rx + rw].copy()
        return EyePairFrame(
            frame_bgr=frame_bgr,
            left_eye_bgr=left_eye,
            right_eye_bgr=right_eye,
            boxes=[tuple(int(v) for v in left_box), tuple(int(v) for v in right_box)],
            valid=True,
        )


class TrackedEyePairExtractor:
    def __init__(
        self,
        detector: EyePairExtractor | None = None,
        refresh_interval: int = 12,
        smoothing: float = 0.2,
    ) -> None:
        self.detector = detector or EyePairExtractor()
        self.refresh_interval = max(refresh_interval, 1)
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self._frame_index = 0
        self._tracked_boxes: list[tuple[int, int, int, int]] | None = None

    @staticmethod
    def _blend_box(
        previous_box: tuple[int, int, int, int],
        current_box: tuple[int, int, int, int],
        smoothing: float,
    ) -> tuple[int, int, int, int]:
        blended = []
        for prev_value, curr_value in zip(previous_box, current_box, strict=True):
            value = ((1.0 - smoothing) * prev_value) + (smoothing * curr_value)
            blended.append(int(round(value)))
        return tuple(blended)  # type: ignore[return-value]

    @staticmethod
    def _crop_from_boxes(
        frame_bgr: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> EyePairFrame:
        left_box, right_box = boxes
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        left_eye = frame_bgr[ly : ly + lh, lx : lx + lw].copy()
        right_eye = frame_bgr[ry : ry + rh, rx : rx + rw].copy()
        return EyePairFrame(
            frame_bgr=frame_bgr,
            left_eye_bgr=left_eye,
            right_eye_bgr=right_eye,
            boxes=boxes,
            valid=True,
        )

    def extract(self, frame_bgr: np.ndarray) -> EyePairFrame:
        should_refresh = self._tracked_boxes is None or (self._frame_index % self.refresh_interval == 0)
        self._frame_index += 1

        if not should_refresh and self._tracked_boxes is not None:
            return self._crop_from_boxes(frame_bgr, self._tracked_boxes)

        detected = self.detector.extract(frame_bgr)
        if not detected.valid:
            if self._tracked_boxes is not None:
                return self._crop_from_boxes(frame_bgr, self._tracked_boxes)
            return detected

        if self._tracked_boxes is None:
            self._tracked_boxes = detected.boxes
            return detected

        self._tracked_boxes = [
            self._blend_box(previous_box, current_box, self.smoothing)
            for previous_box, current_box in zip(self._tracked_boxes, detected.boxes, strict=True)
        ]
        return self._crop_from_boxes(frame_bgr, self._tracked_boxes)


class EyePairCamera:
    def __init__(self, camera_index: int = 0, extractor: EyePairExtractor | TrackedEyePairExtractor | None = None) -> None:
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.extractor = extractor or TrackedEyePairExtractor()

    def read(self) -> EyePairFrame | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        return self.extractor.extract(frame)

    def close(self) -> None:
        self.capture.release()
