from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


def preprocess_frame(
    image: np.ndarray,
    output_shape: tuple[int, int],
    align_width: bool = True,
) -> tuple[torch.Tensor, tuple[float, int]]:
    if not align_width:
        raise ValueError("Height-only alignment is not supported.")

    if output_shape[1] != image.shape[1]:
        scale = output_shape[1] / image.shape[1]
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        if output_shape[0] > image.shape[0]:
            pad_height = output_shape[0] - image.shape[0]
            image = np.pad(
                image,
                ((pad_height // 2, pad_height - pad_height // 2), (0, 0)),
            )
        elif output_shape[0] < image.shape[0]:
            trim = image.shape[0] - output_shape[0]
            start = trim // 2
            image = image[start : start + output_shape[0], :]
            pad_height = -trim
        else:
            pad_height = 0
        scale_shift = (scale, pad_height)
    else:
        scale_shift = (1.0, 0)

    std = float(image.std())
    if std < 1e-6:
        image = image.astype(np.float32) - float(image.mean())
    else:
        image = (image - image.mean()) / std
    return torch.from_numpy(image).unsqueeze(0).to(torch.float32), scale_shift


def rescale_to_original(
    seg_map: np.ndarray,
    pupil_ellipse: np.ndarray,
    iris_ellipse: np.ndarray,
    scale_shift: tuple[float, int],
    original_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pupil = pupil_ellipse.copy()
    iris = iris_ellipse.copy()
    pupil[1] = pupil[1] - np.floor(scale_shift[1] // 2)
    pupil[:-1] = pupil[:-1] * (1 / scale_shift[0])
    iris[1] = iris[1] - np.floor(scale_shift[1] // 2)
    iris[:-1] = iris[:-1] * (1 / scale_shift[0])

    if scale_shift[1] < 0:
        seg_map = np.pad(seg_map, ((-scale_shift[1] // 2, -scale_shift[1] // 2), (0, 0)))
    elif scale_shift[1] > 0:
        seg_map = seg_map[scale_shift[1] // 2 : -scale_shift[1] // 2, ...]

    seg_map = cv2.resize(seg_map, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return seg_map, pupil, iris
