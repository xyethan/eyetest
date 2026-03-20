from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from eyetest.models.types import Ellipse2D


def _coerce_ellipse(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    if len(values) != 5:
        raise ValueError("Ellipse values must contain exactly 5 elements")
    cx, cy, major, minor, angle = (float(v) for v in values)
    return cx, cy, major, minor, angle


def adapt_ellseg_ellipse(values: Sequence[float]) -> Ellipse2D:
    cx, cy, major, minor, angle = _coerce_ellipse(values)
    if not all(isfinite(v) for v in (cx, cy, major, minor, angle)):
        return Ellipse2D.invalid()
    if cx < 0 and cy < 0 and major < 0 and minor < 0:
        return Ellipse2D.invalid()
    if major <= 0 or minor <= 0:
        return Ellipse2D.invalid()
    return Ellipse2D(
        center_x=cx,
        center_y=cy,
        major=major,
        minor=minor,
        angle=angle,
        valid=True,
    )


def adapt_ellseg_ellipse_pair(
    left_values: Sequence[float],
    right_values: Sequence[float],
) -> tuple[Ellipse2D, Ellipse2D]:
    return adapt_ellseg_ellipse(left_values), adapt_ellseg_ellipse(right_values)
