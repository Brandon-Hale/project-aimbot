from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..types import Detection

_BOX_COLOR = (0, 255, 0)     # BGR green
_TEXT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    image: np.ndarray,
    detections: Sequence[Detection],
    fps: float | None = None,
) -> np.ndarray:
    """Return a copy of `image` with boxes, labels, and a HUD drawn on it."""
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)

    hud = f"objects: {len(detections)}"
    if fps is not None:
        hud += f"  fps: {fps:.1f}"
    cv2.putText(canvas, hud, (10, 25), _FONT, 0.7, _TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


class OverlayWindow:
    """A mirror window that displays overlaid frames. Press 'q' to quit."""

    def __init__(self, title: str = "gamevision") -> None:
        self._title = title

    def show(self, image: np.ndarray) -> bool:
        cv2.imshow(self._title, image)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
