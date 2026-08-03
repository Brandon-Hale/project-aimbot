from __future__ import annotations

import time
from typing import Iterator

import cv2

from ..types import Frame
from .base import FrameSource


class DeviceSource(FrameSource):
    """Live capture card / webcam via OpenCV VideoCapture device index."""

    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open capture device index {index}")

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(image=image, timestamp=time.perf_counter(), is_live=True)

    def close(self) -> None:
        self._cap.release()
