from __future__ import annotations

import time
from typing import Iterator

import cv2

from ..types import Frame
from .base import FrameSource


class VideoFileSource(FrameSource):
    """Reads frames from a recorded video file.

    mode="fast": yield frames as fast as possible (offline analysis).
    mode="realtime": pace playback to the recording's fps (live-like review).
    """

    def __init__(self, path: str, mode: str = "fast") -> None:
        if mode not in ("fast", "realtime"):
            raise ValueError(f"mode must be 'fast' or 'realtime', got {mode!r}")
        self._mode = mode
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if self._mode == "realtime" and index > 0:
                time.sleep(1.0 / self._fps)
            yield Frame(image=image, timestamp=index / self._fps, is_live=False)
            index += 1

    def close(self) -> None:
        self._cap.release()
