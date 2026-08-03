from __future__ import annotations

import time
from typing import Optional, Protocol

from .types import Detection, Frame
from .sources.base import FrameSource
from .overlay.renderer import OverlayWindow, draw_overlay
from .datalog.logger import DetectionLogger


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]:
        ...


def run_pipeline(
    source: FrameSource,
    detector: Detector,
    window: Optional[OverlayWindow] = None,
    logger: Optional[DetectionLogger] = None,
) -> int:
    """Pull frames, detect, then feed the overlay window and/or logger.

    Returns the number of frames processed. Stops at end of source or when
    the overlay window signals quit (user pressed 'q').
    """
    frame_count = 0
    last = time.perf_counter()
    fps = 0.0
    for frame in source.frames():
        detections = detector.detect(frame)

        now = time.perf_counter()
        dt = now - last
        if dt > 0:
            fps = 1.0 / dt
        last = now

        if logger is not None:
            logger.log(frame.timestamp, detections)

        if window is not None:
            canvas = draw_overlay(frame.image, detections, fps)
            if not window.show(canvas):
                break

        frame_count += 1
    return frame_count
