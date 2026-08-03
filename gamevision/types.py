from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object in a single frame."""
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels (top-left origin)
    track_id: int | None = None


@dataclass
class Frame:
    """One captured frame plus metadata."""
    image: np.ndarray  # BGR, shape (H, W, 3), dtype uint8
    timestamp: float   # seconds; wall-clock for live, frame_index/fps for video
    is_live: bool
