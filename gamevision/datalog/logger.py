from __future__ import annotations

import json
from typing import Sequence, TextIO

from ..types import Detection


class DetectionLogger:
    """Appends one JSON record per frame to a text stream (JSONL)."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def log(self, timestamp: float, detections: Sequence[Detection]) -> None:
        record = {
            "timestamp": timestamp,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "box": list(d.box),
                    "track_id": d.track_id,
                }
                for d in detections
            ],
        }
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
