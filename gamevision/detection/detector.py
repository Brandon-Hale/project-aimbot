from __future__ import annotations

import numpy as np

from ..types import Detection, Frame


def parse_boxes(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    names: dict[int, str],
    track_ids: np.ndarray | None = None,
) -> list[Detection]:
    """Map raw model output arrays into a list of Detection objects."""
    detections: list[Detection] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        class_id = int(cls[i])
        track_id = int(track_ids[i]) if track_ids is not None else None
        detections.append(
            Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=float(conf[i]),
                box=(x1, y1, x2, y2),
                track_id=track_id,
            )
        )
    return detections


class YoloDetector:
    """Runs a YOLO model on frames. Set track=True for stable IDs across frames."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.25, track: bool = False) -> None:
        from ultralytics import YOLO  # imported lazily so unit tests don't need it

        self._model = YOLO(model_path)
        self._conf = conf
        self._track = track

    def detect(self, frame: Frame) -> list[Detection]:
        if self._track:
            results = self._model.track(frame.image, persist=True, conf=self._conf, verbose=False)
        else:
            results = self._model.predict(frame.image, conf=self._conf, verbose=False)
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if (self._track and boxes.id is not None) else None
        return parse_boxes(xyxy, conf, cls, result.names, track_ids)
