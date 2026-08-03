import numpy as np
import pytest

from gamevision.detection.detector import YoloDetector
from gamevision.types import Frame


@pytest.mark.integration
def test_detect_returns_list_on_blank_frame():
    detector = YoloDetector(model_path="yolov8n.pt")
    frame = Frame(image=np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0, is_live=False)
    dets = detector.detect(frame)
    assert isinstance(dets, list)  # blank frame likely yields [], but must be a list
