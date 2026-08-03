from gamevision.types import Detection, Frame
import numpy as np


def test_detection_holds_fields():
    d = Detection(class_id=0, class_name="person", confidence=0.9, box=(1, 2, 3, 4))
    assert d.class_id == 0
    assert d.class_name == "person"
    assert d.confidence == 0.9
    assert d.box == (1, 2, 3, 4)
    assert d.track_id is None


def test_frame_holds_fields():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    f = Frame(image=img, timestamp=1.5, is_live=True)
    assert f.timestamp == 1.5
    assert f.is_live is True
    assert f.image.shape == (4, 4, 3)
