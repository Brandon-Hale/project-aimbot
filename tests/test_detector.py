import numpy as np

from gamevision.detection.detector import parse_boxes


def test_parse_boxes_maps_fields():
    xyxy = np.array([[10, 20, 30, 40]], dtype=float)
    conf = np.array([0.8])
    cls = np.array([0])
    names = {0: "person"}
    dets = parse_boxes(xyxy, conf, cls, names)
    assert len(dets) == 1
    assert dets[0].box == (10, 20, 30, 40)
    assert dets[0].class_id == 0
    assert dets[0].class_name == "person"
    assert dets[0].confidence == 0.8
    assert dets[0].track_id is None


def test_parse_boxes_with_track_ids():
    xyxy = np.array([[0, 0, 1, 1]], dtype=float)
    dets = parse_boxes(xyxy, np.array([0.5]), np.array([2]), {2: "car"}, np.array([7]))
    assert dets[0].class_name == "car"
    assert dets[0].track_id == 7


def test_parse_boxes_unknown_class_falls_back_to_id():
    dets = parse_boxes(np.array([[0, 0, 1, 1]], dtype=float), np.array([0.5]), np.array([99]), {})
    assert dets[0].class_name == "99"
