import io
import json

from gamevision.datalog.logger import DetectionLogger
from gamevision.types import Detection


def test_logger_writes_one_json_line_per_call():
    buf = io.StringIO()
    logger = DetectionLogger(buf)
    logger.log(1.5, [Detection(0, "person", 0.9, (1, 2, 3, 4), track_id=5)])
    logger.log(2.0, [])
    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["timestamp"] == 1.5
    assert first["detections"][0]["class_name"] == "person"
    assert first["detections"][0]["box"] == [1, 2, 3, 4]
    assert first["detections"][0]["track_id"] == 5

    second = json.loads(lines[1])
    assert second["detections"] == []
