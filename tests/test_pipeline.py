import io
import json

from gamevision.pipeline import run_pipeline
from gamevision.sources.video import VideoFileSource
from gamevision.datalog.logger import DetectionLogger
from gamevision.types import Detection, Frame


class FakeDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        return [Detection(0, "person", 0.9, (1, 2, 3, 4))]


def test_pipeline_processes_all_frames_and_logs(sample_video):
    buf = io.StringIO()
    logger = DetectionLogger(buf)
    with VideoFileSource(sample_video, mode="fast") as src:
        count = run_pipeline(src, FakeDetector(), window=None, logger=logger)
    assert count == 10
    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == 10
    assert json.loads(lines[0])["detections"][0]["class_name"] == "person"


def test_pipeline_runs_without_logger(sample_video):
    with VideoFileSource(sample_video, mode="fast") as src:
        count = run_pipeline(src, FakeDetector(), window=None, logger=None)
    assert count == 10
