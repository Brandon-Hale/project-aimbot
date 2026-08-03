import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path):
    """Write a deterministic 10-frame MJPG/.avi clip and return its path."""
    path = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 48))
    assert writer.isOpened(), "MJPG writer unavailable in this OpenCV build"
    for i in range(10):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)
