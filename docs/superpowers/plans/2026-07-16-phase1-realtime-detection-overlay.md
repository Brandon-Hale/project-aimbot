# Phase 1: Real-Time Detection & Overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python tool that reads frames from a recorded video *or* a live screen/device, runs a pretrained YOLO detector on each frame, and shows a mirror-window overlay with boxes, labels, and a HUD — with optional JSONL logging.

**Architecture:** A one-directional pipeline. A swappable `FrameSource` yields `(image, timestamp, is_live)`; a `YoloDetector` maps each frame to a list of `Detection`; the overlay renderer and optional logger are independent consumers of that list. Frame sources are selected at runtime by a CLI flag, so recorded and live inputs share the entire downstream pipeline.

**Tech Stack:** Python 3.10+, `ultralytics` (YOLO), `opencv-python`, `numpy`, `dxcam` (Windows live capture), `pytest`.

## Global Constraints

- **Python >= 3.10** (uses `X | None` typing syntax).
- **Single-player use only.** No multiplayer/online-match support.
- **Display + analytics only.** The pipeline NEVER controls mouse/keyboard or game input. No task may add input automation.
- **Package name:** `gamevision` (flat layout at repo root).
- **Detection box format:** `(x1, y1, x2, y2)` integer pixels, top-left origin.
- **Frame image format:** BGR `numpy.ndarray`, shape `(H, W, 3)`, dtype `uint8` (OpenCV convention).
- **Tests must pass without a GPU or network.** Anything requiring the real YOLO model download is marked `@pytest.mark.integration` and excluded from the default test run.

---

## File Structure

```
project-aimbot/
  pyproject.toml                     # deps, entry point, pytest config
  README.md                          # setup + run instructions
  gamevision/
    __init__.py
    types.py                         # Detection, Frame dataclasses
    pipeline.py                      # run_pipeline(): wires everything
    cli.py                           # argparse entry point -> main()
    sources/
      __init__.py
      base.py                        # FrameSource ABC (+ context manager)
      video.py                       # VideoFileSource (recorded)
      screen.py                      # ScreenSource (dxcam, live)
      device.py                      # DeviceSource (capture card, live)
      factory.py                     # build_source(args) -> FrameSource
    detection/
      __init__.py
      detector.py                    # parse_boxes(), YoloDetector
    overlay/
      __init__.py
      renderer.py                    # draw_overlay(), OverlayWindow
    datalog/
      __init__.py
      logger.py                      # DetectionLogger (JSONL)
  tests/
    conftest.py                      # sample_video fixture
    test_types.py
    test_video_source.py
    test_detector.py
    test_detector_integration.py     # @integration
    test_overlay.py
    test_logger.py
    test_pipeline.py
    test_factory.py
```

**Note on tracking:** the spec lists a tracking module. In Phase 1 this is implemented as a `track=True` option on `YoloDetector` (Ultralytics' built-in ByteTrack via `model.track(persist=True)`), which populates `Detection.track_id`. A standalone tracking module is unnecessary for v1 (YAGNI); revisit only if custom tracking logic is needed later.

**Note on the `datalog` name:** the logger package is named `datalog`, not `logging`, to avoid shadowing Python's stdlib `logging` module.

---

### Task 1: Project scaffold & core types

**Files:**
- Create: `pyproject.toml`
- Create: `gamevision/__init__.py` (empty)
- Create: `gamevision/types.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Detection(class_id: int, class_name: str, confidence: float, box: tuple[int,int,int,int], track_id: int | None = None)` — frozen dataclass.
  - `Frame(image: np.ndarray, timestamp: float, is_live: bool)` — dataclass.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "gamevision"
version = "0.1.0"
description = "Real-time single-player game object detection & overlay"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.3",
    "opencv-python>=4.8",
    "numpy>=1.24",
    "dxcam>=0.0.5; platform_system=='Windows'",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
gamevision = "gamevision.cli:main"

[tool.setuptools.packages.find]
include = ["gamevision*"]

[tool.pytest.ini_options]
markers = [
    "integration: tests that download/run the real YOLO model (slow, needs network/GPU)",
]
addopts = "-m 'not integration'"
```

- [ ] **Step 2: Create empty `gamevision/__init__.py` and `tests/__init__.py`**

Both files are empty.

- [ ] **Step 3: Write the failing test** — `tests/test_types.py`

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pip install -e ".[dev]"` then `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.types'`

- [ ] **Step 5: Write `gamevision/types.py`**

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml gamevision/__init__.py gamevision/types.py tests/__init__.py tests/test_types.py
git commit -m "feat: project scaffold and core Detection/Frame types"
```

---

### Task 2: Frame source interface & recorded video source

**Files:**
- Create: `gamevision/sources/__init__.py` (empty)
- Create: `gamevision/sources/base.py`
- Create: `gamevision/sources/video.py`
- Create: `tests/conftest.py`
- Test: `tests/test_video_source.py`

**Interfaces:**
- Consumes: `Frame` from Task 1.
- Produces:
  - `FrameSource` ABC with `frames() -> Iterator[Frame]`, `close() -> None`, and context-manager support.
  - `VideoFileSource(path: str, mode: str = "fast")` — `mode` is `"fast"` or `"realtime"`; raises `FileNotFoundError` on a bad path and `ValueError` on a bad mode.

- [ ] **Step 1: Create `tests/conftest.py` (shared fixture)**

```python
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
```

- [ ] **Step 2: Write the failing test** — `tests/test_video_source.py`

```python
import pytest

from gamevision.sources.video import VideoFileSource


def test_reads_all_frames(sample_video):
    with VideoFileSource(sample_video, mode="fast") as src:
        frames = list(src.frames())
    assert len(frames) == 10
    assert all(f.is_live is False for f in frames)
    assert frames[0].timestamp == 0.0
    assert frames[0].image.shape == (48, 64, 3)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoFileSource("does_not_exist.avi")


def test_invalid_mode_raises(sample_video):
    with pytest.raises(ValueError):
        VideoFileSource(sample_video, mode="turbo")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_video_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.sources'`

- [ ] **Step 4: Write `gamevision/sources/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..types import Frame


class FrameSource(ABC):
    """Yields frames from some input (recorded or live)."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted (video) or stopped (live)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying resources. Safe to call more than once."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 5: Write `gamevision/sources/video.py`**

```python
from __future__ import annotations

import time
from typing import Iterator

import cv2

from ..types import Frame
from .base import FrameSource


class VideoFileSource(FrameSource):
    """Reads frames from a recorded video file.

    mode="fast": yield frames as fast as possible (offline analysis).
    mode="realtime": pace playback to the recording's fps (live-like review).
    """

    def __init__(self, path: str, mode: str = "fast") -> None:
        if mode not in ("fast", "realtime"):
            raise ValueError(f"mode must be 'fast' or 'realtime', got {mode!r}")
        self._mode = mode
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if self._mode == "realtime" and index > 0:
                time.sleep(1.0 / self._fps)
            yield Frame(image=image, timestamp=index / self._fps, is_live=False)
            index += 1

    def close(self) -> None:
        self._cap.release()
```

- [ ] **Step 6: Create empty `gamevision/sources/__init__.py`**

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_video_source.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add gamevision/sources/ tests/conftest.py tests/test_video_source.py
git commit -m "feat: FrameSource interface and recorded VideoFileSource"
```

---

### Task 3: YOLO detector

**Files:**
- Create: `gamevision/detection/__init__.py` (empty)
- Create: `gamevision/detection/detector.py`
- Test: `tests/test_detector.py`
- Test: `tests/test_detector_integration.py`

**Interfaces:**
- Consumes: `Detection`, `Frame` from Task 1.
- Produces:
  - `parse_boxes(xyxy, conf, cls, names, track_ids=None) -> list[Detection]` — pure mapping from arrays to `Detection`s. `names: dict[int, str]`.
  - `YoloDetector(model_path="yolov8n.pt", conf=0.25, track=False)` with `detect(frame: Frame) -> list[Detection]`.

- [ ] **Step 1: Write the failing unit test** — `tests/test_detector.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.detection'`

- [ ] **Step 3: Write `gamevision/detection/detector.py`**

```python
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
```

- [ ] **Step 4: Create empty `gamevision/detection/__init__.py`**

- [ ] **Step 5: Run unit test to verify it passes**

Run: `pytest tests/test_detector.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Write the integration test** — `tests/test_detector_integration.py`

```python
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
```

- [ ] **Step 7: Run the integration test explicitly (downloads yolov8n, ~6 MB)**

Run: `pytest tests/test_detector_integration.py -m integration -v`
Expected: PASS (1 test). Skipped by default because `addopts` excludes `integration`.

- [ ] **Step 8: Commit**

```bash
git add gamevision/detection/ tests/test_detector.py tests/test_detector_integration.py
git commit -m "feat: YoloDetector with pure parse_boxes mapping and tracking option"
```

---

### Task 4: Overlay renderer

**Files:**
- Create: `gamevision/overlay/__init__.py` (empty)
- Create: `gamevision/overlay/renderer.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `Detection` from Task 1.
- Produces:
  - `draw_overlay(image, detections, fps=None) -> np.ndarray` — pure; returns a NEW image (does not mutate input).
  - `OverlayWindow(title="gamevision")` with `show(image) -> bool` (False = user pressed `q`) and `close()`.

- [ ] **Step 1: Write the failing test** — `tests/test_overlay.py`

```python
import numpy as np

from gamevision.overlay.renderer import draw_overlay
from gamevision.types import Detection


def test_draw_overlay_returns_new_image_same_shape():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [Detection(0, "person", 0.9, (10, 10, 50, 50))]
    out = draw_overlay(img, dets, fps=30.0)
    assert out.shape == img.shape
    assert out is not img          # must not mutate caller's frame
    assert img.sum() == 0          # original untouched
    assert out.sum() > 0           # boxes/labels drawn


def test_draw_overlay_no_detections_still_draws_hud():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_overlay(img, [], fps=None)
    assert out.sum() > 0           # HUD text is always drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.overlay'`

- [ ] **Step 3: Write `gamevision/overlay/renderer.py`**

```python
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..types import Detection

_BOX_COLOR = (0, 255, 0)     # BGR green
_TEXT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    image: np.ndarray,
    detections: Sequence[Detection],
    fps: float | None = None,
) -> np.ndarray:
    """Return a copy of `image` with boxes, labels, and a HUD drawn on it."""
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)

    hud = f"objects: {len(detections)}"
    if fps is not None:
        hud += f"  fps: {fps:.1f}"
    cv2.putText(canvas, hud, (10, 25), _FONT, 0.7, _TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


class OverlayWindow:
    """A mirror window that displays overlaid frames. Press 'q' to quit."""

    def __init__(self, title: str = "gamevision") -> None:
        self._title = title

    def show(self, image: np.ndarray) -> bool:
        cv2.imshow(self._title, image)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
```

- [ ] **Step 4: Create empty `gamevision/overlay/__init__.py`**

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_overlay.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add gamevision/overlay/ tests/test_overlay.py
git commit -m "feat: overlay renderer with boxes, labels, and HUD"
```

---

### Task 5: Detection logger

**Files:**
- Create: `gamevision/datalog/__init__.py` (empty)
- Create: `gamevision/datalog/logger.py`
- Test: `tests/test_logger.py`

**Interfaces:**
- Consumes: `Detection` from Task 1.
- Produces:
  - `DetectionLogger(stream)` with `log(timestamp: float, detections: Sequence[Detection]) -> None`, writing one JSON object per line.

- [ ] **Step 1: Write the failing test** — `tests/test_logger.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.datalog'`

- [ ] **Step 3: Write `gamevision/datalog/logger.py`**

```python
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
```

- [ ] **Step 4: Create empty `gamevision/datalog/__init__.py`**

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_logger.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Commit**

```bash
git add gamevision/datalog/ tests/test_logger.py
git commit -m "feat: JSONL detection logger"
```

---

### Task 6: Pipeline wiring

**Files:**
- Create: `gamevision/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `FrameSource` (Task 2), `OverlayWindow`/`draw_overlay` (Task 4), `DetectionLogger` (Task 5), `Detection`/`Frame` (Task 1).
- Produces:
  - `Detector` Protocol with `detect(frame: Frame) -> list[Detection]` (satisfied by `YoloDetector`).
  - `run_pipeline(source, detector, window=None, logger=None) -> int` — returns frames processed. Stops when the source ends or `window.show()` returns False.

- [ ] **Step 1: Write the failing test** — `tests/test_pipeline.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.pipeline'`

- [ ] **Step 3: Write `gamevision/pipeline.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add gamevision/pipeline.py tests/test_pipeline.py
git commit -m "feat: run_pipeline wiring source -> detector -> overlay/logger"
```

---

### Task 7: Live sources, source factory, and CLI

**Files:**
- Create: `gamevision/sources/screen.py`
- Create: `gamevision/sources/device.py`
- Create: `gamevision/sources/factory.py`
- Create: `gamevision/cli.py`
- Create: `README.md`
- Test: `tests/test_factory.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `ScreenSource(monitor=0, region=None)` — live, `dxcam`-based (Windows).
  - `DeviceSource(index=0)` — live, capture card / webcam via OpenCV.
  - `build_source(args) -> FrameSource` — dispatches on `args.source` in `{"video","screen","device"}`.
  - `main(argv=None) -> int` — CLI entry point (registered as the `gamevision` command).

- [ ] **Step 1: Write `gamevision/sources/screen.py`**

```python
from __future__ import annotations

import time
from typing import Iterator, Optional

from ..types import Frame
from .base import FrameSource


class ScreenSource(FrameSource):
    """Live screen capture via dxcam (Windows). Drops stale frames to stay current."""

    def __init__(self, monitor: int = 0, region: Optional[tuple[int, int, int, int]] = None) -> None:
        import dxcam  # lazy import: only needed for live screen capture

        self._camera = dxcam.create(output_idx=monitor, output_color="BGR")
        if self._camera is None:
            raise RuntimeError(f"Failed to initialize dxcam for monitor {monitor}")
        self._region = region

    def frames(self) -> Iterator[Frame]:
        while True:
            image = self._camera.grab(region=self._region)
            if image is None:
                # No new frame since last grab; yield the CPU briefly.
                time.sleep(0.001)
                continue
            yield Frame(image=image, timestamp=time.perf_counter(), is_live=True)

    def close(self) -> None:
        try:
            self._camera.release()
        except Exception:
            pass
```

- [ ] **Step 2: Write `gamevision/sources/device.py`**

```python
from __future__ import annotations

import time
from typing import Iterator

import cv2

from ..types import Frame
from .base import FrameSource


class DeviceSource(FrameSource):
    """Live capture card / webcam via OpenCV VideoCapture device index."""

    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open capture device index {index}")

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(image=image, timestamp=time.perf_counter(), is_live=True)

    def close(self) -> None:
        self._cap.release()
```

- [ ] **Step 3: Write the failing test** — `tests/test_factory.py`

```python
import argparse

import pytest

from gamevision.sources.factory import build_source
from gamevision.sources.video import VideoFileSource


def _args(**overrides):
    ns = argparse.Namespace(source="video", path=None, index=0, monitor=0, mode="fast")
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_factory_builds_video_source(sample_video):
    src = build_source(_args(source="video", path=sample_video))
    assert isinstance(src, VideoFileSource)
    src.close()


def test_factory_video_requires_path():
    with pytest.raises(ValueError):
        build_source(_args(source="video", path=None))


def test_factory_rejects_unknown_source():
    with pytest.raises(ValueError):
        build_source(_args(source="bogus"))
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gamevision.sources.factory'`

- [ ] **Step 5: Write `gamevision/sources/factory.py`**

```python
from __future__ import annotations

from argparse import Namespace

from .base import FrameSource
from .video import VideoFileSource


def build_source(args: Namespace) -> FrameSource:
    """Construct the FrameSource selected by args.source.

    Live sources are imported lazily so that recorded-video and unit-test
    runs never require dxcam or a camera device.
    """
    kind = args.source
    if kind == "video":
        if not args.path:
            raise ValueError("--source video requires --path")
        return VideoFileSource(args.path, mode=args.mode)
    if kind == "screen":
        from .screen import ScreenSource

        return ScreenSource(monitor=args.monitor)
    if kind == "device":
        from .device import DeviceSource

        return DeviceSource(index=args.index)
    raise ValueError(f"Unknown source: {kind!r}")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_factory.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Write `gamevision/cli.py`**

```python
from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack

from .detection.detector import YoloDetector
from .datalog.logger import DetectionLogger
from .overlay.renderer import OverlayWindow
from .pipeline import run_pipeline
from .sources.factory import build_source


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gamevision",
        description="Real-time single-player game object detection & overlay.",
    )
    p.add_argument("--source", choices=["screen", "video", "device"], default="video")
    p.add_argument("--path", help="video file path (for --source video)")
    p.add_argument("--index", type=int, default=0, help="device index (for --source device)")
    p.add_argument("--monitor", type=int, default=0, help="monitor index (for --source screen)")
    p.add_argument("--mode", choices=["fast", "realtime"], default="realtime",
                   help="video playback pace (video source only)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--track", action="store_true", help="assign stable track IDs")
    p.add_argument("--log", help="write detections to this JSONL file")
    p.add_argument("--no-window", action="store_true", help="run without the overlay window")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with ExitStack() as stack:
        source = build_source(args)
        stack.callback(source.close)

        detector = YoloDetector(model_path=args.model, conf=args.conf, track=args.track)

        window = None
        if not args.no_window:
            window = OverlayWindow()
            stack.callback(window.close)

        logger = None
        if args.log:
            handle = stack.enter_context(open(args.log, "w", encoding="utf-8"))
            logger = DetectionLogger(handle)

        count = run_pipeline(source, detector, window=window, logger=logger)

    print(f"Processed {count} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Write `README.md`**

````markdown
# gamevision

Real-time object detection & overlay for **single-player** games. Reads a
recorded video or a live screen/capture-device feed, runs a YOLO detector, and
shows a mirror window with bounding boxes, labels, and a HUD. Optionally logs
detections to JSONL for offline analysis.

> Scope: single-player, display + analytics only. This tool does not control
> mouse/keyboard or game input, and is not for online multiplayer use.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

**GPU note:** `pip` installs the CPU build of PyTorch by default. For real-time
performance on your NVIDIA GPU, install the CUDA build from
https://pytorch.org/get-started/locally/ before/after the step above, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
# Recorded video (great for development — reproducible, no game needed)
gamevision --source video --path clip.mp4 --mode realtime

# Analyze a whole recording as fast as possible, logging detections
gamevision --source video --path match.mp4 --mode fast --no-window --log run.jsonl

# Live screen capture (Windows)
gamevision --source screen --monitor 0

# Capture card / webcam
gamevision --source device --index 0
```

Press `q` in the overlay window to quit.

## Test

```bash
pytest                       # fast unit tests
pytest -m integration        # runs the real YOLO model (downloads ~6 MB)
```
````

- [ ] **Step 9: Run the full unit test suite**

Run: `pytest -v`
Expected: PASS (all tests; integration tests deselected)

- [ ] **Step 10: Manual end-to-end verification**

1. Obtain any short `.mp4` with people/cars in it (or record a clip).
2. Run: `gamevision --source video --path <clip>.mp4 --mode realtime`
3. Confirm: a window opens, green boxes + labels appear on detected objects, and the HUD shows `objects: N  fps: NN.N`.
4. Run with logging: `gamevision --source video --path <clip>.mp4 --no-window --log run.jsonl`, then confirm `run.jsonl` has one JSON line per frame.
5. (If on Windows with a game/app open) Run: `gamevision --source screen` and confirm the live mirror window tracks your screen.

- [ ] **Step 11: Commit**

```bash
git add gamevision/sources/screen.py gamevision/sources/device.py gamevision/sources/factory.py gamevision/cli.py tests/test_factory.py README.md
git commit -m "feat: live sources, source factory, and CLI entry point"
```

---

## Self-Review

**1. Spec coverage:**
- Swappable recorded + live input → Tasks 2 (video) and 7 (screen/device/factory). ✓
- Real-time YOLO detection, pretrained first → Task 3. ✓
- Tracking (stable IDs) → Task 3 `track=True` (documented deviation from a separate module). ✓
- Mirror-window overlay with boxes + labels + HUD → Task 4. ✓
- Optional JSONL logging → Task 5, wired in Task 6. ✓
- Live vs recorded handling (fast/realtime, timestamps, drop-stale, EOF) → Tasks 2 & 7. ✓
- Error handling (bad file/model/device) → Tasks 2, 3, 7 raise explicit errors. ✓
- Testing strategy (per-module unit tests + end-to-end) → every task; e2e in Tasks 6 & 7. ✓
- Non-goals (single-player, no input automation) → Global Constraints + README. ✓
- Analysis/visualization module and custom-training pipeline are Phase 3 / Phase 2 — intentionally out of this plan's scope.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/vague steps. Every code step shows complete code. ✓

**3. Type consistency:** `Detection`/`Frame` field names, `parse_boxes` signature, `detect()` return type, `run_pipeline` params, and `build_source` dispatch all match across tasks. The `Detector` Protocol in Task 6 matches `YoloDetector.detect` from Task 3. ✓

---

## Notes for the implementer

- Work the tasks in order; each ends green and committed.
- Real-time FPS depends on the CUDA PyTorch build (see README GPU note). On CPU the pipeline still runs correctly, just slower — fine for development against recorded clips.
- `dxcam` is Windows-only and guarded by a platform marker in `pyproject.toml`; the video path needs neither `dxcam` nor a GPU.
