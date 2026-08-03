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
