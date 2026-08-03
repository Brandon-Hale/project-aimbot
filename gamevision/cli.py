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
