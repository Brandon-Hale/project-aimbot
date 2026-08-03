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
