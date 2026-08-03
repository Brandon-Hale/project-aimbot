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
