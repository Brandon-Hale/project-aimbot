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
