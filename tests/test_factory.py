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
