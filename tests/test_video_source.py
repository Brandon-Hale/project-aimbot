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
