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
