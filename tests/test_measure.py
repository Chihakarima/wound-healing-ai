import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from measure import area_px_to_cm2, hausdorff_distance_px, wound_area_px


def test_wound_area_px_counts_nonzero_pixels():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255  # 3x3 = 9 pixels
    assert wound_area_px(mask) == 9


def test_wound_area_px_empty_mask_is_zero():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert wound_area_px(mask) == 0


def test_area_px_to_cm2_conversion():
    # 100 px de côté pour 1 cm -> 1 cm^2 = 100*100 = 10000 px^2
    assert area_px_to_cm2(10000, pixels_per_cm=100) == pytest.approx(1.0)


def _square_mask(size=20, box=(5, 15, 5, 15)):
    mask = np.zeros((size, size), dtype=np.uint8)
    y0, y1, x0, x1 = box
    mask[y0:y1, x0:x1] = 1
    return mask


def test_hausdorff_distance_zero_for_identical_masks():
    mask = _square_mask()
    assert hausdorff_distance_px(mask, mask.copy()) == pytest.approx(0.0)


def test_hausdorff_distance_detects_a_shifted_boundary():
    a = _square_mask(box=(5, 15, 5, 15))
    b = _square_mask(box=(5, 15, 5, 16))  # 1 px de plus sur la droite
    assert hausdorff_distance_px(a, b) == pytest.approx(1.0)


def test_hausdorff_distance_none_when_a_mask_is_empty():
    empty = np.zeros((20, 20), dtype=np.uint8)
    mask = _square_mask()
    assert hausdorff_distance_px(empty, mask) is None
