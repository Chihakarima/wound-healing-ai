import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from measure import area_px_to_cm2, wound_area_px


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
