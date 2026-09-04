import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import confusion_counts, dice_score, iou_score


def test_confusion_counts_perfect_match():
    pred = torch.tensor([[[True, True], [False, False]]])
    target = pred.clone()
    tp, fp, fn, tn = confusion_counts(pred, target)
    assert tp.item() == 2
    assert fp.item() == 0
    assert fn.item() == 0
    assert tn.item() == 2


def test_dice_and_iou_perfect_match_are_one():
    tp, fp, fn = torch.tensor(10.0), torch.tensor(0.0), torch.tensor(0.0)
    assert dice_score(tp, fp, fn).item() == pytest.approx(1.0)
    assert iou_score(tp, fp, fn).item() == pytest.approx(1.0)


def test_dice_and_iou_known_values():
    # 3 vrais positifs, 1 faux positif, 1 faux négatif
    tp, fp, fn = torch.tensor(3.0), torch.tensor(1.0), torch.tensor(1.0)
    assert iou_score(tp, fp, fn).item() == pytest.approx(3 / 5)
    assert dice_score(tp, fp, fn).item() == pytest.approx(6 / 8)
