import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from statistical_comparison import bootstrap_ci_mean_diff, compare_metric, load_paired_metric


def test_compare_metric_detects_a_consistent_one_sided_difference():
    ids = [str(i) for i in range(14)]
    unet_vals = np.full(14, 0.9)
    baseline_vals = np.full(14, 0.6)
    # légère variation pour éviter des différences toutes strictement identiques
    unet_vals = unet_vals + np.linspace(0, 0.05, 14)

    result = compare_metric(ids, unet_vals, baseline_vals, "dice")

    assert result["n"] == 14
    assert result["n_positive_diff"] == 14
    assert result["n_negative_diff"] == 0
    assert result["mean_diff"] == pytest.approx(unet_vals.mean() - baseline_vals.mean())
    assert result["wilcoxon_p_value"] < 0.05


def test_bootstrap_ci_mean_diff_brackets_the_observed_mean():
    diffs = np.array([0.2, 0.25, 0.3, 0.22, 0.28, 0.26, 0.24])
    lo, hi = bootstrap_ci_mean_diff(diffs, n_bootstrap=2000, seed=0)
    assert lo <= diffs.mean() <= hi


def test_load_paired_metric_raises_on_mismatched_ids(tmp_path):
    unet_path = tmp_path / "unet.json"
    baseline_path = tmp_path / "baseline.json"
    unet_path.write_text(json.dumps({"per_image": [{"id": "1", "dice": 0.9}, {"id": "2", "dice": 0.8}]}))
    baseline_path.write_text(json.dumps({"per_image": [{"id": "1", "dice": 0.5}]}))

    with pytest.raises(ValueError):
        load_paired_metric(str(unet_path), str(baseline_path), metric="dice")


def test_load_paired_metric_matches_by_id_regardless_of_order(tmp_path):
    unet_path = tmp_path / "unet.json"
    baseline_path = tmp_path / "baseline.json"
    unet_path.write_text(json.dumps({"per_image": [{"id": "2", "dice": 0.8}, {"id": "1", "dice": 0.9}]}))
    baseline_path.write_text(json.dumps({"per_image": [{"id": "1", "dice": 0.4}, {"id": "2", "dice": 0.5}]}))

    ids, unet_vals, baseline_vals = load_paired_metric(str(unet_path), str(baseline_path), metric="dice")

    assert ids == ["1", "2"]
    assert list(unet_vals) == pytest.approx([0.9, 0.8])
    assert list(baseline_vals) == pytest.approx([0.4, 0.5])
