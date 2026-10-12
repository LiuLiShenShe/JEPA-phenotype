import numpy as np
import pandas as pd
import pytest

from m3.metrics import TARGETS, bootstrap, paired_difference, summarize
from evaluate_m3 import oracle_predictions
from test_m3_dataset import fixture_project


def prediction_fixture():
    records = []
    for model in ("B3-2", "B3-3"):
        for plant, count, error in [("a", 1, 1), ("b", 9, 3)]:
            for index in range(count):
                record = {"model": model, "seed": 17, "horizon": 2, "plant_uid": plant, "genotype": "G1"}
                for target in TARGETS:
                    record[f"true_{target}"] = 10
                    record[f"pred_{target}"] = 10 + error + (model == "B3-2")
                records.append(record)
    return pd.DataFrame(records)


def test_macro_equal_plants_not_overlapping_windows():
    summary, plants = summarize(prediction_fixture(), replicates=100)
    group = summary.loc[summary.model.eq("B3-3")]
    assert np.allclose(group.plant_macro_mae, 2)
    assert len(plants) == 4
    pairs = paired_difference(prediction_fixture(), "B3-3", "B3-2", "plant_height", replicates=100)
    assert pairs.left_minus_right_mae.iloc[0] == -1
    assert pairs.ci95_low.iloc[0] == pairs.ci95_high.iloc[0] == -1


def test_bootstrap_reproducible_and_rejects_nonfinite():
    assert bootstrap([1, 2, 3], replicates=100) == bootstrap([1, 2, 3], replicates=100)
    with pytest.raises(ValueError):
        bootstrap([1, np.nan])


def test_growth_trend_uses_all_three_actual_das(tmp_path):
    metadata, windows, cache = fixture_project(tmp_path)
    predictions = oracle_predictions(windows)
    trend = predictions.loc[predictions.model.eq("B3-1")]
    np.testing.assert_allclose(trend.pred_plant_height, trend.true_plant_height)
    np.testing.assert_allclose(trend.pred_plant_width, trend.true_plant_width)
