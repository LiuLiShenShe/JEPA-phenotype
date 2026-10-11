"""Equal-plant statistics and clustered uncertainty, not window bootstrap."""

import numpy as np
import pandas as pd
import pytest

from training.metrics import cluster_interval, evaluate, plant_scores


def test_plant_macro_mae_is_not_window_weighted():
    rows = pd.DataFrame({"plant_uid":["short","long","long","long"],"genotype":["G1"]*4})
    targets = np.zeros((4,2))
    predicted = np.array([[4.,.4],[1.,.1],[1.,.1],[1.,.1]])
    metrics,plants = evaluate(rows,targets,predicted,bootstrap_replicates=300)
    assert metrics["leaf_count"]["window_mae"]==1.75
    assert metrics["leaf_count"]["plant_macro_mae"]==2.5
    assert metrics["leaf_count"]["plant_macro_rmse"]==2.5
    assert metrics["leaf_count"]["window_rmse"]==pytest.approx(np.sqrt(19/4))
    assert metrics["bootstrap_unit"]=="plant_uid"
    assert len(plants)==4


def test_cluster_interval_is_deterministic():
    first = cluster_interval([1,2,4,8],seed=42,replicates=500)
    assert first==cluster_interval([1,2,4,8],seed=42,replicates=500)
    assert first[0] <= np.mean([1,2,4,8]) <= first[1]


def test_genotype_macro_uses_plants():
    rows = pd.DataFrame({"plant_uid":["a","a","b","c"],"genotype":["g1","g1","g1","g2"]})
    metrics,_ = evaluate(rows,np.zeros((4,2)),np.array([[2,2],[2,2],[6,6],[9,9]]))
    assert metrics["genotypes"]["g1"]["leaf_count"]["plant_macro_mae"]==4
    assert metrics["genotypes"]["g2"]["leaf_count"]["plants"]==1


def test_paired_difference_uses_matching_independent_plants():
    plants = pd.DataFrame({"plant_uid":["a","b","c"],"mlp":[1.,2.,3.],"gru":[.5,1.5,2.5]})
    assert cluster_interval(plants.gru-plants.mlp)==[-.5,-.5]
