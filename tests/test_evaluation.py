import numpy as np
import pandas as pd

from src.evaluation import regression_metrics, split_at


def test_temporal_split_has_no_overlap():
    index = pd.date_range("2024-01-01", periods=72, freq="h")
    features = pd.DataFrame({"target": np.arange(72)}, index=index)
    train, test = split_at(features, "2024-01-03")

    assert train.index.max() < test.index.min()
    assert len(train) == 48
    assert len(test) == 24


def test_metrics_are_reproducible():
    metrics = regression_metrics(np.array([100, 200]), np.array([90, 220]))
    assert metrics["MAE"] == 15.0
    assert metrics["WAPE_pct"] == 10.0

