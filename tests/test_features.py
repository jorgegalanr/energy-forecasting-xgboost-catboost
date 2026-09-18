import numpy as np
import pandas as pd

from src.features import build_causal_features


def test_rolling_features_exclude_current_target():
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    frame = pd.DataFrame({"Energy": np.arange(1, 201, dtype=float)}, index=index)

    features = build_causal_features(frame)
    timestamp = features.index[0]
    position = frame.index.get_loc(timestamp)

    expected_mean = frame["Energy"].iloc[position - 24 : position].mean()
    assert features.loc[timestamp, "rolling_mean_24"] == expected_mean
    assert features.loc[timestamp, "lag_1"] == frame["Energy"].iloc[position - 1]
    assert features.loc[timestamp, "target"] == frame["Energy"].iloc[position]

