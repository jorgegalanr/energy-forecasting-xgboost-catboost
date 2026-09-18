import numpy as np
import pandas as pd

from src.recursive import feature_row, recursive_forecast, recursive_naive


class IncrementPreviousValue:
    def predict(self, row: pd.DataFrame) -> np.ndarray:
        return np.array([row.iloc[0]["lag_1"] + 1])


def test_recursive_forecast_reuses_its_own_predictions():
    history_index = pd.date_range("2024-01-01", periods=200, freq="h")
    history = pd.Series(np.arange(200, dtype=float), index=history_index)
    future = pd.date_range(history_index[-1] + pd.Timedelta(hours=1), periods=3, freq="h")

    predictions = recursive_forecast(IncrementPreviousValue(), history, future)

    np.testing.assert_array_equal(predictions, np.array([200.0, 201.0, 202.0]))


def test_feature_row_matches_causal_history():
    history = list(np.arange(1, 201, dtype=float))
    row = feature_row(pd.Timestamp("2024-02-01 12:00:00"), history)

    assert row.iloc[0]["lag_1"] == 200.0
    assert row.iloc[0]["lag_168"] == 33.0
    assert row.iloc[0]["rolling_mean_24"] == np.mean(history[-24:])


def test_recursive_weekly_naive_repeats_last_week():
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    history = pd.Series(np.arange(200, dtype=float), index=index)
    predictions = recursive_naive(history, steps=170, lag=168)

    assert predictions[0] == history.iloc[-168]
    assert predictions[168] == predictions[0]
