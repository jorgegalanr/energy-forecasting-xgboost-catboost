from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def split_at(features: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundary = pd.Timestamp(test_start)
    train = features.loc[features.index < boundary].copy()
    test = features.loc[features.index >= boundary].copy()
    if train.empty or test.empty:
        raise ValueError("La fecha de corte debe dejar observaciones en train y test")
    if train.index.max() >= test.index.min():
        raise AssertionError("La división temporal no es cronológica")
    return train, test


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(true - pred)
    denominator = np.abs(true).sum()
    wape = float(absolute_error.sum() / denominator * 100) if denominator else np.nan
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "WAPE_pct": wape,
    }


def naive_from_lag(test: pd.DataFrame, lag: int) -> np.ndarray:
    """Devuelve una referencia ingenua basada en un valor pasado observado."""
    column = f"lag_{lag}"
    if column not in test.columns:
        raise ValueError(f"No existe la variable {column}")
    return test[column].to_numpy()
