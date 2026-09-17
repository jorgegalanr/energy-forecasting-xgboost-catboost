from __future__ import annotations

import numpy as np
import pandas as pd


LAGS = (1, 2, 24, 48, 168)
ROLLING_WINDOWS = (24, 168)


def build_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Crea variables disponibles justo antes de predecir cada hora.

    Las estadísticas móviles se calculan sobre ``Energy.shift(1)`` para que el
    valor objetivo de la hora t nunca participe en sus propias variables.
    """
    if "Energy" not in frame.columns:
        raise ValueError("Se requiere la columna Energy")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser DatetimeIndex")

    target = frame["Energy"].astype(float)
    features = pd.DataFrame(index=frame.index)

    for lag in LAGS:
        features[f"lag_{lag}"] = target.shift(lag)

    known_history = target.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = known_history.rolling(window=window, min_periods=window)
        features[f"rolling_mean_{window}"] = rolling.mean()
        features[f"rolling_std_{window}"] = rolling.std()

    hour = frame.index.hour.to_numpy()
    day_of_week = frame.index.dayofweek.to_numpy()
    day_of_year = frame.index.dayofyear.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["target"] = target

    return features.dropna()

