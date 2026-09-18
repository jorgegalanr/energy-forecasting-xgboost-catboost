from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import LAGS, ROLLING_WINDOWS


def feature_row(timestamp: pd.Timestamp, history: list[float]) -> pd.DataFrame:
    """Construye las variables de una hora usando solo el historial disponible."""
    minimum_history = max(max(LAGS), max(ROLLING_WINDOWS))
    if len(history) < minimum_history:
        raise ValueError(f"Se requieren al menos {minimum_history} observaciones")

    values: dict[str, float] = {}
    for lag in LAGS:
        values[f"lag_{lag}"] = float(history[-lag])
    for window in ROLLING_WINDOWS:
        sample = np.asarray(history[-window:], dtype=float)
        values[f"rolling_mean_{window}"] = float(sample.mean())
        values[f"rolling_std_{window}"] = float(sample.std(ddof=1))

    values["hour_sin"] = float(np.sin(2 * np.pi * timestamp.hour / 24))
    values["hour_cos"] = float(np.cos(2 * np.pi * timestamp.hour / 24))
    values["dow_sin"] = float(np.sin(2 * np.pi * timestamp.dayofweek / 7))
    values["dow_cos"] = float(np.cos(2 * np.pi * timestamp.dayofweek / 7))
    values["year_sin"] = float(np.sin(2 * np.pi * timestamp.dayofyear / 365.25))
    values["year_cos"] = float(np.cos(2 * np.pi * timestamp.dayofyear / 365.25))
    return pd.DataFrame([values], index=[timestamp])


def recursive_forecast(model, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    """Genera un forecast multi-step sin consultar objetivos del periodo futuro."""
    if history.empty or future_index.empty:
        raise ValueError("El historial y el horizonte futuro no pueden estar vacíos")
    if history.index.max() >= future_index.min():
        raise ValueError("El historial debe terminar antes del horizonte futuro")

    values = history.astype(float).tolist()
    predictions: list[float] = []
    for timestamp in future_index:
        row = feature_row(timestamp, values)
        prediction = float(np.asarray(model.predict(row)).reshape(-1)[0])
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions)


def recursive_naive(history: pd.Series, steps: int, lag: int) -> np.ndarray:
    """Extiende recursivamente una referencia basada en un retardo."""
    if lag < 1 or len(history) < lag:
        raise ValueError("El lag debe ser positivo y existir en el historial")
    values = history.astype(float).tolist()
    predictions: list[float] = []
    for _ in range(steps):
        prediction = float(values[-lag])
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions)

