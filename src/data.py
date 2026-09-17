from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    source_rows: int
    duplicate_timestamps: int
    inserted_hours: int
    start: pd.Timestamp
    end: pd.Timestamp


def load_energy_data(path: str | Path) -> tuple[pd.DataFrame, DataQualityReport]:
    """Carga la serie, elimina duplicados y completa huecos horarios internos."""
    frame = pd.read_csv(path)
    required = {"Datetime", "Energy"}
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise ValueError(f"Columnas obligatorias ausentes: {missing}")

    source_rows = len(frame)
    frame = frame.loc[:, ["Datetime", "Energy"]].copy()
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="raise")
    frame["Energy"] = pd.to_numeric(frame["Energy"], errors="raise")
    frame = frame.sort_values("Datetime").set_index("Datetime")

    duplicate_timestamps = int(frame.index.duplicated(keep="first").sum())
    frame = frame.loc[~frame.index.duplicated(keep="first")]

    complete_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    inserted_hours = len(complete_index.difference(frame.index))
    frame = frame.reindex(complete_index)
    frame.index.name = "Datetime"
    frame["Energy"] = frame["Energy"].interpolate(method="time", limit_area="inside")

    if frame["Energy"].isna().any():
        raise ValueError("Quedan valores ausentes después de completar la serie")
    if (frame["Energy"] <= 0).any():
        raise ValueError("El consumo debe ser positivo")

    report = DataQualityReport(
        source_rows=source_rows,
        duplicate_timestamps=duplicate_timestamps,
        inserted_hours=inserted_hours,
        start=frame.index.min(),
        end=frame.index.max(),
    )
    return frame, report

