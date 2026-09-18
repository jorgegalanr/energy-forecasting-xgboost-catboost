from pathlib import Path

import pandas as pd

from src.data import load_energy_data


def test_loader_removes_duplicates_and_fills_hourly_gap(tmp_path: Path):
    source = pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 03:00:00",
            ],
            "Energy": [10.0, 12.0, 99.0, 16.0],
        }
    )
    path = tmp_path / "energy.csv"
    source.to_csv(path, index=False)

    frame, report = load_energy_data(path)

    assert report.duplicate_timestamps == 1
    assert report.inserted_hours == 1
    assert frame.loc["2024-01-01 01:00:00", "Energy"] == 12.0
    assert frame.loc["2024-01-01 02:00:00", "Energy"] == 14.0


def test_repository_dataset_profile():
    frame, report = load_energy_data("data/energy_train.csv")

    assert report.source_rows == 124_870
    assert report.duplicate_timestamps == 2
    assert report.inserted_hours == 28
    assert report.start == pd.Timestamp("2002-04-01 01:00:00")
    assert report.end == pd.Timestamp("2016-06-30 00:00:00")
    assert len(frame) == 124_896
