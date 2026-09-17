from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_energy_data
from src.evaluation import naive_from_lag, regression_metrics, split_at
from src.features import build_causal_features
from src.modeling import build_catboost, build_xgboost


DEFAULT_DATA = Path("data/energy_train.csv")
DEFAULT_OUTPUT = Path("reports/generated")
TEST_START = "2016-01-01 00:00:00"


def run(data_path: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, quality = load_energy_data(data_path)
    features = build_causal_features(frame)
    train, test = split_at(features, TEST_START)

    feature_columns = [column for column in features.columns if column != "target"]
    x_train, y_train = train[feature_columns], train["target"]
    x_test, y_test = test[feature_columns], test["target"]

    predictions: dict[str, object] = {
        "Persistence (1h)": naive_from_lag(test, 1),
        "Seasonal naive (168h)": naive_from_lag(test, 168),
    }
    models = {"XGBoost": build_xgboost(), "CatBoost": build_catboost()}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_test)

    rows = []
    prediction_frame = pd.DataFrame({"actual": y_test}, index=test.index)
    for name, values in predictions.items():
        prediction_frame[name] = values
        rows.append({"model": name, **regression_metrics(y_test, values)})

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index_label="Datetime")

    last_week = prediction_frame.tail(24 * 7)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(last_week.index, last_week["actual"], label="Real", linewidth=2)
    ax.plot(last_week.index, last_week[metrics.loc[0, "model"]], label=metrics.loc[0, "model"])
    ax.set(title="Forecast una hora por delante — última semana de test", ylabel="Consumo")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "forecast_last_week.png", dpi=160)
    plt.close(fig)

    importance = pd.Series(
        models["XGBoost"].feature_importances_, index=feature_columns
    ).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot.barh(ax=ax)
    ax.set(title="Importancia de variables — XGBoost", xlabel="Importancia")
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_feature_importance.png", dpi=160)
    plt.close(fig)

    print(
        f"Datos: {quality.start} — {quality.end} | "
        f"duplicados={quality.duplicate_timestamps} | huecos={quality.inserted_hours}"
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecasting horario reproducible")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data, arguments.output)
