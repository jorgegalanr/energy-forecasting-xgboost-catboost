from __future__ import annotations

from catboost import CatBoostRegressor
from xgboost import XGBRegressor


def build_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def build_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=700,
        learning_rate=0.06,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )

