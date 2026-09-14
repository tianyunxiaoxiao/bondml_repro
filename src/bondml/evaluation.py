"""Recursive training, validation, testing, and predictor importance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from bondml.config import BENCHMARK_PREDICTORS, DETERMINING_FEATURE_SET, FULL_FEATURE_SET
from bondml.metrics import mse, oos_r2
from bondml.models import MODEL_NAMES, model_candidates


FEATURE_SCENARIOS = {
    "benchmark": BENCHMARK_PREDICTORS,
    "eem_determining": DETERMINING_FEATURE_SET,
    "eem_resulting": FULL_FEATURE_SET,
}


@dataclass(frozen=True)
class RecursiveConfig:
    """Configuration for recursive paper-style evaluation."""

    first_test_year: int = 2003
    last_test_year: int = 2021
    random_state: int = 20260306
    model_names: tuple[str, ...] = tuple(MODEL_NAMES)
    scenarios: tuple[str, ...] = ("benchmark", "eem_determining", "eem_resulting")


def run_recursive_evaluation(panel: pd.DataFrame, config: RecursiveConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window validation and out-of-sample testing."""

    cfg = config or RecursiveConfig()
    yearly_rows: list[dict[str, object]] = []

    for scenario in cfg.scenarios:
        features = FEATURE_SCENARIOS[scenario]
        for model_name in cfg.model_names:
            for test_year in range(cfg.first_test_year, cfg.last_test_year + 1):
                split = _split_panel(panel, test_year)
                if split is None:
                    continue
                train, valid, test = split
                X_train, y_train = train[features], train["realized_excess_return"]
                X_valid, y_valid = valid[features], valid["realized_excess_return"]
                X_test, y_test = test[features], test["realized_excess_return"]

                candidate, validation_r2 = _select_candidate(model_name, X_train, y_train, X_valid, y_valid, cfg.random_state + test_year)
                final_model = clone(candidate.estimator)
                train_valid = pd.concat([train, valid], ignore_index=True)
                final_model.fit(train_valid[features], train_valid["realized_excess_return"])
                prediction = _predict(final_model, X_test)

                yearly_rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "test_year": test_year,
                        "selected_candidate": candidate.label,
                        "validation_r2": validation_r2,
                        "oos_r2": oos_r2(y_test.to_numpy(), prediction),
                        "mse": mse(y_test.to_numpy(), prediction),
                        "n_train": len(train),
                        "n_validation": len(valid),
                        "n_test": len(test),
                    }
                )

    yearly = pd.DataFrame(yearly_rows)
    performance = (
        yearly.groupby(["scenario", "model"], as_index=False)
        .agg(
            mean_oos_r2=("oos_r2", "mean"),
            mean_mse=("mse", "mean"),
            median_validation_r2=("validation_r2", "median"),
            years=("test_year", "nunique"),
        )
        .sort_values(["scenario", "mean_oos_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return performance, yearly


def compute_predictor_importance(
    panel: pd.DataFrame,
    config: RecursiveConfig | None = None,
    scenario: str = "eem_resulting",
) -> pd.DataFrame:
    """Compute predictor importance by zeroing each feature in test samples."""

    cfg = config or RecursiveConfig()
    features = FEATURE_SCENARIOS[scenario]
    rows: list[dict[str, object]] = []

    for model_name in cfg.model_names:
        yearly_importances: list[dict[str, float]] = []
        for test_year in range(cfg.first_test_year, cfg.last_test_year + 1):
            split = _split_panel(panel, test_year)
            if split is None:
                continue
            train, valid, test = split
            X_train, y_train = train[features], train["realized_excess_return"]
            X_valid, y_valid = valid[features], valid["realized_excess_return"]
            X_test, y_test = test[features], test["realized_excess_return"].to_numpy()

            candidate, _ = _select_candidate(model_name, X_train, y_train, X_valid, y_valid, cfg.random_state + 10_000 + test_year)
            train_valid = pd.concat([train, valid], ignore_index=True)
            model = clone(candidate.estimator)
            model.fit(train_valid[features], train_valid["realized_excess_return"])
            baseline = oos_r2(y_test, _predict(model, X_test))

            drops = {}
            for feature in features:
                zeroed = X_test.copy()
                zeroed[feature] = 0.0
                drops[feature] = baseline - oos_r2(y_test, _predict(model, zeroed))
            yearly_importances.append(drops)

        if not yearly_importances:
            continue
        averaged = pd.DataFrame(yearly_importances).mean().clip(lower=0.0)
        total = averaged.sum()
        normalized = averaged / total if total > 0 else averaged
        for feature, importance in normalized.items():
            rows.append({"scenario": scenario, "model": model_name, "feature": feature, "importance": float(importance)})

    return pd.DataFrame(rows)


def rank_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Rank predictors by total model contribution."""

    ranks = importance.copy()
    ranks["rank"] = ranks.groupby("model")["importance"].rank(method="average", ascending=True)
    totals = ranks.groupby("feature", as_index=False)["rank"].sum().rename(columns={"rank": "total_rank"})
    return ranks.merge(totals, on="feature").sort_values(["total_rank", "feature"], ascending=[False, True]).reset_index(drop=True)


def _split_panel(panel: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    train = panel[panel["year"] <= test_year - 3]
    valid = panel[(panel["year"] >= test_year - 2) & (panel["year"] <= test_year - 1)]
    test = panel[panel["year"] == test_year]
    if train.empty or valid.empty or test.empty:
        return None
    return train, valid, test


def _select_candidate(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, random_state: int):
    candidates = model_candidates(model_name, X_train.shape[1], random_state)
    best_candidate = candidates[0]
    best_score = -np.inf
    for candidate in candidates:
        model = clone(candidate.estimator)
        model.fit(X_train, y_train)
        score = oos_r2(y_valid.to_numpy(), _predict(model, X_valid))
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate, float(best_score)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float).ravel()
