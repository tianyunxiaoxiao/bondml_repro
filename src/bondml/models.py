"""Model factories for the paper's machine-learning methods."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


MODEL_NAMES = [
    "OLS",
    "PLS",
    "PCR",
    "LASSO",
    "RIDGE",
    "ENet",
    "GLM",
    "RF",
    "GBDT",
    "NN1",
    "NN2",
    "NN3",
    "NN4",
    "NN5",
]


@dataclass(frozen=True)
class CandidateModel:
    """A named estimator candidate used in recursive validation."""

    label: str
    estimator: RegressorMixin


def model_candidates(model_name: str, n_features: int, random_state: int) -> list[CandidateModel]:
    """Return compact validation candidates for a model family."""

    if model_name == "OLS":
        return [CandidateModel("ols", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]))]

    if model_name == "PLS":
        components = _component_grid(n_features)
        return [
            CandidateModel(
                f"n_components={k}",
                Pipeline([("scaler", StandardScaler()), ("model", PLSRegression(n_components=k))]),
            )
            for k in components
        ]

    if model_name == "PCR":
        components = _component_grid(n_features)
        return [
            CandidateModel(
                f"n_components={k}",
                Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=k)), ("model", LinearRegression())]),
            )
            for k in components
        ]

    if model_name == "LASSO":
        return [
            CandidateModel(
                f"alpha={alpha}",
                Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=10_000, random_state=random_state))]),
            )
            for alpha in (0.001, 0.01)
        ]

    if model_name == "RIDGE":
        return [
            CandidateModel(
                f"alpha={alpha}",
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))]),
            )
            for alpha in (1.0, 10.0)
        ]

    if model_name == "ENet":
        return [
            CandidateModel(
                f"alpha={alpha},l1_ratio={ratio}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=10_000, random_state=random_state)),
                    ]
                ),
            )
            for alpha in (0.001, 0.01)
            for ratio in (0.35, 0.7)
        ]

    if model_name == "GLM":
        return [
            CandidateModel(
                f"knots={knots},alpha={alpha}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("spline", SplineTransformer(n_knots=knots, degree=3, include_bias=False)),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
            for knots in (3,)
            for alpha in (1.0, 10.0)
        ]

    if model_name == "RF":
        return [
            CandidateModel(
                f"trees={trees},depth={depth}",
                RandomForestRegressor(
                    n_estimators=trees,
                    max_depth=depth,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
            for trees in (60,)
            for depth in (5,)
        ]

    if model_name == "GBDT":
        return [
            CandidateModel(
                f"lr={learning_rate},depth={depth}",
                HistGradientBoostingRegressor(
                    max_iter=90,
                    learning_rate=learning_rate,
                    max_depth=depth,
                    min_samples_leaf=10,
                    random_state=random_state,
                ),
            )
            for learning_rate in (0.08,)
            for depth in (3,)
        ]

    if model_name.startswith("NN"):
        layers = int(model_name[-1])
        width = 8
        hidden = tuple(width for _ in range(layers))
        return [
            CandidateModel(
                f"layers={layers},alpha={alpha}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=hidden,
                                activation="relu",
                                solver="adam",
                                alpha=alpha,
                                learning_rate_init=0.003,
                                max_iter=180,
                                early_stopping=True,
                                n_iter_no_change=12,
                                tol=1e-3,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
            for alpha in (0.0005,)
        ]

    raise ValueError(f"Unknown model name: {model_name}")


def _component_grid(n_features: int) -> list[int]:
    max_components = max(1, min(n_features, 13))
    raw = [1, 4, 8]
    return sorted({k for k in raw if k <= max_components})
