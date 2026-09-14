"""Evaluation metrics for return prediction."""

from __future__ import annotations

import numpy as np


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R2 against the paper's zero-excess-return benchmark."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum(y_true**2)
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared prediction error."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
