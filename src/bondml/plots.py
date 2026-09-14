"""Plotting utilities for paper-style synthetic results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bondml.config import PREDICTOR_GROUPS


GROUP_COLORS = {
    "benchmark": "#e9c46a",
    "eem_determining": "#f4a261",
    "eem_resulting": "#d62828",
}


def plot_realized_vs_eem(panel: pd.DataFrame, path: Path) -> None:
    """Save delta-binned realized and EEM return panels."""

    buckets = [
        ("All", panel),
        ("<= 30 days", panel[panel["option_maturity"] <= 30]),
        ("60 days", panel[panel["option_maturity"] == 60]),
        ("180 days", panel[panel["option_maturity"] == 180]),
        ("365 days", panel[panel["option_maturity"] == 365]),
        ("730 days", panel[panel["option_maturity"] == 730]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    bins = np.linspace(-0.95, 0.95, 20)

    for ax, (title, sample) in zip(axes.ravel(), buckets, strict=True):
        if sample.empty:
            ax.set_title(title)
            continue
        binned = sample.copy()
        binned["delta_bin"] = pd.cut(binned["option_delta"], bins=bins, include_lowest=True)
        grouped = (
            binned.groupby("delta_bin", observed=True)
            .agg(
                delta=("option_delta", "mean"),
                realized=("realized_excess_return", "mean"),
                realized_std=("realized_excess_return", "std"),
                eem=("EEM_ret", "mean"),
            )
            .dropna()
        )
        ax.plot(grouped["delta"], grouped["realized"], color="#2f55d4", label="Realized return")
        ax.fill_between(
            grouped["delta"],
            grouped["realized"] - grouped["realized_std"],
            grouped["realized"] + grouped["realized_std"],
            color="#8da0ff",
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(grouped["delta"], grouped["eem"], color="#2a9d55", linestyle="--", label="EEM return")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Delta")
        ax.set_ylabel("Return")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_yearly_r2(yearly: pd.DataFrame, path: Path, scenario: str = "eem_resulting") -> None:
    """Save yearly out-of-sample R2 patterns."""

    sample = yearly[yearly["scenario"] == scenario]
    pivot = sample.pivot_table(index="test_year", columns="model", values="oos_r2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in pivot.columns:
        ax.plot(pivot.index, pivot[column] * 100.0, linewidth=1.4, label=column)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title(f"Yearly out-of-sample R2: {scenario}")
    ax.set_xlabel("Test year")
    ax.set_ylabel("R2 (%)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_importance_by_model(importance: pd.DataFrame, path: Path) -> None:
    """Save normalized predictor importance bars by model."""

    models = list(dict.fromkeys(importance["model"]))
    n_cols = 3
    n_rows = int(np.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(7, 2.7 * n_rows)))
    axes_flat = np.asarray(axes).ravel()
    color_map = _feature_color_map()

    for ax, model in zip(axes_flat, models, strict=False):
        sample = importance[importance["model"] == model].sort_values("importance", ascending=True).tail(12)
        colors = [color_map.get(feature, "#999999") for feature in sample["feature"]]
        ax.barh(sample["feature"], sample["importance"], color=colors)
        ax.set_title(model)
        ax.set_xlabel("Normalized importance")
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes_flat[len(models) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_importance_ranks(ranks: pd.DataFrame, path: Path) -> None:
    """Save heatmap of model-specific predictor ranks."""

    ordered_features = ranks.sort_values("total_rank", ascending=False)["feature"].drop_duplicates().tolist()
    pivot = ranks.pivot_table(index="feature", columns="model", values="rank", aggfunc="mean").reindex(ordered_features)
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrBr")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Predictor importance ranks")
    fig.colorbar(image, ax=ax, label="Rank within model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _feature_color_map() -> dict[str, str]:
    colors = {}
    for group, features in PREDICTOR_GROUPS.items():
        for feature in features:
            colors[feature] = GROUP_COLORS[group]
    return colors
