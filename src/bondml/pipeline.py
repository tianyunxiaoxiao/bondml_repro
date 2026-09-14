"""End-to-end synthetic reproduction pipeline."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from bondml.data import SyntheticDataConfig, generate_synthetic_panel, summarize_panel, validate_panel
from bondml.evaluation import RecursiveConfig, compute_predictor_importance, rank_importance, run_recursive_evaluation
from bondml.models import MODEL_NAMES
from bondml.plots import plot_importance_by_model, plot_importance_ranks, plot_realized_vs_eem, plot_yearly_r2


def run_pipeline(
    output: Path,
    rows_per_year: int = 120,
    seed: int = 20260306,
    model_names: tuple[str, ...] = tuple(MODEL_NAMES),
    scenarios: tuple[str, ...] = ("benchmark", "eem_determining", "eem_resulting"),
) -> dict[str, Path]:
    """Generate synthetic data, fit models, and write final artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    panel = generate_synthetic_panel(SyntheticDataConfig(rows_per_year=rows_per_year, seed=seed))
    validate_panel(panel)

    recursive_config = RecursiveConfig(model_names=model_names, scenarios=scenarios, random_state=seed)
    performance, yearly = run_recursive_evaluation(panel, recursive_config)
    importance = compute_predictor_importance(
        panel,
        RecursiveConfig(model_names=model_names, scenarios=("eem_resulting",), random_state=seed),
    )
    ranks = rank_importance(importance)
    summary = summarize_panel(panel)

    artifacts = {
        "panel": output / "synthetic_panel.csv",
        "summary": output / "summary_statistics.csv",
        "performance": output / "model_performance.csv",
        "yearly": output / "yearly_performance.csv",
        "importance": output / "predictor_importance.csv",
        "ranks": output / "importance_ranks.csv",
        "fig_realized_vs_eem": output / "fig_realized_vs_eem.png",
        "fig_yearly_r2": output / "fig_yearly_r2.png",
        "fig_importance_by_model": output / "fig_importance_by_model.png",
        "fig_importance_ranks": output / "fig_importance_ranks.png",
    }

    panel.to_csv(artifacts["panel"], index=False)
    summary.to_csv(artifacts["summary"], index=False)
    performance.to_csv(artifacts["performance"], index=False)
    yearly.to_csv(artifacts["yearly"], index=False)
    importance.to_csv(artifacts["importance"], index=False)
    ranks.to_csv(artifacts["ranks"], index=False)

    plot_realized_vs_eem(panel, artifacts["fig_realized_vs_eem"])
    plot_yearly_r2(yearly, artifacts["fig_yearly_r2"])
    plot_importance_by_model(importance, artifacts["fig_importance_by_model"])
    plot_importance_ranks(ranks, artifacts["fig_importance_ranks"])

    return artifacts


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    parser = argparse.ArgumentParser(description="Run the synthetic bondML reproduction workflow.")
    parser.add_argument("--output", type=Path, default=Path("outputs/run_default"))
    parser.add_argument("--rows-per-year", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--models", type=str, default=",".join(MODEL_NAMES), help="Comma-separated model names.")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="benchmark,eem_determining,eem_resulting",
        help="Comma-separated feature scenarios.",
    )
    args = parser.parse_args()

    model_names = tuple(item.strip() for item in args.models.split(",") if item.strip())
    scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    artifacts = run_pipeline(args.output, args.rows_per_year, args.seed, model_names, scenarios)

    print(f"Wrote {len(artifacts)} artifacts to {args.output}")


if __name__ == "__main__":
    main()
