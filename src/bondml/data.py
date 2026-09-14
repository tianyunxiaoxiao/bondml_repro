"""Synthetic data generation matching the paper's variable design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bondml.config import FULL_FEATURE_SET
from bondml.eem import TRADING_DAYS, black_scholes_delta, black_scholes_price, eem_expected_price, eem_expected_return


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for the synthetic option panel."""

    start_year: int = 1998
    end_year: int = 2021
    rows_per_year: int = 360
    seed: int = 20260306
    trim_quantile: float = 0.99


def generate_synthetic_panel(config: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """Generate a deterministic synthetic S&P 500 index option panel."""

    cfg = config or SyntheticDataConfig()
    rng = np.random.default_rng(cfg.seed)
    frames: list[pd.DataFrame] = []

    for year in range(cfg.start_year, cfg.end_year + 1):
        n_rows = cfg.rows_per_year
        year_progress = year - cfg.start_year
        crisis = _crisis_intensity(year)
        day_of_year = rng.integers(1, 253, size=n_rows)

        market_level = 1000.0 * np.exp(0.045 * year_progress + 0.04 * np.sin(year_progress / 2.5))
        underlying_price = market_level * np.exp(rng.normal(0.0, 0.08 + 0.05 * crisis, size=n_rows))
        if_call = rng.binomial(1, 0.5, size=n_rows).astype(float)
        option_maturity = rng.choice([14, 30, 60, 91, 126, 180, 252, 365, 547, 730], size=n_rows).astype(float)
        holding_period = option_maturity.copy()

        base_vol = 0.13 + 0.025 * np.sin(year_progress / 2.0) + 0.13 * crisis
        underlying_vol = np.clip(base_vol + rng.gamma(1.8, 0.025, size=n_rows), 0.05, 1.25)
        risk_free = np.clip(0.018 + 0.012 * np.cos(year_progress / 3.0) - 0.008 * crisis + rng.normal(0, 0.004, n_rows), 0.001, 0.08)
        underlying_ret = np.clip(0.065 + 0.025 * np.sin(year_progress / 4.0) - 0.09 * crisis + rng.normal(0, 0.025, n_rows), -0.18, 0.20)

        signed_delta_bucket = rng.uniform(0.10, 0.90, size=n_rows)
        target_delta = np.where(if_call >= 0.5, signed_delta_bucket, -signed_delta_bucket)
        moneyness_noise = rng.normal(0.0, 0.045 + 0.02 * crisis, size=n_rows)
        option_moneyness = np.clip(1.0 + (0.52 - np.abs(target_delta)) * np.where(if_call >= 0.5, 0.38, -0.38) + moneyness_noise, 0.45, 2.50)
        strike_price = underlying_price / option_moneyness

        impl_vol = np.clip(
            underlying_vol
            + 0.035 * (1.0 - np.abs(target_delta))
            + 0.015 * (1.0 - if_call)
            + rng.normal(0, 0.015, n_rows),
            0.03,
            1.35,
        )
        bs_price = black_scholes_price(underlying_price, strike_price, risk_free, impl_vol, option_maturity, if_call)
        option_delta = black_scholes_delta(underlying_price, strike_price, risk_free, impl_vol, option_maturity, if_call)
        impl_premium = np.clip(0.025 + 0.20 * crisis + 0.08 * (impl_vol - underlying_vol) + rng.normal(0, 0.05, n_rows), -0.45, 0.85)

        eem_price = eem_expected_price(
            underlying_price,
            strike_price,
            underlying_ret,
            risk_free,
            underlying_vol,
            holding_period,
            option_maturity,
            if_call,
        )
        eem_ret = eem_expected_return(
            underlying_price,
            strike_price,
            underlying_ret,
            risk_free,
            underlying_vol,
            holding_period,
            option_maturity,
            if_call,
        )

        macro = _macro_predictors(year_progress, crisis, rng, n_rows)
        realized = _synthetic_realized_return(
            eem_ret=eem_ret,
            option_delta=option_delta,
            impl_premium=impl_premium,
            macro=macro,
            if_call=if_call,
            crisis=crisis,
            rng=rng,
        )

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(year * 1000 + day_of_year, format="%Y%j"),
                "year": year,
                "realized_excess_return": realized,
                "underlying_price": underlying_price,
                "strike_price": strike_price,
                "underlying_ret": underlying_ret,
                "risk_free": risk_free,
                "underlying_vol": underlying_vol,
                "holding_period": holding_period,
                "option_maturity": option_maturity,
                "if_call": if_call,
                "EEM_price": eem_price,
                "EEM_ret": eem_ret,
                "impl_vol": impl_vol,
                "impl_premium": impl_premium,
                "BS_price": bs_price,
                "option_delta": option_delta,
                "option_moneyness": option_moneyness,
                **macro,
            }
        )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "if_call"]).reset_index(drop=True)
    panel = _trim_extreme_returns(panel, cfg.trim_quantile)
    return panel


def summarize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return paper-style summary statistics for all, call, and put samples."""

    rows = []
    variables = [
        "realized_excess_return",
        "EEM_ret",
        "option_maturity",
        "option_moneyness",
        "impl_vol",
        "underlying_vol",
        "option_delta",
    ]
    samples = {
        "All Options": panel,
        "Call Options": panel[panel["if_call"] >= 0.5],
        "Put Options": panel[panel["if_call"] < 0.5],
    }
    for sample_name, sample in samples.items():
        stats = sample[variables].agg(["mean", "std", "min", "median", "max", "skew"]).T
        stats["q10"] = sample[variables].quantile(0.10)
        stats["q25"] = sample[variables].quantile(0.25)
        stats["q75"] = sample[variables].quantile(0.75)
        stats["q90"] = sample[variables].quantile(0.90)
        stats["sample"] = sample_name
        stats["n"] = len(sample)
        rows.append(stats.reset_index(names="variable"))
    result = pd.concat(rows, ignore_index=True)
    ordered = ["sample", "n", "variable", "mean", "std", "min", "q10", "q25", "median", "q75", "q90", "max", "skew"]
    return result[ordered]


def validate_panel(panel: pd.DataFrame) -> None:
    """Raise a useful error if generated data violates core assumptions."""

    required = {"date", "year", "realized_excess_return", *FULL_FEATURE_SET}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    if panel[list(required.difference({"date"}))].isna().any().any():
        raise ValueError("Panel contains missing numeric values.")
    if not np.isfinite(panel[list(required.difference({"date"}))].to_numpy()).all():
        raise ValueError("Panel contains non-finite numeric values.")
    if not (panel.loc[panel["if_call"] >= 0.5, "option_delta"] > 0).all():
        raise ValueError("Call deltas must be positive.")
    if not (panel.loc[panel["if_call"] < 0.5, "option_delta"] < 0).all():
        raise ValueError("Put deltas must be negative.")


def _crisis_intensity(year: int) -> float:
    intensities = {
        2001: 0.35,
        2002: 0.45,
        2008: 1.00,
        2009: 0.75,
        2011: 0.35,
        2015: 0.25,
        2020: 0.95,
        2021: 0.35,
    }
    return intensities.get(year, 0.0)


def _macro_predictors(year_progress: int, crisis: float, rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
    common_cycle = np.sin(year_progress / 3.0)
    return {
        "dp": 0.025 + 0.004 * common_cycle + 0.003 * crisis + rng.normal(0, 0.002, n_rows),
        "ep": 0.045 + 0.008 * np.cos(year_progress / 4.0) - 0.006 * crisis + rng.normal(0, 0.004, n_rows),
        "bm": 0.32 + 0.04 * common_cycle + 0.05 * crisis + rng.normal(0, 0.025, n_rows),
        "ntis": -0.005 + 0.018 * np.sin(year_progress / 2.2) - 0.025 * crisis + rng.normal(0, 0.015, n_rows),
        "tms": 0.018 + 0.01 * np.cos(year_progress / 2.8) - 0.012 * crisis + rng.normal(0, 0.006, n_rows),
        "dfy": 0.008 + 0.015 * crisis + rng.normal(0, 0.004, n_rows),
        "svar": 0.018 + 0.055 * crisis + rng.gamma(1.5, 0.006, n_rows),
    }


def _synthetic_realized_return(
    eem_ret: np.ndarray,
    option_delta: np.ndarray,
    impl_premium: np.ndarray,
    macro: dict[str, np.ndarray],
    if_call: np.ndarray,
    crisis: float,
    rng: np.random.Generator,
) -> np.ndarray:
    eem_signal = np.tanh(np.clip(eem_ret, -5.0, 5.0) / 2.0)
    abs_delta = np.abs(option_delta)
    nonlinear_eem = (
        0.38 * np.sin(2.2 * eem_signal)
        + 0.72 * ((eem_signal > 0.18) & (abs_delta > 0.32) & (abs_delta < 0.78))
        - 0.58 * ((eem_signal < -0.22) & (impl_premium > 0.08))
    )
    nonlinear_delta = 1.35 * np.sign(option_delta) * (abs_delta - 0.52) ** 2
    macro_signal = 3.2 * macro["ntis"] - 1.7 * macro["dfy"] + 1.2 * macro["tms"] - 0.7 * macro["svar"]
    macro_threshold = 0.34 * ((macro["ntis"] > 0.0) & (macro["tms"] > 0.0)) - 0.42 * (
        (macro["dfy"] > 0.018) & (macro["svar"] > 0.045)
    )
    call_put_tilt = np.where(if_call >= 0.5, 0.12, -0.22)
    crisis_drag = -0.48 * crisis * (1.0 - abs_delta) - 0.34 * crisis * (impl_premium > 0.12)
    noise = rng.standard_t(df=5, size=len(eem_ret)) * (0.28 + 0.22 * crisis)
    realized = nonlinear_eem + nonlinear_delta - 0.30 * impl_premium + macro_signal + macro_threshold + call_put_tilt + crisis_drag + noise
    return np.clip(realized, -1.25, 6.25)


def _trim_extreme_returns(panel: pd.DataFrame, trim_quantile: float) -> pd.DataFrame:
    realized_abs = panel["realized_excess_return"].abs()
    eem_abs = panel["EEM_ret"].abs()
    realized_cutoff = realized_abs.quantile(trim_quantile)
    eem_cutoff = eem_abs.quantile(trim_quantile)
    trimmed = panel[(realized_abs <= realized_cutoff) & (eem_abs <= eem_cutoff)].copy()
    return trimmed.reset_index(drop=True)
