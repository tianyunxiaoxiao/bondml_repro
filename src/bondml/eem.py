"""Black-Scholes and EEM formulas used by the reproduction workflow."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252.0


def _as_array(value: np.ndarray | float) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _safe_time(days: np.ndarray | float) -> np.ndarray:
    return np.maximum(_as_array(days) / TRADING_DAYS, 1.0 / TRADING_DAYS)


def _holding_time(days: np.ndarray | float) -> np.ndarray:
    return np.maximum(_as_array(days) / TRADING_DAYS, 0.0)


def black_scholes_price(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    risk_free: np.ndarray | float,
    volatility: np.ndarray | float,
    maturity_days: np.ndarray | float,
    if_call: np.ndarray | float,
) -> np.ndarray:
    """Return European option prices under Black-Scholes."""

    spot = np.maximum(_as_array(spot), 1e-8)
    strike = np.maximum(_as_array(strike), 1e-8)
    risk_free = _as_array(risk_free)
    volatility = np.maximum(_as_array(volatility), 1e-6)
    maturity = _safe_time(maturity_days)
    if_call = _as_array(if_call)

    sigma_sqrt_t = volatility * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (risk_free + 0.5 * volatility**2) * maturity) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discount = np.exp(-risk_free * maturity)
    call = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    put = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return np.where(if_call >= 0.5, call, put)


def black_scholes_delta(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    risk_free: np.ndarray | float,
    volatility: np.ndarray | float,
    maturity_days: np.ndarray | float,
    if_call: np.ndarray | float,
) -> np.ndarray:
    """Return Black-Scholes option delta."""

    spot = np.maximum(_as_array(spot), 1e-8)
    strike = np.maximum(_as_array(strike), 1e-8)
    risk_free = _as_array(risk_free)
    volatility = np.maximum(_as_array(volatility), 1e-6)
    maturity = _safe_time(maturity_days)
    if_call = _as_array(if_call)

    sigma_sqrt_t = volatility * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (risk_free + 0.5 * volatility**2) * maturity) / sigma_sqrt_t
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(if_call >= 0.5, call_delta, put_delta)


def eem_expected_price(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    expected_return: np.ndarray | float,
    risk_free: np.ndarray | float,
    volatility: np.ndarray | float,
    holding_days: np.ndarray | float,
    maturity_days: np.ndarray | float,
    if_call: np.ndarray | float,
) -> np.ndarray:
    """Return the EEM expected future option price at the holding horizon."""

    spot = np.maximum(_as_array(spot), 1e-8)
    strike = np.maximum(_as_array(strike), 1e-8)
    expected_return = _as_array(expected_return)
    risk_free = _as_array(risk_free)
    volatility = np.maximum(_as_array(volatility), 1e-6)
    maturity = _safe_time(maturity_days)
    holding = np.minimum(_holding_time(holding_days), maturity)
    if_call = _as_array(if_call)

    remaining = np.maximum(maturity - holding, 0.0)
    sigma_sqrt_t = volatility * np.sqrt(maturity)
    drift_term = expected_return * holding + risk_free * remaining
    d1_hat = (np.log(spot / strike) + drift_term + 0.5 * volatility**2 * maturity) / sigma_sqrt_t
    d2_hat = d1_hat - sigma_sqrt_t
    expected_spot = spot * np.exp(expected_return * holding)
    discount = np.exp(-risk_free * remaining)

    call = expected_spot * norm.cdf(d1_hat) - strike * discount * norm.cdf(d2_hat)
    put = strike * discount * norm.cdf(-d2_hat) - expected_spot * norm.cdf(-d1_hat)
    return np.where(if_call >= 0.5, call, put)


def eem_expected_return(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    expected_return: np.ndarray | float,
    risk_free: np.ndarray | float,
    volatility: np.ndarray | float,
    holding_days: np.ndarray | float,
    maturity_days: np.ndarray | float,
    if_call: np.ndarray | float,
) -> np.ndarray:
    """Return EEM expected option excess return over the holding horizon."""

    current_price = black_scholes_price(
        spot=spot,
        strike=strike,
        risk_free=risk_free,
        volatility=volatility,
        maturity_days=maturity_days,
        if_call=if_call,
    )
    future_price = eem_expected_price(
        spot=spot,
        strike=strike,
        expected_return=expected_return,
        risk_free=risk_free,
        volatility=volatility,
        holding_days=holding_days,
        maturity_days=maturity_days,
        if_call=if_call,
    )
    holding = np.minimum(_holding_time(holding_days), _safe_time(maturity_days))
    return future_price / np.maximum(current_price, 1e-8) - 1.0 - risk_free * holding
