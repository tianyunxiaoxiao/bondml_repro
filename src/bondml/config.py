"""Shared feature groups and constants."""

BENCHMARK_OPTION_PREDICTORS = [
    "impl_vol",
    "impl_premium",
    "BS_price",
    "option_delta",
    "option_moneyness",
]

BENCHMARK_MACRO_PREDICTORS = [
    "dp",
    "ep",
    "bm",
    "ntis",
    "tms",
    "dfy",
    "svar",
]

EEM_DETERMINING_PREDICTORS = [
    "underlying_price",
    "strike_price",
    "underlying_ret",
    "risk_free",
    "underlying_vol",
    "holding_period",
    "option_maturity",
    "if_call",
]

EEM_RESULTING_PREDICTORS = [
    "EEM_price",
    "EEM_ret",
]

BENCHMARK_PREDICTORS = BENCHMARK_OPTION_PREDICTORS + BENCHMARK_MACRO_PREDICTORS

DETERMINING_FEATURE_SET = BENCHMARK_PREDICTORS + EEM_DETERMINING_PREDICTORS

FULL_FEATURE_SET = DETERMINING_FEATURE_SET + EEM_RESULTING_PREDICTORS

PREDICTOR_GROUPS = {
    "benchmark": BENCHMARK_PREDICTORS,
    "eem_determining": EEM_DETERMINING_PREDICTORS,
    "eem_resulting": EEM_RESULTING_PREDICTORS,
}
