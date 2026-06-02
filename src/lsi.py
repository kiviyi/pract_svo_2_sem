import pandas as pd


MODULE_WEIGHTS = {
    "m1": 0.20,
    "m2": 0.30,
    "m3": 0.25,
    "m5": 0.25,
}


def stress_from_mad_series(series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    series = series.clip(0, 5)
    return (series * 20).clip(0, 100)


def get_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def get_status(lsi: float) -> str:
    if lsi < 40:
        return "Норма"
    if lsi < 70:
        return "Напряжение"
    return "Стресс"


def apply_double_count_correction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tax_flag = get_col(df, "m4_tax_week_flag")
    quarter_flag = get_col(df, "m4_end_of_quarter_flag")
    seasonal_pressure = ((tax_flag == 1) | (quarter_flag == 1)).astype(int)
    scores_to_check = ["m1_score", "m2_score", "m5_score"]
    premiums = {}
    for col in scores_to_check:
        if col in df.columns:
            tax_mean = df.loc[seasonal_pressure == 1, col].mean()
            non_tax_mean = df.loc[seasonal_pressure == 0, col].mean()
            premiums[col] = max(0, tax_mean - non_tax_mean)
    df["m1_adjusted_score"] = df["m1_score"]
    df["m2_adjusted_score"] = df["m2_score"]
    df["m3_adjusted_score"] = df["m3_score"]
    df["m5_adjusted_score"] = df["m5_score"]
    for col, premium in premiums.items():
        if premium > 0:
            adj_col = col.replace("_score", "_adjusted_score")
            df[adj_col] = (df[adj_col] - seasonal_pressure * premium).clip(lower=0)
    df["double_count_correction_flag"] = seasonal_pressure
    return df


def calculate_module_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["m1_score"] = (
        stress_from_mad_series(get_col(df, "m1_mad_score_spread"))
        + stress_from_mad_series(get_col(df, "m1_mad_score_ruonia"))
        + get_col(df, "m1_flag_end_of_period") * 10
    )
    df["m2_score"] = (
        stress_from_mad_series(get_col(df, "m2_mad_score_cover_ratio_proxy"))
        + stress_from_mad_series(get_col(df, "m2_mad_score_rate_spread_proxy"))
        + get_col(df, "m2_flag_demand") * 15
    )
    df["m3_score"] = (
        stress_from_mad_series(get_col(df, "m3_mad_score_cover_ratio"))
        + stress_from_mad_series(get_col(df, "m3_mad_score_yield_spread"))
        + get_col(df, "m3_flag_nedospros") * 20
        - get_col(df, "m3_flag_perespros") * 10
    )
    df["m4_score"] = (
        get_col(df, "m4_tax_week_flag") * 10
        + get_col(df, "m4_main_tax_day_flag") * 10
        + get_col(df, "m4_vat_period_flag") * 5
        + get_col(df, "m4_profit_tax_flag") * 5
        + get_col(df, "m4_end_of_month_flag") * 5
        + get_col(df, "m4_end_of_quarter_flag") * 10
    )
    df["m5_score"] = (
        stress_from_mad_series(get_col(df, "m5_mad_score_liquidity_deficit"))
        + stress_from_mad_series(get_col(df, "m5_mad_score_weekly_delta"))
        + stress_from_mad_series(get_col(df, "m5_mad_score_monthly_delta"))
        + get_col(df, "m5_flag_budget_drain") * 20
    )
    df = apply_double_count_correction(df)
    return df


def calculate_lsi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = calculate_module_scores(df)
    seasonal_factor = get_col(df, "m4_seasonal_factor")
    m4_score_raw = get_col(df, "m4_score")
    df["m1_contribution"] = df["m1_adjusted_score"] * MODULE_WEIGHTS["m1"]
    df["m2_contribution"] = df["m2_adjusted_score"] * MODULE_WEIGHTS["m2"]
    df["m3_contribution"] = df["m3_adjusted_score"] * MODULE_WEIGHTS["m3"]
    df["m5_contribution"] = df["m5_adjusted_score"] * MODULE_WEIGHTS["m5"]
    df["raw_lsi_before_seasonal"] = (
        df["m1_contribution"]
        + df["m2_contribution"]
        + df["m3_contribution"]
        + df["m5_contribution"]
    )
    df["raw_lsi"] = df["raw_lsi_before_seasonal"] * seasonal_factor + m4_score_raw * 0.1
    df["m4_contribution"] = df["raw_lsi"] - df["raw_lsi_before_seasonal"]
    df["m4_contribution"] = df["m4_contribution"].clip(lower=0)
    df["lsi"] = df["raw_lsi"].clip(0, 100)
    df["status"] = df["lsi"].apply(get_status)
    return df


def prepare_lsi_dataset(
    m1: pd.DataFrame,
    m2: pd.DataFrame,
    m3: pd.DataFrame,
    m4: pd.DataFrame,
    m5: pd.DataFrame,
) -> pd.DataFrame:
    df = m1.copy()
    df = df.merge(m2, on="date", how="outer")
    df = df.merge(m3, on="date", how="outer")
    df = df.merge(m4, on="date", how="outer")
    df = df.merge(m5, on="date", how="outer")
    df = (
        df.sort_values("date")
        .groupby("date", as_index=False)
        .mean(numeric_only=True)
    )
    df = df.fillna(0)
    return df
