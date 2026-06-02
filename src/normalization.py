import numpy as np
import pandas as pd
from typing import Optional


def mad(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    median = values.median()
    return float(np.median(np.abs(values - median)))


def mad_score(
    series: pd.Series,
    min_mad: float = 1e-9,
) -> pd.Series:
    median = series.median()
    mad_value = mad(series)
    if pd.isna(mad_value) or mad_value < min_mad:
        return pd.Series(0.0, index=series.index)
    return (series - median) / mad_value


def rolling_mad_score(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    window_days: int = 365 * 3,
    min_periods: int = 30,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(date_col)
    if output_col is None:
        output_col = f"mad_score_{value_col}"
    values = []
    for idx, row in df.iterrows():
        current_date = row[date_col]
        start_date = current_date - pd.Timedelta(days=window_days)
        history = df[
            (df[date_col] < current_date)
            & (df[date_col] >= start_date)
        ][value_col]
        if len(history.dropna()) < min_periods:
            values.append(np.nan)
            continue
        median = history.median()
        mad_value = mad(history)
        if pd.isna(mad_value) or mad_value < 1e-9:
            values.append(0.0)
        else:
            values.append(float((row[value_col] - median) / mad_value))
    df[output_col] = values
    return df


def rolling_mad_scores(
    df: pd.DataFrame,
    value_cols: list[str],
    date_col: str = "date",
    window_days: int = 365 * 3,
    min_periods: int = 30,
    prefix: str = "mad_score_",
) -> pd.DataFrame:
    df = df.copy()
    for col in value_cols:
        output_col = f"{prefix}{col}"
        df = rolling_mad_score(
            df,
            value_col=col,
            date_col=date_col,
            window_days=window_days,
            min_periods=min_periods,
            output_col=output_col,
        )
    return df


def normalize_columns_mad(
    df: pd.DataFrame,
    columns: list[str],
    prefix: str = "mad_score_",
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{prefix}{col}"] = mad_score(df[col])
    return df


def clip_scores(
    df: pd.DataFrame,
    columns: list[str],
    lower: float = -5.0,
    upper: float = 5.0,
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)
    return df
