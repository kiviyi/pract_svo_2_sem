import pandas as pd

from src.normalization import rolling_mad_scores
from src.modules.common import clean_numeric_series


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != "nan"]).strip()
            for col in df.columns
        ]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    return df


def _prepare_repo_debt(repo_debt_df: pd.DataFrame) -> pd.DataFrame:
    df = repo_debt_df.copy()
    if all(str(c).isdigit() for c in df.columns):
        header = df.iloc[0].astype(str).tolist()
        df = df.iloc[3:].copy()
        df.columns = header
    df = _normalize_columns(df)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    date_col = None
    for col in df.columns:
        if "дата" in str(col).lower():
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 5:
                date_col = col
                break
    if date_col is None:
        best_col = None
        best_count = -1
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            count = parsed.notna().sum()
            if count > best_count:
                best_count = count
                best_col = col
        if best_count <= 5:
            raise ValueError("M2: не удалось определить колонку даты")
        date_col = best_col
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    val_col = _find_repo_outstanding_column(df, date_col)
    result = pd.DataFrame({
        "date": df["date"],
        "repo_outstanding": pd.to_numeric(clean_numeric_series(df[val_col]), errors="coerce"),
    })
    result = result.dropna(subset=["date", "repo_outstanding"])
    result = result.sort_values("date")
    return result


def _find_repo_outstanding_column(df: pd.DataFrame, date_col: str) -> str:
    candidates = []
    for col in df.columns:
        if col == date_col:
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        parsed = pd.to_numeric(clean_numeric_series(series), errors="coerce")
        valid_count = parsed.notna().sum()
        if valid_count > 5:
            candidates.append((col, valid_count, parsed.median()))
    if not candidates:
        raise ValueError("M2: не найдена числовая колонка РЕПО")
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _prepare_key_rate(key_rate_df: pd.DataFrame) -> pd.DataFrame:
    key = key_rate_df.copy()
    key["date"] = pd.to_datetime(key["Дата"], errors="coerce", dayfirst=True)
    key["key_rate"] = pd.to_numeric(key["Ставка"], errors="coerce") / 100
    key = key.dropna(subset=["date"])
    key = key.sort_values("date")
    return key[["date", "key_rate"]]


def calculate_m2(
    repo_debt_df: pd.DataFrame,
    key_rate_df: pd.DataFrame,
) -> pd.DataFrame:
    debt = _prepare_repo_debt(repo_debt_df)
    debt["repo_rolling_mean_90d"] = (
        debt["repo_outstanding"].rolling(90, min_periods=10).mean()
    )
    debt["cover_ratio_proxy"] = (
        debt["repo_outstanding"] / debt["repo_rolling_mean_90d"]
    ).replace([float("inf"), -float("inf")], pd.NA)
    debt["repo_delta_7d"] = debt["repo_outstanding"].diff(7)
    debt["repo_delta_30d"] = debt["repo_outstanding"].diff(30)
    debt["rate_spread_proxy"] = (
        debt["repo_delta_7d"] / debt["repo_outstanding"].rolling(90, min_periods=10).mean()
    ).replace([float("inf"), -float("inf")], pd.NA)
    for col in ["repo_delta_7d", "repo_delta_30d"]:
        lo = debt[col].quantile(0.01)
        hi = debt[col].quantile(0.99)
        debt[col] = debt[col].clip(lo, hi)
    debt["repo_rolling_7"] = debt["repo_outstanding"].rolling(7, min_periods=3).mean()
    debt["repo_rolling_30"] = debt["repo_outstanding"].rolling(30, min_periods=10).mean()
    key = _prepare_key_rate(key_rate_df)
    df = debt.merge(key, on="date", how="left")
    df["key_rate"] = df["key_rate"].ffill().bfill()
    mad_cols = [
        "repo_outstanding", "repo_delta_7d", "repo_delta_30d",
        "repo_rolling_7", "repo_rolling_30",
        "cover_ratio_proxy", "rate_spread_proxy",
    ]
    df = rolling_mad_scores(df, value_cols=mad_cols, prefix="m2_mad_score_")
    df["m2_flag_demand"] = (
        (df.get("m2_mad_score_cover_ratio_proxy", pd.Series(0, index=df.index)) > 2.0)
        | (df.get("m2_mad_score_repo_delta_7d", pd.Series(0, index=df.index)) > 2.0)
        | (df.get("m2_mad_score_repo_outstanding", pd.Series(0, index=df.index)) > 2.0)
    ).astype(int)
    return df[
        [
            "date",
            "repo_outstanding",
            "cover_ratio_proxy",
            "rate_spread_proxy",
            "repo_delta_7d",
            "repo_delta_30d",
            "repo_rolling_7",
            "repo_rolling_30",
            "key_rate",
            "m2_mad_score_repo_outstanding",
            "m2_mad_score_cover_ratio_proxy",
            "m2_mad_score_rate_spread_proxy",
            "m2_mad_score_repo_delta_7d",
            "m2_mad_score_repo_delta_30d",
            "m2_mad_score_repo_rolling_7",
            "m2_mad_score_repo_rolling_30",
            "m2_flag_demand",
        ]
    ]
