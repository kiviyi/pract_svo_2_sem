import pandas as pd

from src.normalization import rolling_mad_score
from src.modules.common import clean_numeric_series


def _prepare_reserves(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    header_idx = None
    for i, row in df.iterrows():
        row_str = row.astype(str).tolist()
        if "фактические" in str(row_str[1]).lower():
            header_idx = i
            break
    if header_idx is None:
        header_idx = 5
    df.columns = df.iloc[header_idx].astype(str).tolist()
    df = df.iloc[header_idx + 1:].copy()
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(clean_numeric_series(df[col]), errors="coerce")
    return df


def _prepare_ruonia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = None
    val_col = None
    for col in df.columns:
        col_str = str(col).lower()
        if "дата" in col_str and "ставки" in col_str:
            date_col = col
        if "ruonia" in col_str.lower():
            val_col = col
    if date_col is not None and val_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        values = pd.to_numeric(clean_numeric_series(df[val_col]), errors="coerce")
        result = pd.DataFrame({"date": dates, "ruonia": values / 100})
        result = result.dropna(subset=["date", "ruonia"])
        result = result.drop_duplicates(subset=["date"]).sort_values("date")
        return result
    dates = pd.to_datetime(df.iloc[0, 1:].astype(str), errors="coerce", dayfirst=True)
    for _, row in df.iterrows():
        label = str(row.iloc[0]).lower()
        if "ruonia" in label:
            values = pd.to_numeric(row.iloc[1:], errors="coerce")
            result = pd.DataFrame({"date": dates, "ruonia": values / 100})
            result = result.dropna(subset=["date", "ruonia"])
            result = result.drop_duplicates(subset=["date"]).sort_values("date")
            return result
    return pd.DataFrame(columns=["date", "ruonia"])


def _end_of_period_flag(dates: pd.Series) -> pd.Series:
    return (dates.dt.day.between(6, 10)).astype(int)


def calculate_m1(
    reserves_df: pd.DataFrame,
    ruonia_df: pd.DataFrame,
) -> pd.DataFrame:
    df = _prepare_reserves(reserves_df)
    num_cols = df.select_dtypes("number").columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("M1: не найдены числовые колонки резервов")
    df["actual_reserves"] = df[num_cols[0]]
    df["required_reserves"] = df[num_cols[1]]
    df["spread"] = df["actual_reserves"] - df["required_reserves"]
    df = rolling_mad_score(df, value_col="spread", output_col="m1_mad_score_spread")
    ruonia = _prepare_ruonia(ruonia_df)
    if not ruonia.empty:
        ruonia = rolling_mad_score(ruonia, value_col="ruonia", output_col="m1_mad_score_ruonia")
        df = pd.merge_asof(
            df.sort_values("date"),
            ruonia[["date", "ruonia", "m1_mad_score_ruonia"]].sort_values("date"),
            on="date",
            direction="nearest",
        )
    else:
        df["ruonia"] = 0.0
        df["m1_mad_score_ruonia"] = 0.0
    df["m1_flag_end_of_period"] = _end_of_period_flag(df["date"])
    return df[
        [
            "date",
            "actual_reserves",
            "required_reserves",
            "spread",
            "ruonia",
            "m1_mad_score_spread",
            "m1_mad_score_ruonia",
            "m1_flag_end_of_period",
        ]
    ]
