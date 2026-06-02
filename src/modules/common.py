import re
import pandas as pd


def to_number(value):
    if pd.isna(value):
        return pd.NA

    value = str(value)
    value = value.replace("\xa0", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")
    value = re.sub(r"[^\d\.\-]", "", value)

    if value in {"", "-", ".", "-."}:
        return pd.NA

    return pd.to_numeric(value, errors="coerce")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    return series.apply(to_number)


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def find_col(df: pd.DataFrame, keywords: list[str]):
    for col in df.columns:
        col_lower = str(col).lower()
        if all(k.lower() in col_lower for k in keywords):
            return col
    return None