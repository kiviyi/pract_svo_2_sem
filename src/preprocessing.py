import re
from typing import Optional

import pandas as pd


def clean_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = col.replace("\n", " ")
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\wа-яА-ЯёЁ_]+", "", col)
    return col


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def parse_date_column(
    df: pd.DataFrame,
    date_col: str,
    dayfirst: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        dayfirst=dayfirst,
    )
    return df


def clean_numeric_value(value):
    if pd.isna(value):
        return pd.NA

    value = str(value)
    value = value.replace("\xa0", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")
    value = re.sub(r"[^\d\.\-]", "", value)

    if value in {"", "-", ".", "-."}:
        return pd.NA

    return float(value)


def parse_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_value)

    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.dropna(how="all")


def basic_clean(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = df.copy()

    df = drop_empty_rows(df)
    df = clean_columns(df)

    if date_col is not None:
        date_col = clean_column_name(date_col)
        df = parse_date_column(df, date_col)

    if numeric_cols is not None:
        numeric_cols = [clean_column_name(col) for col in numeric_cols]
        df = parse_numeric_columns(df, numeric_cols)

    return df


def save_processed(
    df: pd.DataFrame,
    path: str,
) -> None:
    df.to_csv(path, index=False)