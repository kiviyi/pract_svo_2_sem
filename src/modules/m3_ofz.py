import pandas as pd

from src.normalization import rolling_mad_scores
from src.modules.common import clean_numeric_series


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean_numeric_series(series), errors="coerce")


def calculate_m3(ofz_df: pd.DataFrame) -> pd.DataFrame:
    df = ofz_df.copy()
    df = df[df["Дата аукциона"] != "Дата аукциона"]
    df = df[df["Дата аукциона"] != "1"]
    df["date"] = pd.to_datetime(df["Дата аукциона"], errors="coerce", dayfirst=False)
    df["offered_volume"] = to_float(df["Объем предложения"])
    df["demand_volume"] = to_float(df["Совокупный объем спроса по номиналу"])
    df["placed_volume"] = to_float(df["Объем размещения по номиналу"])
    df["auction_yield"] = to_float(df["Доходность по средне-взвешенной цене"])
    df = df.dropna(subset=["date"])
    df["cover_ratio"] = df["demand_volume"] / df["offered_volume"]
    df["cover_ratio"] = df["cover_ratio"].replace([float("inf"), -float("inf")], pd.NA)
    rolling_median_yield = df["auction_yield"].rolling(10, min_periods=3).median()
    df["yield_spread"] = df["auction_yield"] - rolling_median_yield
    df = rolling_mad_scores(df, value_cols=["cover_ratio", "yield_spread"], prefix="m3_mad_score_")
    df["m3_flag_nedospros"] = (df["cover_ratio"] < 1.2).astype(int)
    df["m3_flag_perespros"] = (df["cover_ratio"] > 2.0).astype(int)
    return df[
        [
            "date",
            "offered_volume",
            "demand_volume",
            "placed_volume",
            "cover_ratio",
            "auction_yield",
            "yield_spread",
            "m3_mad_score_cover_ratio",
            "m3_mad_score_yield_spread",
            "m3_flag_nedospros",
            "m3_flag_perespros",
        ]
    ]
