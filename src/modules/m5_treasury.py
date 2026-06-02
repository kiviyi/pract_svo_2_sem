import pandas as pd

from src.normalization import rolling_mad_scores
from src.modules.common import clean_numeric_series


def _parse_treasury_deposits(ts_df: pd.DataFrame) -> pd.DataFrame:
    ts = ts_df.copy()
    if len(ts) < 2:
        return pd.DataFrame(columns=["date", "num_placements"])
    if "date" in ts.columns and "num_placements" in ts.columns:
        ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
        ts["num_placements"] = pd.to_numeric(ts["num_placements"], errors="coerce").fillna(0)
        return ts[["date", "num_placements"]].dropna(subset=["date"])
    date_col = None
    for col in ts.columns:
        if "дата" in str(col).lower() or "date" in str(col).lower():
            date_col = col
            break
    if date_col is None:
        date_col = ts.columns[0]
    ts["date"] = pd.to_datetime(ts[date_col], errors="coerce", dayfirst=True)
    ts = ts.dropna(subset=["date"])
    if "num_placements" in ts.columns:
        ts["num_placements"] = pd.to_numeric(ts["num_placements"], errors="coerce").fillna(0)
    else:
        ts["num_placements"] = 0
    return ts[["date", "num_placements"]]


def calculate_m5(
    bank_liquidity_df: pd.DataFrame,
    treasury_df=None,
) -> pd.DataFrame:
    liq = bank_liquidity_df.copy()
    date_cols = [c for c in liq.columns if str(c).lower().startswith("дата")]
    date_col = date_cols[0] if date_cols else liq.columns[0]
    liq["date"] = pd.to_datetime(liq[date_col], errors="coerce", dayfirst=True)
    liq = liq.dropna(subset=["date"])
    liq_cols = [c for c in liq.columns if "дефицит" in str(c).lower() and "профицит" in str(c).lower()]
    liquidity_col = liq_cols[0] if liq_cols else liq.columns[1]
    liq["liquidity_deficit"] = pd.to_numeric(
        clean_numeric_series(liq[liquidity_col]), errors="coerce"
    )
    liq = liq.dropna(subset=["liquidity_deficit"])
    liq = liq.sort_values("date")
    liq["weekly_delta"] = liq["liquidity_deficit"].diff(7)
    liq["monthly_delta"] = liq["liquidity_deficit"].diff(30)
    liq["liquidity_deficit_rolling_7"] = liq["liquidity_deficit"].rolling(7, min_periods=3).mean()
    liq["liquidity_deficit_rolling_30"] = liq["liquidity_deficit"].rolling(30, min_periods=10).mean()
    if treasury_df is not None:
        t_data = _parse_treasury_deposits(treasury_df)
        if len(t_data) > 0:
            liq = liq.merge(t_data, on="date", how="left")
            liq["num_placements"] = liq["num_placements"].ffill().bfill().fillna(0)
            liq["placements_delta_7d"] = liq["num_placements"].diff(7)
        else:
            liq["num_placements"] = 0.0
            liq["placements_delta_7d"] = 0.0
    else:
        liq["num_placements"] = 0.0
        liq["placements_delta_7d"] = 0.0
    mad_cols = [
        "liquidity_deficit", "weekly_delta", "monthly_delta",
        "liquidity_deficit_rolling_7", "liquidity_deficit_rolling_30",
        "num_placements", "placements_delta_7d",
    ]
    liq = rolling_mad_scores(liq, value_cols=mad_cols, prefix="m5_mad_score_")
    liq["m5_flag_budget_drain"] = (
        (liq["weekly_delta"] > 300)
        | (liq["monthly_delta"] > 500)
        | (liq.get("m5_mad_score_liquidity_deficit", pd.Series(0, index=liq.index)) > 2.0)
        | (liq.get("m5_mad_score_num_placements", pd.Series(0, index=liq.index)) > 2.0)
    ).astype(int)
    return liq[
        [
            "date",
            "liquidity_deficit",
            "weekly_delta",
            "monthly_delta",
            "liquidity_deficit_rolling_7",
            "liquidity_deficit_rolling_30",
            "num_placements",
            "placements_delta_7d",
            "m5_mad_score_liquidity_deficit",
            "m5_mad_score_weekly_delta",
            "m5_mad_score_monthly_delta",
            "m5_mad_score_liquidity_deficit_rolling_7",
            "m5_mad_score_liquidity_deficit_rolling_30",
            "m5_mad_score_num_placements",
            "m5_mad_score_placements_delta_7d",
            "m5_flag_budget_drain",
        ]
    ]
