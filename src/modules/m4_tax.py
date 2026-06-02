import pandas as pd


TAX_DAYS = [25, 28]


def _parse_fns_calendar(fns_df: pd.DataFrame) -> pd.DataFrame:
    fns = fns_df.copy()
    if len(fns) < 2:
        return pd.DataFrame(columns=["date", "event"])
    events = []
    for _, row in fns.iterrows():
        for col in fns.columns:
            val = str(row[col])
            if "налог" in val.lower() or "ндс" in val.lower() or "прибыль" in val.lower():
                events.append({"raw": val})
    result = pd.DataFrame(events)
    return result


def calculate_m4(fns_df=None) -> pd.DataFrame:
    dates = pd.date_range("2014-01-01", pd.Timestamp.today(), freq="D")
    df = pd.DataFrame({"date": dates})
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["m4_tax_week_flag"] = df["day"].between(20, 30).astype(int)
    df["m4_main_tax_day_flag"] = df["day"].isin(TAX_DAYS).astype(int)
    df["m4_vat_period_flag"] = df["day"].between(25, 28).astype(int)
    df["m4_profit_tax_flag"] = df["day"].between(26, 30).astype(int)
    df["m4_end_of_month_flag"] = df["date"].dt.is_month_end.astype(int)
    df["m4_end_of_quarter_flag"] = (
        df["month"].isin([3, 6, 9, 12]) & df["day"].between(25, 31)
    ).astype(int)
    if fns_df is not None:
        fns_events = _parse_fns_calendar(fns_df)
        if len(fns_events) > 0:
            df["m4_source"] = "mixed"
        else:
            df["m4_source"] = "synthetic"
    else:
        df["m4_source"] = "synthetic"
    df["m4_seasonal_factor"] = 1.0
    df.loc[df["m4_tax_week_flag"] == 1, "m4_seasonal_factor"] = 1.15
    df.loc[df["m4_main_tax_day_flag"] == 1, "m4_seasonal_factor"] = 1.25
    df.loc[df["m4_end_of_quarter_flag"] == 1, "m4_seasonal_factor"] = 1.4
    return df[
        [
            "date",
            "m4_tax_week_flag",
            "m4_main_tax_day_flag",
            "m4_vat_period_flag",
            "m4_profit_tax_flag",
            "m4_end_of_month_flag",
            "m4_end_of_quarter_flag",
            "m4_seasonal_factor",
            "m4_source",
        ]
    ]
