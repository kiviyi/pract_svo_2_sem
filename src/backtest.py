import pandas as pd


STRESS_PERIODS = {
    "december_2014": ("2014-12-01", "2014-12-31"),
    "feb_march_2022": ("2022-02-01", "2022-03-31"),
    "august_2023": ("2023-08-01", "2023-09-15"),
}

VALIDATION_PERIODS = {
    "holdout_2024_2025": ("2024-01-01", "2025-12-31"),
}


def main():
    df = pd.read_csv("data/final/lsi_ml.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    rows = []
    for name, (start, end) in STRESS_PERIODS.items():
        part = df[df["date"].between(start, end)]
        is_train = pd.to_datetime(end) < pd.to_datetime("2024-01-01")
        rows.append({
            "period": name,
            "type": "calibration" if is_train else "validation",
            "start": start,
            "end": end,
            "mean_lsi": part["lsi"].mean(),
            "max_lsi": part["lsi"].max(),
            "mean_ml_lsi": part["ml_lsi"].mean(),
            "max_ml_lsi": part["ml_lsi"].max(),
            "pct_stress_days": (part["status"] == "Стресс").mean() * 100,
            "days": len(part),
        })
    for name, (start, end) in VALIDATION_PERIODS.items():
        part = df[df["date"].between(start, end)]
        rows.append({
            "period": name,
            "type": "holdout_validation",
            "start": start,
            "end": end,
            "mean_lsi": part["lsi"].mean(),
            "max_lsi": part["lsi"].max(),
            "mean_ml_lsi": part["ml_lsi"].mean(),
            "max_ml_lsi": part["ml_lsi"].max(),
            "pct_stress_days": (part["status"] == "Стресс").mean() * 100,
            "days": len(part),
        })
    result = pd.DataFrame(rows)
    result.to_csv("data/final/backtest_report.csv", index=False)
    print(result)
    print("\nПримечание:")
    print("- 'calibration' периоды использовались при обучении ML-модели")
    print("- 'holdout_validation' периоды НЕ использовались при обучении")
    print("- Ожидается, что на holdout LSI будет ниже (отсутствие стресса)")


if __name__ == "__main__":
    main()
