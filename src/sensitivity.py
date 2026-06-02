from pathlib import Path

import pandas as pd


BASE_WEIGHTS = {
    "m1": 0.20,
    "m2": 0.30,
    "m3": 0.25,
    "m5": 0.25,
}


def calculate_weighted_lsi(df: pd.DataFrame, weights: dict) -> pd.Series:
    raw = (
        df.get("m1_adjusted_score", pd.Series(0, index=df.index)) * weights["m1"]
        + df.get("m2_adjusted_score", pd.Series(0, index=df.index)) * weights["m2"]
        + df.get("m3_adjusted_score", pd.Series(0, index=df.index)) * weights["m3"]
        + df.get("m5_adjusted_score", pd.Series(0, index=df.index)) * weights["m5"]
    )
    return raw.clip(0, 100)


def normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def main() -> None:
    Path("data/final").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/final/lsi.csv")
    rows = []
    for module in BASE_WEIGHTS:
        for change in [-0.2, 0.2]:
            weights = BASE_WEIGHTS.copy()
            weights[module] = weights[module] * (1 + change)
            weights = normalize_weights(weights)
            scenario_name = f"{module}_{int(change * 100)}pct"
            scenario_lsi = calculate_weighted_lsi(df, weights)
            seasonal = df.get("m4_seasonal_factor", pd.Series(1.0, index=df.index))
            scenario_lsi = (scenario_lsi * seasonal).clip(0, 100)
            base_col = "raw_lsi_before_seasonal" if "raw_lsi_before_seasonal" in df.columns else "raw_lsi"
            rows.append({
                "scenario": scenario_name,
                "changed_module": module,
                "change_pct": change * 100,
                "mean_lsi": scenario_lsi.mean(),
                "max_lsi": scenario_lsi.max(),
                "latest_lsi": scenario_lsi.iloc[-1],
                "base_latest_lsi": df[base_col].iloc[-1],
                "latest_lsi_clipped": scenario_lsi.clip(0, 100).iloc[-1],
                "latest_diff": scenario_lsi.iloc[-1] - df[base_col].iloc[-1],
            })
    result = pd.DataFrame(rows)
    result.to_csv("data/final/sensitivity_report.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
