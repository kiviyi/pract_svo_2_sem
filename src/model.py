from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
BASE_FEATURES = [
    "m1_mad_score_spread",
    "m1_mad_score_ruonia",
    "m2_mad_score_repo_outstanding",
    "m2_mad_score_cover_ratio_proxy",
    "m2_mad_score_rate_spread_proxy",
    "m2_mad_score_repo_delta_7d",
    "m2_mad_score_repo_delta_30d",
    "m2_mad_score_repo_rolling_7",
    "m2_mad_score_repo_rolling_30",
    "m3_mad_score_cover_ratio",
    "m3_mad_score_yield_spread",
    "m4_tax_week_flag",
    "m4_end_of_month_flag",
    "m4_end_of_quarter_flag",
    "m4_main_tax_day_flag",
    "m4_vat_period_flag",
    "m4_profit_tax_flag",
    "m4_seasonal_factor",
    "m5_mad_score_liquidity_deficit",
    "m5_mad_score_weekly_delta",
    "m5_mad_score_monthly_delta",
    "m5_mad_score_liquidity_deficit_rolling_7",
    "m5_mad_score_liquidity_deficit_rolling_30",
    "m5_mad_score_deposit_volume",
    "m5_mad_score_deposit_delta_7d",
]

LAG_FEATURES = [
    "m1_mad_score_spread",
    "m2_mad_score_repo_outstanding",
    "m2_mad_score_repo_delta_7d",
    "m3_mad_score_cover_ratio",
    "m3_mad_score_yield_spread",
    "m5_mad_score_liquidity_deficit",
    "m5_mad_score_weekly_delta",
]


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stress_target"] = (
        df["date"].between("2014-11-15", "2015-01-15")
        | df["date"].between("2022-02-01", "2022-04-15")
        | df["date"].between("2023-08-01", "2023-09-15")
    ).astype(int)
    return df


def prepare_dataset(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = (
        df.sort_values("date")
        .groupby("date", as_index=False)
        .mean(numeric_only=True)
    )
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df[BASE_FEATURES] = (
        df[BASE_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-100, 100)
    )
    feature_cols = list(dict.fromkeys(BASE_FEATURES))

    for col in LAG_FEATURES:
        if col not in df.columns:
            continue

        roll_name = f"{col}_rolling_7"

        if roll_name not in df.columns:
            df[roll_name] = (
                df[col]
                .rolling(7, min_periods=1)
                .mean()
            )

        feature_cols.append(roll_name)

    feature_cols = list(dict.fromkeys(feature_cols))

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df[feature_cols] = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-100, 100)
    )

    df = create_target(df)
    return df, feature_cols


def train_model(df: pd.DataFrame):
    df, feature_cols = prepare_dataset(df)
    X_all = df[feature_cols].copy()
    used_features = [
        col for col in feature_cols
        if X_all[col].nunique() > 1
    ]
    if not used_features:
        raise ValueError("Нет информативных признаков для обучения")
    X = X_all[used_features]
    y = df["stress_target"]
    dates = df["date"]
    split_date = "2024-01-01"
    train_mask = dates < split_date
    test_mask = dates >= split_date
    X_train = X[train_mask]
    y_train = y[train_mask]
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    df["stress_probability"] = model.predict_proba(X)[:, 1]
    df["ml_lsi"] = (df["stress_probability"] * 100).clip(0, 100)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X[test_mask], y[test_mask]) if test_mask.any() else 0.0
    print(f"Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
    return model, df, used_features


def get_feature_importance(
    model,
    used_features: list[str],
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": used_features,
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False)


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("data/final").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/final/lsi.csv")
    model, result, used_features = train_model(df)
    importance = get_feature_importance(model, used_features)
    if "status" in df.columns:
        status_map = df[["date", "status"]].drop_duplicates(subset=["date"])
        status_map["date"] = pd.to_datetime(status_map["date"], errors="coerce")
        result = result.merge(status_map, on="date", how="left")
    joblib.dump(
        {
            "model": model,
            "features": used_features,
        },
        "models/lsi_model.pkl",
    )
    result.to_csv("data/final/lsi_ml.csv", index=False)
    importance.to_csv("data/final/feature_importance.csv", index=False)
    print("Model saved: models/lsi_model.pkl")
    print("Result saved: data/final/lsi_ml.csv")
    print("Feature importance saved: data/final/feature_importance.csv")
    print("\nUsed features:")
    print(used_features)
    print("\nLatest ML LSI:")
    print(
        result[
            [
                "date",
                "ml_lsi",
                "stress_probability",
                "stress_target",
            ]
        ].tail(10)
    )
    print("\nTop feature importance:")
    print(importance.head(20))
    print("\nОбоснование выбора RandomForestClassifier:")
    print("- Устойчив к выбросам и не требует нормализации признаков")
    print("- Встроенная оценка важности признаков (feature_importances_)")
    print("- Работает с бинарной целевой переменной (stress vs normal)")
    print("- class_weight='balanced' компенсирует дисбаланс классов")
    print("- max_depth=5 и min_samples_leaf=5 предотвращают переобучение")


if __name__ == "__main__":
    main()
