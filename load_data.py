from pathlib import Path

import pandas as pd

from src.modules.m1_reserves import calculate_m1
from src.modules.m2_repo import calculate_m2
from src.modules.m3_ofz import calculate_m3
from src.modules.m4_tax import calculate_m4 as calculate_m4_synthetic
from src.modules.m5_treasury import calculate_m5
from src.lsi import (
    prepare_lsi_dataset,
    calculate_lsi,
)


RAW_DIR = Path("data/raw")
FINAL_DIR = Path("data/final")

FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)


def main():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    reserves = load_csv("cbr_required_reserves.csv")
    ruonia = load_csv("cbr_ruonia.csv")
    repo_debt = load_csv("cbr_repo_debt.csv")
    key_rate = load_csv("cbr_key_rate.csv")
    bank_liquidity = load_csv("cbr_bank_liquidity.csv")
    ofz = load_csv("minfin_ofz_auctions.csv")
    print("OK")
    print("=" * 60)
    print("Calculating modules...")
    print("=" * 60)
    m1 = calculate_m1(reserves_df=reserves, ruonia_df=ruonia)
    print("M1 done")
    m2 = calculate_m2(repo_debt_df=repo_debt, key_rate_df=key_rate)
    print("M2 done")
    m3 = calculate_m3(ofz_df=ofz)
    print("M3 done")
    try:
        fns_data = load_csv("fns_tax_calendar.csv")
        m4 = calculate_m4_synthetic(fns_df=fns_data)
        print("M4 done (with FNS calendar)")
    except (FileNotFoundError, Exception) as e:
        m4 = calculate_m4_synthetic()
        print(f"M4 done (synthetic, FNS not available: {e})")
    try:
        treasury_data = load_csv("treasury_deposits.csv")
        m5 = calculate_m5(bank_liquidity_df=bank_liquidity, treasury_df=treasury_data)
        print("M5 done (with treasury deposits)")
    except (FileNotFoundError, Exception) as e:
        m5 = calculate_m5(bank_liquidity_df=bank_liquidity)
        print(f"M5 done (without treasury, error: {e})")
    print("=" * 60)
    print("Building LSI dataset...")
    print("=" * 60)
    dataset = prepare_lsi_dataset(m1=m1, m2=m2, m3=m3, m4=m4, m5=m5)
    result = calculate_lsi(dataset)
    output_path = FINAL_DIR / "lsi.csv"
    result.to_csv(output_path, index=False)
    print("=" * 60)
    print("Finished")
    print("=" * 60)
    print(f"Saved: {output_path}")
    print("\nLatest values:\n")
    cols = ["date", "lsi", "status"]
    existing_cols = [c for c in cols if c in result.columns]
    print(result[existing_cols].tail(10))


if __name__ == "__main__":
    main()
