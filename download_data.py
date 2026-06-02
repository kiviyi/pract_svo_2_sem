import sys

from src.loaders import CBRLoader, FNSLoader, MinfinLoader, TreasuryLoader


def safe_load(loader, method_name: str, label: str):
    try:
        getattr(loader, method_name)()
        print(f"✓ {label}")
    except Exception as e:
        print(f"✗ {label}: {e}", file=sys.stderr)


def main() -> None:
    cbr = CBRLoader()
    fns = FNSLoader()
    minfin = MinfinLoader()
    treasury = TreasuryLoader()

    safe_load(cbr, "load_required_reserves", "Резервы ЦБ")
    safe_load(cbr, "load_ruonia", "RUONIA")
    safe_load(cbr, "load_repo", "РЕПО")
    safe_load(cbr, "load_repo_debt", "Задолженность РЕПО")
    safe_load(cbr, "load_key_rate", "Ключевая ставка")
    safe_load(cbr, "load_bank_liquidity", "Ликвидность банков")

    safe_load(fns, "load_tax_calendar", "Календарь ФНС")
    safe_load(minfin, "load_ofz_auctions", "ОФЗ")
    safe_load(treasury, "load_treasury_deposits", "Депозиты Казначейства")

    print("\nЗагрузка данных завершена")


if __name__ == "__main__":
    main()
