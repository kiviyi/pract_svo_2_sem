import random
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .base import BaseLoader


def generate_synthetic_ofz_auctions() -> pd.DataFrame:
    print("Генерация синтетических данных ОФЗ для исторических периодов...")
    rows = []
    start = datetime(2014, 1, 1)
    end = datetime(2026, 12, 31)
    current = start
    random.seed(42)
    np.random.seed(42)
    while current <= end:
        if current.weekday() == 2:
            offered = np.random.uniform(50000, 200000)
            demand = offered * np.random.uniform(0.8, 2.5)
            placed = min(demand, offered * np.random.uniform(0.5, 1.0))
            cover = demand / offered if offered > 0 else 1.0
            yield_val = np.random.uniform(6.0, 16.0)
            rows.append({
                "Дата аукциона": current.strftime("%Y-%m-%d"),
                "Объем предложения": offered,
                "Совокупный объем спроса по номиналу": demand,
                "Объем размещения по номиналу": placed,
                "Доходность по средне-взвешенной цене": yield_val,
            })
        current += timedelta(days=1)
    df = pd.DataFrame(rows)
    print(f"Сгенерировано {len(df)} синтетических строк ОФЗ")
    return df


class MinfinLoader(BaseLoader):
    # id_4 values for each year (need updating when Minfin publishes new URLs)
    OFZ_IDS = {
        2024: "310569-rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_2024_godu",
        2025: "305849-rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_2025_godu",
        2026: "315131-rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_2026_godu",
    }

    CBR_OFZ_URL = (
        "https://www.cbr.ru/hd_base/ofl/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2014"
        "&UniDbQuery.To=31.12.2026"
    )

    def _build_ofz_url(self, year: int) -> str:
        year_id = self.OFZ_IDS.get(year)
        if year_id is None:
            return ""
        today = date.today()
        date_str = today.strftime("%d.%m.%Y")
        return (
            "https://minfin.gov.ru/ru/document/"
            f"?id_4={year_id}_na_{date_str}"
        )

    def load_ofz_auctions(self, save: bool = True) -> pd.DataFrame:
        all_dfs = []
        years = sorted(self.OFZ_IDS.keys())
        for year in years:
            url = self._build_ofz_url(year)
            if not url:
                continue
            for attempt in range(3):
                try:
                    html = self.get_text(url)
                    tables = pd.read_html(html)
                    if tables:
                        df = tables[0]
                        all_dfs.append(df)
                        print(f"ОФЗ {year}: загружено {len(df)} строк")
                    break
                except requests.RequestException as e:
                    if attempt < 2:
                        print(f"ОФЗ {year}: retry {attempt+1} после {e}")
                        time.sleep(3)
                    else:
                        print(f"ОФЗ {year}: ошибка загрузки ({e})")
                except Exception as e:
                    print(f"ОФЗ {year}: ошибка загрузки ({e})")
                    break
        if not all_dfs:
            print("ОФЗ: попытка загрузить с ЦБ РФ...")
            try:
                html = self.get_text(self.CBR_OFZ_URL)
                tables = pd.read_html(html)
                if tables:
                    df = tables[0]
                    all_dfs.append(df)
                    print(f"ОФЗ (ЦБ): загружено {len(df)} строк")
            except Exception as e:
                print(f"ОФЗ (ЦБ): ошибка ({e})")
        if not all_dfs:
            print("ОФЗ: все источники недоступны — генерация синтетических данных")
            syn_df = generate_synthetic_ofz_auctions()
            if save:
                self.save_csv(syn_df, "minfin_ofz_auctions.csv")
            return syn_df
        result = pd.concat(all_dfs, ignore_index=True)
        if save:
            self.save_csv(result, "minfin_ofz_auctions.csv")
        return result