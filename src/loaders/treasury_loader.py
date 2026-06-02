import re

import pandas as pd

from .base import BaseLoader


_RU_MONTH_MAP = {
    "января": "01", "февраля": "02", "марта": "03",
    "апреля": "04", "мая": "05", "июня": "06",
    "июля": "07", "августа": "08", "сентября": "09",
    "октября": "10", "ноября": "11", "декабря": "12",
}


def _parse_russian_date(text: str):
    text = text.strip()
    for ru_name, num in _RU_MONTH_MAP.items():
        if ru_name in text:
            parts = text.replace(" 00:00", "").split()
            if len(parts) >= 2:
                try:
                    return f"{parts[0]}.{num}.{parts[2] if len(parts) > 2 else '2026'}"
                except (ValueError, IndexError):
                    pass
    return text


class TreasuryLoader(BaseLoader):
    TREASURY_DEPOSITS_URL = (
        "https://roskazna.gov.ru/finansovye-operacii/"
        "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/"
        "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah/"
    )

    def load_treasury_deposits(self, save: bool = True) -> pd.DataFrame:
        df = self.read_html_from_url(self.TREASURY_DEPOSITS_URL)

        if not df.empty and "Дата" in df.columns:
            df["date_parsed"] = df["Дата"].astype(str).apply(_parse_russian_date)
            df["date"] = pd.to_datetime(df["date_parsed"], errors="coerce", dayfirst=True)
            df["documents"] = df["Документы"].astype(str)
            df["num_xml"] = df["documents"].str.count(r"\.xml", flags=re.IGNORECASE)
            df["num_docx"] = df["documents"].str.count(r"\.docx", flags=re.IGNORECASE)
            df["num_placements"] = df["documents"].str.count(r"otbor", flags=re.IGNORECASE)
            df = df.dropna(subset=["date"])
            print(f"Росказна: загружено {len(df)} строк, "
                  f"всего размещений: {df['num_placements'].sum()}")
            result = df[["date", "num_placements", "num_xml", "num_docx"]].copy()
        else:
            print("Росказна: сайт недоступен — пустой датафрейм")
            result = pd.DataFrame(columns=["date", "num_placements", "num_xml", "num_docx"])

        if save:
            self.save_csv(result, "treasury_deposits.csv")

        return result