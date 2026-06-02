from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .base import BaseLoader


_RU_MONTH_NAMES = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}


def generate_synthetic_tax_calendar(start_year=2014, end_year=2026):
    """Генерирует синтетический налоговый календарь на основе правил РФ."""
    rows = []
    for year in range(start_year, end_year + 1):
        months = [
            (1, 15, 28),
            (2, 15, 28),
            (3, 15, 28, 31),
            (4, 15, 28),
            (5, 15, 28),
            (6, 15, 28),
            (7, 15, 28),
            (8, 15, 28),
            (9, 15, 28, 30),
            (10, 15, 28),
            (11, 15, 28),
            (12, 15, 28, 31),
        ]
        for month_data in months:
            month_num = month_data[0]
            base_day = month_data[1]
            main_day = month_data[2]
            tax_dates = [
                (base_day, "НДС"),
                (main_day, "Налог на прибыль"),
            ]
            if len(month_data) > 3:
                tax_dates.append((month_data[3], "НДФЛ"))
            for day, tax_type in tax_dates:
                try:
                    dt = date(year, month_num, day)
                    rows.append({
                        "Дата": dt.isoformat(),
                        "Тип налога": tax_type,
                        "Описание": f"Уплата {tax_type}",
                    })
                except ValueError:
                    pass
    df = pd.DataFrame(rows)
    print(f"FNS: сгенерировано {len(df)} строк синтетического календаря ({start_year}-{end_year})")
    return df


def _parse_russian_date(text: str) -> Optional[date]:
    text = text.strip()
    for ru_name, num in _RU_MONTH_NAMES.items():
        if ru_name in text:
            parts = text.replace(" 00:00", "").split()
            if len(parts) >= 2:
                try:
                    day = int(parts[0])
                    return date(2026, num, day)
                except (ValueError, IndexError):
                    pass
    return None


class FNSLoader(BaseLoader):
    TAX_CALENDAR_URL = "https://www.nalog.gov.ru/rn77/calendar/"

    def load_tax_calendar(self, save: bool = True) -> pd.DataFrame:
        df = self.read_html_from_url(self.TAX_CALENDAR_URL)

        if df.empty or len(df) < 5:
            print("FNS: сайт не вернул данных — генерация синтетического календаря")
            df = generate_synthetic_tax_calendar()

        if save:
            self.save_csv(df, "fns_tax_calendar.csv")

        return df