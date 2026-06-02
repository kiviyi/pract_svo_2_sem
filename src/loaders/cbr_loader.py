from io import StringIO

import pandas as pd

from .base import BaseLoader


class CBRLoader(BaseLoader):
    REQUIRED_RESERVES_URL = (
        "https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx"
    )

    RUONIA_URL = (
        "https://www.cbr.ru/hd_base/ruonia/dynamics/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2014"
        "&UniDbQuery.To=31.12.2026"
    )
    REPO_URL = "https://www.cbr.ru/hd_base/repo/"
    KEY_RATE_URL = (
        "https://www.cbr.ru/hd_base/keyrate/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2014"
        "&UniDbQuery.To=31.12.2026"
    )

    BANK_LIQUIDITY_URL = (
        "https://www.cbr.ru/hd_base/bliquidity/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2014"
        "&UniDbQuery.To=31.12.2026"
    )

    REPO_DEBT_URL = (
        "https://www.cbr.ru/hd_base/repo_debt/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2014"
        "&UniDbQuery.To=31.12.2026"
    )

    def load_required_reserves(self, save: bool = True) -> pd.DataFrame:
        df = self.read_excel_from_url(self.REQUIRED_RESERVES_URL)

        if save:
            self.save_csv(df, "cbr_required_reserves.csv")

        return df

    def load_ruonia(self, save: bool = True) -> pd.DataFrame:
        df = self.read_html_from_url(self.RUONIA_URL)

        if save:
            self.save_csv(df, "cbr_ruonia.csv")

        return df

    def load_repo(self, save: bool = True) -> pd.DataFrame:
        df = self.read_html_from_url(self.REPO_URL)

        if save:
            self.save_csv(df, "cbr_repo.csv")

        return df

    def load_repo_debt(self, save: bool = True) -> pd.DataFrame:
        html = self.get_text(self.REPO_DEBT_URL)
        tables = pd.read_html(StringIO(html))

        if not tables:
            raise ValueError("Не найдены таблицы на странице repo_debt")

        best_table = None
        best_score = -1

        for table in tables:
            table_text = table.astype(str).head(10).to_string().lower()
            cols_text = " ".join(map(str, table.columns)).lower()

            score = 0

            if "дата" in table_text or "дата" in cols_text:
                score += 10

            if "репо" in table_text or "репо" in cols_text:
                score += 10

            if "требован" in table_text or "требован" in cols_text:
                score += 10

            score += min(len(table), 100)

            if score > best_score:
                best_score = score
                best_table = table

        df = best_table.copy()

        if save:
            self.save_csv(df, "cbr_repo_debt.csv")

        return df

    def load_key_rate(self, save: bool = True) -> pd.DataFrame:
        df = self.read_html_from_url(self.KEY_RATE_URL)

        if save:
            self.save_csv(df, "cbr_key_rate.csv")

        return df

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        flat = []
        for col in df.columns:
            parts = [str(c).strip() for c in col if str(c).strip() not in ("nan", "")]
            flat.append(" / ".join(parts))
        df = df.copy()
        df.columns = flat
        return df

    def load_bank_liquidity(self, save: bool = True) -> pd.DataFrame:
        html = self.get_text(self.BANK_LIQUIDITY_URL)
        tables = pd.read_html(StringIO(html))

        if not tables:
            raise ValueError("Не найдены таблицы bank_liquidity")

        df = self._flatten_columns(tables[0])

        if save:
            self.save_csv(df, "cbr_bank_liquidity.csv")

        return df