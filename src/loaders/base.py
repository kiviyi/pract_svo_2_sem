from io import BytesIO, StringIO
from pathlib import Path
import subprocess

import pandas as pd
import requests


class BaseLoader:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def get(self, url: str) -> requests.Response:
        response = requests.get(
            url,
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response

    def get_text(self, url: str) -> str:
        try:
            response = self.get(url)
            return response.text
        except requests.exceptions.SSLError:
            return self.get_text_with_curl(url)

    def get_bytes(self, url: str) -> bytes:
        try:
            response = self.get(url)
            return response.content
        except requests.exceptions.SSLError:
            return self.get_bytes_with_curl(url)

    def get_text_with_curl(self, url: str) -> str:
        result = subprocess.run(
            [
                "curl",
                "-k",          # игнорировать SSL certificate errors
                "-L",
                "--http1.1",
                "--tlsv1.2",
                "-A",
                self.headers["User-Agent"],
                url,
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout.decode("utf-8", errors="ignore")


    def get_bytes_with_curl(self, url: str) -> bytes:
        result = subprocess.run(
            [
                "curl",
                "-k",          # игнорировать SSL certificate errors
                "-L",
                "--http1.1",
                "--tlsv1.2",
                "-A",
                self.headers["User-Agent"],
                url,
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def read_excel_from_url(self, url: str, **kwargs) -> pd.DataFrame:
        content = self.get_bytes(url)
        return pd.read_excel(BytesIO(content), **kwargs)

    def read_html_from_url(self, url: str, table_index: int = 0) -> pd.DataFrame:
        html = self.get_text(url)
        tables = pd.read_html(StringIO(html))

        if not tables:
            raise ValueError(f"На странице не найдены таблицы: {url}")

        return tables[table_index]

    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.raw_dir / filename
        df.to_csv(path, index=False)
        return path