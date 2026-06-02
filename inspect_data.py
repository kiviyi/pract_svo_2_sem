from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def main():
    for path in RAW_DIR.glob("*.csv"):
        print("\n" + "=" * 100)
        print(path.name)

        df = pd.read_csv(path)

        print("\nColumns:")
        for col in df.columns:
            print("-", col)

        print("\nHead:")
        print(df.head(3))


if __name__ == "__main__":
    main()