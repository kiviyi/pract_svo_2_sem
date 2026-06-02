import subprocess
import sys


commands = [
    ["python3", "download_data.py"],
    ["python3", "load_data.py"],
    ["python3", "src/model.py"],
    ["python3", "src/backtest.py"],
    ["python3", "src/sensitivity.py"],
]


def main():
    for command in commands:
        print("\n" + "=" * 60)
        print("Running:", " ".join(command))
        print("=" * 60)

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: {' '.join(command)} завершился с ошибкой {e.returncode}", file=sys.stderr)
            print("Продолжаем со следующим шагом...")

    print("\nГотово")


if __name__ == "__main__":
    main()
