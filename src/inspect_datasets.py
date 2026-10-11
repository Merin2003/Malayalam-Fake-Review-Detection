
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in: {RAW_DIR}")
    print("Copy your three original CSV files into data/raw first.")
    raise SystemExit(1)

for file_path in csv_files:
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nDuplicate complete rows:", df.duplicated().sum())

    # Print value counts for columns with relatively few unique values.
    print("\nPossible label columns:")
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 10:
            print(f"\nColumn: {col}")
            print(df[col].value_counts(dropna=False).to_string())
