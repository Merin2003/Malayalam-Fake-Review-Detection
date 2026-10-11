
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUGMENTATION_DIR = PROJECT_ROOT / "data" / "augmentation"

files = [
    AUGMENTATION_DIR / "native_100.csv",
    AUGMENTATION_DIR / "translation_100.csv",
]

for file_path in files:
    print("\n" + "=" * 60)
    print("FILE:", file_path.name)

    if not file_path.exists():
        print("ERROR: File not found!")
        continue

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="utf-8")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum())

    # Find the likely review-text column
    candidates = [
        col for col in df.columns
        if col.strip().lower() in [
            "review_text", "review", "text", "reviews"
        ]
    ]

    if candidates:
        text_col = candidates[0]
        text = df[text_col].fillna("").astype(str).str.strip()

        print("\nEmpty reviews:", text.eq("").sum())
        print(
            "Exact duplicate reviews:",
            text[text.ne("")].str.casefold().duplicated().sum()
        )

        print("\nFirst 5 reviews:")
        for i, review in enumerate(text.head(5), start=1):
            print(f"{i}. {review}")
    else:
        print("\nReview-text column not identified.")
        print("Please check the column names above.")
