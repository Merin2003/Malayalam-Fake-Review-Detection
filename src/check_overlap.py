
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT = PROJECT_DIR / "results" / "duplicate_review_candidates.csv"

if not REPORT.exists():
    raise FileNotFoundError(
        "Duplicate report not found. Run prepare_dataset.py first."
    )

df = pd.read_csv(REPORT, encoding="utf-8-sig")

print("Rows listed in duplicate report:", len(df))

print("\nDuplicate candidates by source:")
print(df["source_dataset"].value_counts().to_string())

print("\nDuplicate groups across sources:")

df["review_key"] = (
    df["review_text"]
    .astype("string")
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.casefold()
)

groups = df.groupby("review_key")

cross_source_groups = groups.filter(
    lambda group: group["source_dataset"].nunique() > 1
)

if cross_source_groups.empty:
    print("No exact-text duplicates across different sources.")
else:
    summary = (
        cross_source_groups.groupby("source_dataset")
        .size()
        .rename("duplicate_rows")
        .reset_index()
    )
    print(summary.to_string(index=False))

print("\nLabel combinations among duplicate candidates:")
print(
    df.groupby(["source_dataset", "label"])
    .size()
    .to_string()
)
