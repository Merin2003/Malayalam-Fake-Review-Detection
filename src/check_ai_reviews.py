
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUG = ROOT / "data" / "augmentation"
PROCESSED = ROOT / "data" / "processed"

native_path = AUG / "native_100.csv"
translation_path = AUG / "translation_100.csv"

native = pd.read_csv(native_path, encoding="utf-8-sig")
translation = pd.read_csv(translation_path, encoding="utf-8-sig")

required = {"review_id", "review_text", "label", "category", "style", "generator"}

for name, df in [("Native", native), ("Translation", translation)]:
    missing_columns = required - set(df.columns)

    print(f"\n--- {name} reviews ---")
    print("Rows:", len(df))
    print("Label counts:")
    print(df["label"].value_counts(dropna=False).to_string())

    if missing_columns:
        print("Missing expected columns:", sorted(missing_columns))

    print("Unique review IDs:", df["review_id"].nunique())
    print("Unique review texts:",
          df["review_text"].fillna("").str.strip().str.casefold().nunique())

# Check labels
for name, df in [("Native", native), ("Translation", translation)]:
    labels = set(df["label"].dropna().unique())
    print(f"{name}: labels are {labels}")
    if labels != {2}:
        print("WARNING: Expected every AI-generated review to have label 2.")

# Compare the two AI files
native_text = set(
    native["review_text"].dropna().str.strip().str.casefold()
)
translation_text = set(
    translation["review_text"].dropna().str.strip().str.casefold()
)

cross_duplicates = native_text & translation_text
print("\nExact duplicates across AI files:", len(cross_duplicates))

# Compare against existing training data only
train_path = PROCESSED / "train_original.csv"

if train_path.exists():
    train = pd.read_csv(train_path, encoding="utf-8-sig")

    train_text = set(
        train["review_text"].dropna().str.strip().str.casefold()
    )

    ai_text = native_text | translation_text
    overlap = ai_text & train_text

    print("Exact overlaps with original training data:", len(overlap))

    if overlap:
        pd.DataFrame({"review_text": sorted(overlap)}).to_csv(
            ROOT / "results" / "ai_training_overlap.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print("Overlap report saved to results/ai_training_overlap.csv")
else:
    print("Training file not found:", train_path)

print("\nCheck complete. No original data has been modified.")
