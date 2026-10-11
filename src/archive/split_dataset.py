
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "cleaned" / "master_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Keep only the two verified existing classes for now.
df = df[df["label"].isin([0, 1])].copy()

# Remove rows without usable review text.
df = df.dropna(subset=["review_text"])
df = df[df["review_text"].astype(str).str.strip() != ""]

# First: reserve 20% for test.
train_validation, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

# Second: reserve 20% of the remaining 80% for validation.
train, validation = train_test_split(
    train_validation,
    test_size=0.20,
    random_state=42,
    stratify=train_validation["label"],
)

train.to_csv(OUTPUT_DIR / "train_original.csv", index=False, encoding="utf-8-sig")
validation.to_csv(OUTPUT_DIR / "validation_original.csv", index=False, encoding="utf-8-sig")
test.to_csv(OUTPUT_DIR / "test_original.csv", index=False, encoding="utf-8-sig")

print("Original dataset split completed.")
print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Test rows: {len(test)}")

for name, part in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(f"\n{name} class distribution:")
    print(part["label"].value_counts().sort_index().to_string())
