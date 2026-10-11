
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_DIR / "data" / "processed" / "train_original.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "augmentation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")

# Keep only verified human-labeled classes.
df = df[df["label"].isin([0, 1])].copy()

# Create a separate augmentation working file.
df["augmentation_type"] = "original"
df["authorship"] = "human"
df["review_language_style"] = "existing_translation"

output_file = OUTPUT_DIR / "augmentation_working.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Augmentation working file created successfully.")
print(f"Reviews: {len(df)}")
print(f"Saved to: {output_file}")
print("\nClass distribution:")
print(df["label"].value_counts().sort_index().to_string())
print("\nNext: add verified new examples separately.")
