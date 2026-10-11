
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUG = ROOT / "data" / "augmentation"

native_path = AUG / "native_100.csv"
translation_path = AUG / "translation_100.csv"
output_path = AUG / "ai_generated_reviews.csv"

native = pd.read_csv(native_path, encoding="utf-8-sig")
translation = pd.read_csv(translation_path, encoding="utf-8-sig")

# Preserve the origin and style of every review
native["generation_method"] = "native_malayalam"
translation["generation_method"] = "translation_style_malayalam"

combined = pd.concat(
    [native, translation],
    ignore_index=True,
    sort=False
)

# Check required data before saving
combined["review_text"] = combined["review_text"].fillna("").astype(str).str.strip()

if combined["review_text"].eq("").any():
    raise ValueError("Empty reviews found. Fix them before combining.")

if not combined["label"].eq(2).all():
    raise ValueError("Expected label 2 for every AI-generated review.")

if combined["review_text"].str.casefold().duplicated().any():
    raise ValueError("Duplicate review texts found across the combined file.")

# Give every row a unique ID in the combined file
combined["review_id"] = [
    f"AI_{i:04d}" for i in range(1, len(combined) + 1)
]

combined.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Combined AI reviews:", len(combined))
print("\nGeneration method counts:")
print(combined["generation_method"].value_counts())
print("\nLabel counts:")
print(combined["label"].value_counts())
print("\nSaved to:", output_path)
print("\nOriginal files were not modified.")
