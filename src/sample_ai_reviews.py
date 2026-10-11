
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "augmentation" / "ai_generated_reviews.csv"

df = pd.read_csv(FILE, encoding="utf-8-sig")

print("Total AI reviews:", len(df))
print("\nReviews by generation method:")
print(df["generation_method"].value_counts())

# Show 10 reviews from each method for manual quality checking
for method in ["native_malayalam", "translation_style_malayalam"]:
    subset = df[df["generation_method"] == method].head(10)

    print(f"\n{'=' * 70}")
    print(f"METHOD: {method}")
    print("=" * 70)

    for _, row in subset.iterrows():
        print(f"\nID: {row['review_id']}")
        print(f"Category: {row['category']}")
        print(f"Review: {row['review_text']}")
