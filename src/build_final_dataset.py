"""
Script to combine cleaned original human reviews (master_dataset.csv)
and reviewed AI-generated reviews (ai_generated_reviews_reviewed.csv)
into the final 3-class dataset (final_dataset.csv).
Generates class and source distribution summaries.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "data" / "cleaned" / "master_dataset.csv"
AI_PATH = ROOT / "data" / "augmentation" / "ai_generated_reviews_reviewed.csv"

FINAL_DATASET_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
CLASS_DIST_PATH = ROOT / "results" / "final_class_distribution.csv"
SOURCE_DIST_PATH = ROOT / "results" / "final_source_distribution.csv"

LABEL_MAP = {
    0: "Human Real",
    1: "Human Fake",
    2: "AI Generated"
}

def main():
    print("Loading datasets...")
    master_df = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    ai_df = pd.read_csv(AI_PATH, encoding="utf-8-sig")

    print(f"Master dataset rows: {len(master_df)}")
    print(f"AI dataset rows: {len(ai_df)}")

    # Ensure metadata alignment for Master dataset
    master_df["source_type"] = "human"
    master_df["original_source"] = master_df["source_dataset"]
    master_df["class_name"] = master_df["label"].map(LABEL_MAP)

    # Ensure metadata alignment for AI dataset
    ai_df["label"] = 2
    ai_df["class_name"] = "AI Generated"
    ai_df["source_type"] = "ai_generated"
    ai_df["original_source"] = ai_df["generation_method"]

    # Common unified columns
    common_cols = [
        "review_id",
        "review_text",
        "ratings",
        "label",
        "class_name",
        "category",
        "source_type",
        "original_source"
    ]

    final_df = pd.concat([master_df[common_cols], ai_df[common_cols]], ignore_index=True)

    # Clean whitespaces & verify missing values
    final_df["review_text"] = final_df["review_text"].fillna("").astype(str).str.strip()
    missing_texts = (final_df["review_text"] == "").sum()
    if missing_texts > 0:
        raise ValueError(f"Found {missing_texts} empty review texts in final dataset!")

    # Save final dataset
    FINAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_DATASET_PATH, index=False, encoding="utf-8-sig")
    print(f"\nFinal 3-class dataset saved to {FINAL_DATASET_PATH} with {len(final_df)} rows.")

    # Class distribution summary
    class_dist = final_df.groupby(["label", "class_name", "source_type"]).size().reset_index(name="count")
    class_dist["percentage"] = (class_dist["count"] / len(final_df) * 100).round(2)
    CLASS_DIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    class_dist.to_csv(CLASS_DIST_PATH, index=False, encoding="utf-8-sig")
    print("\n--- Final Class Distribution ---")
    print(class_dist.to_string(index=False))

    # Source distribution summary
    source_dist = final_df.groupby(["source_type", "original_source", "class_name"]).size().reset_index(name="count")
    source_dist.to_csv(SOURCE_DIST_PATH, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
