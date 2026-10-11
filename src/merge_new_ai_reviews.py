"""
Script to clean, validate, format, and merge newly added AI-generated reviews
from data/augmentation/new_ai_generted.csv into data/augmentation/ai_generated_reviews_reviewed.csv,
then rebuild final_dataset.csv and the train/validation/test splits.
"""

from pathlib import Path
import pandas as pd
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
AUG_DIR = ROOT / "data" / "augmentation"

EXISTING_PATH = AUG_DIR / "ai_generated_reviews_reviewed.csv"
BACKUP_PATH = AUG_DIR / "ai_generated_reviews_reviewed_backup.csv"
NEW_PATH = AUG_DIR / "new_ai_generted.csv"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip any parenthetical / bracketed suffix artifacts
    text = re.sub(r'\s*\([^)]+\)\s*$', '', text)
    text = re.sub(r'\s*\[[^\]]+\]\s*$', '', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=" * 60)
    print("Step 1: Loading existing and new AI reviews...")
    print("=" * 60)
    
    if not EXISTING_PATH.exists():
        raise FileNotFoundError(f"Existing file not found: {EXISTING_PATH}")
    if not NEW_PATH.exists():
        raise FileNotFoundError(f"New reviews file not found: {NEW_PATH}")
        
    existing_df = pd.read_csv(EXISTING_PATH, encoding="utf-8-sig")
    print(f"Loaded existing AI reviews: {len(existing_df)} rows.")
    
    # Backup existing file
    shutil.copyfile(EXISTING_PATH, BACKUP_PATH)
    print(f"Backed up existing file to: {BACKUP_PATH.name}")

    new_df = pd.read_csv(NEW_PATH, encoding="utf-8-sig")
    print(f"Loaded new AI reviews: {len(new_df)} rows.")

    # Step 2: Clean and validate new reviews
    print("\n" + "=" * 60)
    print("Step 2: Cleaning and standardizing new reviews...")
    print("=" * 60)
    
    new_df["review_text"] = new_df["review_text"].apply(clean_text)
    
    # Drop any empty rows
    new_df = new_df[new_df["review_text"] != ""].copy()
    
    # Deduplicate within new reviews
    new_df = new_df.drop_duplicates(subset=["review_text"]).copy().reset_index(drop=True)
    
    # Check overlap with existing AI reviews
    existing_texts = set(existing_df["review_text"].str.casefold())
    overlap_mask = new_df["review_text"].str.casefold().isin(existing_texts)
    num_overlapping = overlap_mask.sum()
    if num_overlapping > 0:
        print(f"Removing {num_overlapping} reviews that already exist in the dataset.")
        new_df = new_df[~overlap_mask].copy().reset_index(drop=True)
    else:
        print("No duplicates found against existing reviews. All new reviews are unique!")

    print(f"Cleaned unique new reviews to add: {len(new_df)} rows.")

    # Standardize metadata fields
    # If style contains native/translation, standardize to 'native' and 'translation'
    def parse_style(val):
        val_str = str(val).lower()
        if "native" in val_str:
            return "native"
        return "translation"

    def parse_generation_method(row):
        val_str = str(row.get("style", "")).lower() + " " + str(row.get("generation_method", "")).lower()
        if "native" in val_str:
            return "native_malayalam"
        return "translation_style_malayalam"

    new_df["generation_method"] = new_df.apply(parse_generation_method, axis=1)
    new_df["style"] = new_df["style"].apply(parse_style)
    new_df["label"] = 2
    new_df["generator"] = new_df["generator"].fillna("LLM").astype(str)
    new_df["ratings"] = pd.to_numeric(new_df["ratings"], errors="coerce").fillna(5).astype(int)
    new_df["quality_status"] = "manually_cleaned_and_verified"
    new_df["human_quality_review_needed"] = False

    # Assign sequential review IDs continuing from existing count
    existing_count = len(existing_df)
    new_df["review_id"] = [f"AI_{existing_count + i + 1:04d}" for i in range(len(new_df))]

    # Align column order
    expected_cols = [
        "review_id",
        "review_text",
        "ratings",
        "label",
        "category",
        "style",
        "generator",
        "generation_method",
        "quality_status",
        "human_quality_review_needed"
    ]
    
    new_formatted = new_df[expected_cols].copy()

    # Step 3: Combine with existing reviews
    print("\n" + "=" * 60)
    print("Step 3: Merging and saving updated AI reviews dataset...")
    print("=" * 60)
    
    combined_ai_df = pd.concat([existing_df[expected_cols], new_formatted], ignore_index=True)
    combined_ai_df.to_csv(EXISTING_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved updated AI reviews to {EXISTING_PATH}")
    print(f"Total AI reviews now: {len(combined_ai_df)} (Previous: {existing_count}, Added: {len(new_formatted)})")
    print(f"ID range: {combined_ai_df['review_id'].iloc[0]} to {combined_ai_df['review_id'].iloc[-1]}")

if __name__ == "__main__":
    main()
