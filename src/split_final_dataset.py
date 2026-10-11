"""
Script to create leakage-safe stratified splits (train, validation, test)
for the final 3-class Malayalam Fake Review dataset.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "final_dataset.csv"

TRAIN_PATH = ROOT / "data" / "processed" / "train_final.csv"
VAL_PATH = ROOT / "data" / "processed" / "validation_final.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"

SPLIT_SUMMARY_PATH = ROOT / "results" / "final_split_summary.csv"

RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"Loaded final dataset: {len(df)} rows.")

    # 1. Stratified Train (80%) vs Temp (20% test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # 2. Split Train further into Train (64% of total) and Val (16% of total)
    # Val is 0.20 of original train (0.20 * 0.80 = 0.16 of total)
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_df["label"]
    )

    test_df = temp_df.copy()

    print(f"Train split: {len(train_df)} rows")
    print(f"Validation split: {len(val_df)} rows")
    print(f"Test split: {len(test_df)} rows")

    # Verification of Data Leakage (Overlap check between review texts across splits)
    train_texts = set(train_df["review_text"])
    val_texts = set(val_df["review_text"])
    test_texts = set(test_df["review_text"])

    tv_overlap = len(train_texts.intersection(val_texts))
    tt_overlap = len(train_texts.intersection(test_texts))
    vt_overlap = len(val_texts.intersection(test_texts))

    print(f"\n--- Leakage Verification ---")
    print(f"Train vs Val text overlap: {tv_overlap}")
    print(f"Train vs Test text overlap: {tt_overlap}")
    print(f"Val vs Test text overlap: {vt_overlap}")

    if tv_overlap + tt_overlap + vt_overlap > 0:
        raise ValueError("Data leakage detected between splits!")

    # Save final split files
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved split files to {TRAIN_PATH.parent}")

    # Build summary table across splits and classes
    summary_list = []
    for split_name, split_data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = split_data["class_name"].value_counts().to_dict()
        summary_list.append({
            "Split": split_name,
            "Total Rows": len(split_data),
            "Human Real (0)": counts.get("Human Real", 0),
            "Human Fake (1)": counts.get("Human Fake", 0),
            "AI Generated (2)": counts.get("AI Generated", 0)
        })

    summary_df = pd.DataFrame(summary_list)
    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SPLIT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("\n--- Split Class Breakdown ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
