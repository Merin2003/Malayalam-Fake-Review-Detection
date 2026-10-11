
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "cleaned"
RESULTS_DIR = PROJECT_DIR / "results"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "20kto3k (1).csv",
    "mal20kto1000 (1).csv",
    "malayalam sample aug 27 (1).csv",
]

# New final label mapping
HUMAN_REAL = 0
HUMAN_FAKE = 1
AI_GENERATED = 2

LABEL_NAMES = {
    0: "Human Real",
    1: "Human Fake",
    2: "AI Generated",
}


def read_dataset(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"review_text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )

    df["source_dataset"] = filename

    if filename == "malayalam sample aug 27 (1).csv":
        df["translation_method"] = "human"
        df["proofreading_status"] = "human_translated"
    else:
        df["translation_method"] = "machine"
        df["proofreading_status"] = "pending"

    return df


# Read and combine the original datasets.
frames = [read_dataset(name) for name in DATASETS]
combined = pd.concat(frames, ignore_index=True, sort=False)

print("Rows before cleaning:", len(combined))

# Preserve the source labels before changing their meanings.
combined["original_label"] = combined["label"]
numeric_labels = pd.to_numeric(combined["label"], errors="coerce")

invalid = numeric_labels.isna() | ~numeric_labels.isin([0, 1])

if invalid.any():
    combined.loc[invalid].to_csv(
        RESULTS_DIR / "invalid_label_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raise ValueError(
        f"{int(invalid.sum())} rows have missing or invalid original labels. "
        "Check results/invalid_label_rows.csv."
    )

combined["original_label"] = numeric_labels.astype(int)

# Original dataset: 0 = fake, 1 = genuine.
# New dataset:      0 = Human Real, 1 = Human Fake.
combined["label"] = combined["original_label"].map({
    0: HUMAN_FAKE,
    1: HUMAN_REAL,
}).astype(int)

# Clean blank review text and normalise whitespace.
combined["review_text"] = combined["review_text"].astype("string").str.strip()

blank = combined["review_text"].isna() | combined["review_text"].eq("")
print("Missing or blank review texts:", int(blank.sum()))

combined = combined.loc[~blank].copy()

combined["review_text"] = (
    combined["review_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Case-insensitive exact-text key for duplicate checks.
combined["review_key"] = combined["review_text"].str.casefold()

duplicate_mask = combined.duplicated(
    subset=["review_key"], keep=False
)

combined.loc[duplicate_mask].sort_values("review_key").to_csv(
    RESULTS_DIR / "duplicate_review_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "Rows involved in exact-text duplicates:",
    int(duplicate_mask.sum()),
)

# A single review text should not have different target labels.
label_counts = combined.groupby("review_key")["label"].nunique()
conflicting_keys = label_counts[label_counts > 1].index

conflicts = combined[combined["review_key"].isin(conflicting_keys)]

conflicts.to_csv(
    RESULTS_DIR / "conflicting_label_reviews.csv",
    index=False,
    encoding="utf-8-sig",
)

if len(conflicting_keys) > 0:
    raise ValueError(
        f"Found {len(conflicting_keys)} review texts with conflicting labels. "
        "Review results/conflicting_label_reviews.csv before continuing."
    )

# Keep one copy of each exact review text.
cleaned = combined.drop_duplicates(
    subset=["review_key"], keep="first"
).copy()

cleaned = cleaned.drop(columns=["review_key"])

cleaned.insert(
    0,
    "review_id",
    [f"MR{i:05d}" for i in range(1, len(cleaned) + 1)],
)

cleaned["class_name"] = cleaned["label"].map(LABEL_NAMES)

master_path = CLEAN_DIR / "master_dataset.csv"

cleaned.to_csv(
    master_path,
    index=False,
    encoding="utf-8-sig",
)

# Save the new three-class distribution.
class_counts = (
    cleaned.groupby(["label", "class_name"])
    .size()
    .reset_index(name="count")
    .sort_values("label")
)

class_counts.to_csv(
    RESULTS_DIR / "class_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

# Save source and translation metadata counts.
source_counts = (
    cleaned.groupby(
        ["source_dataset", "translation_method", "proofreading_status"],
        dropna=False,
    )
    .size()
    .reset_index(name="count")
)

source_counts.to_csv(
    RESULTS_DIR / "source_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nDataset preparation completed.")
print("Master dataset:", master_path)
print("Rows after cleaning:", len(cleaned))
print("\nClass distribution:")
print(class_counts.to_string(index=False))
print("\nSource distribution:")
print(source_counts.to_string(index=False))
print("\nAI-generated examples included: 0")
print("Label 2 is reserved for verified AI-generated reviews.")

