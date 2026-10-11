"""
update_quality_check.py
-----------------------
Appends new AI reviews (AI_0061 onwards) to the quality check CSV.
Performs automated checks for:
  - possible_contradiction: positive rating (>=4) with negative words, or negative rating (<=2) with positive words
  - generic_or_template_like: short review (<30 chars) or repeating filler phrases
  - human_quality_review_needed: flagged if either above check fails
"""

import pandas as pd
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
AUGMENTATION_CSV = "data/augmentation/ai_generated_reviews_reviewed.csv"
QUALITY_CSV      = "results/ai_review_quality_check_reviewed.csv"

# ── Load files ─────────────────────────────────────────────────────────────────
aug_df     = pd.read_csv(AUGMENTATION_CSV)
quality_df = pd.read_csv(QUALITY_CSV)

# IDs already in quality check
existing_ids = set(quality_df["review_id"].tolist())

# New reviews not yet in quality check
new_reviews = aug_df[~aug_df["review_id"].isin(existing_ids)].copy()
print(f"Found {len(new_reviews)} new reviews to add to quality check.")

if new_reviews.empty:
    print("Nothing to add. Quality check is already up to date.")
    exit()

# ── Automated quality checks ──────────────────────────────────────────────────

# Malayalam negative sentiment keywords
NEGATIVE_WORDS = [
    "മോശം", "പ്രശ്നം", "കേടായ", "നഷ്ടം", "ഖേദം", "നിരാശ",
    "തകരാർ", "പോരാ", "ദോഷം", "ബുദ്ധിമുട്ട്", "കഷ്ടം",
    "disappointed", "bad", "worst", "terrible", "poor"
]

# Malayalam positive sentiment keywords
POSITIVE_WORDS = [
    "നല്ല", "മികച്ച", "അടിപൊളി", "കിടു", "സൂപ്പർ", "ഇഷ്ടം",
    "ശുപാർശ", "ഫ്രഷ്", "കൊള്ളാം", "ഹാപ്പി", "excellent",
    "great", "good", "amazing", "wonderful", "best"
]

# Generic/template-like filler phrases
GENERIC_PHRASES = [
    "ഈ ഉൽപ്പന്നം", "വളരെ നല്ലത്", "ഞാൻ ശുപാർശ ചെയ്യുന്നു",
    "നല്ല ഉൽപ്പന്നം", "good product", "nice product"
]


def check_contradiction(row):
    """Flag if rating contradicts sentiment in review text."""
    text    = str(row["review_text"]).lower()
    rating  = int(row["ratings"])

    has_negative = any(w.lower() in text for w in NEGATIVE_WORDS)
    has_positive = any(w.lower() in text for w in POSITIVE_WORDS)

    # High rating but negative words, OR low rating but positive words
    if rating >= 4 and has_negative and not has_positive:
        return True
    if rating <= 2 and has_positive and not has_negative:
        return True
    return False


def check_generic(row):
    """Flag if review is too short or uses known generic/template phrases."""
    text = str(row["review_text"])
    # Too short
    if len(text.strip()) < 30:
        return True
    # Contains known generic filler
    for phrase in GENERIC_PHRASES:
        if phrase.lower() in text.lower():
            return True
    return False


new_reviews["possible_contradiction"]      = new_reviews.apply(check_contradiction, axis=1)
new_reviews["generic_or_template_like"]    = new_reviews.apply(check_generic, axis=1)
new_reviews["human_quality_review_needed"] = (
    new_reviews["possible_contradiction"] | new_reviews["generic_or_template_like"]
)

# Ensure quality_status is set
new_reviews["quality_status"] = "manually_cleaned_and_verified"

# ── Select only the columns matching the quality check file ───────────────────
cols = list(quality_df.columns)
new_reviews_out = new_reviews[cols]

# ── Append and save ───────────────────────────────────────────────────────────
updated_df = pd.concat([quality_df, new_reviews_out], ignore_index=True)
updated_df.to_csv(QUALITY_CSV, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
total          = len(updated_df)
flagged_contra = new_reviews["possible_contradiction"].sum()
flagged_generic = new_reviews["generic_or_template_like"].sum()
flagged_human  = new_reviews["human_quality_review_needed"].sum()

print(f"\nDONE - Quality check updated: {QUALITY_CSV}")
print(f"   Total reviews now : {total}")
print(f"   New reviews added : {len(new_reviews)}")
print(f"\n-- Flags in new reviews --")
print(f"   Possible contradictions       : {flagged_contra}")
print(f"   Generic/template-like         : {flagged_generic}")
print(f"   Human review needed (any flag): {flagged_human}")
