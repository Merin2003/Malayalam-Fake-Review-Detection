
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "augmentation" / "ai_generated_reviews.csv"
OUTPUT = ROOT / "results" / "ai_review_quality_check.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig")

# These are flags for manual inspection, not automatic judgments.
checks = {
    "possible_contradiction": [
        ("പ്രശ്നം", "അടിപൊളി എക്സ്പീരിയൻസ്"),
        ("വാങ്ങല്ലേ", "വാങ്ങിക്കോളൂ"),
        ("പണം നഷ്ടപ്പെട്ടു", "വീണ്ടും വാങ്ങും"),
        ("ശുപാർശ ചെയ്യുന്നു", "പോരായ്മ"),
    ],
    "generic_or_template_like": [
        "വളരെ നല്ലത്",
        "മികച്ച അനുഭവം",
        "വളരെ മികച്ചത്",
        "എല്ലാവർക്കും ഉപകാരപ്പെടും",
    ],
}

for column, phrases in checks.items():
    if column == "possible_contradiction":
        df[column] = df["review_text"].fillna("").apply(
            lambda text: any(
                first in text and second in text
                for first, second in phrases
            )
        )
    else:
        df[column] = df["review_text"].fillna("").apply(
            lambda text: sum(phrase in text for phrase in phrases) >= 2
        )

df["human_quality_review_needed"] = True
df["quality_status"] = "pending_manual_review"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("Reviews assessed:", len(df))
print("Possible contradictions flagged:",
      int(df["possible_contradiction"].sum()))
print("Potentially generic reviews flagged:",
      int(df["generic_or_template_like"].sum()))
print("Saved quality-control report:", OUTPUT)
print("Original AI dataset remains unchanged.")
