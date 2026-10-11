"""
Utility script to test and predict individual Malayalam reviews
using the trained TF-IDF + Linear SVM model.
"""

from pathlib import Path
import sys
import joblib

# Ensure Malayalam text displays correctly in the terminal.
if sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower().replace("-", "") != "utf8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"

LABEL_MAP = {
    0: "Human Real (Genuine)",
    1: "Human Fake (Deceptive)",
    2: "AI Generated (Synthetic)",
}


def predict(review_text: str):
    """Predict the class of one Malayalam review."""

    review_text = review_text.strip()

    if not review_text:
        raise ValueError("Review text cannot be empty. Please enter a review.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please check that the trained model file exists."
        )

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([review_text])[0]
    decision_scores = model.decision_function([review_text])[0]

    print("\n" + "=" * 60)
    print(f'Review: "{review_text}"')
    print(
        f"Predicted Class: {prediction} -> "
        f"{LABEL_MAP.get(prediction, 'Unknown')}"
    )
    print(
        "Decision scores across classes [Real, Fake, AI]:",
        decision_scores.round(3),
    )
    print("Note: Decision scores are not probabilities.")
    print("=" * 60 + "\n")

    return prediction


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a Malayalam review: ")

    try:
        predict(text)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        sys.exit(1)