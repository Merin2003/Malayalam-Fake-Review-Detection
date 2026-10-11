"""
Utility script to test and predict individual Malayalam reviews
using the trained TF-IDF + Linear SVM model.
"""

from pathlib import Path
import sys
import joblib

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"

LABEL_MAP = {
    0: "Human Real (Genuine)",
    1: "Human Fake (Deceptive)",
    2: "AI Generated (Synthetic)"
}

def predict(review_text: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")
    
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([review_text])[0]
    confidence_decision = model.decision_function([review_text])[0]
    
    print("\n" + "=" * 60)
    print(f"Review: \"{review_text}\"")
    print(f"Predicted Class: {prediction} -> {LABEL_MAP.get(prediction, 'Unknown')}")
    print("Decision scores across classes [Real, Fake, AI]:", confidence_decision.round(3))
    print("=" * 60 + "\n")
    return prediction

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        # Example review
        text = "ഡെലിവറി ഭയങ്കര ഫാസ്റ്റ് ആയിരുന്നു. Shoes ക്വാളിറ്റിയുടെ കാര്യത്തിൽ ഒരു കോംപ്രമൈസും ഇല്ല. എനിക്ക് ഒത്തിരി ഇഷ്ടപ്പെട്ടു."
    predict(text)
