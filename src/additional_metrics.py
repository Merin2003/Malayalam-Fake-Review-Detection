
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"
OUTPUT_PATH = ROOT / "results" / "additional_evaluation_metrics.csv"

def main():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    X_test = test_df["review_text"].astype(str).str.strip()
    y_true = test_df["label"].astype(int)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Weighted Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Weighted F1-score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Macro F1-score": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Micro F1-score": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"Metric": name, "Score": value} for name, value in metrics.items()]
    ).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Additional evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f} ({value:.2%})")

    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nPer-class metrics:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Human Real", "Human Fake", "AI Generated"],
            digits=4,
            zero_division=0,
        )
    )

if __name__ == "__main__":
    main()
