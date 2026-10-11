
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"
OUTPUT_PATH = ROOT / "results" / "error_analysis.csv"

LABEL_NAMES = {
    0: "Human Real",
    1: "Human Fake",
    2: "AI Generated",
}


def main():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    X_test = test_df["review_text"].astype(str).str.strip()
    y_true = test_df["label"].astype(int)
    y_pred = model.predict(X_test)

    # Select only reviews classified incorrectly.
    errors = test_df.loc[y_true != y_pred].copy()
    errors["actual_label"] = y_true.loc[errors.index]
    errors["predicted_label"] = y_pred[y_true.to_numpy() != y_pred]
    errors["actual_class"] = errors["actual_label"].map(LABEL_NAMES)
    errors["predicted_class"] = errors["predicted_label"].map(LABEL_NAMES)

    # Save useful columns first, followed by any remaining dataset columns.
    first_columns = [
        "review_id",
        "review_text",
        "actual_label",
        "actual_class",
        "predicted_label",
        "predicted_class",
    ]
    columns = [c for c in first_columns if c in errors.columns]
    columns += [c for c in errors.columns if c not in columns]
    errors = errors[columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Total test reviews: {len(test_df)}")
    print(f"Correct predictions: {(y_true.to_numpy() == y_pred).sum()}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"Error analysis saved to: {OUTPUT_PATH}")

    print("\nErrors by actual class:")
    print(errors["actual_class"].value_counts().to_string())

    print("\nActual class vs predicted class:")
    print(
        pd.crosstab(
            errors["actual_class"],
            errors["predicted_class"],
            dropna=False,
        ).to_string()
    )


if __name__ == "__main__":
    main()
