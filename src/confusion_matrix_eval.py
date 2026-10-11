
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"
RESULTS_DIR = ROOT / "results"

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    X_test = test_df["review_text"].astype(str).str.strip()
    y_test = test_df["label"].astype(int)

    y_pred = model.predict(X_test)

    labels = [0, 1, 2]
    class_names = ["Human Real", "Human Fake", "AI Generated"]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )

    cm_path = RESULTS_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Baseline Linear SVM - Test Confusion Matrix")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    image_path = RESULTS_DIR / "confusion_matrix_baseline.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved:")
    print(cm_df)
    print(f"\nCSV: {cm_path}")
    print(f"Image: {image_path}")

if __name__ == "__main__":
    main()
