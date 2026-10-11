"""
Script to evaluate the trained TF-IDF + Linear SVM baseline model
on the unseen, held-out test dataset (test_final.csv).
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"
REPORT_OUTPUT = ROOT / "results" / "test_evaluation_results.csv"

CLASS_NAMES = ["Human Real (0)", "Human Fake (1)", "AI Fake (2)"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python src/train_baseline.py' first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_PATH}. Run 'python src/split_final_dataset.py' first.")

    print("=" * 65)
    print("Evaluating TF-IDF Model on Held-out Test Set (test_final.csv)")
    print("=" * 65)

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    X_test = test_df["review_text"].astype(str).str.strip()
    y_test = test_df["label"].astype(int)

    print(f"Total Test Samples: {len(test_df)}")
    print(f"Test Label Distribution:")
    print(y_test.value_counts().sort_index().rename(index={0: '0 (Human Real)', 1: '1 (Human Fake)', 2: '2 (AI Fake)'}))

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f"OVERALL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 65)

    print("\nDetailed Classification Report:")
    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report_str)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print("                Pred Real   Pred Fake   Pred AI")
    print(f"Actual Real:      {cm[0][0]:<10} {cm[0][1]:<10} {cm[0][2]:<10}")
    print(f"Actual Fake:      {cm[1][0]:<10} {cm[1][1]:<10} {cm[1][2]:<10}")
    print(f"Actual AI:        {cm[2][0]:<10} {cm[2][1]:<10} {cm[2][2]:<10}")

    # Save to CSV
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_OUTPUT, encoding="utf-8-sig")
    print(f"\nSaved evaluation metrics to: {REPORT_OUTPUT}")

if __name__ == "__main__":
    main()
