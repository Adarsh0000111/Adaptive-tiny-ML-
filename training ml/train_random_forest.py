
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "processed" / "split"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "random_forest.joblib"

TARGET = "Attack_label"
RANDOM_STATE = 42


def main():
    print("Loading training and testing datasets...", flush=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype("int8")

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype("int8")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing features do not match.")

    print(f"Training rows: {len(X_train):,}", flush=True)
    print(f"Testing rows: {len(X_test):,}", flush=True)
    print(f"Input features: {X_train.shape[1]}", flush=True)

    # A deliberately small initial forest to limit resource usage.
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=16,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    print("\nTraining Random Forest...", flush=True)
    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    print("Evaluating model...", flush=True)
    start = perf_counter()
    predictions = model.predict(X_test)
    inference_seconds = perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)

    print("\n========== RANDOM FOREST RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Attack precision: {precision:.4f}")
    print(f"Attack recall: {recall:.4f}")
    print(f"Attack F1-score: {f1:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(
        "Average inference latency: "
        f"{inference_seconds / len(X_test) * 1000:.6f} ms/sample"
    )
    print(f"Model size: {size_mb:.3f} MB")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))

    print(f"\nSaved model: {MODEL_FILE}")


if __name__ == "__main__":
    main()
