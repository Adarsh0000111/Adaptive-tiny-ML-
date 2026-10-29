
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "processed" / "split"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "decision_tree.joblib"

TARGET = "Attack_label"
RANDOM_STATE = 42

def main():
    print("Loading training and testing data...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")

    # Separate inputs from the target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype("int8")

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype("int8")

    # Ensure both sets have the same feature columns
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing feature columns do not match.")

    print(f"Input features: {X_train.shape[1]}")
    print("\nTraining Decision Tree...")

    model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    print("Training complete. Evaluating model...")

    start = perf_counter()
    predictions = model.predict(X_test)
    total_inference_seconds = perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    model_size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)

    print("\n========== DECISION TREE RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Attack precision: {precision:.4f}")
    print(f"Attack recall: {recall:.4f}")
    print(f"Attack F1-score: {f1:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(
        "Average inference latency: "
        f"{total_inference_seconds / len(X_test) * 1000:.6f} ms/sample"
    )
    print(f"Model size: {model_size_mb:.3f} MB")

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )

    print(f"Saved model: {MODEL_FILE}")

if __name__ == "__main__":
    main()
