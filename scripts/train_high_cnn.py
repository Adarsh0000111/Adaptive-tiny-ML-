"""Train/evaluate a 1D CNN candidate for SARA's high laptop tier.

Selection uses only the saved validation split. The previous high artifact is
retained as an MLP baseline, and the active artifact is replaced only when the
CNN stays within two percentage points of its validation accuracy, macro recall,
and macro-F1. The held-out test split is reported only after that decision.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cnn import CNN_ARCHITECTURE_VERSION, PacketCNNClassifier


def metrics(model, x, y) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_precision": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    data = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
    feature_names = data["feature_names"].tolist()
    class_names = data["class_names"].tolist()
    x_train = data["X_train"].astype(np.float32, copy=False)
    y_train = data["y_train_multi"].astype(np.int64, copy=False)
    x_val = data["X_val"].astype(np.float32, copy=False)
    y_val = data["y_val_multi"].astype(np.int64, copy=False)
    x_test = data["X_test"].astype(np.float32, copy=False)
    y_test = data["y_test_multi"].astype(np.int64, copy=False)
    if x_train.shape[1] != 18 or len(class_names) != 6:
        raise ValueError("The high CNN expects the saved 18-feature, six-class packet schema")

    active_path = ROOT / "models/heavy/model.joblib"
    baseline_path = ROOT / "models/heavy/model_mlp_baseline.joblib"
    candidate_path = ROOT / "models/heavy/cnn_candidate.joblib"
    report_path = ROOT / "results/experiments/high_cnn_evaluation.json"
    baseline = joblib.load(active_path)
    if not baseline_path.exists() and type(baseline).__name__ == "MLPClassifier":
        shutil.copy2(active_path, baseline_path)
    baseline_validation = metrics(baseline, x_val, y_val)
    baseline_name = type(baseline).__name__

    model = PacketCNNClassifier(input_features=x_train.shape[1], classes=len(class_names))
    started = time.perf_counter()
    model.fit(x_train, y_train, validation_data=(x_val, y_val))
    training_seconds = time.perf_counter() - started
    candidate_validation = metrics(model, x_val, y_val)
    gate = {key: baseline_validation[key] - 0.02 for key in ("accuracy", "macro_recall", "macro_f1")}
    approved = all(candidate_validation[key] >= gate[key] for key in gate)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate_path)

    test_metrics = metrics(model, x_test, y_test) if approved else None
    if approved:
        shutil.copy2(candidate_path, active_path)

    report = {
        "model": "1D CNN over ordered packet features",
        "architecture_version": CNN_ARCHITECTURE_VERSION,
        "input_semantics": "18 ordered packet fields treated as a one-dimensional feature axis; not temporal packet sequences",
        "data": {"training_samples": len(y_train), "validation_samples": len(y_val), "test_samples": len(y_test),
                 "feature_names": feature_names, "class_names": class_names,
                 "test_split_used_for_model_selection": False},
        "training": {"epochs_run": len(model.training_history_), "training_seconds": training_seconds,
                     "batch_size": model.batch_size, "learning_rate": model.learning_rate,
                     "class_weighting": "square-root inverse frequency from training labels"},
        "baseline_model": baseline_name,
        "baseline_validation": baseline_validation,
        "candidate_cnn_validation": candidate_validation,
        "promotion_gate": {"rule": "each validation metric must be within 0.02 of the current high-tier model",
                           "minimums": gate},
        "promoted_to_high_tier": approved,
        "promotion_reason": "CNN passed the validation gate and is now the active high-tier artifact." if approved
                            else "CNN did not pass the validation gate; current high-tier artifact remains unchanged.",
        "candidate_artifact": str(candidate_path.relative_to(ROOT)),
        "active_artifact": str(active_path.relative_to(ROOT)) if approved else f"{active_path.relative_to(ROOT)} (existing high-tier model retained)",
        "candidate_test_metrics": test_metrics,
        "training_history": model.training_history_,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"promoted": approved, "reason": report["promotion_reason"],
                      "baseline_model": baseline_name,
                      "baseline_validation": baseline_validation,
                      "candidate_cnn_validation": candidate_validation,
                      "candidate_cnn_test": test_metrics,
                      "epochs_run": len(model.training_history_),
                      "training_seconds": round(training_seconds, 2),
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
