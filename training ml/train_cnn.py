
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "processed" / "cnn"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train_scaled.npz"
TEST_FILE = DATA_DIR / "test_scaled.npz"
MODEL_FILE = MODEL_DIR / "cnn_model.keras"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prepared CNN datasets...", flush=True)
    with np.load(TRAIN_FILE) as data:
        X_train = data["X"].astype(np.float32)
        y_train = data["y"].astype(np.int32)

    with np.load(TEST_FILE) as data:
        X_test = data["X"].astype(np.float32)
        y_test = data["y"].astype(np.int32)

    print("Training shape:", X_train.shape, flush=True)
    print("Test shape:", X_test.shape, flush=True)

    # CNN input shape: samples × features × channels
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=3, padding="same", activation="relu"
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=3, padding="same", activation="relu"
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
    ]

    print("\nStarting CNN training...", flush=True)
    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=8,
        batch_size=256,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    training_seconds = time.perf_counter() - start

    print("\nEvaluating on the held-out test set...", flush=True)
    probabilities = model.predict(X_test, batch_size=256, verbose=1)
    y_pred = (probabilities.ravel() >= 0.5).astype(np.int32)

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_FILE)
    model_size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)

    print(f"\nTraining time: {training_seconds:.2f} seconds")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved model size: {model_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
