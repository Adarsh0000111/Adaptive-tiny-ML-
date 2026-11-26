"""Leakage-aware preprocessing for the official packet-level Edge-IIoTset CSV."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import load_config
from src.data.edge_packet_schema import (
    NUMERIC_FEATURES, PROTOCOL_CATEGORIES, PROTOCOL_FEATURE, SCHEMA_VERSION,
    prepare_official_rows,
)

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.seed = int(self.config.get("seed", 42))
        self.test_size = float(self.config["dataset"].get("test_size", .2))
        self.val_size = float(self.config["dataset"].get("val_size", .1))
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.cat_encoder = OneHotEncoder(categories=[PROTOCOL_CATEGORIES], handle_unknown="ignore", sparse_output=False)

    def fit_transform(self, df: pd.DataFrame, source_path: str | None = None):
        rows, metadata = prepare_official_rows(df)
        if rows.empty:
            raise ValueError("No supported official Edge-IIoTset rows remain after schema/label mapping")
        y = self.label_encoder.fit_transform(rows["Attack_type"])
        self.class_names = self.label_encoder.classes_.tolist()
        if min(np.bincount(y)) < 3:
            raise ValueError("Each mapped class needs at least three rows for stratified train/validation/test splitting")
        Xn = rows[NUMERIC_FEATURES].to_numpy(dtype=np.float64)
        Xc = rows[[PROTOCOL_FEATURE]]
        X_train_n, X_rest_n, X_train_c, X_rest_c, y_train, y_rest = train_test_split(
            Xn, Xc, y, test_size=self.test_size + self.val_size, random_state=self.seed, stratify=y)
        val_relative = self.val_size / (self.test_size + self.val_size)
        X_val_n, X_test_n, X_val_c, X_test_c, y_val, y_test = train_test_split(
            X_rest_n, X_rest_c, y_rest, test_size=1-val_relative, random_state=self.seed, stratify=y_rest)
        train_n = self.scaler.fit_transform(self.imputer.fit_transform(X_train_n))
        val_n = self.scaler.transform(self.imputer.transform(X_val_n))
        test_n = self.scaler.transform(self.imputer.transform(X_test_n))
        train_c = self.cat_encoder.fit_transform(X_train_c)
        val_c = self.cat_encoder.transform(X_val_c)
        test_c = self.cat_encoder.transform(X_test_c)
        self.feature_names = NUMERIC_FEATURES + self.cat_encoder.get_feature_names_out([PROTOCOL_FEATURE]).tolist()
        X_train, X_val, X_test = (np.hstack((a, b)).astype(np.float32) for a, b in ((train_n, train_c), (val_n, val_c), (test_n, test_c)))

        processed_path = Path(self.config["paths"]["processed_data"]).resolve()
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(processed_path, X_train=X_train, X_val=X_val, X_test=X_test,
                            y_train_multi=y_train, y_val_multi=y_val, y_test_multi=y_test,
                            y_train_bin=(y_train != self.class_names.index("Normal")).astype(np.int8),
                            y_val_bin=(y_val != self.class_names.index("Normal")).astype(np.int8),
                            y_test_bin=(y_test != self.class_names.index("Normal")).astype(np.int8),
                            feature_names=np.array(self.feature_names), class_names=np.array(self.class_names))
        transformer_path = Path(self.config["paths"]["models_dir"]) / "preprocessing_pipeline.joblib"
        transformer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": SCHEMA_VERSION, "imputer": self.imputer, "scaler": self.scaler,
                     "label_encoder": self.label_encoder, "cat_encoder": self.cat_encoder,
                     "feature_names": self.feature_names, "class_names": self.class_names,
                     "num_cols": NUMERIC_FEATURES, "cat_cols": [PROTOCOL_FEATURE]}, transformer_path)
        source_sha = None
        if source_path and Path(source_path).is_file():
            digest = hashlib.sha256()
            with open(source_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_sha = digest.hexdigest()
        manifest = {"schema_version": SCHEMA_VERSION, "source_path": str(Path(source_path).resolve()) if source_path else None,
                    "source_sha256": source_sha, "source_dataset": "Edge-IIoTset ML-EdgeIIoT-dataset.csv",
                    "source_attribution": "Mohamed Amine Ferrag et al., Edge-IIoTset; DOI: 10.1109/ACCESS.2022.3165809",
                    "source_license": "CC BY-NC-SA 4.0", "split": {"seed": self.seed, "train": len(y_train), "validation": len(y_val), "test": len(y_test)},
                    "features": self.feature_names, "classes": self.class_names, "preprocessing": metadata,
                    "evaluation_caveat": "Packet-level stratified random split; dataset has no capture/session identifier, so session-independent generalization is not established."}
        manifest_path = Path(self.config["paths"].get("results_dir", "results")) / "experiments" / "data_provenance.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
                "y_train_multi": y_train, "y_val_multi": y_val, "y_test_multi": y_test,
                "y_train_bin": (y_train != self.class_names.index("Normal")).astype(np.int8),
                "y_val_bin": (y_val != self.class_names.index("Normal")).astype(np.int8),
                "y_test_bin": (y_test != self.class_names.index("Normal")).astype(np.int8),
                "feature_names": self.feature_names, "class_names": self.class_names,
                "schema_version": SCHEMA_VERSION, "metadata": metadata}


def run_preprocessing(config_path="config.yaml", dataset_path: str | None = None):
    from src.data.download import acquire_dataset
    df, resolved_path = acquire_dataset(config_path, dataset_path=dataset_path, return_path=True)
    return PreprocessingPipeline(config_path).fit_transform(df, source_path=resolved_path)


if __name__ == "__main__":
    data = run_preprocessing()
    print("Train/validation/test:", data["X_train"].shape, data["X_val"].shape, data["X_test"].shape)
