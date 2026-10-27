import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.drop_features = self.config["dataset"]["drop_features"]
        self.target_col = self.config["dataset"]["target_col"]
        self.binary_col = self.config["dataset"]["binary_col"]
        self.test_size = self.config["dataset"]["test_size"]
        self.val_size = self.config["dataset"]["val_size"]
        self.seed = self.config.get("seed", 42)
        
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        
        self.feature_names = None
        self.class_names = None
        
    def fit_transform(self, df):
        logger.info("Starting data preprocessing pipeline (safeguarded against data leakage)...")
        
        # 1. Drop irrelevant metadata features
        drop_cols = [c for c in self.drop_features if c in df.columns]
        logger.info(f"Dropping high-cardinality metadata features: {drop_cols}")
        df_clean = df.drop(columns=drop_cols)
        
        # 2. Extract targets
        y_multi_raw = df_clean[self.target_col].values
        y_bin = df_clean[self.binary_col].values
        
        df_features = df_clean.drop(columns=[self.target_col, self.binary_col])
        
        # Encode multi-class target
        y_multi = self.label_encoder.fit_transform(y_multi_raw)
        self.class_names = self.label_encoder.classes_.tolist()
        logger.info(f"Encoded attack categories ({len(self.class_names)}): {self.class_names}")
        
        # Identify numerical and categorical columns
        cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Split data into train, val, test BEFORE fit to prevent data leakage
        # Train: 70%, Val: 10%, Test: 20%
        X_train_raw, X_temp_raw, y_train_multi, y_temp_multi, y_train_bin, y_temp_bin = train_test_split(
            df_features, y_multi, y_bin,
            test_size=(self.test_size + self.val_size),
            random_state=self.seed,
            stratify=y_multi
        )
        
        val_ratio_relative = self.val_size / (self.test_size + self.val_size)
        X_val_raw, X_test_raw, y_val_multi, y_test_multi, y_val_bin, y_test_bin = train_test_split(
            X_temp_raw, y_temp_multi, y_temp_bin,
            test_size=(1.0 - val_ratio_relative),
            random_state=self.seed,
            stratify=y_temp_multi
        )
        
        logger.info(f"Dataset split ratios -> Train: {len(X_train_raw)} (70%), Val: {len(X_val_raw)} (10%), Test: {len(X_test_raw)} (20%)")
        
        # Fit Imputer and Scaler ONLY on Training set
        X_train_num = X_train_raw[num_cols].values
        X_val_num = X_val_raw[num_cols].values
        X_test_num = X_test_raw[num_cols].values
        
        X_train_num_imp = self.imputer.fit_transform(X_train_num)
        X_val_num_imp = self.imputer.transform(X_val_num)
        X_test_num_imp = self.imputer.transform(X_test_num)
        
        X_train_num_scaled = self.scaler.fit_transform(X_train_num_imp)
        X_val_num_scaled = self.scaler.transform(X_val_num_imp)
        X_test_num_scaled = self.scaler.transform(X_test_num_imp)
        
        # Categorical encoding
        if len(cat_cols) > 0:
            X_train_cat = self.cat_encoder.fit_transform(X_train_raw[cat_cols])
            X_val_cat = self.cat_encoder.transform(X_val_raw[cat_cols])
            X_test_cat = self.cat_encoder.transform(X_test_raw[cat_cols])
            
            cat_feature_names = self.cat_encoder.get_feature_names_out(cat_cols).tolist()
            self.feature_names = num_cols + cat_feature_names
            
            X_train = np.hstack([X_train_num_scaled, X_train_cat])
            X_val = np.hstack([X_val_num_scaled, X_val_cat])
            X_test = np.hstack([X_test_num_scaled, X_test_cat])
        else:
            self.feature_names = num_cols
            X_train = X_train_num_scaled
            X_val = X_val_num_scaled
            X_test = X_test_num_scaled
            
        logger.info(f"Final Preprocessed Feature Dimension: {X_train.shape[1]} features")
        
        # Save processed dataset
        processed_path = os.path.abspath(self.config["paths"]["processed_data"])
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        np.savez_compressed(
            processed_path,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train_multi=y_train_multi, y_val_multi=y_val_multi, y_test_multi=y_test_multi,
            y_train_bin=y_train_bin, y_val_bin=y_val_bin, y_test_bin=y_test_bin,
            feature_names=np.array(self.feature_names),
            class_names=np.array(self.class_names)
        )
        logger.info(f"Processed dataset saved to {processed_path}")
        
        # Save fitted transformers
        transformer_path = os.path.join(self.config["paths"]["models_dir"], "preprocessing_pipeline.joblib")
        os.makedirs(os.path.dirname(transformer_path), exist_ok=True)
        joblib.dump({
            "imputer": self.imputer,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "cat_encoder": self.cat_encoder,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "drop_features": drop_cols
        }, transformer_path)
        logger.info(f"Preprocessing artifacts saved to {transformer_path}")
        
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train_multi": y_train_multi, "y_val_multi": y_val_multi, "y_test_multi": y_test_multi,
            "y_train_bin": y_train_bin, "y_val_bin": y_val_bin, "y_test_bin": y_test_bin,
            "feature_names": self.feature_names, "class_names": self.class_names
        }

def run_preprocessing(config_path="config.yaml"):
    from src.data.download import acquire_dataset
    df = acquire_dataset(config_path)
    pipeline = PreprocessingPipeline(config_path)
    return pipeline.fit_transform(df)

if __name__ == "__main__":
    data_dict = run_preprocessing()
    print("X_train shape:", data_dict["X_train"].shape)
    print("X_test shape:", data_dict["X_test"].shape)
