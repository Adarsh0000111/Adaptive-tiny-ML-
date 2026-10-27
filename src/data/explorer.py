import os
import json
import logging
import pandas as pd
import numpy as np
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def explore_dataset(df, config_path="config.yaml"):
    config = load_config(config_path)
    target_col = config["dataset"]["target_col"]
    binary_col = config["dataset"]["binary_col"]
    
    total_samples = len(df)
    total_features = len(df.columns)
    
    # Missing value analysis
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    
    # Duplicate analysis
    duplicate_rows = int(df.duplicated().sum())
    
    # Class distribution
    class_distribution = df[target_col].value_counts().to_dict()
    binary_distribution = df[binary_col].value_counts().to_dict()
    
    # Numerical vs Categorical features
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    summary = {
        "dataset_name": config["dataset"]["name"],
        "total_samples": total_samples,
        "total_features": total_features,
        "duplicate_rows": duplicate_rows,
        "total_missing_values": total_missing,
        "missing_by_feature": {k: v for k, v in missing_counts.items() if v > 0},
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "class_distribution": class_distribution,
        "binary_distribution": {str(k): v for k, v in binary_distribution.items()}
    }
    
    output_summary_path = config["paths"]["data_summary"]
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("--- DATASET EXPLORATION SUMMARY ---")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Total Features: {total_features}")
    logger.info(f"Duplicates Detected: {duplicate_rows}")
    logger.info(f"Missing Values: {total_missing}")
    logger.info(f"Class Distribution: {class_distribution}")
    logger.info(f"Summary saved to {output_summary_path}")
    
    return summary

if __name__ == "__main__":
    from src.data.download import acquire_dataset
    df = acquire_dataset()
    explore_dataset(df)
