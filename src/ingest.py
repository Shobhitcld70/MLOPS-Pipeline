"""
Data Ingestion Module
Supports: CSV file, URL, or synthetic data generation for demo.
In production: swap load_from_source() with your DB / S3 / API call.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

log = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH", "")   # set to CSV path or URL in production


def ingest_data() -> pd.DataFrame:
    """
    Load raw data. Priority:
    1. DATA_PATH env var (CSV file or HTTP URL)
    2. Synthetic data (for demo/testing)
    """
    if DATA_PATH:
        log.info(f"Loading data from: {DATA_PATH}")
        return _load_from_source(DATA_PATH)
    log.info("DATA_PATH not set — generating synthetic dataset")
    return _generate_synthetic()


def _load_from_source(path: str) -> pd.DataFrame:
    if path.startswith("http"):
        df = pd.read_csv(path)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def _generate_synthetic() -> pd.DataFrame:
    """
    Generates a realistic binary classification dataset.
    10 features: 5 informative, 2 redundant, 3 noise.
    Mimics a churn / fraud / approval prediction scenario.
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        flip_y=0.05,
        random_state=42,
    )
    feature_names = [
        "tenure_months", "monthly_charges", "total_charges",
        "num_products", "support_calls",
        "redundant_1", "redundant_2",
        "noise_1", "noise_2", "noise_3",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Inject realistic nulls (~3%)
    for col in ["monthly_charges", "support_calls"]:
        null_idx = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        df.loc[null_idx, col] = np.nan

    log.info(f"Synthetic dataset: {df.shape} | target balance: {df['target'].value_counts().to_dict()}")
    return df
