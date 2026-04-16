"""
Preprocessing Module
Builds a scikit-learn Pipeline: imputation → scaling → feature selection.
Returns train/val splits and the fitted preprocessor (saved for inference).
"""

import logging
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

log = logging.getLogger(__name__)

TARGET_COL      = os.getenv("TARGET_COL", "target")
VAL_SIZE        = float(os.getenv("VAL_SIZE", "0.2"))
RANDOM_STATE    = int(os.getenv("RANDOM_STATE", "42"))
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def preprocess(df: pd.DataFrame):
    """
    Clean → split → fit preprocessor on train → transform both splits.
    Returns: X_train, X_val, y_train, y_val, fitted_preprocessor
    """
    df = _clean(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Split → train: {len(X_train)}, val: {len(X_val)}")

    preprocessor = _build_preprocessor(X_train.shape[1])
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t   = preprocessor.transform(X_val)

    os.makedirs("models", exist_ok=True)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    log.info(f"Preprocessor saved → {PREPROCESSOR_PATH}")

    return X_train_t, X_val_t, y_train.values, y_val.values, preprocessor


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    # Drop exact duplicates
    df = df.drop_duplicates()
    # Drop rows where target is null
    df = df.dropna(subset=[TARGET_COL])
    log.info(f"Cleaning: {initial} → {len(df)} rows (dropped {initial - len(df)})")
    return df


def _build_preprocessor(n_features: int) -> Pipeline:
    k = min(8, n_features)   # select top-k features
    return Pipeline([
        ("imputer",   SimpleImputer(strategy="median")),
        ("scaler",    StandardScaler()),
        ("selector",  SelectKBest(score_func=f_classif, k=k)),
    ])
