"""
Drift Detection Module
Compares incoming data distribution against a saved reference baseline.
Uses Population Stability Index (PSI) — industry standard for drift detection.
PSI < 0.1  → no drift
PSI 0.1-0.2 → moderate drift
PSI > 0.2  → significant drift — retrain
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REFERENCE_PATH   = "models/reference_data.pkl"
TARGET_COL       = os.getenv("TARGET_COL", "target")
PSI_BINS         = int(os.getenv("PSI_BINS", "10"))
DRIFT_THRESHOLD  = float(os.getenv("DRIFT_THRESHOLD", "0.1"))


def check_drift(df: pd.DataFrame) -> float:
    """
    Compute average PSI across all numeric features.
    If no reference exists, save current data as reference and return 0.0.
    """
    features = [c for c in df.columns if c != TARGET_COL and df[c].dtype in [np.float64, np.int64, float, int]]

    if not os.path.exists(REFERENCE_PATH):
        log.info("No reference data found — saving current data as baseline.")
        _save_reference(df[features])
        return 0.0

    reference = _load_reference()
    psi_scores = []

    for col in features:
        if col not in reference.columns:
            continue
        try:
            psi = _compute_psi(reference[col].dropna().values, df[col].dropna().values)
            psi_scores.append(psi)
            log.debug(f"  PSI({col}) = {psi:.4f}")
        except Exception as e:
            log.warning(f"PSI failed for {col}: {e}")

    avg_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
    log.info(f"Average PSI across {len(psi_scores)} features: {avg_psi:.4f} (threshold: {DRIFT_THRESHOLD})")

    # Update reference after computing drift
    _save_reference(df[features])
    return avg_psi


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = None) -> float:
    bins = bins or PSI_BINS
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    ref_pct = ref_counts / (len(reference) + 1e-10)
    cur_pct = cur_counts / (len(current)   + 1e-10)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _save_reference(df: pd.DataFrame):
    os.makedirs("models", exist_ok=True)
    with open(REFERENCE_PATH, "wb") as f:
        pickle.dump(df, f)


def _load_reference() -> pd.DataFrame:
    with open(REFERENCE_PATH, "rb") as f:
        return pickle.load(f)
