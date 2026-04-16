"""
Evaluation Module
Computes classification metrics and saves a confusion matrix plot.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)

log = logging.getLogger(__name__)

PROMOTION_AUC_THRESHOLD = float(os.getenv("PROMOTION_AUC_THRESHOLD", "0.75"))
ARTIFACTS_DIR = "models/artifacts"


def evaluate_model(model, X_val, y_val) -> dict:
    """
    Evaluate model on validation set. Returns metrics dict for MLflow logging.
    Also saves confusion matrix as artifact.
    """
    y_pred      = model.predict(X_val)
    y_prob      = model.predict_proba(X_val)[:, 1]

    metrics = {
        "val_accuracy":  round(accuracy_score(y_val, y_pred),          4),
        "val_roc_auc":   round(roc_auc_score(y_val, y_prob),           4),
        "val_f1":        round(f1_score(y_val, y_pred, zero_division=0), 4),
        "val_precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "val_recall":    round(recall_score(y_val, y_pred, zero_division=0),    4),
        "promotable":    int(roc_auc_score(y_val, y_prob) >= PROMOTION_AUC_THRESHOLD),
    }

    log.info(f"Evaluation results:\n{classification_report(y_val, y_pred)}")
    _save_confusion_matrix(y_val, y_pred)

    return metrics


def _save_confusion_matrix(y_true, y_pred):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    path = f"{ARTIFACTS_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=100)
    plt.close()
    log.info(f"Confusion matrix saved → {path}")
