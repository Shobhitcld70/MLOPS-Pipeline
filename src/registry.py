"""
Model Registry Module
Registers model to MLflow Model Registry if metrics pass threshold.
Handles versioning and stage promotion (Staging → Production).
"""

import os
import logging
import pickle
import mlflow
import mlflow.sklearn

log = logging.getLogger(__name__)

PROMOTION_AUC_THRESHOLD = float(os.getenv("PROMOTION_AUC_THRESHOLD", "0.75"))
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "mlops-pipeline-model")


def register_model(model, preprocessor, metrics: dict, run_id: str) -> bool:
    """
    Register model + preprocessor to MLflow registry if AUC meets threshold.
    Returns True if registered.
    """
    auc = metrics.get("val_roc_auc", 0)
    if auc < PROMOTION_AUC_THRESHOLD:
        log.warning(f"AUC {auc} below threshold {PROMOTION_AUC_THRESHOLD} — not registering.")
        return False

    log.info(f"AUC {auc} passes threshold — registering model '{MODEL_NAME}'")

    # Log model artifact to MLflow
    mlflow.sklearn.log_model(
        sk_model        = model,
        artifact_path   = "model",
        registered_model_name = MODEL_NAME,
    )

    # Log preprocessor as artifact
    preprocessor_path = "models/preprocessor.pkl"
    if os.path.exists(preprocessor_path):
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

    # Log confusion matrix if exists
    cm_path = "models/artifacts/confusion_matrix.png"
    if os.path.exists(cm_path):
        mlflow.log_artifact(cm_path, artifact_path="artifacts")

    log.info(f"Model registered to MLflow registry as '{MODEL_NAME}'")
    return True


def load_latest_model():
    """
    Load the latest Production-stage model from MLflow registry.
    Falls back to local pickle if registry unavailable.
    """
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        log.info(f"Loaded model from registry: {model_uri}")
        return model
    except Exception as e:
        log.warning(f"Registry load failed ({e}), falling back to local model.")
        with open("models/model.pkl", "rb") as f:
            return pickle.load(f)


def load_preprocessor():
    with open("models/preprocessor.pkl", "rb") as f:
        return pickle.load(f)
