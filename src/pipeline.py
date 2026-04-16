"""
MLOps Pipeline — Main Orchestrator
Runs: ingest → preprocess → train → evaluate → register → serve
"""

import os
import logging
import mlflow
from src.ingest import ingest_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate_model
from src.registry import register_model
from src.drift import check_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME     = os.getenv("EXPERIMENT_NAME", "mlops-pipeline")
DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.1"))


def run_pipeline(retrain: bool = False):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"MLflow run started: {run_id}")

        # 1. Ingest
        log.info("Step 1/6 — Data ingestion")
        raw = ingest_data()
        mlflow.log_param("n_samples_raw", len(raw))

        # 2. Drift detection (skip on first run / forced retrain)
        if not retrain:
            log.info("Step 2/6 — Drift detection")
            drift_score = check_drift(raw)
            mlflow.log_metric("drift_score", drift_score)
            if drift_score < DRIFT_THRESHOLD:
                log.info(f"Drift score {drift_score:.4f} below threshold — skipping retrain.")
                mlflow.set_tag("retrain_triggered", "false")
                return {"status": "skipped", "drift_score": drift_score, "run_id": run_id}
            log.info(f"Drift score {drift_score:.4f} above threshold — retraining.")
            mlflow.set_tag("retrain_triggered", "true")
        else:
            log.info("Step 2/6 — Forced retrain, skipping drift check")
            mlflow.set_tag("retrain_triggered", "forced")

        # 3. Preprocess
        log.info("Step 3/6 — Preprocessing")
        X_train, X_val, y_train, y_val, preprocessor = preprocess(raw)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val",   len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])

        # 4. Train
        log.info("Step 4/6 — Training")
        model, params = train_model(X_train, y_train)
        mlflow.log_params(params)

        # 5. Evaluate
        log.info("Step 5/6 — Evaluation")
        metrics = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        log.info(f"Metrics: {metrics}")

        # 6. Register if good enough
        log.info("Step 6/6 — Model registration")
        registered = register_model(model, preprocessor, metrics, run_id)
        mlflow.set_tag("model_registered", str(registered))

        log.info("Pipeline complete.")
        return {
            "status":     "trained",
            "run_id":     run_id,
            "metrics":    metrics,
            "registered": registered,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retrain")
    args = parser.parse_args()
    result = run_pipeline(retrain=args.retrain)
    print(result)
