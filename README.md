# MLOps Pipeline — Production-Ready ML System

A fully automated, end-to-end MLOps pipeline covering data ingestion, preprocessing,
model training with hyperparameter tuning, evaluation, drift detection, model registry,
REST API serving, and CI/CD deployment via Jenkins + Kubernetes.

---

## Architecture

```
Data Source
    │
    ▼
[1] Ingest          ← CSV / URL / synthetic data
    │
    ▼
[2] Drift Detection ← PSI-based comparison vs reference baseline
    │  (skip retrain if drift < threshold)
    ▼
[3] Preprocess      ← Imputation → Scaling → Feature Selection (sklearn Pipeline)
    │
    ▼
[4] Train           ← GradientBoosting / RandomForest + RandomizedSearchCV
    │
    ▼
[5] Evaluate        ← Accuracy, ROC-AUC, F1, Precision, Recall + Confusion Matrix
    │
    ▼
[6] Register        ← MLflow Model Registry (promote if AUC ≥ threshold)
    │
    ▼
[7] Serve           ← Flask REST API (predict / batch / health / pipeline trigger)
    │
    ▼
[8] CI/CD           ← Jenkins → Docker → Kubernetes
```

---

## Quick Start

### Local (no Docker)
```bash
pip install -r requirements.txt
python -m src.pipeline --retrain       # run full pipeline
python src/app.py                       # start API server
```

### Docker
```bash
docker build -t mlops-pipeline .
docker run -p 5000:5000 mlops-pipeline
```

### Full Stack (API + MLflow UI)
```bash
docker-compose up
# API:      http://localhost:5000
# MLflow:   http://localhost:5001
```

---

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/health`         | Liveness check                     |
| GET    | `/model/info`     | Current model metadata             |
| POST   | `/predict`        | Single prediction                  |
| POST   | `/predict/batch`  | Batch predictions                  |
| POST   | `/pipeline/run`   | Trigger full pipeline run          |

### Example — Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure_months": 12, "monthly_charges": 65.0,
      "total_charges": 780.0, "num_products": 2,
      "support_calls": 1, "redundant_1": 0.3,
      "redundant_2": -0.1, "noise_1": 0.0,
      "noise_2": 0.1, "noise_3": -0.2
    }
  }'
# {"prediction": 0, "probability": 0.2341}
```

### Example — Trigger Retrain
```bash
curl -X POST http://localhost:5000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"retrain": true}'
```

---

## Configuration (Environment Variables)

| Variable                   | Default                  | Description                        |
|----------------------------|--------------------------|------------------------------------|
| `MLFLOW_TRACKING_URI`      | `sqlite:///mlflow.db`    | MLflow backend URI                 |
| `EXPERIMENT_NAME`          | `mlops-pipeline`         | MLflow experiment name             |
| `DATA_PATH`                | `` (synthetic)           | Path or URL to CSV data            |
| `TARGET_COL`               | `target`                 | Target column name                 |
| `MODEL_TYPE`               | `gradient_boosting`      | `gradient_boosting` or `random_forest` |
| `TUNING_N_ITER`            | `20`                     | RandomizedSearchCV iterations      |
| `CV_FOLDS`                 | `5`                      | Cross-validation folds             |
| `DRIFT_THRESHOLD`          | `0.1`                    | PSI threshold to trigger retrain   |
| `PROMOTION_AUC_THRESHOLD`  | `0.75`                   | Min AUC to register model          |
| `VAL_SIZE`                 | `0.2`                    | Validation split ratio             |
| `PORT`                     | `5000`                   | API server port                    |

---

## Running Tests
```bash
pytest tests/ -v
# 33 tests — ingest, preprocess, train, evaluate, drift, API
```

---

## Project Structure
```
mlops_pipeline/
├── src/
│   ├── pipeline.py     # Main orchestrator
│   ├── ingest.py       # Data ingestion
│   ├── preprocess.py   # sklearn Pipeline preprocessing
│   ├── train.py        # Model training + tuning
│   ├── evaluate.py     # Metrics + confusion matrix
│   ├── drift.py        # PSI drift detection
│   ├── registry.py     # MLflow model registry
│   └── app.py          # Flask REST API
├── tests/
│   └── test_pipeline.py  # 33 tests
├── configs/
│   └── k8s-deployment.yaml
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── requirements.txt
└── README.md
```

---

## Tech Stack
Python · scikit-learn · MLflow · Flask · Docker · Kubernetes · Jenkins · Terraform (infra)
