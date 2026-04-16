"""
Test Suite — MLOps Pipeline
Tests: ingest, preprocess, train, evaluate, drift, API endpoints
Run: pytest tests/ -v
"""

import os
import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest     import ingest_data, _generate_synthetic
from src.preprocess import preprocess, _clean
from src.train      import train_model
from src.evaluate   import evaluate_model
from src.drift      import check_drift, _compute_psi


# ── FIXTURES ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_df():
    return _generate_synthetic()

@pytest.fixture(scope="module")
def processed(raw_df):
    return preprocess(raw_df)

@pytest.fixture(scope="module")
def trained_model(processed):
    X_train, X_val, y_train, y_val, _ = processed
    # Fast training for tests
    os.environ["TUNING_N_ITER"] = "3"
    os.environ["CV_FOLDS"] = "2"
    model, params = train_model(X_train, y_train)
    return model, params, X_val, y_val


# ── INGEST TESTS ──────────────────────────────────────────────────────────────
class TestIngest:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_has_target_column(self, raw_df):
        assert "target" in raw_df.columns

    def test_correct_row_count(self, raw_df):
        assert len(raw_df) == 2000

    def test_correct_feature_count(self, raw_df):
        assert raw_df.shape[1] == 11   # 10 features + target

    def test_target_is_binary(self, raw_df):
        assert set(raw_df["target"].unique()).issubset({0, 1})

    def test_has_injected_nulls(self, raw_df):
        assert raw_df.isnull().any().any()


# ── PREPROCESS TESTS ──────────────────────────────────────────────────────────
class TestPreprocess:
    def test_returns_five_values(self, processed):
        assert len(processed) == 5

    def test_train_val_shapes(self, processed):
        X_train, X_val, y_train, y_val, _ = processed
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0]   == len(y_val)

    def test_no_nulls_after_preprocess(self, processed):
        X_train, X_val, _, _, _ = processed
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_val).any()

    def test_val_size_approx_20pct(self, processed):
        X_train, X_val, _, _, _ = processed
        total = X_train.shape[0] + X_val.shape[0]
        val_ratio = X_val.shape[0] / total
        assert 0.18 <= val_ratio <= 0.22

    def test_preprocessor_saved(self):
        assert os.path.exists("models/preprocessor.pkl")

    def test_preprocessor_is_pipeline(self, processed):
        from sklearn.pipeline import Pipeline
        _, _, _, _, preprocessor = processed
        assert isinstance(preprocessor, Pipeline)

    def test_clean_removes_duplicates(self, raw_df):
        duped = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
        cleaned = _clean(duped)
        assert len(cleaned) == len(raw_df)


# ── TRAIN TESTS ───────────────────────────────────────────────────────────────
class TestTrain:
    def test_model_has_predict(self, trained_model):
        model, _, _, _ = trained_model
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, trained_model):
        model, _, _, _ = trained_model
        assert hasattr(model, "predict_proba")

    def test_params_logged(self, trained_model):
        _, params, _, _ = trained_model
        assert "model_type" in params
        assert "cv_best_auc" in params

    def test_cv_auc_reasonable(self, trained_model):
        _, params, _, _ = trained_model
        assert params["cv_best_auc"] > 0.5   # better than random

    def test_model_saved(self):
        assert os.path.exists("models/model.pkl")


# ── EVALUATE TESTS ────────────────────────────────────────────────────────────
class TestEvaluate:
    def test_returns_all_metrics(self, trained_model):
        model, _, X_val, y_val = trained_model
        metrics = evaluate_model(model, X_val, y_val)
        for key in ["val_accuracy", "val_roc_auc", "val_f1", "val_precision", "val_recall"]:
            assert key in metrics

    def test_metrics_in_valid_range(self, trained_model):
        model, _, X_val, y_val = trained_model
        metrics = evaluate_model(model, X_val, y_val)
        for key in ["val_accuracy", "val_roc_auc", "val_f1"]:
            assert 0.0 <= metrics[key] <= 1.0

    def test_auc_above_random(self, trained_model):
        model, _, X_val, y_val = trained_model
        metrics = evaluate_model(model, X_val, y_val)
        assert metrics["val_roc_auc"] > 0.5

    def test_confusion_matrix_saved(self, trained_model):
        model, _, X_val, y_val = trained_model
        evaluate_model(model, X_val, y_val)
        assert os.path.exists("models/artifacts/confusion_matrix.png")


# ── DRIFT TESTS ───────────────────────────────────────────────────────────────
class TestDrift:
    def test_psi_zero_same_distribution(self):
        data = np.random.normal(0, 1, 1000)
        psi  = _compute_psi(data, data)
        assert psi < 0.01

    def test_psi_high_different_distribution(self):
        ref  = np.random.normal(0, 1, 1000)
        cur  = np.random.normal(5, 1, 1000)   # very different
        psi  = _compute_psi(ref, cur)
        assert psi > 0.1

    def test_check_drift_returns_float(self, raw_df):
        # Remove reference to force fresh baseline
        if os.path.exists("models/reference_data.pkl"):
            os.remove("models/reference_data.pkl")
        score = check_drift(raw_df)
        assert isinstance(score, float)
        assert score == 0.0   # first run always returns 0

    def test_check_drift_second_run(self, raw_df):
        score = check_drift(raw_df)  # same data → low drift
        assert isinstance(score, float)
        assert score < 0.1


# ── API TESTS ─────────────────────────────────────────────────────────────────
class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow_test.db"
        from src.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_has_status_ok(self, client):
        res  = client.get("/health")
        data = json.loads(res.data)
        assert data["status"] == "ok"

    def test_pipeline_run_endpoint(self, client):
        res  = client.post("/pipeline/run",
                           data=json.dumps({"retrain": True}),
                           content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["status"] in ("trained", "skipped")

    def test_predict_endpoint(self, client):
        # Run pipeline first to ensure model exists
        client.post("/pipeline/run",
                    data=json.dumps({"retrain": True}),
                    content_type="application/json")
        features = {
            "tenure_months": 12, "monthly_charges": 50.0,
            "total_charges": 600.0, "num_products": 2,
            "support_calls": 1, "redundant_1": 0.5,
            "redundant_2": -0.3, "noise_1": 0.1,
            "noise_2": -0.2, "noise_3": 0.0,
        }
        res  = client.post("/predict",
                           data=json.dumps({"features": features}),
                           content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_batch_endpoint(self, client):
        records = [
            {"tenure_months": 6,  "monthly_charges": 30.0, "total_charges": 180.0,
             "num_products": 1, "support_calls": 0, "redundant_1": 0.2,
             "redundant_2": 0.1, "noise_1": 0.0, "noise_2": 0.0, "noise_3": 0.0},
            {"tenure_months": 24, "monthly_charges": 80.0, "total_charges": 1920.0,
             "num_products": 3, "support_calls": 3, "redundant_1": -0.5,
             "redundant_2": 0.8, "noise_1": 0.3, "noise_2": -0.1, "noise_3": 0.2},
        ]
        res  = client.post("/predict/batch",
                           data=json.dumps({"records": records}),
                           content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["n_records"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_missing_features(self, client):
        res = client.post("/predict",
                          data=json.dumps({}),
                          content_type="application/json")
        assert res.status_code == 400

    def test_model_info_endpoint(self, client):
        res  = client.get("/model/info")
        assert res.status_code in (200, 503)
