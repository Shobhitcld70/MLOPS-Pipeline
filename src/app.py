"""
Flask REST API — Model Serving
Endpoints:
  GET  /health          → liveness check
  GET  /model/info      → current model metadata
  POST /predict         → single prediction
  POST /predict/batch   → batch predictions
  POST /pipeline/run    → trigger full pipeline run
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from src.registry import load_latest_model, load_preprocessor
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at startup
try:
    _model        = load_latest_model()
    _preprocessor = load_preprocessor()
    log.info("Model and preprocessor loaded successfully.")
except Exception as e:
    log.warning(f"Could not load model at startup: {e}")
    _model = _preprocessor = None


def _reload():
    global _model, _preprocessor
    _model        = load_latest_model()
    _preprocessor = load_preprocessor()


# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model_loaded":  _model is not None,
    }), 200


# ── MODEL INFO ────────────────────────────────────────────────────────────────
@app.route("/model/info", methods=["GET"])
def model_info():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 503
    return jsonify({
        "model_type":   type(_model).__name__,
        "n_features_in": getattr(_model, "n_features_in_", "unknown"),
        "classes":      _model.classes_.tolist() if hasattr(_model, "classes_") else [],
    }), 200


# ── SINGLE PREDICT ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "No model loaded. Run /pipeline/run first."}), 503

    body = request.get_json(force=True)
    if not body or "features" not in body:
        return jsonify({"error": "Request body must contain 'features' dict"}), 400

    try:
        df   = pd.DataFrame([body["features"]])
        X    = _preprocessor.transform(df)
        pred = int(_model.predict(X)[0])
        prob = float(_model.predict_proba(X)[0][1])
        return jsonify({"prediction": pred, "probability": round(prob, 4)}), 200
    except Exception as e:
        log.error(f"/predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ── BATCH PREDICT ─────────────────────────────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if _model is None:
        return jsonify({"error": "No model loaded."}), 503

    body = request.get_json(force=True)
    if not body or "records" not in body:
        return jsonify({"error": "Request body must contain 'records' list"}), 400

    try:
        df    = pd.DataFrame(body["records"])
        X     = _preprocessor.transform(df)
        preds = _model.predict(X).tolist()
        probs = _model.predict_proba(X)[:, 1].tolist()
        return jsonify({
            "predictions":   preds,
            "probabilities": [round(p, 4) for p in probs],
            "n_records":     len(preds),
        }), 200
    except Exception as e:
        log.error(f"/predict/batch error: {e}")
        return jsonify({"error": str(e)}), 500


# ── TRIGGER PIPELINE ──────────────────────────────────────────────────────────
@app.route("/pipeline/run", methods=["POST"])
def trigger_pipeline():
    body   = request.get_json(force=True) or {}
    retrain = body.get("retrain", False)
    try:
        result = run_pipeline(retrain=retrain)
        if result.get("status") == "trained":
            _reload()
        return jsonify(result), 200
    except Exception as e:
        log.error(f"/pipeline/run error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
