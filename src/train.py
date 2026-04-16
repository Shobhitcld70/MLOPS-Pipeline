"""
Training Module
Trains a GradientBoostingClassifier with RandomizedSearchCV for hyperparameter tuning.
Fully configurable via environment variables.
"""

import os
import logging
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

log = logging.getLogger(__name__)

MODEL_TYPE   = os.getenv("MODEL_TYPE", "gradient_boosting")   # or "random_forest"
N_ITER       = int(os.getenv("TUNING_N_ITER", "20"))
CV_FOLDS     = int(os.getenv("CV_FOLDS", "5"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MODEL_PATH   = "models/model.pkl"

PARAM_GRIDS = {
    "gradient_boosting": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [3, 4, 5],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.7, 0.8, 1.0],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2", None],
    },
    "random_forest": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2"],
        "bootstrap":         [True, False],
    },
}


def train_model(X_train, y_train):
    """
    Run RandomizedSearchCV, return best estimator and its params.
    """
    base = _get_base_model()
    param_grid = PARAM_GRIDS.get(MODEL_TYPE, PARAM_GRIDS["gradient_boosting"])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator   = base,
        param_distributions = param_grid,
        n_iter      = N_ITER,
        scoring     = "roc_auc",
        cv          = cv,
        refit       = True,
        n_jobs      = -1,
        random_state= RANDOM_STATE,
        verbose     = 0,
    )

    log.info(f"Training {MODEL_TYPE} with {N_ITER}-iter RandomizedSearchCV, {CV_FOLDS}-fold CV ...")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    best_params = search.best_params_
    best_params["model_type"] = MODEL_TYPE
    best_params["cv_best_auc"] = round(search.best_score_, 4)

    log.info(f"Best CV AUC: {best_params['cv_best_auc']} | params: {best_params}")

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)
    log.info(f"Model saved → {MODEL_PATH}")

    return best, best_params


def _get_base_model():
    if MODEL_TYPE == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE)
    return GradientBoostingClassifier(random_state=RANDOM_STATE)
