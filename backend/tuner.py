"""
tuner.py — Hyperparameter tuning for SSL models.

Tunes SSL models on labeled points only, while passing the unlabeled
data pool alongside so that the semi-supervised learning behavior is
maintained during cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from backend.config import (
    MIN_LABELED_FOR_3CV,
    RANDOM_SEED,
    TUNING_CV_FOLDS,
    TUNING_CV_FOLDS_SMALL,
    TUNING_N_ITER,
    TUNING_SCORING,
)
from backend.models.base import BaseSSLModel

logger = logging.getLogger(__name__)


# ── Tuning Wrapper ─────────────────────────────────────────────────

class SSLTuningWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper that supplies the unlabeled pool during CV fold fitting.

    Why this wrapper is needed:
        RandomizedSearchCV operates on the labeled pool (X_labeled, y_labeled)
        to evaluate validation performance on true known labels. However, an
        SSL model needs access to the unlabeled data pool to learn semi-supervised
        structure. This wrapper holds the unlabeled data pool in memory. During
        each CV split's `fit(X_fold_labeled, y_fold_labeled)`, it stitches the
        fold's labeled points together with the fixed unlabeled pool (with label
        -1), ensuring the SSL model sees labeled + unlabeled samples without
        leaking the CV validation fold's labels.
    """

    def __init__(
        self,
        base_estimator: Optional[BaseSSLModel] = None,
        X_unlabeled: Optional[np.ndarray] = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.X_unlabeled = X_unlabeled

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SSLTuningWrapper":
        """Fit by combining labeled fold points with the unlabeled pool."""
        if self.base_estimator is None:
            raise ValueError("base_estimator cannot be None.")

        # Ensure base estimator is cloned fresh
        self.estimator_ = clone(self.base_estimator)

        if self.X_unlabeled is not None and len(self.X_unlabeled) > 0:
            X_combined = np.vstack([X, self.X_unlabeled])
            y_unlabeled = np.full(len(self.X_unlabeled), -1, dtype=y.dtype)
            y_combined = np.concatenate([y, y_unlabeled])
        else:
            X_combined = X
            y_combined = y

        self.estimator_.fit(X_combined, y_combined)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using the fitted base estimator."""
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the fitted base estimator."""
        return self.estimator_.predict_proba(X)


# ── Helpers for Param Grids ────────────────────────────────────────

def _prefix_param_grid(
    param_grid: Union[Dict[str, Any], List[Dict[str, Any]]],
    prefix: str = "base_estimator__",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Prefix all parameter grid keys for pipeline/wrapper delegation."""
    if isinstance(param_grid, list):
        return [
            {f"{prefix}{k}": v for k, v in grid.items()}
            for grid in param_grid
        ]
    return {f"{prefix}{k}": v for k, v in param_grid.items()}


def _unprefix_best_params(
    best_params: Dict[str, Any],
    prefix: str = "base_estimator__",
) -> Dict[str, Any]:
    """Strip prefix from best parameters dict."""
    cleaned: Dict[str, Any] = {}
    for k, v in best_params.items():
        if k.startswith(prefix):
            cleaned[k[len(prefix):]] = v
        else:
            cleaned[k] = v
    return cleaned


# ── Main Tuning Function ───────────────────────────────────────────

def tune_model(
    model_class: Type[BaseSSLModel],
    X_train: np.ndarray,
    y_masked: np.ndarray,
    label_fraction: float,
) -> Dict[str, Any]:
    """Tune an SSL model class using RandomizedSearchCV on labeled points.

    Args:
        model_class: Class of the model to instantiate and tune.
        X_train: All training features (labeled + unlabeled).
        y_masked: Labels where -1 marks unlabeled points.
        label_fraction: Current fraction of labeled points.

    Returns:
        dict: {
            "best_params": dict,
            "best_cv_score": float,
            "default_params": dict,
            "status": "SUCCESS" | "TUNING_FAILED",
        }
    """
    default_instance = model_class()
    default_params = default_instance.get_params()
    raw_param_grid = default_instance.get_param_grid()

    # Partition labeled and unlabeled
    labeled_mask = (y_masked != -1)
    n_labeled = int(labeled_mask.sum())
    X_l = X_train[labeled_mask]
    y_l = y_masked[labeled_mask]
    X_u = X_train[~labeled_mask]

    # Return default params if insufficient data or empty grid
    if n_labeled < 4 or not raw_param_grid:
        return {
            "best_params": default_params,
            "best_cv_score": 0.0,
            "default_params": default_params,
            "status": "DEFAULT_ONLY",
        }

    # Determine CV splits
    cv_splits = (
        TUNING_CV_FOLDS_SMALL
        if n_labeled < MIN_LABELED_FOR_3CV
        else TUNING_CV_FOLDS
    )
    # Check minimum class count for StratifiedKFold
    min_class_count = min(np.bincount(y_l)) if len(np.unique(y_l)) > 1 else n_labeled
    cv_splits = max(2, min(cv_splits, int(min_class_count)))

    try:
        wrapper = SSLTuningWrapper(
            base_estimator=default_instance,
            X_unlabeled=X_u,
        )
        prefixed_grid = _prefix_param_grid(raw_param_grid)

        # Count total combinations if discrete
        search = RandomizedSearchCV(
            estimator=wrapper,
            param_distributions=prefixed_grid,
            n_iter=TUNING_N_ITER,
            scoring=TUNING_SCORING,
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED),
            random_state=RANDOM_SEED,
            refit=True,
            error_score=np.nan,
            n_jobs=1,  # Keep sequential to prevent thread lock issues
        )

        search.fit(X_l, y_l)

        best_params = _unprefix_best_params(search.best_params_)
        best_score = float(search.best_score_) if not np.isnan(search.best_score_) else 0.0

        logger.info(
            "Tuned %s (frac=%.2f): best F1=%.4f",
            default_instance.get_name(), label_fraction, best_score,
        )
        return {
            "best_params": best_params,
            "best_cv_score": round(best_score, 4),
            "default_params": default_params,
            "status": "SUCCESS",
        }

    except Exception as exc:
        logger.warning(
            "Tuning crashed for %s (frac=%.2f): %s. Falling back to defaults.",
            default_instance.get_name(), label_fraction, exc,
        )
        return {
            "best_params": default_params,
            "best_cv_score": 0.0,
            "default_params": default_params,
            "status": "TUNING_FAILED",
        }
