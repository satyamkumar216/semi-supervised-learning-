"""
team2_label_propagation.py — Label Propagation model for SSL-MedBench.

Wraps Team 2's scratch Label Propagation implementation, featuring an
RBF affinity kernel, row-normalized transition matrix, iterative label
propagation, and hard clamping on labeled data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.utils.validation import check_is_fitted

# =====================================================================
# Imports and Configuration Defaults
# =====================================================================

_CURRENT_DIR = Path(__file__).resolve().parent
_SSL_MEDBENCH_DIR = _CURRENT_DIR.parent.parent
_PROJECT_ROOT = _SSL_MEDBENCH_DIR.parent

for _dir_path in [_SSL_MEDBENCH_DIR, _PROJECT_ROOT, _CURRENT_DIR.parent]:
    if str(_dir_path) not in sys.path:
        sys.path.insert(0, str(_dir_path))

try:
    from backend.config import (
        LABEL_PROP_GAMMA,
        LABEL_PROP_MAX_ITER,
        LABEL_PROP_TOL,
    )
    from backend.models.base import BaseSSLModel
except ImportError:
    try:
        from ..config import (
            LABEL_PROP_GAMMA,
            LABEL_PROP_MAX_ITER,
            LABEL_PROP_TOL,
        )
        from .base import BaseSSLModel
    except (ImportError, ValueError):
        try:
            from ssl_medbench.backend.config import (
                LABEL_PROP_GAMMA,
                LABEL_PROP_MAX_ITER,
                LABEL_PROP_TOL,
            )
            from ssl_medbench.backend.models.base import BaseSSLModel
        except ImportError:
            LABEL_PROP_GAMMA: float = 0.5
            LABEL_PROP_MAX_ITER: int = 1000
            LABEL_PROP_TOL: float = 1e-5
            from base import BaseSSLModel  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_PARAM_GRID_GAMMA: List[float] = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
DEFAULT_PARAM_GRID_MAX_ITER: List[int] = [500, 1000, 2000]


# =====================================================================
# Core Algorithm — Team 2 Scratch Implementation
# =====================================================================


class LabelPropagationScratch:
    """Team 2 core scratch Label Propagation algorithm."""

    def __init__(self, gamma: float = 0.5, max_iter: int = 1000, tol: float = 1e-5) -> None:
        """Initialize the scratch label propagation algorithm.

        Args:
            gamma: RBF kernel bandwidth coefficient.
            max_iter: Maximum number of propagation iterations.
            tol: Convergence tolerance for label distribution changes.
        """
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit label propagation on graph and predict labels for all samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,), with -1 for unlabeled.

        Returns:
            1-D array of predicted labels for all samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).copy()
        labeled = (y != -1)
        if not labeled.any():
            raise ValueError("At least one labeled sample is required.")
        classes = np.unique(y[labeled])
        class_idx = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((len(X), len(classes)))
        for i in np.where(labeled)[0]:
            Y[i, class_idx[y[i]]] = 1.0
        F = Y.copy()
        sq = np.sum(X ** 2, axis=1, keepdims=True)
        d2 = np.maximum(sq + sq.T - 2 * X @ X.T, 0.0)
        W = np.exp(-self.gamma * d2)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        S = np.divide(W, rs, out=np.zeros_like(W), where=rs != 0)
        for it in range(self.max_iter):
            F_new = S @ F
            F_new[labeled] = Y[labeled]
            change = np.max(np.abs(F_new - F))
            F = F_new
            if change < self.tol:
                break
        pred = classes[np.argmax(F, axis=1)]
        pred[labeled] = y[labeled]
        self.classes_ = classes
        self.label_distributions_ = F
        self.n_iter_ = it + 1
        return pred


# =====================================================================
# Model Wrapper — SSL-MedBench BaseSSLModel Integration
# =====================================================================


class LabelPropagationModel(BaseSSLModel):
    """SSL-MedBench wrapper for Team 2's Label Propagation algorithm.

    Inherits from ``BaseSSLModel`` to ensure full compatibility with
    scikit-learn estimators and randomized hyperparameter search.
    """

    def __init__(
        self,
        gamma: float = LABEL_PROP_GAMMA,
        max_iter: int = LABEL_PROP_MAX_ITER,
        tol: float = LABEL_PROP_TOL,
    ) -> None:
        """Initialize the Label Propagation benchmark model.

        Args:
            gamma: RBF kernel bandwidth coefficient.
            max_iter: Maximum number of propagation iterations.
            tol: Convergence tolerance for label change.
        """
        super().__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelPropagationModel":
        """Fit the Label Propagation model on semi-supervised data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array with -1 indicating unlabeled samples.

        Returns:
            self: Fitted model instance.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).copy()

        self.n_features_in_ = X_arr.shape[1]
        self.scratch_model_ = LabelPropagationScratch(
            gamma=self.gamma,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.train_predictions_ = self.scratch_model_.fit_predict(X_arr, y_arr)
        self.classes_ = self.scratch_model_.classes_
        self.label_distributions_ = self.scratch_model_.label_distributions_
        self.n_iter_ = self.scratch_model_.n_iter_
        self.X_train_ = X_arr
        self.y_train_ = y_arr

        logger.info(
            "Fitted %s: n_samples=%d, n_features=%d, n_classes=%d, n_iter=%d",
            self.get_name(),
            len(X_arr),
            self.n_features_in_,
            len(self.classes_),
            self.n_iter_,
        )
        return self

    def _is_training_data(self, X: np.ndarray) -> bool:
        """Check if feature matrix X corresponds to the training set.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            True if X matches X_train_, False otherwise.
        """
        if X is self.X_train_:
            return True
        if X.shape != self.X_train_.shape:
            return False
        return bool(np.array_equal(X, self.X_train_))

    def _compute_pairwise_sq_distances(self, X_eval: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between X_eval and X_train_.

        Args:
            X_eval: Feature matrix of shape (n_eval, n_features).

        Returns:
            Distance matrix of shape (n_eval, n_train).
        """
        sq_eval = np.sum(X_eval ** 2, axis=1, keepdims=True)
        sq_train = np.sum(self.X_train_ ** 2, axis=1, keepdims=True)
        cross_term = 2.0 * (X_eval @ self.X_train_.T)
        return np.maximum(sq_eval + sq_train.T - cross_term, 0.0)

    def _compute_similarity_weights(self, X_eval: np.ndarray) -> np.ndarray:
        """Compute row-normalized RBF affinity weights between X_eval and X_train_.

        Args:
            X_eval: Feature matrix of shape (n_eval, n_features).

        Returns:
            Affinity weights of shape (n_eval, n_train) where rows sum to 1.
        """
        sq_dists = self._compute_pairwise_sq_distances(X_eval)
        weights = np.exp(-self.gamma * sq_dists)
        row_sums = weights.sum(axis=1, keepdims=True)

        zero_rows = (row_sums.ravel() == 0.0)
        if np.any(zero_rows):
            nearest_indices = np.argmin(sq_dists[zero_rows], axis=1)
            weights[zero_rows, nearest_indices] = 1.0
            row_sums = weights.sum(axis=1, keepdims=True)

        return weights / row_sums

    def _normalize_probabilities(self, probas: np.ndarray) -> np.ndarray:
        """Ensure class probability rows sum to 1.0.

        Args:
            probas: Matrix of shape (n_samples, n_classes).

        Returns:
            Normalized matrix of shape (n_samples, n_classes).
        """
        row_sums = probas.sum(axis=1, keepdims=True)
        uniform_value = 1.0 / len(self.classes_)
        return np.divide(
            probas,
            row_sums,
            out=np.full_like(probas, uniform_value),
            where=row_sums > 0,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities for input features X.

        For training data, returns normalized label distributions from
        propagation. For test data, computes RBF similarity-weighted
        sum of training label distributions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        check_is_fitted(self, ["classes_", "X_train_"])
        X_arr = np.asarray(X, dtype=float)
        if self._is_training_data(X_arr):
            return self._normalize_probabilities(self.label_distributions_.copy())

        weights = self._compute_similarity_weights(X_arr)
        raw_proba = weights @ self.label_distributions_
        return self._normalize_probabilities(raw_proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input features X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D array of predicted class labels.
        """
        check_is_fitted(self, ["classes_", "X_train_"])
        X_arr = np.asarray(X, dtype=float)
        if self._is_training_data(X_arr):
            return self.train_predictions_.copy()

        probas = self.predict_proba(X_arr)
        best_indices = np.argmax(probas, axis=1)
        return self.classes_[best_indices]

    def get_name(self) -> str:
        """Return the benchmark display name for this model.

        Returns:
            Human-readable name 'Label Propagation'.
        """
        return "Label Propagation"

    def get_param_grid(self) -> Dict[str, Any]:
        """Return the hyperparameter search space for tuning.

        Returns:
            Dictionary mapping hyperparameter names to candidate lists.
        """
        return {
            "gamma": list(DEFAULT_PARAM_GRID_GAMMA),
            "max_iter": list(DEFAULT_PARAM_GRID_MAX_ITER),
        }

    def get_convergence_trace(self) -> List[float]:
        """Return per-iteration metric trace.

        Label Propagation is non-iterative from the benchmark trace
        perspective, so an empty list is returned.

        Returns:
            Empty list.
        """
        return []


# =====================================================================
# Self-Test Execution
# =====================================================================


def _run_self_test() -> None:
    """Execute self-test on load_breast_cancer with 10% labeled data."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("Running self-test for LabelPropagationModel...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rng = np.random.RandomState(42)
    y_tr_masked = np.full_like(y_train, -1)
    n_labeled = max(1, int(0.10 * len(y_train)))
    labeled_idx = rng.choice(len(y_train), size=n_labeled, replace=False)
    y_tr_masked[labeled_idx] = y_train[labeled_idx]

    # Ensure all classes are present in labeled set
    for c in np.unique(y_train):
        if not np.any(y_tr_masked == c):
            c_indices = np.where(y_train == c)[0]
            pick = rng.choice(c_indices)
            y_tr_masked[pick] = c

    model = LabelPropagationModel(gamma=0.5, max_iter=1000, tol=1e-5)
    model.fit(X_tr, y_tr_masked)
    test_preds = model.predict(X_te)
    test_proba = model.predict_proba(X_te)

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average="macro")
    auc = roc_auc_score(y_test, test_proba[:, 1])

    print(f"Model Name:     {model.get_name()}")
    print(f"Iterations:     {model.n_iter_}")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Macro F1:  {f1:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")

    assert test_preds.shape == (len(X_te),), "Predictions shape mismatch"
    assert test_proba.shape == (len(X_te), 2), "Probabilities shape mismatch"
    assert np.allclose(test_proba.sum(axis=1), 1.0), "Probabilities must sum to 1.0"
    print("Self-test passed successfully!")


if __name__ == "__main__":
    _run_self_test()
