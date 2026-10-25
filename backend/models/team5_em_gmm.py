"""
team5_em_gmm.py — Team 5: Semi-Supervised EM with Gaussian Mixture Models.

Wraps the core SemiSupervisedEMGMMCore algorithm into the SSL-MedBench
BaseSSLModel architecture, supporting class-label alignment, scikit-learn
estimator API compatibility, and convergence logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture

# Add project root and backend to sys.path for standalone script execution
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
for _path_entry in [str(_PROJECT_ROOT), str(_BACKEND_DIR), str(_CURRENT_DIR)]:
    if _path_entry not in sys.path:
        sys.path.insert(0, _path_entry)

# ── Safe Config and Base Imports ─────────────────────────────────────
try:
    from backend.config import EM_GMM_MAX_ITER, EM_GMM_TOL, RANDOM_SEED
except ImportError:
    try:
        from ..config import EM_GMM_MAX_ITER, EM_GMM_TOL, RANDOM_SEED
    except ImportError:
        EM_GMM_MAX_ITER = 100
        EM_GMM_TOL = 1e-4
        RANDOM_SEED = 42

try:
    from .base import BaseSSLModel
except ImportError:
    try:
        from backend.models.base import BaseSSLModel
    except ImportError:
        try:
            from ssl_medbench.backend.models.base import BaseSSLModel
        except ImportError:
            from base import BaseSSLModel

logger = logging.getLogger(__name__)


# =====================================================================
# Core Algorithm (Team 5 Original Implementation)
# =====================================================================

class SemiSupervisedEMGMMCore:
    """Original Team 5 Semi-Supervised EM with Gaussian Mixture Models."""
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.gmm = None
        self.log_likelihood_trace = []

    def fit(self, X, y_masked):
        labeled_mask = y_masked != -1
        X_labeled = X[labeled_mask]
        y_labeled = y_masked[labeled_mask]
        self.gmm = GaussianMixture(
            n_components=self.n_components, covariance_type='full',
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state, warm_start=False)
        self.gmm.fit(X_labeled)
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            responsibilities = self.gmm.predict_proba(X)
            for cls in np.unique(y_labeled):
                cls_labeled_mask = (y_masked == cls)
                hard_gamma = np.zeros(self.n_components)
                hard_gamma[int(cls)] = 1.0
                responsibilities[cls_labeled_mask] = hard_gamma
            self._m_step(X, responsibilities)
            ll = self.gmm.score(X)
            self.log_likelihood_trace.append(ll)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def _m_step(self, X, responsibilities):
        N, K = responsibilities.shape
        eps = 1e-8
        means = np.zeros((K, X.shape[1]))
        covariances = np.zeros((K, X.shape[1], X.shape[1]))
        weights = np.zeros(K)
        for k in range(K):
            r_k = responsibilities[:, k]
            N_k = r_k.sum() + eps
            means[k] = (r_k[:, np.newaxis] * X).sum(axis=0) / N_k
            diff = X - means[k]
            covariances[k] = (r_k[:, np.newaxis] * diff).T @ diff / N_k
            covariances[k] += np.eye(X.shape[1]) * 1e-6
            weights[k] = N_k / N
        self.gmm.means_ = means
        self.gmm.covariances_ = covariances
        self.gmm.weights_ = weights
        from numpy.linalg import cholesky
        precisions_chol = np.zeros_like(covariances)
        for k in range(K):
            try:
                cov_chol = cholesky(covariances[k])
                precisions_chol[k] = np.linalg.solve(cov_chol, np.eye(X.shape[1])).T
            except np.linalg.LinAlgError:
                precisions_chol[k] = np.eye(X.shape[1])
        self.gmm.precisions_cholesky_ = precisions_chol

    def predict(self, X):
        return self.gmm.predict(X)

    def predict_proba(self, X):
        return self.gmm.predict_proba(X)

    def get_convergence_trace(self):
        return self.log_likelihood_trace


# =====================================================================
# Benchmark Wrapper (EMGMMModel)
# =====================================================================

class EMGMMModel(BaseSSLModel):
    """Semi-Supervised Gaussian Mixture Model wrapper for SSL-MedBench.

    Wraps the Team 5 semi-supervised EM-GMM implementation, handling
    dynamic cluster sizing, label alignment for inverted cluster
    indices, and scikit-learn estimator interface compliance.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = EM_GMM_MAX_ITER,
        tol: float = EM_GMM_TOL,
        random_state: int = RANDOM_SEED,
    ) -> None:
        """Initialize the EMGMMModel estimator.

        Args:
            n_components: Number of mixture components (classes).
            max_iter: Maximum number of EM iterations.
            tol: Convergence tolerance for log-likelihood.
            random_state: Random state seed for reproducible initialization.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.core_: Optional[SemiSupervisedEMGMMCore] = None
        self.label_mapping_: Dict[int, int] = {}
        self.is_flipped_: bool = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EMGMMModel":
        """Fit the semi-supervised EM-GMM model on X with masked labels y.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array with -1 marking unlabeled samples.

        Returns:
            self: The fitted EMGMMModel estimator.
        """
        labeled_mask = (y != -1)
        unique_labeled = np.unique(y[labeled_mask])

        if len(unique_labeled) > 1:
            effective_components = len(unique_labeled)
            self.classes_ = np.sort(unique_labeled)
        elif len(unique_labeled) == 1:
            effective_components = max(self.n_components, 2)
            self.classes_ = np.array([unique_labeled[0], 1 - unique_labeled[0]])
        else:
            effective_components = self.n_components
            self.classes_ = np.arange(self.n_components)

        self.core_ = SemiSupervisedEMGMMCore(
            n_components=effective_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.core_.fit(X, y)
        self._align_labels(X, y, labeled_mask)
        return self

    def _align_labels(
        self, X: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray
    ) -> None:
        """Check and rectify component label inversions on labeled data.

        Args:
            X: Feature matrix.
            y: Target label array with -1 for unlabeled.
            labeled_mask: Boolean mask of labeled data points.
        """
        if not np.any(labeled_mask):
            self.is_flipped_ = False
            self.label_mapping_ = {int(c): int(c) for c in self.classes_}
            return

        raw_preds = self.core_.predict(X[labeled_mask])
        y_labeled = y[labeled_mask]
        acc = float(np.mean(raw_preds == y_labeled))
        logger.info("EM+GMM initial labeled data accuracy: %.4f", acc)

        if acc < 0.5 and len(self.classes_) == 2:
            logger.info("Flipping binary class labels (initial accuracy %.4f < 0.5)", acc)
            self.is_flipped_ = True
            c0, c1 = int(self.classes_[0]), int(self.classes_[1])
            self.label_mapping_ = {c0: c1, c1: c0}
        else:
            self.is_flipped_ = False
            self.label_mapping_ = {int(c): int(c) for c in self.classes_}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D array of predicted class labels.
        """
        if self.core_ is None:
            raise ValueError("The model is not fitted yet. Call 'fit' before predicting.")

        raw_preds = self.core_.predict(X)
        if self.is_flipped_:
            return np.array(
                [self.label_mapping_.get(int(p), int(p)) for p in raw_preds],
                dtype=raw_preds.dtype,
            )
        return raw_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities for test samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        if self.core_ is None:
            raise ValueError("The model is not fitted yet. Call 'fit' before predicting.")

        proba = self.core_.predict_proba(X)
        if self.is_flipped_ and proba.shape[1] == 2:
            return proba[:, [1, 0]]
        return proba

    def get_name(self) -> str:
        """Return the benchmark-standard model display name."""
        return "EM + GMM"

    def get_param_grid(self) -> Dict[str, Any]:
        """Return the hyperparameter search space for tuning.

        Returns:
            Dictionary mapping hyperparameter names to candidate values.
        """
        return {
            "max_iter": [50, 100, 200],
            "tol": [1e-3, 1e-4, 1e-5],
        }

    def get_convergence_trace(self) -> List[float]:
        """Return the per-iteration log-likelihood trace from EM.

        Returns:
            List of log-likelihood values recorded per iteration.
        """
        if self.core_ is None:
            return []
        return list(self.core_.get_convergence_trace())


# =====================================================================
# Self-Test Entry Point
# =====================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("Self-test: EMGMMModel on Breast Cancer (10% labeled)")
    print("=" * 60)

    dataset = load_breast_cancer()
    X_raw, y_true = dataset.data, dataset.target
    X_norm = StandardScaler().fit_transform(X_raw)

    # Mask 90% of labels to retain 10% supervision
    rng = np.random.RandomState(RANDOM_SEED)
    y_semi = y_true.copy()
    unlabeled_mask = rng.rand(len(y_true)) > 0.10
    y_semi[unlabeled_mask] = -1

    num_labeled = int(np.sum(y_semi != -1))
    print(f"Total samples: {len(y_true)}")
    print(f"Labeled samples: {num_labeled} ({num_labeled / len(y_true):.1%})")

    model = EMGMMModel(
        n_components=2,
        max_iter=EM_GMM_MAX_ITER,
        tol=EM_GMM_TOL,
        random_state=RANDOM_SEED,
    )
    print(f"Fitting model: {model.get_name()}...")
    model.fit(X_norm, y_semi)

    predictions = model.predict(X_norm)
    probabilities = model.predict_proba(X_norm)
    trace = model.get_convergence_trace()

    acc = float(accuracy_score(y_true, predictions))
    f1 = float(f1_score(y_true, predictions, average="macro"))

    print(f"Model Name: {model.get_name()}")
    print(f"Param Grid: {model.get_param_grid()}")
    print(f"Accuracy on full dataset: {acc:.4f}")
    print(f"Macro F1 on full dataset: {f1:.4f}")
    print(f"Labels inverted / flipped: {model.is_flipped_}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Convergence iterations: {len(trace)}")
    if trace:
        print(f"Final log-likelihood: {trace[-1]:.4f}")

    assert len(predictions) == len(y_true), "Prediction length mismatch"
    assert probabilities.shape == (len(y_true), 2), "Probability shape mismatch"
    assert len(trace) > 0, "Convergence trace should not be empty"
    assert acc > 0.5, f"Accuracy too low ({acc:.4f})"
    print("EMGMMModel self-test passed successfully!")
