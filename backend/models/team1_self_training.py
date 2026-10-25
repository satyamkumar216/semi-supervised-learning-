"""
team1_self_training.py — Self-Training with Random Forest classifier.

Wraps scikit-learn's SelfTrainingClassifier using RandomForestClassifier
as the underlying base estimator for benchmark evaluation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils.validation import check_is_fitted

# ── Dynamic Path Resolution & Imports ───────────────────────────────
_CURRENT_DIR = Path(__file__).resolve().parent
_SSL_MEDBENCH_DIR = _CURRENT_DIR.parent.parent
_PROJECT_ROOT = _SSL_MEDBENCH_DIR.parent

for _dir_path in [_SSL_MEDBENCH_DIR, _PROJECT_ROOT]:
    if str(_dir_path) not in sys.path:
        sys.path.insert(0, str(_dir_path))

try:
    from backend.config import (
        RANDOM_SEED,
        SELF_TRAINING_MAX_ITER,
        SELF_TRAINING_N_ESTIMATORS,
        SELF_TRAINING_THRESHOLD,
    )
except ImportError:
    try:
        from ssl_medbench.backend.config import (
            RANDOM_SEED,
            SELF_TRAINING_MAX_ITER,
            SELF_TRAINING_N_ESTIMATORS,
            SELF_TRAINING_THRESHOLD,
        )
    except ImportError:
        # Fallback constants for standalone execution
        SELF_TRAINING_THRESHOLD: float = 0.90
        SELF_TRAINING_MAX_ITER: int = 10
        SELF_TRAINING_N_ESTIMATORS: int = 100
        RANDOM_SEED: int = 42

try:
    from .base import BaseSSLModel
except (ImportError, ValueError):
    try:
        from backend.models.base import BaseSSLModel
    except ImportError:
        try:
            from ssl_medbench.backend.models.base import BaseSSLModel
        except ImportError:
            from base import BaseSSLModel

logger = logging.getLogger(__name__)

# ── Search Space Constants ──────────────────────────────────────────
SEARCH_THRESHOLDS: List[float] = [0.6, 0.7, 0.8, 0.9, 0.95]
SEARCH_MAX_ITERS: List[int] = [5, 10, 15, 20]
SEARCH_N_ESTIMATORS: List[int] = [50, 100, 200]
UNLABELED_SENTINEL: int = -1


# ── Model Implementation ─────────────────────────────────────────────
class SelfTrainingModel(BaseSSLModel):
    """Self-Training SSL model using a Random Forest base classifier.

    Wraps sklearn's ``SelfTrainingClassifier`` and ``RandomForestClassifier``.
    Adheres to scikit-learn estimator conventions and benchmark interface.
    """

    def __init__(
        self,
        threshold: float = SELF_TRAINING_THRESHOLD,
        max_iter: int = SELF_TRAINING_MAX_ITER,
        n_estimators: int = SELF_TRAINING_N_ESTIMATORS,
        random_state: Optional[int] = RANDOM_SEED,
    ) -> None:
        """Initialize SelfTrainingModel with explicit hyperparameters.

        Args:
            threshold: Probability confidence threshold for pseudo-labeling.
            max_iter: Maximum self-training iterations to run.
            n_estimators: Number of trees in the underlying Random Forest.
            random_state: Seed controlling random number generation.
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state

    # ── Fitting & Inference ─────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelfTrainingModel":
        """Fit the Self-Training classifier on labeled and unlabeled data.

        Unlabeled instances are indicated by label value -1.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target label vector of shape (n_samples,); -1 for unlabeled.

        Returns:
            Fitted SelfTrainingModel instance.
        """
        logger.info(
            "Fitting SelfTrainingModel (threshold=%.2f, max_iter=%d, n_estimators=%d)",
            self.threshold,
            self.max_iter,
            self.n_estimators,
        )

        base_rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.classifier_ = SelfTrainingClassifier(
            base_rf,
            threshold=self.threshold,
            max_iter=self.max_iter,
        )
        self.classifier_.fit(X, y)
        self.classes_ = self.classifier_.classes_
        self.convergence_trace_ = self._extract_convergence_trace()

        logger.info(
            "Self-training complete. Iterations trace: %s",
            self.convergence_trace_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test features X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D array of predicted class labels.
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities from the underlying Random Forest.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(X)

    # ── Benchmark Interface ─────────────────────────────────────────

    def get_name(self) -> str:
        """Return the benchmark human-readable model name.

        Returns:
            Exact string 'Self-Training (RF)'.
        """
        return "Self-Training (RF)"

    def get_param_grid(self) -> Dict[str, Any]:
        """Return hyperparameter search space for tuning.

        Returns:
            Dictionary mapping hyperparameter names to value distributions.
        """
        return {
            "threshold": SEARCH_THRESHOLDS,
            "max_iter": SEARCH_MAX_ITERS,
            "n_estimators": SEARCH_N_ESTIMATORS,
        }

    # ── Diagnostic Tracing ──────────────────────────────────────────

    def _extract_convergence_trace(self) -> List[float]:
        """Extract per-iteration pseudo-label counts from labeled_iter_.

        Returns:
            List of pseudo-label counts added per iteration, or [] if none.
        """
        if not hasattr(self, "classifier_"):
            return []

        labeled_iter = getattr(self.classifier_, "labeled_iter_", None)
        if labeled_iter is None or len(labeled_iter) == 0:
            return []

        max_iter_val = int(np.max(labeled_iter))
        if max_iter_val < 1:
            return []

        return [
            float(np.sum(labeled_iter == iteration_idx))
            for iteration_idx in range(1, max_iter_val + 1)
        ]

    def get_convergence_trace(self) -> List[float]:
        """Return list of per-iteration pseudo-label counts.

        Returns:
            Per-iteration pseudo-label counts, or [] if unavailable.
        """
        if hasattr(self, "convergence_trace_"):
            return list(self.convergence_trace_)
        return self._extract_convergence_trace()


# ── Self-Test ───────────────────────────────────────────────────────

def _run_self_test() -> None:
    """Run verification self-test on breast cancer dataset with 10% labels."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score

    print("=== Running SelfTrainingModel Self-Test ===")
    test_label_fraction: float = 0.10

    # 1. Load dataset
    features, labels = load_breast_cancer(return_X_y=True)
    n_samples: int = len(labels)

    # 2. Mask 90% of labels with UNLABELED_SENTINEL (-1)
    rng = np.random.RandomState(RANDOM_SEED)
    mask = rng.rand(n_samples) > test_label_fraction
    masked_labels = labels.copy()
    masked_labels[mask] = UNLABELED_SENTINEL

    labeled_count: int = int(np.sum(masked_labels != UNLABELED_SENTINEL))
    print(
        f"Dataset samples: {n_samples}, Initially labeled: {labeled_count} "
        f"({test_label_fraction * 100:.1f}%)"
    )

    # 3. Instantiate model
    model = SelfTrainingModel()
    print(f"Model name: {model.get_name()}")
    print(f"Param grid: {model.get_param_grid()}")

    # 4. Check convergence trace before fit
    initial_trace = model.get_convergence_trace()
    assert initial_trace == [], f"Expected empty trace before fit, got: {initial_trace}"

    # 5. Fit model
    model.fit(features, masked_labels)

    # 6. Evaluate predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    acc: float = float(accuracy_score(labels, predictions))
    trace = model.get_convergence_trace()

    print(f"Convergence trace: {trace}")
    print(f"Total pseudo-labels added: {sum(trace):.0f}")
    print(f"Overall accuracy on full dataset: {acc:.4f}")
    print(f"Probabilities shape: {probabilities.shape}")

    # 7. Assertions
    assert model.get_name() == "Self-Training (RF)", "Model name mismatch"
    assert len(predictions) == n_samples, "Prediction count mismatch"
    assert probabilities.shape == (n_samples, 2), "Probability matrix shape mismatch"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities do not sum to 1"
    assert isinstance(trace, list), "Convergence trace should be a list"
    assert len(trace) > 0, "Expected at least one self-training iteration"
    print("=== Self-Test Passed Successfully! ===")


if __name__ == "__main__":
    _run_self_test()
