"""
team4_s3vm.py — Transductive Support Vector Machine (TSVM / S3VM) wrapper.

Wraps Team 4's Transductive SVM into the SSL-MedBench BaseSSLModel interface,
making it compatible with scikit-learn estimators, pipelines, and hyperparameter
tuners (e.g. RandomizedSearchCV).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.svm import SVC

import sys
from pathlib import Path

# Ensure package and workspace paths are accessible when run as standalone script
_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent
_ssl_medbench_dir = _backend_dir.parent
_workspace_root = _ssl_medbench_dir.parent

for _p in (_workspace_root, _ssl_medbench_dir, _backend_dir, _current_dir):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

try:
    from ssl_medbench.backend.models.base import BaseSSLModel
except ImportError:
    try:
        from backend.models.base import BaseSSLModel
    except ImportError:
        try:
            from .base import BaseSSLModel
        except (ImportError, ValueError):
            from base import BaseSSLModel  # type: ignore

try:
    from ssl_medbench.backend.config import (
        TSVM_C,
        TSVM_C_U,
        TSVM_GAMMA,
        TSVM_KERNEL,
        TSVM_MAX_SWAPS,
        TSVM_N_STEPS,
    )
except ImportError:
    try:
        from backend.config import (  # type: ignore
            TSVM_C,
            TSVM_C_U,
            TSVM_GAMMA,
            TSVM_KERNEL,
            TSVM_MAX_SWAPS,
            TSVM_N_STEPS,
        )
    except ImportError:
        try:
            from config import (  # type: ignore
                TSVM_C,
                TSVM_C_U,
                TSVM_GAMMA,
                TSVM_KERNEL,
                TSVM_MAX_SWAPS,
                TSVM_N_STEPS,
            )
        except ImportError:
            TSVM_C: float = 1.0
            TSVM_C_U: float = 1.0
            TSVM_KERNEL: str = "rbf"
            TSVM_GAMMA: str = "scale"
            TSVM_N_STEPS: int = 10
            TSVM_MAX_SWAPS: int = 50

logger = logging.getLogger(__name__)


# ====================================================================
# Original Core Implementation (Team 4)
# ====================================================================


class TSVMCore:
    """Original Team 4 Transductive SVM."""
    def __init__(self, C=1.0, C_u=1.0, kernel='rbf', gamma='scale', class_weight=None,
                 n_steps=10, max_swaps=50, balance=True, ramp_up=True):
        self.C, self.C_u, self.kernel, self.gamma = C, C_u, kernel, gamma
        self.class_weight, self.n_steps, self.max_swaps = class_weight, n_steps, max_swaps
        self.balance, self.ramp_up = balance, ramp_up

    def _svc(self):
        return SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, class_weight=self.class_weight)

    def _init_labels(self, scores, pos_ratio):
        if not self.balance:
            return np.where(scores > 0, 1.0, -1.0)
        n_pos = int(np.clip(round(pos_ratio * len(scores)), 1, len(scores) - 1))
        y_u = -np.ones(len(scores))
        y_u[np.argsort(-scores)[:n_pos]] = 1
        return y_u

    @staticmethod
    def _find_swap(y_u, slack):
        pos, neg = np.where(y_u == 1)[0], np.where(y_u == -1)[0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        i, j = pos[np.argmax(slack[pos])], neg[np.argmax(slack[neg])]
        return (i, j) if slack[i] > 0 and slack[j] > 0 and slack[i] + slack[j] > 2 else None

    def fit(self, X_l, y_l, X_u):
        self.classes_ = np.unique(y_l)
        y_l = np.where(y_l == self.classes_[1], 1.0, -1.0) if len(self.classes_) > 1 else y_l.astype(float)
        self.n_swaps_ = 0
        clf = self._svc().fit(X_l, y_l)
        if len(X_u) == 0:
            self.clf_, self.transduced_labels_ = clf, np.array([])
            return self
        y_u = self._init_labels(clf.decision_function(X_u), (y_l == 1).mean())
        X_all = np.vstack([X_l, X_u])
        schedule = (np.linspace(self.C_u / self.n_steps, self.C_u, self.n_steps)
                    if self.ramp_up else [self.C_u])
        for c_star in schedule:
            w = np.concatenate([np.ones(len(X_l)), np.full(len(X_u), c_star / self.C)])
            for it in range(self.max_swaps + 1):
                clf = self._svc().fit(X_all, np.concatenate([y_l, y_u]), sample_weight=w)
                if it == self.max_swaps:
                    break
                slack = np.maximum(0.0, 1.0 - y_u * clf.decision_function(X_u))
                swap = self._find_swap(y_u, slack)
                if swap is None:
                    break
                i, j = swap
                y_u[i], y_u[j] = -1.0, 1.0
                self.n_swaps_ += 1
        self.clf_ = clf
        self.transduced_labels_ = np.where(y_u == 1, self.classes_[1], self.classes_[0]) if len(self.classes_) > 1 else y_u
        return self

    def decision_function(self, X):
        return self.clf_.decision_function(X)


# ====================================================================
# Benchmark S3VM Model Wrapper
# ====================================================================


class S3VMModel(BaseSSLModel):
    """Semi-Supervised Support Vector Machine (S3VM / TSVM) model wrapper.

    Wraps Team 4's TSVMCore implementation to provide a scikit-learn
    compatible estimator conforming to BaseSSLModel.
    """

    def __init__(
        self,
        C: float = TSVM_C,
        C_u: float = TSVM_C_U,
        kernel: str = TSVM_KERNEL,
        gamma: Union[str, float] = TSVM_GAMMA,
        n_steps: int = TSVM_N_STEPS,
        max_swaps: int = TSVM_MAX_SWAPS,
    ) -> None:
        """Initialise S3VMModel with regularisation and schedule parameters.

        Args:
            C: Regularisation parameter for labeled samples.
            C_u: Maximum penalty weight for unlabeled slack.
            kernel: Kernel specification ('rbf', 'linear', 'poly', etc.).
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            n_steps: Number of annealing steps in the C_u ramp-up schedule.
            max_swaps: Maximum pair swaps per annealing step.
        """
        self.C = C
        self.C_u = C_u
        self.kernel = kernel
        self.gamma = gamma
        self.n_steps = n_steps
        self.max_swaps = max_swaps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "S3VMModel":
        """Fit the Transductive SVM model.

        Separates labeled (y != -1) and unlabeled samples, instantiates and fits
        the core TSVMCore estimator, and records fitted attributes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels array with -1 marking unlabeled samples.

        Returns:
            self: The fitted model instance.
        """
        logger.info("Fitting S3VMModel with %d total samples", len(y))
        labeled_mask = y != -1
        X_l: np.ndarray = X[labeled_mask]
        y_l: np.ndarray = y[labeled_mask]
        X_u: np.ndarray = X[~labeled_mask]

        if len(y_l) == 0:
            raise ValueError("Cannot fit S3VMModel without labeled data (all y == -1).")

        self.classes_: np.ndarray = np.unique(y_l)
        if len(self.classes_) > 2:
            raise ValueError(
                f"S3VMModel supports binary classification; got {len(self.classes_)} classes."
            )

        core = TSVMCore(
            C=self.C,
            C_u=self.C_u,
            kernel=self.kernel,
            gamma=self.gamma,
            n_steps=self.n_steps,
            max_swaps=self.max_swaps,
        )
        core.fit(X_l, y_l, X_u)

        self.core_: TSVMCore = core
        self.transduced_labels_: np.ndarray = core.transduced_labels_
        self.n_swaps_: int = core.n_swaps_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values for samples in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D array of decision scores.
        """
        if not hasattr(self, "core_") or self.core_ is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        return np.asarray(self.core_.decision_function(X)).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Thresholds decision values at 0 and maps back to original class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D array of predicted class labels.
        """
        decision_values = self.decision_function(X)
        if len(self.classes_) == 1:
            return np.full(X.shape[0], self.classes_[0])

        pos_mask = decision_values >= 0.0
        return np.where(pos_mask, self.classes_[1], self.classes_[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities using Platt-style sigmoid calibration.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with probability estimates.
        """
        decision_values = self.decision_function(X)
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 1), dtype=float)

        return self._decision_to_proba_binary(decision_values)

    def get_name(self) -> str:
        """Return the benchmark display name for this model."""
        return "S3VM (TSVM)"

    def get_param_grid(self) -> List[Dict[str, Any]]:
        """Return hyperparameter search space for tuning.

        Returns a list of dicts to handle kernel-dependent parameter spaces
        (linear vs. rbf with gamma).

        Returns:
            List of hyperparameter parameter distributions.
        """
        return [
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "C_u": [0.01, 0.1, 1.0, 5.0],
                "kernel": ["linear"],
                "n_steps": [5, 10],
            },
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "C_u": [0.01, 0.1, 1.0, 5.0],
                "kernel": ["rbf"],
                "gamma": [0.01, 0.1, 1.0, "scale"],
                "n_steps": [5, 10],
            },
        ]

    def get_convergence_trace(self) -> List[float]:
        """Return per-iteration metric convergence trace.

        Returns:
            Empty list as TSVM does not log metric traces during training.
        """
        return []


# ====================================================================
# Self-Test Execution
# ====================================================================


if __name__ == "__main__":
    from sklearn.base import clone
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Running S3VMModel self-test on breast cancer dataset...")
    data = load_breast_cancer()
    X_raw, y_raw = data.data, data.target

    # Scale features for SVM stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_raw,
        test_size=0.20,
        random_state=42,
        stratify=y_raw,
    )

    # Mask to keep exactly 10% labeled training samples
    rng = np.random.RandomState(42)
    y_train_semi = y_train.copy()
    n_labeled = max(2, int(0.10 * len(y_train)))
    unlabeled_idx = rng.choice(
        len(y_train),
        size=len(y_train) - n_labeled,
        replace=False,
    )
    y_train_semi[unlabeled_idx] = -1

    n_actual_l = np.sum(y_train_semi != -1)
    n_actual_u = np.sum(y_train_semi == -1)
    print(f"Dataset: {len(X_train)} train samples ({n_actual_l} labeled, {n_actual_u} unlabeled), {len(X_test)} test samples")

    # Instantiate model
    model = S3VMModel(
        C=1.0,
        C_u=1.0,
        kernel="rbf",
        gamma="scale",
        n_steps=5,
        max_swaps=20,
    )

    # Test interface methods before fit
    assert model.get_name() == "S3VM (TSVM)", f"Unexpected name: {model.get_name()}"
    assert model.get_convergence_trace() == [], "Convergence trace must be empty list"
    assert len(model.get_param_grid()) == 2, "Param grid must return 2 kernel configurations"

    # Test sklearn clone compatibility
    cloned_model = clone(model)
    assert cloned_model.C == model.C
    assert cloned_model.kernel == model.kernel

    # Fit model
    model.fit(X_train, y_train_semi)
    assert hasattr(model, "classes_")
    assert hasattr(model, "transduced_labels_")
    assert hasattr(model, "n_swaps_")

    # Evaluate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    assert y_pred.shape == y_test.shape, f"Shape mismatch: {y_pred.shape} vs {y_test.shape}"
    assert y_proba.shape == (len(X_test), 2), f"Proba shape mismatch: {y_proba.shape}"
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities must sum to 1.0"
    assert np.all(y_proba >= 0.0) and np.all(y_proba <= 1.0), "Probabilities must be in [0, 1]"

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba[:, 1])

    print("--- S3VM Self-Test Results ---")
    print(f"Swaps performed: {model.n_swaps_}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro F1:       {f1:.4f}")
    print(f"ROC AUC:        {auc:.4f}")

    # Sanity check: with 10% labels on scaled breast cancer, S3VM should easily exceed 0.80 F1
    assert f1 > 0.80, f"F1 score {f1:.4f} is lower than expected baseline (>0.80)"
    print("All S3VMModel tests passed successfully!")
