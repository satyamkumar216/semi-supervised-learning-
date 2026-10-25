"""
team3_co_training.py — Co-Training with Support Vector Machines (Team 3 wrapper).

Co-Training partitions the feature space into two conditionally independent
views and trains an SVM on each view. At each iteration, high-confidence
predictions from one classifier are transferred as pseudo-labels to enrich the
training pool of the other classifier, iteratively expanding the labeled set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

# ── Imports with Config Fallback ─────────────────────────────────────

try:
    from backend.config import (
        COTRAIN_C,
        COTRAIN_N,
        COTRAIN_N_ITER,
        COTRAIN_P,
        COTRAIN_U,
        RANDOM_SEED,
    )
    from backend.models.base import BaseSSLModel
except ImportError:
    try:
        from ..config import (
            COTRAIN_C,
            COTRAIN_N,
            COTRAIN_N_ITER,
            COTRAIN_P,
            COTRAIN_U,
            RANDOM_SEED,
        )
        from .base import BaseSSLModel
    except (ImportError, ValueError):
        import sys
        from pathlib import Path

        _current_dir = Path(__file__).resolve().parent
        _backend_dir = _current_dir.parent
        if str(_backend_dir) not in sys.path:
            sys.path.insert(0, str(_backend_dir))
        if str(_current_dir) not in sys.path:
            sys.path.insert(0, str(_current_dir))

        try:
            from config import (  # type: ignore
                COTRAIN_C,
                COTRAIN_N,
                COTRAIN_N_ITER,
                COTRAIN_P,
                COTRAIN_U,
                RANDOM_SEED,
            )
        except ImportError:
            COTRAIN_P = 1
            COTRAIN_N = 1
            COTRAIN_U = 30
            COTRAIN_N_ITER = 15
            COTRAIN_C = 1.0
            RANDOM_SEED = 42

        try:
            from base import BaseSSLModel  # type: ignore
        except ImportError:
            from models.base import BaseSSLModel  # type: ignore

logger = logging.getLogger(__name__)

# ── Hyperparameter Defaults & Constants ──────────────────────────────
DEFAULT_VIEW_SPLIT_RATIO: float = 0.5
PARAM_GRID_C: List[float] = [0.1, 0.5, 1.0, 5.0, 10.0]
PARAM_GRID_N_ITER: List[int] = [5, 10, 15, 20]
PARAM_GRID_U: List[int] = [20, 30, 50]


# ── Original Core Algorithm (Verbatim) ──────────────────────────────

class CoTrainingSVMCore:
    """Original Team 3 co-training SVM with two-view learning."""
    def __init__(self, p=1, n=1, u=30, n_iter=15, C=1.0):
        self.p = p
        self.n = n
        self.u = u
        self.n_iter = n_iter
        self.C = C
        self.clf1 = SVC(C=self.C, kernel='rbf')
        self.clf2 = SVC(C=self.C, kernel='rbf')

    def fit(self, X1_L, X2_L, y_L, X1_U, X2_U):
        L1, L2, y = list(X1_L), list(X2_L), list(y_L)
        U1, U2 = list(X1_U), list(X2_U)
        np.random.seed(42)
        pool_indices = list(np.random.choice(len(U1), size=min(self.u, len(U1)), replace=False))
        pool_indices.sort(reverse=True)
        U1_pool = [U1.pop(i) for i in pool_indices]
        U2_pool = [U2.pop(i) for i in pool_indices]
        for round_num in range(self.n_iter):
            self.clf1.fit(np.array(L1), np.array(y))
            self.clf2.fit(np.array(L2), np.array(y))
            if len(U1_pool) == 0:
                break
            margins1 = self.clf1.decision_function(np.array(U1_pool))
            margins2 = self.clf2.decision_function(np.array(U2_pool))
            picks_clf1_pos = np.argsort(margins1)[-self.p:]
            picks_clf1_neg = np.argsort(margins1)[:self.n]
            picks_clf2_pos = np.argsort(margins2)[-self.p:]
            picks_clf2_neg = np.argsort(margins2)[:self.n]
            selected = {}
            for idx in picks_clf1_pos: selected[idx] = 1
            for idx in picks_clf1_neg: selected[idx] = 0
            for idx in picks_clf2_pos: selected[idx] = 1
            for idx in picks_clf2_neg: selected[idx] = 0
            for idx in sorted(selected.keys(), reverse=True):
                L1.append(U1_pool.pop(idx))
                L2.append(U2_pool.pop(idx))
                y.append(selected[idx])
            replenish_count = min(len(selected), len(U1))
            for _ in range(replenish_count):
                rand_idx = np.random.randint(len(U1))
                U1_pool.append(U1.pop(rand_idx))
                U2_pool.append(U2.pop(rand_idx))
        self.clf1.fit(np.array(L1), np.array(y))
        self.clf2.fit(np.array(L2), np.array(y))
        return self

    def predict(self, X1, X2):
        fused_margin = self.clf1.decision_function(X1) + self.clf2.decision_function(X2)
        return (fused_margin >= 0).astype(int)


# ── Benchmark Model Wrapper ──────────────────────────────────────────

class CoTrainingModel(BaseSSLModel):
    """Benchmark wrapper for Team 3's Co-Training SVM with internal feature splitting.

    Inherits from BaseSSLModel to provide scikit-learn estimator compatibility,
    calibrated probability outputs, and standardized benchmark interfaces.
    """

    def __init__(
        self,
        p: int = COTRAIN_P,
        n: int = COTRAIN_N,
        u: int = COTRAIN_U,
        n_iter: int = COTRAIN_N_ITER,
        C: float = COTRAIN_C,
        view_split_ratio: float = DEFAULT_VIEW_SPLIT_RATIO,
    ) -> None:
        """Initialize CoTrainingModel.

        Args:
            p: Positive pseudo-labels added per iteration per view.
            n: Negative pseudo-labels added per iteration per view.
            u: Size of candidate unlabeled pool buffer.
            n_iter: Number of co-training rounds.
            C: Regularization parameter for the internal RBF SVMs.
            view_split_ratio: Fraction of features allocated to view 1.
        """
        self.p = p
        self.n = n
        self.u = u
        self.n_iter = n_iter
        self.C = C
        self.view_split_ratio = view_split_ratio
        self.core_: Optional[CoTrainingSVMCore] = None
        self.split_idx_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def _compute_split_index(self, n_features: int) -> int:
        """Calculate column boundary separating feature views.

        Args:
            n_features: Total number of input features.

        Returns:
            Column split index separating view 1 and view 2.
        """
        if n_features < 2:
            return 1
        raw_split = int(n_features * self.view_split_ratio)
        return max(1, min(raw_split, n_features - 1))

    def _split_views(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split feature matrix into two views using stored split index.

        Args:
            X: Input feature array of shape (n_samples, n_features).

        Returns:
            Tuple of view arrays (X1, X2).
        """
        if self.split_idx_ is None:
            raise ValueError("Split index has not been initialized.")
        return X[:, : self.split_idx_], X[:, self.split_idx_ :]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CoTrainingModel":
        """Fit co-training model on partially labeled data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels array with -1 marking unlabeled samples.

        Returns:
            Fitted instance of CoTrainingModel.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.split_idx_ = self._compute_split_index(n_features)
        self.X_train_ = X_arr
        self.y_train_ = y_arr

        labeled_mask = y_arr != -1
        if not np.any(labeled_mask):
            raise ValueError("At least one labeled sample is required to fit CoTrainingModel.")

        y_labeled = y_arr[labeled_mask]
        self.classes_ = np.unique(y_labeled)

        X1, X2 = self._split_views(X_arr)
        X1_L, X2_L = X1[labeled_mask], X2[labeled_mask]
        X1_U, X2_U = X1[~labeled_mask], X2[~labeled_mask]

        core = CoTrainingSVMCore(
            p=self.p,
            n=self.n,
            u=self.u,
            n_iter=self.n_iter,
            C=self.C,
        )
        core.fit(X1_L, X2_L, y_labeled, X1_U, X2_U)
        self.core_ = core
        logger.info(
            "Fitted CoTrainingModel: labeled=%d, unlabeled=%d, split_idx=%d",
            len(y_labeled),
            len(X1_U),
            self.split_idx_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for samples in X.

        Args:
            X: Input feature array of shape (n_samples, n_features).

        Returns:
            Predicted 1-D binary array of labels (0 or 1).
        """
        check_is_fitted(self, attributes=["core_", "split_idx_"])
        X_arr = np.asarray(X, dtype=np.float64)
        X1, X2 = self._split_views(X_arr)
        return self.core_.predict(X1, X2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Args:
            X: Input feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) containing class probabilities.
        """
        check_is_fitted(self, attributes=["core_", "split_idx_"])
        X_arr = np.asarray(X, dtype=np.float64)
        X1, X2 = self._split_views(X_arr)
        margin1 = self.core_.clf1.decision_function(X1)
        margin2 = self.core_.clf2.decision_function(X2)
        fused_margin = margin1 + margin2
        return self._decision_to_proba_binary(fused_margin)

    def get_name(self) -> str:
        """Return human-readable model identifier.

        Returns:
            Model name string.
        """
        return "Co-Training (SVM)"

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Return hyperparameter search space for tuning.

        Returns:
            Dictionary mapping hyperparameter names to candidate values.
        """
        return {
            "C": PARAM_GRID_C,
            "n_iter": PARAM_GRID_N_ITER,
            "u": PARAM_GRID_U,
        }

    def get_convergence_trace(self) -> List[float]:
        """Return convergence metric trace.

        Returns:
            Empty list since Co-Training does not track a loss trace.
        """
        return []


# ── Self-Test / Verification ─────────────────────────────────────────

def _run_self_test() -> None:
    """Run verification self-test on breast cancer dataset with 10% labels."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    print("Running CoTrainingModel self-test on Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Mask 90% of training labels as -1 (10% labeled)
    rng = np.random.RandomState(RANDOM_SEED)
    labeled_mask = rng.rand(len(y_train)) < 0.10
    y_train_masked = np.full_like(y_train, -1)
    y_train_masked[labeled_mask] = y_train[labeled_mask]

    n_labeled = int(np.sum(y_train_masked != -1))
    print(f"Total train samples: {len(y_train)}, Labeled: {n_labeled}")

    model = CoTrainingModel(
        p=1,
        n=1,
        u=30,
        n_iter=15,
        C=1.0,
        view_split_ratio=0.5,
    )
    model.fit(X_train, y_train_masked)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba[:, 1])

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Test ROC-AUC:  {auc:.4f}")
    print(f"Model Name:    {model.get_name()}")
    print(f"Param Grid:    {model.get_param_grid()}")
    print(f"Convergence:   {model.get_convergence_trace()}")

    assert y_pred.shape == y_test.shape, "Prediction shape mismatch"
    assert y_proba.shape == (len(y_test), 2), "Probability shape mismatch"
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities must sum to 1"
    assert acc > 0.70, f"Expected accuracy > 0.70, got {acc:.4f}"
    print("Self-test PASSED successfully!")


if __name__ == "__main__":
    _run_self_test()
