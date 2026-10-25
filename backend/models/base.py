"""
base.py — Abstract base class for all SSL models in SSL-MedBench.

Every team model inherits ``BaseSSLModel``, which in turn inherits
scikit-learn's ``BaseEstimator`` and ``ClassifierMixin``.  This makes
every model compatible with ``RandomizedSearchCV`` (``get_params``,
``set_params``, ``clone``).

Required methods that each subclass must implement:
    fit(X, y)            — y uses -1 for unlabeled samples.
    predict(X)           — class labels.
    predict_proba(X)     — class probabilities (N, n_classes).
    get_name()           — human-readable model name.
    get_param_grid()     — search space dict for the tuner.
    get_convergence_trace() — list of per-iteration scores (empty for
                              non-iterative models).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class BaseSSLModel(BaseEstimator, ClassifierMixin):
    """Abstract base for every SSL model in the benchmark.

    Inherits sklearn's ``BaseEstimator`` + ``ClassifierMixin`` so that
    ``get_params``, ``set_params``, ``clone``, ``score`` all work out
    of the box with ``RandomizedSearchCV``.
    """

    # ── abstract interface ────────────────────────────────────────

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSSLModel":
        """Fit the model.  ``y`` uses -1 for unlabeled points.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels; -1 marks unlabeled samples.

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            1-D array of predicted labels.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-probability estimates.

        If the underlying algorithm does not natively produce
        probabilities, the subclass should implement a fallback
        (e.g. softmaxed decision-function values).

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes).
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a short, human-readable model name."""

    @abstractmethod
    def get_param_grid(self) -> Dict[str, Any]:
        """Return the hyperparameter search space for the tuner.

        Returns:
            dict mapping parameter names to lists / distributions.
        """

    @abstractmethod
    def get_convergence_trace(self) -> List[float]:
        """Return per-iteration metric trace.

        Non-iterative models return ``[]``.
        """

    # ── shared helpers ────────────────────────────────────────────

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """Row-wise softmax of a 2-D scores array.

        Useful as a fallback ``predict_proba`` for models that only
        provide raw decision-function values.

        Args:
            scores: Array of shape (n_samples, n_classes).

        Returns:
            Probability array of the same shape.
        """
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    @staticmethod
    def _decision_to_proba_binary(decision_values: np.ndarray) -> np.ndarray:
        """Convert 1-D binary decision-function values to (N, 2) probabilities.

        Uses the sigmoid function: p = 1 / (1 + exp(-decision)).

        Args:
            decision_values: 1-D array of decision scores.

        Returns:
            Array of shape (n_samples, 2).
        """
        prob_pos = 1.0 / (1.0 + np.exp(-np.clip(decision_values, -500, 500)))
        return np.column_stack([1.0 - prob_pos, prob_pos])
