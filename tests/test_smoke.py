"""
test_smoke.py — Smoke tests for SSL-MedBench models and core components.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from backend.data_utils import mask_labels
from backend.exporter import export_enriched_dataset, export_winner_model
from backend.models import MODEL_REGISTRY
from backend.scorer import compute_final_score, compute_scores
from backend.tuner import tune_model


@pytest.fixture
def tiny_dataset():
    """Generate a tiny standardized medical dataset for fast testing."""
    data = load_breast_cancer()
    X = data.data[:60]
    y = data.target[:60]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_masked = mask_labels(y, label_fraction=0.20, random_state=42)
    return X_scaled, y, y_masked


@pytest.mark.parametrize("model_name,model_class", MODEL_REGISTRY.items())
def test_model_fit_and_predict(model_name, model_class, tiny_dataset):
    """Test that all registered SSL models fit and predict accurately without errors."""
    X, _, y_masked = tiny_dataset
    model = model_class()

    # Fit model on masked labels
    model.fit(X, y_masked)

    # Predict class labels
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    # Predict class probabilities
    probs = model.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-3)

    # Model name and trace
    assert isinstance(model.get_name(), str)
    assert isinstance(model.get_convergence_trace(), list)


def test_tuner_on_tiny_data(tiny_dataset):
    """Test that hyperparameter tuner runs on a model without crashing."""
    X, _, y_masked = tiny_dataset
    first_model_class = next(iter(MODEL_REGISTRY.values()))
    result = tune_model(first_model_class, X, y_masked, label_fraction=0.20)

    assert "best_params" in result
    assert "best_cv_score" in result
    assert result["status"] in ("SUCCESS", "DEFAULT_ONLY", "TUNING_FAILED")


def test_scorer():
    """Test compute_scores and compute_final_score."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.2, 0.8],
    ])
    scores = compute_scores(y_true, y_pred, y_proba, train_time=0.1)
    assert "accuracy" in scores
    assert "f1_macro" in scores
    assert "auc_roc" in scores
    assert "train_time" in scores
