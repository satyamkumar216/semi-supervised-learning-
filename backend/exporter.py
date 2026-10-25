"""
exporter.py — Export functionality for winning SSL model and enriched datasets.

Supports:
- Mode A: Retraining winner on full data with best params & saving to winner_model.pkl.
- Mode B: Pseudo-labeling unlabeled data above confidence threshold to enriched_dataset.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.config import (
    OUTPUT_DIR,
    PSEUDO_LABEL_CONFIDENCE_THRESHOLD,
    RANDOM_SEED,
)
from backend.models import MODEL_REGISTRY
from backend.models.base import BaseSSLModel

logger = logging.getLogger(__name__)


# ── Mode A: Export Model ───────────────────────────────────────────

def export_winner_model(
    winner_name: str,
    best_params: Dict[str, Any],
    X: np.ndarray,
    y_masked: np.ndarray,
    job_id: str,
) -> Path:
    """Retrain the winning model on the dataset and save with scaler.

    Args:
        winner_name: Registered model name.
        best_params: Tuned hyperparameter dictionary.
        X: Full feature matrix.
        y_masked: Labels with -1 for unlabeled points.
        job_id: Unique benchmark run identifier.

    Returns:
        Path to the saved winner_model.pkl file.
    """
    model_class = MODEL_REGISTRY[winner_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model: BaseSSLModel = model_class(**best_params)
    model.fit(X_scaled, y_masked)

    save_payload = {
        "model_name": winner_name,
        "best_params": best_params,
        "scaler": scaler,
        "model": model,
    }

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / "winner_model.pkl"
    joblib.dump(save_payload, out_path)
    logger.info("Saved winning model to %s", out_path)
    return out_path


# ── Mode B: Export Enriched Dataset ────────────────────────────────

def export_enriched_dataset(
    winner_name: str,
    best_params: Dict[str, Any],
    raw_df: pd.DataFrame,
    target_column: str,
    X: np.ndarray,
    y_masked: np.ndarray,
    job_id: str,
    confidence_threshold: float = PSEUDO_LABEL_CONFIDENCE_THRESHOLD,
) -> Tuple[Path, Dict[str, int]]:
    """Generate pseudo-labels for unlabeled rows and export enriched CSV.

    Args:
        winner_name: Winning model name.
        best_params: Tuned parameters.
        raw_df: Original unmodified DataFrame.
        target_column: Target column name.
        X: Feature matrix.
        y_masked: Labels with -1 for unlabeled rows.
        job_id: Unique benchmark identifier.
        confidence_threshold: Minimum probability threshold for pseudo-labels.

    Returns:
        (path_to_csv, stats_dict)
    """
    model_class = MODEL_REGISTRY[winner_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model: BaseSSLModel = model_class(**best_params)
    model.fit(X_scaled, y_masked)

    unlabeled_mask = (y_masked == -1)
    df_out = raw_df.copy()
    df_out["label_source"] = "original"

    added_count = 0
    discarded_count = 0

    if unlabeled_mask.any():
        X_unlabeled = X_scaled[unlabeled_mask]
        unlabeled_indices = np.where(unlabeled_mask)[0]

        try:
            probs = model.predict_proba(X_unlabeled)
            max_conf = np.max(probs, axis=1)
            preds = model.predict(X_unlabeled)

            for idx, conf, pred in zip(unlabeled_indices, max_conf, preds):
                if conf >= confidence_threshold:
                    df_out.iloc[idx, df_out.columns.get_loc(target_column)] = pred
                    df_out.iloc[idx, df_out.columns.get_loc("label_source")] = "pseudo"
                    added_count += 1
                else:
                    discarded_count += 1
        except Exception as exc:
            logger.warning("Pseudo-labeling failed: %s", exc)
            discarded_count = int(unlabeled_mask.sum())

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_csv = job_dir / "enriched_dataset.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved enriched dataset to %s (added: %d, discarded: %d)",
                out_csv, added_count, discarded_count)

    stats = {
        "pseudo_labels_added": added_count,
        "pseudo_labels_discarded": discarded_count,
        "total_unlabeled": int(unlabeled_mask.sum()),
    }
    return out_csv, stats
