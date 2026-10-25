"""
scorer.py — Metric computation and final composite scoring.

Two public functions:
    compute_scores()       — per-model metrics (accuracy, F1, AUC, time).
    compute_final_score()  — composite Score across all models.

Dependency order: config -> scorer
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from backend.config import (
    WEIGHT_AUC,
    WEIGHT_EFFICIENCY,
    WEIGHT_F1,
    WEIGHT_ROBUSTNESS,
    ROBUSTNESS_HIGH_FRAC,
    ROBUSTNESS_LOW_FRAC,
)

logger = logging.getLogger(__name__)


# ── Per-Model Metrics ──────────────────────────────────────────────

def compute_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    train_time: float,
) -> Dict[str, float]:
    """Compute classification metrics for a single model run.

    Args:
        y_true:     Ground-truth labels (1-D, no -1 values).
        y_pred:     Predicted labels (1-D).
        y_proba:    Predicted probabilities, shape (N, n_classes).
                    May be None if the model failed to produce them.
        train_time: Wall-clock training time in seconds.

    Returns:
        Dict with keys: accuracy, f1_macro, auc_roc, train_time.
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    auc = _safe_auc(y_true, y_proba)

    scores = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "auc_roc": round(auc, 4),
        "train_time": round(train_time, 4),
    }
    return scores


def _safe_auc(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> float:
    """Compute AUC-ROC safely, returning 0.5 on failure.

    Binary:       uses proba[:, 1].
    Multiclass:   uses one-vs-rest macro averaging.

    Args:
        y_true:  Ground-truth labels.
        y_proba: Probability matrix or None.

    Returns:
        AUC-ROC score, or 0.5 if computation fails.
    """
    if y_proba is None:
        return 0.5

    try:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # Binary: use probability of the positive class
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            return float(roc_auc_score(y_true, y_proba.ravel()))
        else:
            # Multiclass: one-vs-rest, macro
            return float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            ))
    except (ValueError, IndexError) as exc:
        logger.warning("AUC computation failed: %s", exc)
        return 0.5


# ── Composite Final Score ──────────────────────────────────────────

def compute_final_score(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the composite Score for each model.

    Formula (weights from config.py):
        Score = W_F1 * F1 + W_AUC * AUC + W_ROBUST * Robustness
              + W_EFF * Efficiency

    Robustness = 1 - (F1_at_high_frac - F1_at_low_frac), clipped [0,1].
    Efficiency = F1 / train_time, then min-max normalised to [0,1].

    Args:
        results_df: DataFrame with columns:
            model_name, label_fraction, f1_macro, auc_roc, train_time.
            Each row is the *mean* across folds for one model × fraction.

    Returns:
        DataFrame with one row per model_name, including:
            f1_mean, auc_mean, robustness, efficiency_raw,
            efficiency_norm, final_score.
    """
    models = results_df["model_name"].unique()
    rows = []

    for name in models:
        subset = results_df[results_df["model_name"] == name]

        # Mean F1 and AUC across all fractions
        f1_mean = float(subset["f1_macro"].mean())
        auc_mean = float(subset["auc_roc"].mean())

        # Robustness
        robustness = _compute_robustness(subset)

        # Efficiency (raw — normalised later)
        time_mean = float(subset["train_time"].mean())
        efficiency_raw = f1_mean / max(time_mean, 1e-6)

        rows.append({
            "model_name": name,
            "f1_mean": round(f1_mean, 4),
            "auc_mean": round(auc_mean, 4),
            "robustness": round(robustness, 4),
            "efficiency_raw": round(efficiency_raw, 4),
            "train_time_mean": round(time_mean, 4),
        })

    score_df = pd.DataFrame(rows)

    # Min-max normalise efficiency across models
    eff_vals = score_df["efficiency_raw"].values
    eff_min, eff_max = eff_vals.min(), eff_vals.max()
    if eff_max - eff_min > 1e-12:
        score_df["efficiency_norm"] = (
            (eff_vals - eff_min) / (eff_max - eff_min)
        ).round(4)
    else:
        score_df["efficiency_norm"] = 0.5  # all equal

    # Composite score
    score_df["final_score"] = (
        WEIGHT_F1 * score_df["f1_mean"]
        + WEIGHT_AUC * score_df["auc_mean"]
        + WEIGHT_ROBUSTNESS * score_df["robustness"]
        + WEIGHT_EFFICIENCY * score_df["efficiency_norm"]
    ).round(4)

    score_df.sort_values("final_score", ascending=False, inplace=True)
    score_df.reset_index(drop=True, inplace=True)
    score_df.insert(0, "rank", range(1, len(score_df) + 1))

    logger.info("Final scores computed for %d models.", len(score_df))
    return score_df


def _compute_robustness(subset: pd.DataFrame) -> float:
    """Compute robustness from a model's results across fractions.

    Robustness = 1 - (F1_at_high_frac - F1_at_low_frac), clipped [0, 1].

    Args:
        subset: Rows for one model across label fractions.

    Returns:
        Robustness score in [0, 1].
    """
    high_rows = subset[
        np.isclose(subset["label_fraction"], ROBUSTNESS_HIGH_FRAC)
    ]
    low_rows = subset[
        np.isclose(subset["label_fraction"], ROBUSTNESS_LOW_FRAC)
    ]

    f1_high = float(high_rows["f1_macro"].mean()) if len(high_rows) else 0.0
    f1_low = float(low_rows["f1_macro"].mean()) if len(low_rows) else 0.0

    robustness = 1.0 - (f1_high - f1_low)
    return float(np.clip(robustness, 0.0, 1.0))
