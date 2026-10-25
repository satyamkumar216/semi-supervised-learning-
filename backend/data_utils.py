"""
data_utils.py — Data loading, health reporting, preprocessing, and label masking.

Dependency order: config -> data_utils
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.config import DEFAULT_LABEL_FRACTIONS, RANDOM_SEED

logger = logging.getLogger(__name__)


# ── Custom Errors ──────────────────────────────────────────────────

class InvalidDatasetError(Exception):
    """Raised when the uploaded CSV is invalid or cannot be processed."""


class TargetColumnError(Exception):
    """Raised when the specified target column is missing or unusable."""


# ── Data Loading ───────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        pandas DataFrame.

    Raises:
        InvalidDatasetError: If the file cannot be read.
    """
    try:
        df = pd.read_csv(path, na_values=["?", "", "NA", "N/A"])
        logger.info("Loaded CSV with shape %s from %s", df.shape, path)
        return df
    except Exception as exc:
        raise InvalidDatasetError(f"Cannot read CSV: {exc}") from exc


# ── Data Health Report ─────────────────────────────────────────────

def build_health_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a data health summary.

    Args:
        df: Raw DataFrame.

    Returns:
        Dict with shape, columns, dtypes, missing counts, class info.
    """
    report: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {
            col: int(cnt)
            for col, cnt in df.isnull().sum().items()
            if cnt > 0
        },
        "missing_pct": round(
            float(df.isnull().sum().sum()) / max(df.size, 1) * 100, 2
        ),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(
            df.select_dtypes(include=["object", "category", "bool"]).columns
        ),
    }
    logger.info("Health report: %d rows, %d cols, %.1f%% missing",
                report["n_rows"], report["n_columns"], report["missing_pct"])
    return report


# ── Preprocessing ──────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Clean the DataFrame and separate features from the target.

    Steps:
        1. Validate target column exists and is discrete.
        2. Drop constant columns.
        3. Encode categorical features with LabelEncoder.
        4. Impute remaining NaNs with column medians.
        5. Encode the target to consecutive integers starting at 0.

    Args:
        df: Raw DataFrame.
        target_column: Name of the label column.

    Returns:
        (X, y, feature_names) where X is float64, y is int64.

    Raises:
        TargetColumnError: If the target column is invalid.
    """
    if target_column not in df.columns:
        raise TargetColumnError(
            f"Column '{target_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    target_series = df[target_column]
    n_unique = target_series.nunique(dropna=True)
    if n_unique < 2:
        raise TargetColumnError(
            f"Target '{target_column}' has fewer than 2 unique values."
        )
    if n_unique > 50:
        raise TargetColumnError(
            f"Target '{target_column}' has {n_unique} unique values — "
            "looks continuous, not a classification target."
        )

    features_df = df.drop(columns=[target_column]).copy()

    # Drop constant columns
    constant_cols = [c for c in features_df.columns
                     if features_df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        logger.info("Dropping %d constant columns: %s",
                     len(constant_cols), constant_cols)
        features_df.drop(columns=constant_cols, inplace=True)

    # Encode categorical features
    cat_cols = features_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        features_df[col] = le.fit_transform(
            features_df[col].astype(str).fillna("__MISSING__")
        )

    # Impute NaNs with median
    for col in features_df.columns:
        if features_df[col].isnull().any():
            median_val = features_df[col].median()
            features_df[col].fillna(median_val, inplace=True)

    X = features_df.values.astype(np.float64)
    feature_names = list(features_df.columns)

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(target_series.fillna(-1).astype(str)).astype(np.int64)

    logger.info("Preprocessed: X %s, y %s, %d classes",
                X.shape, y.shape, len(le_target.classes_))
    return X, y, feature_names


# ── Class Distribution ─────────────────────────────────────────────

def class_distribution(y: np.ndarray) -> Dict[int, int]:
    """Count samples per class.

    Args:
        y: Label array.

    Returns:
        Dict mapping class label to count.
    """
    unique, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


# ── Label Masking ──────────────────────────────────────────────────

def mask_labels(
    y: np.ndarray,
    label_fraction: float,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Mask labels for the SSL scenario.

    Keep ``label_fraction`` of labels per class (stratified), set the
    rest to -1 (unlabeled).  At least 2 samples per class are kept.

    Args:
        y: Ground-truth label array (1-D).
        label_fraction: Fraction in (0, 1) of labels to keep.
        random_state: RNG seed for reproducibility.

    Returns:
        Copy of y with unlabeled positions set to -1.
    """
    rng = np.random.RandomState(random_state)
    y_masked = np.full_like(y, fill_value=-1)

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        n_keep = max(2, int(len(cls_indices) * label_fraction))
        n_keep = min(n_keep, len(cls_indices))  # safety
        keep = rng.choice(cls_indices, size=n_keep, replace=False)
        y_masked[keep] = y[keep]

    n_labeled = int((y_masked != -1).sum())
    logger.info("Masked labels: fraction=%.2f, labeled=%d, unlabeled=%d",
                label_fraction, n_labeled, len(y) - n_labeled)
    return y_masked


# ── Random-Baseline F1 ────────────────────────────────────────────

def random_baseline_f1(y: np.ndarray) -> float:
    """Compute expected macro-F1 of a random classifier.

    Uses the class distribution to compute the expected F1 per class
    when predictions are drawn from the marginal distribution, then
    averages.

    Args:
        y: True labels (only labeled points, no -1).

    Returns:
        Expected macro-F1 score.
    """
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(RANDOM_SEED)
    y_pred = rng.choice(np.unique(y), size=len(y))
    return float(f1_score(y, y_pred, average="macro", zero_division=0.0))
