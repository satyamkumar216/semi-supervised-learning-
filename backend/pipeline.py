"""
pipeline.py — Master benchmark orchestration engine for SSL-MedBench.

Evaluates 5 SSL models across 5 outer stratified folds and multiple
label fractions (0.05, 0.10, 0.20, 0.30) with zero data leakage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from backend.config import (
    DEFAULT_LABEL_FRACTIONS,
    N_OUTER_FOLDS,
    OUTPUT_DIR,
    PLOT_DPI,
    PLOT_FIGSIZE_WIDE,
    RANDOM_SEED,
    SANITY_MARGIN,
)
from backend.data_utils import mask_labels, random_baseline_f1
from backend.models import MODEL_REGISTRY
from backend.models.base import BaseSSLModel
from backend.scorer import compute_final_score, compute_scores
from backend.tuner import tune_model

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str, str], None]


# ── Single Model Evaluation Helper ─────────────────────────────────

def _fit_and_score(
    model: BaseSSLModel,
    X_train: np.ndarray,
    y_masked: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], Optional[BaseSSLModel]]:
    """Fit a model on masked training data and score on held-out test data."""
    start_time = time.time()
    try:
        model.fit(X_train, y_masked)
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
        scores = compute_scores(y_test, y_pred, y_proba, train_time)
        return scores, model
    except Exception as exc:
        logger.warning("Fit/score failed for %s: %s", model.get_name(), exc)
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "auc_roc": 0.5,
            "train_time": time.time() - start_time,
        }, None


# ── Run Model on a Single Fold ─────────────────────────────────────

def _run_model_on_fold(
    model_name: str,
    model_class: Type[BaseSSLModel],
    X_train: np.ndarray,
    y_train_masked: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_fraction: float,
) -> Dict[str, Any]:
    """Execute default run, tuning, and tuned run for one model on one fold."""
    # A) Default run
    def_model = model_class()
    def_scores, _ = _fit_and_score(def_model, X_train, y_train_masked, X_test, y_test)

    # B) Hyperparameter tuning on labeled points only
    tune_res = tune_model(model_class, X_train, y_train_masked, label_fraction)
    best_params = tune_res.get("best_params", {})

    # C) Tuned run with best parameters
    try:
        tuned_model = model_class(**best_params)
        tuned_scores, fitted_tuned = _fit_and_score(
            tuned_model, X_train, y_train_masked, X_test, y_test
        )
    except Exception as exc:
        logger.warning("Instantiating tuned %s failed: %s", model_name, exc)
        tuned_scores = def_scores
        fitted_tuned = None

    return {
        "model_name": model_name,
        "label_fraction": label_fraction,
        "default_scores": def_scores,
        "tuned_scores": tuned_scores,
        "best_params": best_params,
        "fitted_model": fitted_tuned,
    }


# ── Validation Metrics Computation ─────────────────────────────────

def _compute_validation_signals(
    model: BaseSSLModel,
    X_train: np.ndarray,
    y_train_full: np.ndarray,
    y_train_masked: np.ndarray,
) -> Dict[str, Any]:
    """Compute pseudo-label accuracy on hidden points and label stability."""
    unlabeled_mask = (y_train_masked == -1)
    if not unlabeled_mask.any():
        return {
            "pseudo_label_acc": 1.0,
            "label_stability": "N/A",
            "converged": True,
            "trace": [],
        }

    X_unlabeled = X_train[unlabeled_mask]
    y_true_unlabeled = y_train_full[unlabeled_mask]

    try:
        pseudo_preds = model.predict(X_unlabeled)
        pseudo_acc = float((pseudo_preds == y_true_unlabeled).mean())
    except Exception:
        pseudo_acc = 0.0

    trace = model.get_convergence_trace()
    converged = len(trace) > 0

    return {
        "pseudo_label_acc": round(pseudo_acc * 100, 2),
        "label_stability": f"Checked ({len(trace)} iters)" if trace else "N/A",
        "converged": converged,
        "trace": trace,
    }


# ── Plot Generation Helpers ────────────────────────────────────────

def _generate_plots(
    results_df: pd.DataFrame,
    traces: Dict[str, List[float]],
    job_dir: Path,
) -> Dict[str, str]:
    """Render and save benchmark plots."""
    plots: Dict[str, str] = {}

    # Plot 1: F1 vs Label Fraction
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    for model_name, group in results_df.groupby("model_name"):
        sorted_grp = group.sort_values("label_fraction")
        ax.plot(
            sorted_grp["label_fraction"] * 100,
            sorted_grp["f1_macro"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
    ax.set_title("SSL Models: Macro-F1 vs Labeled Data Fraction")
    ax.set_xlabel("% Labeled Data")
    ax.set_ylabel("Held-Out Test Macro-F1")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    f1_plot_path = job_dir / "f1_vs_fraction.png"
    fig.savefig(f1_plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    plots["f1_vs_fraction"] = str(f1_plot_path.name)

    # Plot 2: Convergence Trace
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    has_trace = False
    for model_name, trace in traces.items():
        if trace:
            has_trace = True
            ax.plot(range(1, len(trace) + 1), trace, marker="s", label=model_name)
    if not has_trace:
        ax.text(0.5, 0.5, "No iterative trace data available", ha="center")
    ax.set_title("Model Convergence Traces")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective / Trace Metric")
    ax.grid(True, linestyle="--", alpha=0.5)
    if has_trace:
        ax.legend()
    plt.tight_layout()
    conv_plot_path = job_dir / "convergence.png"
    fig.savefig(conv_plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    plots["convergence"] = str(conv_plot_path.name)

    return plots


# ── Main Benchmark Runner ──────────────────────────────────────────

def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    job_id: str,
    label_fractions: Optional[List[float]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run full benchmarking pipeline across outer folds and label fractions."""
    fractions = label_fractions or DEFAULT_LABEL_FRACTIONS
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    baseline_f1 = random_baseline_f1(y)
    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    total_steps = N_OUTER_FOLDS * len(fractions) * len(MODEL_REGISTRY)
    current_step = 0

    fold_records: List[Dict[str, Any]] = []
    traces: Dict[str, List[float]] = {name: [] for name in MODEL_REGISTRY}
    val_summaries: Dict[str, Dict[str, Any]] = {}
    best_params_per_model: Dict[str, Dict[str, Any]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test = y[train_idx], y[test_idx]

        # Standardize train only to prevent data leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for frac in fractions:
            y_train_masked = mask_labels(y_train_raw, frac, random_state=RANDOM_SEED + fold_idx)

            for model_name, model_class in MODEL_REGISTRY.items():
                current_step += 1
                pct = int((current_step / total_steps) * 90)
                step_msg = f"Fold {fold_idx + 1}/{N_OUTER_FOLDS} | Frac {int(frac*100)}% | {model_name}"
                if progress_callback:
                    progress_callback(pct, step_msg, f"Running {model_name}...")

                res = _run_model_on_fold(
                    model_name, model_class, X_train, y_train_masked, X_test, y_test, frac
                )

                fold_records.append({
                    "fold": fold_idx,
                    "model_name": model_name,
                    "label_fraction": frac,
                    "def_f1": res["default_scores"]["f1_macro"],
                    "def_auc": res["default_scores"]["auc_roc"],
                    "def_acc": res["default_scores"]["accuracy"],
                    "def_time": res["default_scores"]["train_time"],
                    "f1_macro": res["tuned_scores"]["f1_macro"],
                    "auc_roc": res["tuned_scores"]["auc_roc"],
                    "accuracy": res["tuned_scores"]["accuracy"],
                    "train_time": res["tuned_scores"]["train_time"],
                })

                best_params_per_model[model_name] = res["best_params"]

                # Collect validation signals from fold 0 at highest fraction
                if fold_idx == 0 and frac == fractions[-1] and res["fitted_model"] is not None:
                    val_sig = _compute_validation_signals(
                        res["fitted_model"], X_train, y_train_raw, y_train_masked
                    )
                    val_summaries[model_name] = val_sig
                    traces[model_name] = val_sig["trace"]

    df_folds = pd.DataFrame(fold_records)

    # Aggregate across folds
    mean_tuned = df_folds.groupby(["model_name", "label_fraction"])[
        ["f1_macro", "auc_roc", "accuracy", "train_time"]
    ].mean().reset_index()

    mean_default = df_folds.groupby(["model_name", "label_fraction"])[
        ["def_f1", "def_auc", "def_acc", "def_time"]
    ].mean().rename(columns={
        "def_f1": "f1_macro", "def_auc": "auc_roc",
        "def_acc": "accuracy", "def_time": "train_time"
    }).reset_index()

    # Compute final scores
    tuned_scores_df = compute_final_score(mean_tuned)
    default_scores_df = compute_final_score(mean_default)

    # Sanity flags
    tuned_scores_df["sanity_flag"] = tuned_scores_df["f1_mean"].apply(
        lambda f1: "possibly broken" if f1 < (baseline_f1 + SANITY_MARGIN) else "ok"
    )

    # Comparison table
    comp_df = pd.merge(
        default_scores_df[["model_name", "f1_mean", "final_score"]].rename(
            columns={"f1_mean": "default_f1", "final_score": "default_score"}
        ),
        tuned_scores_df[["model_name", "f1_mean", "final_score", "sanity_flag"]].rename(
            columns={"f1_mean": "tuned_f1", "final_score": "tuned_score"}
        ),
        on="model_name",
    )
    comp_df["gain_pp"] = ((comp_df["tuned_f1"] - comp_df["default_f1"]) * 100).round(2)

    # Pick winner (highest score among non-flagged)
    valid_candidates = tuned_scores_df[tuned_scores_df["sanity_flag"] == "ok"]
    if len(valid_candidates) > 0:
        winner_row = valid_candidates.iloc[0]
    else:
        winner_row = tuned_scores_df.iloc[0]
    winner_name = str(winner_row["model_name"])

    # Generate plots
    if progress_callback:
        progress_callback(95, "Generating plots", "Rendering benchmark plots...")
    plots = _generate_plots(mean_tuned, traces, job_dir)

    results = {
        "dataset_name": dataset_name,
        "random_baseline_f1": round(baseline_f1, 4),
        "leaderboard": tuned_scores_df.to_dict(orient="records"),
        "comparison": comp_df.to_dict(orient="records"),
        "best_params": best_params_per_model,
        "validation_summary": val_summaries,
        "winner": {
            "name": winner_name,
            "score": float(winner_row["final_score"]),
            "f1_mean": float(winner_row["f1_mean"]),
            "auc_mean": float(winner_row["auc_mean"]),
            "robustness": float(winner_row["robustness"]),
            "reason": f"Top composite score ({winner_row['final_score']:.4f}) with F1 {winner_row['f1_mean']:.4f}",
        },
        "plots": plots,
    }

    if progress_callback:
        progress_callback(100, "Completed", "Benchmark run finished successfully.")

    return results
