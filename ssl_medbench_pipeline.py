"""
ssl_medbench_pipeline.py — Standalone executable master pipeline runner for SSL-MedBench.

Can be run directly from terminal:
    python ssl_medbench_pipeline.py
or
    python ssl_medbench_pipeline.py --csv path/to/dataset.csv --target target_col
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from backend.data_utils import build_health_report, class_distribution, mask_labels, preprocess
from backend.exporter import export_enriched_dataset, export_winner_model
from backend.pipeline import run_benchmark


def format_report(
    results: dict,
    dataset_name: str,
    n_samples: int,
    n_features: int,
    class_dist: dict,
    exported_output: str,
) -> str:
    """Format benchmark results into the standardized SSL-MedBench text report."""
    lines = [
        "═══════════════════════════════════════════════",
        "SSL-MedBench Report",
        "═══════════════════════════════════════════════",
        f"Dataset     : {dataset_name}",
        f"Samples     : {n_samples} | Features: {n_features}",
        f"Class Balance: {class_dist}",
        "───────────────────────────────────────────────",
        "LEADERBOARD (Tuned Models)",
        f"{'Rank':<5}| {'Model':<20}| {'F1':<6}| {'AUC':<6}| {'Robust':<7}| Score",
    ]

    for item in results["leaderboard"]:
        lines.append(
            f"  {item['rank']:<3}| {item['model_name']:<20}| "
            f"{item['f1_mean']:<6.2f}| {item['auc_mean']:<6.2f}| "
            f"{item['robustness']:<7.2f}| {item['final_score']:<5.2f}"
        )

    lines.extend([
        "───────────────────────────────────────────────",
        "DEFAULT vs TUNED COMPARISON",
        f"{'Model':<20}| {'Default F1':<11}| {'Tuned F1':<9}| Gain",
    ])

    for comp in results["comparison"]:
        sign = "+" if comp["gain_pp"] >= 0 else ""
        gain_str = f"{sign}{int(round(comp['gain_pp']))}pp"
        lines.append(
            f"{comp['model_name']:<20}|   {comp['default_f1']:<9.2f}|  {comp['tuned_f1']:<7.2f}| {gain_str}"
        )

    winner = results["winner"]
    lines.extend([
        "───────────────────────────────────────────────",
        f"WINNER: {winner['name']}",
        f"Reason: {winner['reason']}",
        "───────────────────────────────────────────────",
        "VALIDATION SUMMARY",
        f"{'Model':<20}| {'Pseudo-Label Acc':<17}| {'Label Stability':<16}| Converged",
    ])

    for model_name, val in results["validation_summary"].items():
        acc_str = f"{val['pseudo_label_acc']}%"
        conv_str = "Yes" if val["converged"] else "No"
        lines.append(
            f"{model_name:<20}|     {acc_str:<13}|    {val['label_stability']:<12}|   {conv_str}"
        )

    lines.extend([
        "───────────────────────────────────────────────",
        "OUTPUT",
        f"→ {exported_output}",
        "═══════════════════════════════════════════════",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SSL-MedBench Master Pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to input medical CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Name of target label column")
    parser.add_argument("--export-mode", type=str, choices=["model", "dataset"], default="model",
                        help="Export Mode A ('model') or Mode B ('dataset')")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        df = pd.read_csv(csv_path)
        target_col = args.target or df.columns[-1]
        dataset_name = csv_path.name
        X, y, _ = preprocess(df, target_col)
    else:
        dataset_name = "Breast Cancer Wisconsin (UCI)"
        b_data = load_breast_cancer()
        X = b_data.data
        y = b_data.target
        df = pd.DataFrame(X, columns=b_data.feature_names)
        df["target"] = y
        target_col = "target"

    health = build_health_report(df)
    class_dist = class_distribution(y)
    job_id = "cli_run"

    print("\nStarting SSL-MedBench Benchmark...")
    print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")

    results = run_benchmark(
        X=X,
        y=y,
        dataset_name=dataset_name,
        job_id=job_id,
        progress_callback=lambda pct, step, msg: print(f"[{pct:>2}%] {step} - {msg}"),
    )

    winner_name = results["winner"]["name"]
    best_params = results["best_params"].get(winner_name, {})
    y_masked = mask_labels(y, label_fraction=0.10)

    if args.export_mode == "model":
        out_path = export_winner_model(winner_name, best_params, X, y_masked, job_id)
        exported_output = out_path.name
    else:
        out_path, _ = export_enriched_dataset(winner_name, best_params, df, target_col, X, y_masked, job_id)
        exported_output = out_path.name

    report_text = format_report(
        results=results,
        dataset_name=dataset_name,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        class_dist=class_dist,
        exported_output=exported_output,
    )

    print("\n" + report_text)

    # Save to ssl_medbench_report.txt
    report_file = Path("ssl_medbench_report.txt")
    report_file.write_text(report_text)
    print(f"\nReport written to {report_file.resolve()}")


if __name__ == "__main__":
    main()
