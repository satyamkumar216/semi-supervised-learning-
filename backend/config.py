"""
config.py — Central configuration for SSL-MedBench.

ALL constants live here: random seeds, label fractions, tuning budget,
scoring weights, file paths, and thresholds.  Nothing is hard-coded
elsewhere in the project.
"""

from pathlib import Path
from typing import Dict, List

# ── Paths ──────────────────────────────────────────────────────────
# Root of the outputs directory (plots, models, CSVs, reports)
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Random Seeds ───────────────────────────────────────────────────
RANDOM_SEED: int = 42          # master seed used for splits, masking, models
N_OUTER_FOLDS: int = 5         # number of outer stratified CV folds
TEST_SIZE: float = 0.20        # held-out test fraction (used only in export)

# ── Label Fractions ────────────────────────────────────────────────
# Fractions of training labels to keep; the rest become -1 (unlabeled)
DEFAULT_LABEL_FRACTIONS: List[float] = [0.05, 0.10, 0.20, 0.30]

# ── Tuning Budget ──────────────────────────────────────────────────
TUNING_N_ITER: int = 20        # RandomizedSearchCV iterations
TUNING_SCORING: str = "f1_macro"
TUNING_CV_FOLDS: int = 3       # cross-val folds inside the tuner
TUNING_CV_FOLDS_SMALL: int = 2 # fallback when labeled samples < MIN_LABELED_FOR_3CV
MIN_LABELED_FOR_3CV: int = 30  # threshold to switch from 3-fold to 2-fold

# ── Scorer Weights ─────────────────────────────────────────────────
# Score = W_F1*F1 + W_AUC*AUC + W_ROBUST*Robustness + W_EFF*Efficiency
WEIGHT_F1: float = 0.40
WEIGHT_AUC: float = 0.30
WEIGHT_ROBUSTNESS: float = 0.20
WEIGHT_EFFICIENCY: float = 0.10

# Robustness is computed between these two label fractions:
ROBUSTNESS_HIGH_FRAC: float = 0.30
ROBUSTNESS_LOW_FRAC: float = 0.05

# ── Sanity Check ───────────────────────────────────────────────────
# If a model's tuned F1 is below (random_baseline_f1 + this margin),
# it is flagged "possibly broken" and excluded from winner selection.
SANITY_MARGIN: float = 0.05

# ── Exporter ───────────────────────────────────────────────────────
# Minimum confidence to keep a pseudo-label in the enriched dataset.
PSEUDO_LABEL_CONFIDENCE_THRESHOLD: float = 0.80

# ── Self-Training Defaults ─────────────────────────────────────────
SELF_TRAINING_THRESHOLD: float = 0.90
SELF_TRAINING_MAX_ITER: int = 10
SELF_TRAINING_N_ESTIMATORS: int = 100

# ── Label Propagation Defaults ─────────────────────────────────────
LABEL_PROP_GAMMA: float = 0.5
LABEL_PROP_MAX_ITER: int = 1000
LABEL_PROP_TOL: float = 1e-5

# ── Co-Training Defaults ──────────────────────────────────────────
COTRAIN_P: int = 1             # positive pseudo-labels per round
COTRAIN_N: int = 1             # negative pseudo-labels per round
COTRAIN_U: int = 30            # candidate buffer size
COTRAIN_N_ITER: int = 15       # co-training rounds
COTRAIN_C: float = 1.0         # SVM regularisation

# ── S3VM / TSVM Defaults ──────────────────────────────────────────
TSVM_C: float = 1.0
TSVM_C_U: float = 1.0
TSVM_KERNEL: str = "rbf"
TSVM_GAMMA: str = "scale"
TSVM_N_STEPS: int = 10
TSVM_MAX_SWAPS: int = 50

# ── EM + GMM Defaults ─────────────────────────────────────────────
EM_GMM_MAX_ITER: int = 100
EM_GMM_TOL: float = 1e-4

# ── Plot settings ─────────────────────────────────────────────────
PLOT_DPI: int = 150
PLOT_FIGSIZE_WIDE: tuple = (10, 5)
PLOT_FIGSIZE_SQUARE: tuple = (6, 5)

# ── Logging ────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL: str = "INFO"
