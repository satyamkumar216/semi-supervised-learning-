"""
schemas.py — Pydantic request and response models for SSL-MedBench API.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
# ── Upload Models ──────────────────────────────────────────────────

class DataHealthResponse(BaseModel):
    """Response payload for uploaded dataset health inspection."""
    file_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: List[str]
    missing_pct: float
    health_report: Dict[str, Any]


# ── Run Benchmark Models ───────────────────────────────────────────

class RunBenchmarkRequest(BaseModel):
    """Request payload to initiate benchmarking pipeline."""
    file_id: str
    target_column: str
    label_fractions: Optional[List[float]] = Field(
        default=None,
        description="List of fractions e.g. [0.05, 0.10, 0.20, 0.30]"
    )


class RunBenchmarkResponse(BaseModel):
    """Response returned upon successfully queuing benchmark run."""
    job_id: str
    status: str = "QUEUED"
    message: str = "Benchmark run started in background."


# ── Status Models ──────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    """Response returning current execution state of benchmark job."""
    job_id: str
    state: str  # QUEUED, RUNNING, COMPLETED, FAILED
    progress_pct: int
    current_step: str
    log_lines: List[str]
    error: Optional[str] = None


# ── Results Models ─────────────────────────────────────────────────

class WinnerInfo(BaseModel):
    """Summary of the winning SSL model."""
    name: str
    score: float
    f1_mean: float
    auc_mean: float
    robustness: float
    reason: str


class BenchmarkResultsResponse(BaseModel):
    """Complete benchmark results payload."""
    job_id: str
    dataset_name: str
    random_baseline_f1: float
    leaderboard: List[Dict[str, Any]]
    comparison: List[Dict[str, Any]]
    best_params: Dict[str, Any]
    validation_summary: Dict[str, Any]
    winner: WinnerInfo
    plots: Dict[str, str]
