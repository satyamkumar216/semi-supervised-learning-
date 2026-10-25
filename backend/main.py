"""
main.py — FastAPI application routes for SSL-MedBench.

Routes only, no heavy ML logic. Coordinates background benchmarking jobs
and serves artifacts, plots, and downloadable files.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import LOG_FORMAT, LOG_LEVEL, OUTPUT_DIR
from backend.data_utils import (
    InvalidDatasetError,
    TargetColumnError,
    build_health_report,
    load_csv,
    mask_labels,
    preprocess,
)
from backend.exporter import export_enriched_dataset, export_winner_model
from backend.pipeline import run_benchmark
from backend.schemas import (
    BenchmarkResultsResponse,
    DataHealthResponse,
    JobStatusResponse,
    RunBenchmarkRequest,
    RunBenchmarkResponse,
    WinnerInfo,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Application Setup ──────────────────────────────────────────────

app = FastAPI(
    title="SSL-MedBench API",
    description="Benchmarking Suite for Semi-Supervised Learning in Medicine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job state store (Note: In production, back with Redis or DB)
JOBS: Dict[str, Dict[str, Any]] = {}
UPLOADED_FILES: Dict[str, Path] = {}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Background Worker ──────────────────────────────────────────────

def _benchmark_worker(job_id: str, file_path: Path, target_column: str, fractions: Optional[List[float]]) -> None:
    """Execute benchmarking pipeline and export artifacts in background."""
    job = JOBS[job_id]
    job["state"] = "RUNNING"

    def progress_update(pct: int, step: str, log_msg: str) -> None:
        job["progress_pct"] = pct
        job["current_step"] = step
        job["log_lines"].append(f"[{pct}%] {log_msg}")

    try:
        raw_df = load_csv(file_path)
        X, y, _ = preprocess(raw_df, target_column)
        results = run_benchmark(
            X=X,
            y=y,
            dataset_name=file_path.stem,
            job_id=job_id,
            label_fractions=fractions,
            progress_callback=progress_update,
        )

        # Export Mode A: Model pkl
        winner_name = results["winner"]["name"]
        best_params = results["best_params"].get(winner_name, {})
        y_masked = mask_labels(y, label_fraction=0.10)
        export_winner_model(winner_name, best_params, X, y_masked, job_id)

        # Export Mode B: Enriched dataset CSV
        export_enriched_dataset(
            winner_name, best_params, raw_df, target_column, X, y_masked, job_id
        )

        job["results"] = results
        job["state"] = "COMPLETED"
        job["progress_pct"] = 100
        job["current_step"] = "Finished"
        job["log_lines"].append("[100%] Pipeline finished successfully.")

    except Exception as exc:
        logger.exception("Benchmark job %s failed: %s", job_id, exc)
        job["state"] = "FAILED"
        job["error"] = str(exc)
        job["log_lines"].append(f"[FAILED] Error: {exc}")


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/")
def get_root() -> FileResponse:
    """Serve frontend index.html entry point."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css")
def get_css() -> FileResponse:
    """Serve frontend stylesheet."""
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/app.js")
def get_js() -> FileResponse:
    """Serve frontend client script."""
    return FileResponse(FRONTEND_DIR / "app.js")


@app.post("/upload", response_model=DataHealthResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DataHealthResponse:
    """Upload CSV dataset and return data health inspection metrics."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")

    file_id = str(uuid.uuid4())[:8]
    save_path = OUTPUT_DIR / f"upload_{file_id}_{file.filename}"
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    UPLOADED_FILES[file_id] = save_path
    try:
        df = load_csv(save_path)
        health = build_health_report(df)
        return DataHealthResponse(
            file_id=file_id,
            filename=file.filename,
            n_rows=health["n_rows"],
            n_columns=health["n_columns"],
            columns=health["columns"],
            missing_pct=health["missing_pct"],
            health_report=health,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/run", response_model=RunBenchmarkResponse)
def start_benchmark(req: RunBenchmarkRequest, bg_tasks: BackgroundTasks) -> RunBenchmarkResponse:
    """Initiate benchmarking run as an asynchronous background task."""
    if req.file_id not in UPLOADED_FILES:
        raise HTTPException(status_code=404, detail="Uploaded file not found.")

    file_path = UPLOADED_FILES[req.file_id]
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "state": "QUEUED",
        "progress_pct": 0,
        "current_step": "Initializing",
        "log_lines": ["Job queued."],
        "results": None,
        "error": None,
    }

    bg_tasks.add_task(
        _benchmark_worker, job_id, file_path, req.target_column, req.label_fractions
    )
    return RunBenchmarkResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll execution state, logs, and progress percentage of a job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    job = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        state=job["state"],
        progress_pct=job["progress_pct"],
        current_step=job["current_step"],
        log_lines=job["log_lines"][-25:],  # Return latest 25 log lines
        error=job["error"],
    )


@app.get("/results/{job_id}", response_model=BenchmarkResultsResponse)
def get_job_results(job_id: str) -> BenchmarkResultsResponse:
    """Retrieve complete leaderboard, comparison table, and validation metrics."""
    if job_id not in JOBS or JOBS[job_id]["state"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job is not completed yet.")
    res = JOBS[job_id]["results"]
    return BenchmarkResultsResponse(
        job_id=job_id,
        dataset_name=res["dataset_name"],
        random_baseline_f1=res["random_baseline_f1"],
        leaderboard=res["leaderboard"],
        comparison=res["comparison"],
        best_params=res["best_params"],
        validation_summary=res["validation_summary"],
        winner=WinnerInfo(**res["winner"]),
        plots=res["plots"],
    )


@app.get("/plots/{job_id}/{name}")
def get_job_plot(job_id: str, name: str) -> FileResponse:
    """Serve benchmark plot image files."""
    plot_file = OUTPUT_DIR / job_id / f"{name}.png"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(plot_file, media_type="image/png")


@app.get("/download/{job_id}/model")
def download_model(job_id: str) -> FileResponse:
    """Download retrained winning model pickle."""
    model_file = OUTPUT_DIR / job_id / "winner_model.pkl"
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(
        model_file,
        filename=f"winner_model_{job_id}.pkl",
        media_type="application/octet-stream",
    )


@app.get("/download/{job_id}/dataset")
def download_enriched_dataset(job_id: str) -> FileResponse:
    """Download enriched dataset CSV with pseudo-labels."""
    dataset_file = OUTPUT_DIR / job_id / "enriched_dataset.csv"
    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail="Enriched CSV not found.")
    return FileResponse(
        dataset_file,
        filename=f"enriched_dataset_{job_id}.csv",
        media_type="text/csv",
    )
