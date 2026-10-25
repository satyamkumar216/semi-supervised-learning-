/**
 * app.js — Frontend client controller for SSL-MedBench.
 *
 * Coordinates data ingestion, background job polling, metrics rendering,
 * and artifact downloads.
 */

const API_BASE = window.location.origin;

let currentFileId = null;
let currentJobId = null;
let pollInterval = null;

// ── DOM References ─────────────────────────────────────────────────
const fileInput = document.getElementById("csv-file-input");
const uploadBtn = document.getElementById("upload-btn");
const targetContainer = document.getElementById("target-selector-container");
const targetSelect = document.getElementById("target-column-select");
const healthBox = document.getElementById("health-report-box");
const statRows = document.getElementById("stat-rows");
const statCols = document.getElementById("stat-cols");
const statMissing = document.getElementById("stat-missing");

const runBtn = document.getElementById("run-btn");
const progressContainer = document.getElementById("progress-container");
const progressBarFill = document.getElementById("progress-bar-fill");
const currentStepText = document.getElementById("current-step-text");
const progressPctText = document.getElementById("progress-pct-text");
const terminalLog = document.getElementById("terminal-log");

const sectionResults = document.getElementById("section-results");
const winnerBanner = document.getElementById("winner-banner");
const leaderboardBody = document.querySelector("#leaderboard-table tbody");
const comparisonBody = document.querySelector("#comparison-table tbody");
const validationBody = document.querySelector("#validation-table tbody");
const paramsAccordion = document.getElementById("params-accordion");
const f1PlotImg = document.getElementById("f1-plot-img");
const convPlotImg = document.getElementById("conv-plot-img");

const sectionDownload = document.getElementById("section-download");
const downloadModelBtn = document.getElementById("download-model-btn");
const downloadDatasetBtn = document.getElementById("download-dataset-btn");

const errorBanner = document.getElementById("error-banner");
const infoBanner = document.getElementById("info-banner");

// ── Alert Banner Helpers ───────────────────────────────────────────

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
  setTimeout(() => errorBanner.classList.add("hidden"), 8000);
}

function showInfo(msg) {
  infoBanner.textContent = msg;
  infoBanner.classList.remove("hidden");
  setTimeout(() => infoBanner.classList.add("hidden"), 4000);
}

// ── 1. File Upload & Inspection ───────────────────────────────────

async function uploadFile() {
  if (!fileInput.files.length) {
    showError("Please select a CSV file first.");
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }
    const data = await res.json();
    currentFileId = data.file_id;

    // Populate target select dropdown (default to last column)
    targetSelect.innerHTML = "";
    data.columns.forEach((col, idx) => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      if (idx === data.columns.length - 1) opt.selected = true;
      targetSelect.appendChild(opt);
    });

    statRows.textContent = data.n_rows.toLocaleString();
    statCols.textContent = data.n_columns.toLocaleString();
    statMissing.textContent = `${data.missing_pct}%`;

    targetContainer.classList.remove("hidden");
    healthBox.classList.remove("hidden");
    runBtn.disabled = false;
    showInfo(`Dataset "${data.filename}" uploaded successfully.`);
  } catch (err) {
    showError(err.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload CSV";
  }
}

// ── 2. Run Benchmark & Progress Polling ────────────────────────────

async function startRun() {
  if (!currentFileId) {
    showError("No dataset uploaded.");
    return;
  }
  const targetCol = targetSelect.value;
  runBtn.disabled = true;
  progressContainer.classList.remove("hidden");
  sectionResults.classList.add("hidden");
  sectionDownload.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_id: currentFileId,
        target_column: targetCol,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to start run");
    }
    const data = await res.json();
    currentJobId = data.job_id;
    pollStatus();
    pollInterval = setInterval(pollStatus, 2000);
  } catch (err) {
    showError(err.message);
    runBtn.disabled = false;
  }
}

async function pollStatus() {
  if (!currentJobId) return;

  try {
    const res = await fetch(`${API_BASE}/status/${currentJobId}`);
    if (!res.ok) return;
    const status = await res.json();

    progressBarFill.style.width = `${status.progress_pct}%`;
    progressPctText.textContent = `${status.progress_pct}%`;
    currentStepText.textContent = status.current_step;
    terminalLog.textContent = status.log_lines.join("\n");
    terminalLog.scrollTop = terminalLog.scrollHeight;

    if (status.state === "COMPLETED") {
      clearInterval(pollInterval);
      runBtn.disabled = false;
      showInfo("Benchmark completed!");
      loadResults();
    } else if (status.state === "FAILED") {
      clearInterval(pollInterval);
      runBtn.disabled = false;
      showError(`Job failed: ${status.error}`);
    }
  } catch (err) {
    console.error("Polling error:", err);
  }
}

// ── 3. Render Results ──────────────────────────────────────────────

async function loadResults() {
  try {
    const res = await fetch(`${API_BASE}/results/${currentJobId}`);
    if (!res.ok) throw new Error("Could not retrieve results");
    const results = await res.json();

    renderWinner(results.winner);
    renderLeaderboard(results.leaderboard);
    renderComparison(results.comparison);
    renderValidation(results.validation_summary);
    renderParams(results.best_params);
    renderPlots(results.job_id, results.plots);

    // Setup download links
    downloadModelBtn.href = `${API_BASE}/download/${currentJobId}/model`;
    downloadDatasetBtn.href = `${API_BASE}/download/${currentJobId}/dataset`;

    sectionResults.classList.remove("hidden");
    sectionDownload.classList.remove("hidden");
  } catch (err) {
    showError(err.message);
  }
}

function renderWinner(winner) {
  winnerBanner.innerHTML = `
    <h3>🏆 Winning Model: ${winner.name}</h3>
    <p><strong>Composite Score:</strong> ${winner.score} | <strong>Macro F1:</strong> ${winner.f1_mean} | <strong>AUC-ROC:</strong> ${winner.auc_mean}</p>
    <p><em>${winner.reason}</em></p>
  `;
}

function renderLeaderboard(leaderboard) {
  leaderboardBody.innerHTML = "";
  leaderboard.forEach((item, idx) => {
    const tr = document.createElement("tr");
    if (idx === 0) tr.classList.add("highlight-row");
    const isBroken = item.sanity_flag === "possibly broken";
    tr.innerHTML = `
      <td>${item.rank}</td>
      <td><strong>${item.model_name}</strong></td>
      <td>${item.f1_mean.toFixed(4)}</td>
      <td>${item.auc_mean.toFixed(4)}</td>
      <td>${item.robustness.toFixed(4)}</td>
      <td>${(item.efficiency_norm || 0).toFixed(4)}</td>
      <td><strong>${item.final_score.toFixed(4)}</strong></td>
      <td class="${isBroken ? 'flag-broken' : ''}">${item.sanity_flag}</td>
    `;
    leaderboardBody.appendChild(tr);
  });
}

function renderComparison(comparison) {
  comparisonBody.innerHTML = "";
  comparison.forEach(item => {
    const tr = document.createElement("tr");
    const gainClass = item.gain_pp >= 0 ? "positive-gain" : "negative-gain";
    const sign = item.gain_pp >= 0 ? "+" : "";
    tr.innerHTML = `
      <td><strong>${item.model_name}</strong></td>
      <td>${item.default_f1.toFixed(4)}</td>
      <td>${item.tuned_f1.toFixed(4)}</td>
      <td class="${gainClass}">${sign}${item.gain_pp.toFixed(2)}pp</td>
      <td>${item.default_score.toFixed(4)}</td>
      <td><strong>${item.tuned_score.toFixed(4)}</strong></td>
    `;
    comparisonBody.appendChild(tr);
  });
}

function renderValidation(valSummary) {
  validationBody.innerHTML = "";
  for (const [name, sig] of Object.entries(valSummary)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${name}</strong></td>
      <td>${sig.pseudo_label_acc}%</td>
      <td>${sig.label_stability}</td>
      <td>${sig.converged ? "✓ Yes" : "—"}</td>
    `;
    validationBody.appendChild(tr);
  }
}

function renderParams(params) {
  paramsAccordion.innerHTML = "";
  for (const [name, p] of Object.entries(params)) {
    const item = document.createElement("div");
    item.className = "accordion-item";
    item.innerHTML = `
      <div class="accordion-header">
        <span>${name}</span>
        <span>▼</span>
      </div>
      <div class="accordion-body">
        <pre>${JSON.stringify(p, null, 2)}</pre>
      </div>
    `;
    const header = item.querySelector(".accordion-header");
    const body = item.querySelector(".accordion-body");
    header.addEventListener("click", () => {
      body.classList.toggle("open");
    });
    paramsAccordion.appendChild(item);
  }
}

function renderPlots(jobId, plots) {
  if (plots.f1_vs_fraction) {
    f1PlotImg.src = `${API_BASE}/plots/${jobId}/${plots.f1_vs_fraction.replace('.png', '')}?t=${Date.now()}`;
  }
  if (plots.convergence) {
    convPlotImg.src = `${API_BASE}/plots/${jobId}/${plots.convergence.replace('.png', '')}?t=${Date.now()}`;
  }
}

// ── Event Listeners ────────────────────────────────────────────────
uploadBtn.addEventListener("click", uploadFile);
runBtn.addEventListener("click", startRun);
