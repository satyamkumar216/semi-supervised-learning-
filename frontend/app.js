/**
 * app.js — MediLens client controller.
 * Handles upload, job polling, results rendering and artifact downloads.
 */

// Opened via file:// or an editor preview server (Live Server etc.)? Talk to the FastAPI backend directly.
const DEV_PORTS = ["5500", "5501", "5173", "3000"];
const API_BASE =
  window.location.protocol.startsWith("http") && !DEV_PORTS.includes(window.location.port)
    ? window.location.origin
    : "http://localhost:8000";

function netMessage(err, fallback) {
  return err instanceof TypeError
    ? `Cannot reach the MediLens server at ${API_BASE}. Start it with: uvicorn backend.main:app --reload --port 8000`
    : err.message || fallback;
}
let currentFileId = null;
let currentJobId = null;
let pollInterval = null;

const $ = (id) => document.getElementById(id);
const fileInput = $("csv-file-input");
const uploadBtn = $("upload-btn");
const dropzone = $("dropzone");
const targetContainer = $("target-selector-container");
const targetSelect = $("target-column-select");
const healthBox = $("health-report-box");
const statRows = $("stat-rows");
const statCols = $("stat-cols");
const statMissing = $("stat-missing");
const runBtn = $("run-btn");
const progressContainer = $("progress-container");
const progressBarFill = $("progress-bar-fill");
const currentStepText = $("current-step-text");
const progressPctText = $("progress-pct-text");
const terminalLog = $("terminal-log");
const sectionResults = $("section-results");
const winnerBanner = $("winner-banner");
const leaderboardBody = document.querySelector("#leaderboard-table tbody");
const comparisonBody = document.querySelector("#comparison-table tbody");
const validationBody = document.querySelector("#validation-table tbody");
const paramsAccordion = $("params-accordion");
const f1PlotImg = $("f1-plot-img");
const convPlotImg = $("conv-plot-img");
const sectionDownload = $("section-download");
const downloadModelBtn = $("download-model-btn");
const downloadDatasetBtn = $("download-dataset-btn");
const errorBanner = $("error-banner");
const infoBanner = $("info-banner");

// ── Helpers ────────────────────────────────────────────────────────

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function toast(el, msg, ms) {
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), ms);
}
const showError = (msg) => toast(errorBanner, msg, 8000);
const showInfo = (msg) => toast(infoBanner, msg, 4000);

function setStep(n) {
  document.querySelectorAll("#stepper li").forEach((li) => {
    const s = +li.dataset.step;
    li.classList.toggle("active", s === n);
    li.classList.toggle("done", s < n);
  });
}

function countUp(el, target, suffix = "") {
  const end = parseFloat(String(target).replace(/,/g, ""));
  if (isNaN(end)) { el.textContent = target + suffix; return; }
  const dec = String(target).includes(".") ? 1 : 0;
  const t0 = performance.now();
  const tick = (t) => {
    const p = Math.min((t - t0) / 900, 1);
    const v = end * (1 - Math.pow(1 - p, 3));
    el.textContent = v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function readJsonResponse(response, fallbackMessage) {
  const body = await response.text();
  let data;
  if (body.trim()) {
    try {
      data = JSON.parse(body);
    } catch {
      if (response.ok) throw new Error(`Server returned an invalid or incomplete response (HTTP ${response.status}).`);
    }
  }
  if (!response.ok) {
    const detail = data && typeof data === "object" ? data.detail : null;
    throw new Error(detail || `${fallbackMessage} (HTTP ${response.status}).`);
  }
  if (data === undefined || data === null) throw new Error(`Server returned an empty response (HTTP ${response.status}).`);
  return data;
}

// ── Dropzone ───────────────────────────────────────────────────────

function showPicked() {
  if (!fileInput.files.length) return;
  const f = fileInput.files[0];
  $("drop-title").textContent = f.name;
  $("drop-sub").textContent = `${(f.size / 1024).toFixed(1)} KB · now click Upload CSV`;
}
fileInput.addEventListener("change", showPicked);
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showPicked();
  }
});

// ── 1. Upload ──────────────────────────────────────────────────────

async function uploadFile() {
  if (!fileInput.files.length) return showError("Please select a CSV file first.");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  try {
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
    const data = await readJsonResponse(res, "Upload failed");
    currentFileId = data.file_id;

    targetSelect.innerHTML = "";
    data.columns.forEach((col, idx) => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      if (idx === data.columns.length - 1) opt.selected = true;
      targetSelect.appendChild(opt);
    });

    targetContainer.classList.remove("hidden");
    healthBox.classList.remove("hidden");
    countUp(statRows, data.n_rows);
    countUp(statCols, data.n_columns);
    countUp(statMissing, data.missing_pct, "%");
    runBtn.disabled = false;
    setStep(2);
    showInfo(`Dataset "${data.filename}" uploaded successfully.`);
  } catch (err) {
    console.error("Upload error:", err);
    showError(netMessage(err, "Upload failed"));
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload CSV";
  }
}

// ── 2. Run & poll ──────────────────────────────────────────────────

async function startRun() {
  if (!currentFileId) return showError("No dataset uploaded.");
  runBtn.disabled = true;
  progressContainer.classList.remove("hidden");
  sectionResults.classList.add("hidden");
  sectionDownload.classList.add("hidden");
  setStep(2);
  try {
    const res = await fetch(`${API_BASE}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: currentFileId, target_column: targetSelect.value }),
    });
    const data = await readJsonResponse(res, "Failed to start run");
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
    const status = await readJsonResponse(res, "Could not retrieve benchmark status");
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
    clearInterval(pollInterval);
    pollInterval = null;
    runBtn.disabled = false;
    showError(`Could not retrieve benchmark status: ${err.message}`);
  }
}

// ── 3. Results ─────────────────────────────────────────────────────

async function loadResults() {
  try {
    const res = await fetch(`${API_BASE}/results/${currentJobId}`);
    const results = await readJsonResponse(res, "Could not retrieve results");
    renderWinner(results.winner);
    renderLeaderboard(results.leaderboard);
    renderComparison(results.comparison);
    renderValidation(results.validation_summary);
    renderParams(results.best_params);
    renderPlots(results.job_id, results.plots);

    downloadModelBtn.href = `${API_BASE}/download/${currentJobId}/model`;
    downloadDatasetBtn.href = `${API_BASE}/download/${currentJobId}/dataset`;
    sectionResults.classList.remove("hidden");
    sectionDownload.classList.remove("hidden");
    setStep(4);
    requestAnimationFrame(() => document.querySelectorAll(".scorebar b").forEach((b) => (b.style.width = b.dataset.w)));
    sectionResults.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message);
  }
}

function renderWinner(w) {
  winnerBanner.innerHTML = `
    <span class="tag">Recommended model</span>
    <h3>${esc(w.name)}</h3>
    <div class="kpis"><span>Composite ${esc(w.score)}</span><span>Macro F1 ${esc(w.f1_mean)}</span><span>AUC-ROC ${esc(w.auc_mean)}</span></div>
    <p>${esc(w.reason)}</p>`;
}

function renderLeaderboard(list) {
  leaderboardBody.innerHTML = "";
  list.forEach((it, idx) => {
    const tr = document.createElement("tr");
    if (idx === 0) tr.classList.add("highlight-row");
    const broken = it.sanity_flag === "possibly broken";
    const w = Math.max(0, Math.min(100, it.final_score * 100)).toFixed(1) + "%";
    tr.innerHTML = `
      <td>${esc(it.rank)}</td>
      <td><strong>${esc(it.model_name)}</strong></td>
      <td>${it.f1_mean.toFixed(4)}</td>
      <td>${it.auc_mean.toFixed(4)}</td>
      <td>${it.robustness.toFixed(4)}</td>
      <td>${(it.efficiency_norm || 0).toFixed(4)}</td>
      <td><div class="scorebar"><strong>${it.final_score.toFixed(4)}</strong><i><b data-w="${w}"></b></i></div></td>
      <td class="${broken ? "flag-broken" : ""}">${esc(it.sanity_flag)}</td>`;
    leaderboardBody.appendChild(tr);
  });
}

function renderComparison(list) {
  comparisonBody.innerHTML = "";
  list.forEach((it) => {
    const tr = document.createElement("tr");
    const cls = it.gain_pp >= 0 ? "positive-gain" : "negative-gain";
    const sign = it.gain_pp >= 0 ? "+" : "";
    tr.innerHTML = `
      <td><strong>${esc(it.model_name)}</strong></td>
      <td>${it.default_f1.toFixed(4)}</td>
      <td>${it.tuned_f1.toFixed(4)}</td>
      <td class="${cls}">${sign}${it.gain_pp.toFixed(2)}pp</td>
      <td>${it.default_score.toFixed(4)}</td>
      <td><strong>${it.tuned_score.toFixed(4)}</strong></td>`;
    comparisonBody.appendChild(tr);
  });
}

function renderValidation(val) {
  validationBody.innerHTML = "";
  for (const [name, sig] of Object.entries(val)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${esc(name)}</strong></td>
      <td>${esc(sig.pseudo_label_acc)}%</td>
      <td>${esc(sig.label_stability)}</td>
      <td>${sig.converged ? "&#10003; Yes" : "&mdash;"}</td>`;
    validationBody.appendChild(tr);
  }
}

function renderParams(params) {
  paramsAccordion.innerHTML = "";
  for (const [name, p] of Object.entries(params)) {
    const item = document.createElement("div");
    item.className = "accordion-item";
    item.innerHTML = `
      <div class="accordion-header" tabindex="0"><span>${esc(name)}</span><span>&#9662;</span></div>
      <div class="accordion-body"><pre>${esc(JSON.stringify(p, null, 2))}</pre></div>`;
    const header = item.querySelector(".accordion-header");
    const body = item.querySelector(".accordion-body");
    const toggle = () => body.classList.toggle("open");
    header.addEventListener("click", toggle);
    header.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    paramsAccordion.appendChild(item);
  }
}

function renderPlots(jobId, plots) {
  if (plots.f1_vs_fraction) f1PlotImg.src = `${API_BASE}/plots/${jobId}/${plots.f1_vs_fraction.replace(".png", "")}?t=${Date.now()}`;
  if (plots.convergence) convPlotImg.src = `${API_BASE}/plots/${jobId}/${plots.convergence.replace(".png", "")}?t=${Date.now()}`;
}

// ── Init ───────────────────────────────────────────────────────────

uploadBtn.addEventListener("click", uploadFile);
runBtn.addEventListener("click", startRun);

const io = new IntersectionObserver((entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add("in")), { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
