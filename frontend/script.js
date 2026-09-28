// ─────────────────────────────────────────────────────────────────────────
// AI-Powered Distillation Intelligence — Frontend logic
// Talks to the FastAPI backend exclusively via fetch()/JSON. No secrets,
// no API keys live here — GROQ_API_KEY stays server-side (see backend/.env).
// ─────────────────────────────────────────────────────────────────────────

// When the frontend is served BY the FastAPI app (recommended, single command
// to run everything) this can stay empty/relative. If you open index.html
// directly from disk or from a separate dev server (e.g. Live Server on
// :5500), point this at your backend explicitly.
const API_BASE = (location.port === "5500" || location.protocol === "file:")
  ? "http://127.0.0.1:8000"
  : "";

const INPUT_KEYS = ["pressure", "t1", "L", "V", "D", "B", "F"];

let inputRanges = null;
let lastPrediction = null;   // { prediction, percent, status, ... }
let lastInputs = null;       // raw input values used for lastPrediction
let chatHistory = [];
let metricsChart, flowChart, importanceChart;
let lastBatchResults = null;

// ── Utilities ──────────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

function showToast(message, kind = "danger") {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast show ${kind}`;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { toast.className = "toast"; }, 5000);
}

async function apiFetch(path, options = {}) {
  let resp;
  try {
    resp = await fetch(`${API_BASE}${path}`, {
      headers: options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" } : undefined,
      ...options,
    });
  } catch (err) {
    throw new Error(
      "Could not reach the backend. Is it running? " +
      "(uvicorn backend.main:app --reload) — " + err.message
    );
  }

  let body;
  const text = await resp.text();
  try { body = text ? JSON.parse(text) : {}; }
  catch { throw new Error("Backend returned malformed JSON."); }

  if (!resp.ok) {
    const detail = body && body.detail ? JSON.stringify(body.detail) : `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return body;
}

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
  const dot = $("statusDot");
  const text = $("statusText");
  try {
    const health = await apiFetch("/api/health");
    if (health.status === "ok") {
      dot.className = "status-dot online";
      text.textContent = "Model Online";
    } else {
      dot.className = "status-dot offline";
      text.textContent = `Model Error: ${health.error || "unknown"}`;
    }
  } catch (err) {
    dot.className = "status-dot offline";
    text.textContent = "Backend Unreachable";
  }
}

// ── Build input sliders from /api/input-ranges ──────────────────────────
async function buildInputs() {
  try {
    inputRanges = await apiFetch("/api/input-ranges");
  } catch (err) {
    showToast(err.message);
    return;
  }

  const grid = $("inputGrid");
  grid.innerHTML = "";

  INPUT_KEYS.forEach((key) => {
    const cfg = inputRanges[key];
    const field = document.createElement("div");
    field.className = "input-field" + (key === "F" ? " full" : "");
    field.innerHTML = `
      <label>${cfg.label} <span class="unit">${cfg.unit}</span></label>
      <input type="range" id="in_${key}" min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${cfg.default}">
      <div class="input-value-row"><span>${cfg.min}</span><span class="current-val" id="val_${key}">${cfg.default}</span><span>${cfg.max}</span></div>
    `;
    grid.appendChild(field);
  });

  INPUT_KEYS.forEach((key) => {
    const slider = $(`in_${key}`);
    slider.addEventListener("input", () => {
      $(`val_${key}`).textContent = slider.value;
      updateMassBalance();
    });
  });

  updateMassBalance();
}

function readCurrentInputs() {
  const vals = {};
  INPUT_KEYS.forEach((key) => { vals[key] = parseFloat($(`in_${key}`).value); });
  return vals;
}

function setInputs(vals) {
  INPUT_KEYS.forEach((key) => {
    if (vals[key] === undefined) return;
    $(`in_${key}`).value = vals[key];
    $(`val_${key}`).textContent = vals[key];
  });
  updateMassBalance();
}

function updateMassBalance() {
  const v = readCurrentInputs();
  const gap = Math.abs((v.D + v.B) - v.F);
  const box = $("massBalanceBox");
  const icon = $("mbIcon");
  const text = $("mbText");
  if (gap > 30) {
    box.className = "mass-balance warn";
    icon.textContent = "⚠";
    text.textContent = `Mass balance gap: |D+B−F| = ${gap.toFixed(0)} kmol/hr — check flow rates`;
  } else {
    box.className = "mass-balance";
    icon.textContent = "✓";
    text.textContent = `Mass balance OK — gap: ${gap.toFixed(0)} kmol/hr`;
  }
}

// ── Prediction ────────────────────────────────────────────────────────
async function runPrediction() {
  const runBtn = $("runBtn");
  runBtn.disabled = true;
  runBtn.textContent = "Running…";

  try {
    const inputs = readCurrentInputs();
    const result = await apiFetch("/api/predict", { method: "POST", body: JSON.stringify(inputs) });
    lastPrediction = result;
    lastInputs = inputs;
    renderPrediction(result);
    renderMetricsChart(result);
    renderFlowChart(inputs);
  } catch (err) {
    showToast(err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = '<span class="run-btn-icon">▶</span> Run Prediction';
  }
}

function renderPrediction(result) {
  const box = $("predBox");
  box.className = `pred-box ${result.status_class}`;
  $("predValue").textContent = result.prediction.toFixed(4);
  $("predPercent").textContent = `${result.percent.toFixed(2)} mol%`;

  const emoji = { good: "✅", warn: "⚠️", danger: "🚨" }[result.status_class] || "";
  $("predStatus").textContent = `${emoji} ${result.status}`;

  const pct = Math.min(result.percent, 100);
  $("purityScalePct").textContent = `${pct.toFixed(1)}%`;
  const fill = $("purityBarFill");
  fill.style.width = `${pct}%`;
  const barColor = { good: "#00D4AA", warn: "#F5A623", danger: "#E8445A" }[result.status_class];
  fill.style.background = barColor;

  const metrics = [
    ["Reflux Ratio", result.derived_metrics.reflux_ratio],
    ["Reboiler Intensity", result.derived_metrics.reboiler_intensity],
    ["Condenser Load", result.derived_metrics.condenser_load],
    ["Column Load", result.derived_metrics.column_load],
    ["Separation Duty", result.derived_metrics.separation_duty],
    ["Column Efficiency", result.derived_metrics.column_efficiency],
  ];
  $("metricGrid").innerHTML = metrics.map(([lbl, val]) => `
    <div class="metric-card"><div class="val">${val.toFixed(3)}</div><div class="lbl">${lbl}</div></div>
  `).join("");
}

// ── Charts ────────────────────────────────────────────────────────────
function chartDefaults() {
  Chart.defaults.color = "#7A8BA6";
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.borderColor = "#263049";
}

function renderMetricsChart(result) {
  const ctx = $("metricsChart");
  const labels = ["Reflux Ratio", "Reboiler Int.", "Condenser Load", "Column Load", "Sep. Duty", "Col. Eff."];
  const data = [
    result.derived_metrics.reflux_ratio, result.derived_metrics.reboiler_intensity,
    result.derived_metrics.condenser_load, result.derived_metrics.column_load,
    result.derived_metrics.separation_duty, result.derived_metrics.column_efficiency,
  ];
  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: "#3B8BEB", borderRadius: 4 }] },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, grid: { color: "#1C2436" } }, x: { grid: { display: false } } },
    },
  });
}

function renderFlowChart(inputs) {
  const ctx = $("flowChart");
  const labels = ["L (Reflux)", "V (Vapor)", "D (Distillate)", "B (Bottoms)", "F (Feed)"];
  const data = [inputs.L, inputs.V, inputs.D, inputs.B, inputs.F];
  if (flowChart) flowChart.destroy();
  flowChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: "#00D4AA", borderRadius: 4 }] },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, grid: { color: "#1C2436" } }, x: { grid: { display: false } } },
    },
  });
}

async function renderImportanceChart() {
  try {
    const data = await apiFetch("/api/feature-importance");
    const top = data.items.slice(0, 12);
    const ctx = $("importanceChart");
    if (importanceChart) importanceChart.destroy();
    importanceChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: top.map((i) => i.feature),
        datasets: [{ data: top.map((i) => i.importance), backgroundColor: "#00D4AA", borderRadius: 4 }],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true, grid: { color: "#1C2436" } }, y: { grid: { display: false } } },
      },
    });
  } catch (err) {
    showToast(err.message);
  }
}

// ── Model info / performance ─────────────────────────────────────────
async function renderModelInfo() {
  try {
    const info = await apiFetch("/api/model-info");
    const cards = [
      ["≥ " + info.r2_score.toFixed(2), "R² Score"],
      [info.rmse.toFixed(4), "RMSE"],
      [info.mae.toFixed(4), "MAE"],
      [info.accuracy_band, "Accuracy"],
    ];
    $("perfGrid").innerHTML = cards.map(([val, lbl]) => `
      <div class="perf-card"><div class="val">${val}</div><div class="lbl">${lbl}</div></div>
    `).join("");
  } catch (err) {
    showToast(err.message);
  }
}

// ── Load default example ─────────────────────────────────────────────
function loadDefaultExample() {
  if (!inputRanges) return;
  const defaults = {};
  INPUT_KEYS.forEach((key) => { defaults[key] = inputRanges[key].default; });
  setInputs(defaults);
  showToast("Default example loaded", "success");
}

// ── What-if analysis ──────────────────────────────────────────────────
async function runWhatIf() {
  const current = readCurrentInputs();
  const modified = { ...current, L: parseFloat($("whatifL").value) };

  try {
    const result = await apiFetch("/api/what-if", {
      method: "POST",
      body: JSON.stringify({ current, modified }),
    });
    const rows = [
      ["Reflux Flow (L)", current.L.toFixed(1), modified.L.toFixed(1)],
      ["Predicted Purity", result.current.prediction.toFixed(4), result.modified.prediction.toFixed(4)],
      ["Status", result.current.status, result.modified.status],
      ["Δ Purity", "—", (result.delta >= 0 ? "+" : "") + result.delta.toFixed(4)],
    ];
    $("whatifBody").innerHTML = rows.map(([p, c, m]) => `<tr><td>${p}</td><td>${c}</td><td>${m}</td></tr>`).join("");
  } catch (err) {
    showToast(err.message);
  }
}

// ── Batch prediction ──────────────────────────────────────────────────
async function runBatchPredict() {
  const fileInput = $("csvInput");
  if (!fileInput.files.length) {
    showToast("Choose a CSV file first.", "info");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const result = await apiFetch("/api/batch-predict", { method: "POST", body: formData });
    lastBatchResults = result.results;
    $("batchSummary").textContent =
      `${result.succeeded}/${result.total} rows scored successfully` +
      (result.failed ? ` — ${result.failed} row(s) failed validation` : "");

    $("batchBody").innerHTML = result.results.map((r) => {
      if (r.error) {
        return `<tr><td>${r.row}</td><td colspan="8" class="muted">Error: ${r.error}</td></tr>`;
      }
      const cls = r.status === "OPTIMAL" ? "good" : r.status === "ACCEPTABLE" ? "warn" : "danger";
      return `<tr>
        <td>${r.row}</td><td>${r.pressure}</td><td>${r.t1}</td><td>${r.L}</td>
        <td>${r.V}</td><td>${r.D}</td><td>${r.B}</td><td>${r.F}</td>
        <td>${r.prediction.toFixed(4)}</td>
        <td><span class="status-tag ${cls}">${r.status}</span></td>
      </tr>`;
    }).join("");

    $("downloadResultsBtn").disabled = false;
  } catch (err) {
    showToast(err.message);
  }
}

function downloadBatchResultsCsv() {
  if (!lastBatchResults) return;
  const header = ["row", "pressure", "t1", "L", "V", "D", "B", "F", "prediction", "percent", "status", "error"];
  const lines = [header.join(",")];
  lastBatchResults.forEach((r) => {
    lines.push(header.map((h) => (r[h] !== undefined && r[h] !== null ? r[h] : "")).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "batch_predict_results.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function downloadTemplateCsv() {
  try {
    const resp = await fetch(`${API_BASE}/api/batch-predict/template`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "batch_predict_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast("Could not download template: " + err.message);
  }
}

// ── Chat ──────────────────────────────────────────────────────────────
function appendChatMessage(text, who) {
  const win = $("chatWindow");
  const div = document.createElement("div");
  div.className = `chat-msg ${who}`;
  div.textContent = text;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

async function sendChatMessage() {
  const input = $("chatInput");
  const message = input.value.trim();
  if (!message) return;

  appendChatMessage(message, "user");
  chatHistory.push({ role: "user", content: message });
  input.value = "";
  $("chatSendBtn").disabled = true;

  try {
    const payload = {
      message,
      history: chatHistory.slice(-6),
      params: lastInputs || null,
      prediction: lastPrediction ? lastPrediction.prediction : null,
    };
    const result = await apiFetch("/api/chat", { method: "POST", body: JSON.stringify(payload) });
    appendChatMessage(result.reply, "bot");
    chatHistory.push({ role: "assistant", content: result.reply });
    $("chatSourceBadge").textContent = result.source;
  } catch (err) {
    appendChatMessage("Sorry, I couldn't reach the backend: " + err.message, "bot");
  } finally {
    $("chatSendBtn").disabled = false;
  }
}

// ── Wiring ────────────────────────────────────────────────────────────
async function init() {
  chartDefaults();
  await checkHealth();
  await buildInputs();
  await renderModelInfo();
  await renderImportanceChart();

  $("runBtn").addEventListener("click", runPrediction);
  $("loadDefaultBtn").addEventListener("click", loadDefaultExample);

  $("whatifL").addEventListener("input", () => { $("whatifLVal").textContent = $("whatifL").value; });
  $("whatifBtn").addEventListener("click", runWhatIf);

  $("downloadTemplateBtn").addEventListener("click", downloadTemplateCsv);
  $("downloadResultsBtn").addEventListener("click", downloadBatchResultsCsv);
  $("csvInput").addEventListener("change", runBatchPredict);

  $("chatSendBtn").addEventListener("click", sendChatMessage);
  $("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChatMessage(); });

  setInterval(checkHealth, 20000);
}

document.addEventListener("DOMContentLoaded", init);
