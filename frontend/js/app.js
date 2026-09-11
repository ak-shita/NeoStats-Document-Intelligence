const TYPE_LABELS = {
  invoice: "Invoice",
  balance_sheet: "Balance Sheet",
  profit_and_loss: "Profit & Loss",
  cash_flow: "Cash Flow",
};

const VIEWS = {
  dashboard: { title: "Dashboard", sub: "Processed documents from the persistent store" },
  documents: { title: "Documents", sub: "Every result saved in MySQL" },
  upload: { title: "Upload / Process", sub: "Send a file to the existing API pipeline" },
  history: { title: "History", sub: "Most recently processed first" },
  settings: { title: "Settings", sub: "Connection and appearance" },
  result: { title: "Document result", sub: "Extracted fields, validation, and source evidence" },
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function typeLabel(type) {
  return TYPE_LABELS[type] || type || "—";
}

function formatTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function formatConfidence(value) {
  if (value === null || value === undefined || value === "") return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num <= 1 ? `${Math.round(num * 100)}%` : String(num);
}

function badge(status) {
  const key = String(status || "").toLowerCase();
  const cls = key === "pass" ? "badge-pass" : key === "fail" || key === "failed" ? "badge-fail" : "badge-na";
  return `<span class="badge ${cls}">${escapeHtml(status || "—")}</span>`;
}

function isFieldNode(node) {
  return Boolean(
    node &&
      typeof node === "object" &&
      !Array.isArray(node) &&
      "value" in node &&
      ("confidence" in node || "page_number" in node || "field" in node || "evidence" in node)
  );
}

function fieldMeta(node) {
  const evidence = node.evidence && typeof node.evidence === "object" ? node.evidence : {};
  const page = node.page_number ?? evidence.page_number;
  const source = evidence.source_text;
  const confidence = node.confidence;
  const empty = node.value === null || node.value === undefined || node.value === "";
  const low = typeof confidence === "number" && confidence < 0.7;
  return { page, source, confidence, empty, low };
}

function renderValue(node) {
  if (isFieldNode(node)) {
    const meta = fieldMeta(node);
    const shown = meta.empty
      ? `<span class="missing">Missing / unreadable</span>`
      : meta.low
        ? `<span class="low-conf">${escapeHtml(node.value)} · low confidence</span>`
        : escapeHtml(node.value);
    const bits = [];
    if (meta.confidence !== null && meta.confidence !== undefined && meta.confidence !== "") {
      bits.push(`confidence ${escapeHtml(formatConfidence(meta.confidence))}`);
    }
    if (meta.page !== null && meta.page !== undefined && meta.page !== "") {
      bits.push(`page ${escapeHtml(meta.page)}`);
    }
    const source = meta.source
      ? `<div class="meta-line">Evidence: “${escapeHtml(meta.source)}”</div>`
      : "";
    return `${shown}${bits.length ? `<div class="meta-line">${bits.join(" · ")}</div>` : ""}${source}`;
  }
  if (node === null || node === undefined || node === "") {
    return `<span class="missing">Missing / unreadable</span>`;
  }
  if (typeof node !== "object") return escapeHtml(node);
  return `<code>${escapeHtml(JSON.stringify(node))}</code>`;
}

function humanize(key) {
  return String(key).replaceAll("_", " ");
}

function renderExtracted(data) {
  if (!data || typeof data !== "object") {
    return `<p class="muted">No extracted fields.</p>`;
  }
  const parts = [];

  function walk(node, title) {
    if (Array.isArray(node)) {
      parts.push(renderTable(title, node));
      return;
    }
    if (!node || typeof node !== "object") return;
    const simple = [];
    const nested = [];
    for (const [key, value] of Object.entries(node)) {
      if (key === "document_type") continue;
      if (Array.isArray(value)) nested.push([key, value]);
      else if (isFieldNode(value) || value === null || typeof value !== "object") simple.push([key, value]);
      else nested.push([key, value]);
    }
    if (simple.length) {
      parts.push(`<article class="panel"><h3 class="section-title">${escapeHtml(title)}</h3><div class="kv">`);
      for (const [key, value] of simple) {
        parts.push(
          `<div class="kv-row"><div class="kv-key">${escapeHtml(humanize(key))}</div><div class="kv-val">${renderValue(value)}</div></div>`
        );
      }
      parts.push(`</div></article>`);
    }
    for (const [key, value] of nested) walk(value, humanize(key));
  }

  walk(data, "Extracted fields");
  return parts.join("") || `<p class="muted">No extracted fields.</p>`;
}

function renderTable(title, rows) {
  if (!rows.length) return "";
  if (rows.every((row) => typeof row !== "object" || row === null)) {
    return `<article class="panel"><h3 class="section-title">${escapeHtml(title)}</h3><p>${rows.map(renderValue).join(", ")}</p></article>`;
  }
  const keys = [];
  for (const row of rows) {
    if (row && typeof row === "object") {
      for (const key of Object.keys(row)) {
        if (!keys.includes(key)) keys.push(key);
      }
    }
  }
  const head = keys.map((key) => `<th>${escapeHtml(humanize(key))}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = keys.map((key) => `<td>${renderValue(row ? row[key] : null)}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return `<article class="panel"><h3 class="section-title">${escapeHtml(title)}</h3><div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></article>`;
}

function renderValidation(validation) {
  const checks = (validation && (validation.checks || validation.check)) || [];
  if (!checks.length) {
    return `<article class="panel"><h3 class="section-title">Financial validation</h3><p class="muted">No checks recorded.</p></article>`;
  }
  const rows = checks
    .map((check) => {
      const fail = String(check.status).toUpperCase() === "FAIL";
      return `<tr class="${fail ? "fail-row" : ""}">
        <td>${escapeHtml(humanize(check.name || ""))}</td>
        <td>${escapeHtml(check.formula || "—")}</td>
        <td>${check.calculated_value ?? "—"}</td>
        <td>${check.reported_value ?? escapeHtml(check.reported_raw ?? "—")}</td>
        <td>${check.variance ?? "—"}</td>
        <td>${badge(check.status)}</td>
      </tr>`;
    })
    .join("");
  return `<article class="panel">
    <div class="panel-head">
      <h3 class="section-title">Financial validation</h3>
      <div>${badge(validation.overall_status)}</div>
    </div>
    <div class="table-wrap"><table class="data">
      <thead><tr><th>Check</th><th>Formula</th><th>Calculated</th><th>Reported</th><th>Variance</th><th>Status</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </article>`;
}

function emptyState(message) {
  return `<div class="empty"><div class="empty-mark">N</div><h3>No documents yet</h3><p>${escapeHtml(message)}</p><p><a class="text-link" href="#upload">Process a document</a></p></div>`;
}

function docsTable(docs, emptyMessage) {
  if (!docs.length) return emptyState(emptyMessage);
  const rows = docs
    .map((doc) => {
      const processed = doc.processing_metadata && doc.processing_metadata.processed_at;
      const href = `#result/${encodeURIComponent(doc.document_name)}`;
      return `<tr>
        <td><a class="text-link" href="${href}">${escapeHtml(doc.document_name)}</a></td>
        <td>${escapeHtml(typeLabel(doc.document_type))}</td>
        <td>${badge(doc.processing_status)}</td>
        <td>${badge(doc.validation && doc.validation.overall_status)}</td>
        <td>${escapeHtml(formatConfidence(doc.overall_confidence))}</td>
        <td>${escapeHtml(formatTime(processed))}</td>
      </tr>`;
    })
    .join("");
  return `<div class="table-wrap"><table class="data">
    <thead><tr><th>Document</th><th>Type</th><th>Processing</th><th>Validation</th><th>Confidence</th><th>Processed</th></tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`;
}

function meter(pass, fail, total) {
  if (!total) return "";
  const pw = Math.round((pass / total) * 100);
  const fw = Math.round((fail / total) * 100);
  return `<div class="meter" aria-hidden="true"><span class="pass" style="width:${pw}%"></span><span class="fail" style="width:${fw}%"></span></div>`;
}

function summaryCards(docs) {
  const total = docs.length;
  const passed = docs.filter((d) => d.processing_status === "PASS").length;
  const failed = docs.filter((d) => d.processing_status === "FAILED").length;
  const valFail = docs.filter((d) => d.validation && d.validation.overall_status === "FAIL").length;
  const valPass = docs.filter((d) => d.validation && d.validation.overall_status === "PASS").length;
  const confs = docs.map((d) => d.overall_confidence).filter((v) => typeof v === "number");
  const avg = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null;
  const items = [
    ["Total documents", total, "From GET /api/v1/documents", ""],
    ["Processing PASS / FAILED", `${passed} / ${failed}`, "processing_status", meter(passed, failed, total)],
    ["Validation PASS / FAIL", `${valPass} / ${valFail}`, "validation.overall_status", meter(valPass, valFail, total)],
    ["Average confidence", avg === null ? "—" : formatConfidence(avg), "overall_confidence", ""],
  ];
  return items
    .map(
      ([label, value, hint, extra]) =>
        `<article class="card"><p class="label">${label}</p><p class="value">${escapeHtml(value)}</p><p class="hint">${hint}</p>${extra}</article>`
    )
    .join("");
}

function typeMix(docs) {
  if (!docs.length) {
    return `<p class="muted">No type mix until documents are persisted.</p>`;
  }
  const counts = {};
  for (const doc of docs) {
    const key = doc.document_type || "unknown";
    counts[key] = (counts[key] || 0) + 1;
  }
  const max = Math.max(...Object.values(counts));
  return Object.entries(counts)
    .map(([type, count], idx) => {
      const width = Math.max(8, Math.round((count / max) * 100));
      const cls = idx % 2 ? "sage" : "";
      return `<div class="mix-row"><span>${escapeHtml(typeLabel(type))}</span><div class="mix-bar"><i class="${cls}" style="width:${width}%"></i></div><span>${count}</span></div>`;
    })
    .join("");
}

async function loadDocuments() {
  const payload = await window.NeoStatsAPI.listDocuments();
  return payload.documents || [];
}

function showView(name) {
  document.querySelectorAll(".view").forEach((el) => el.classList.add("is-hidden"));
  const view = $(`view-${name}`) || $("view-dashboard");
  view.classList.remove("is-hidden");
  const meta = VIEWS[name] || VIEWS.dashboard;
  $("page-title").textContent = meta.title;
  $("page-sub").textContent = meta.sub;
  document.querySelectorAll("[data-nav]").forEach((el) => {
    el.classList.toggle("is-active", el.dataset.nav === name);
  });
  $("sidebar").classList.remove("is-open");
  $("sidebar-backdrop").hidden = true;
}

async function renderDashboard() {
  showView("dashboard");
  try {
    const docs = await loadDocuments();
    $("summary-cards").innerHTML = summaryCards(docs);
    $("type-mix").innerHTML = typeMix(docs);
    $("dashboard-table").innerHTML = docsTable(docs.slice(-8).reverse(), "Process a document to populate this dashboard from MySQL.");
  } catch (err) {
    $("summary-cards").innerHTML = "";
    $("type-mix").innerHTML = `<p class="status-box is-err">${escapeHtml(err.message)}</p>`;
    $("dashboard-table").innerHTML = `<p class="status-box is-err">${escapeHtml(err.message)}</p>`;
  }
}

async function renderDocuments() {
  showView("documents");
  try {
    const docs = await loadDocuments();
    $("documents-table").innerHTML = docsTable(docs, "No persisted documents yet.");
  } catch (err) {
    $("documents-table").innerHTML = `<p class="status-box is-err">${escapeHtml(err.message)}</p>`;
  }
}

async function renderHistory() {
  showView("history");
  try {
    const docs = await loadDocuments();
    const sorted = [...docs].sort((a, b) => {
      const ta = Date.parse((a.processing_metadata || {}).processed_at || 0) || 0;
      const tb = Date.parse((b.processing_metadata || {}).processed_at || 0) || 0;
      return tb - ta;
    });
    if (!sorted.length) {
      $("history-list").innerHTML = emptyState("History appears after the API persists a result.");
      return;
    }
    $("history-list").innerHTML = sorted
      .map((doc) => {
        const processed = doc.processing_metadata && doc.processing_metadata.processed_at;
        return `<div class="history-item">
          <div>${escapeHtml(formatTime(processed))}</div>
          <div><a class="text-link" href="#result/${encodeURIComponent(doc.document_name)}">${escapeHtml(doc.document_name)}</a>
            <div class="meta-line">${escapeHtml(typeLabel(doc.document_type))}</div></div>
          <div>${badge(doc.processing_status)} ${badge(doc.validation && doc.validation.overall_status)}</div>
        </div>`;
      })
      .join("");
  } catch (err) {
    $("history-list").innerHTML = `<p class="status-box is-err">${escapeHtml(err.message)}</p>`;
  }
}

async function renderResult(name) {
  showView("result");
  $("page-title").textContent = name;
  $("result-root").innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const doc = await window.NeoStatsAPI.getDocument(name);
    const processed = doc.processing_metadata && doc.processing_metadata.processed_at;
    $("result-root").innerHTML = `
      <div class="cards">
        <article class="card"><p class="label">Type</p><p class="value" style="font-size:20px">${escapeHtml(typeLabel(doc.document_type))}</p></article>
        <article class="card"><p class="label">Processing</p><p class="value">${badge(doc.processing_status)}</p></article>
        <article class="card"><p class="label">Validation</p><p class="value">${badge(doc.validation && doc.validation.overall_status)}</p></article>
        <article class="card"><p class="label">Confidence</p><p class="value" style="font-size:20px">${escapeHtml(formatConfidence(doc.overall_confidence))}</p><p class="hint">${escapeHtml(formatTime(processed))}</p></article>
      </div>
      <div class="result-grid">
        ${renderExtracted(doc.extracted_data)}
        ${renderValidation(doc.validation)}
        <details class="json-block">
          <summary>Raw structured JSON</summary>
          <pre>${escapeHtml(JSON.stringify(doc, null, 2))}</pre>
        </details>
      </div>`;
  } catch (err) {
    $("result-root").innerHTML = `<p class="status-box is-err">${escapeHtml(err.code || "")} ${escapeHtml(err.message)}</p>`;
  }
}

function currentRoute() {
  const hash = (location.hash || "#dashboard").replace(/^#/, "");
  if (hash.startsWith("result/")) {
    return { view: "result", name: decodeURIComponent(hash.slice(7)) };
  }
  return { view: hash || "dashboard" };
}

async function route() {
  const { view, name } = currentRoute();
  if (view === "documents") return renderDocuments();
  if (view === "upload") return showView("upload");
  if (view === "history") return renderHistory();
  if (view === "settings") {
    showView("settings");
    $("api-base").value = window.NeoStatsAPI.apiBase();
    return;
  }
  if (view === "result" && name) return renderResult(name);
  return renderDashboard();
}

async function refreshHealth() {
  const pill = $("health-pill");
  try {
    const { ok, body } = await window.NeoStatsAPI.getHealth();
    if (ok && body.status === "healthy") {
      pill.textContent = "API healthy";
      pill.classList.remove("is-down");
    } else {
      pill.textContent = "API unavailable";
      pill.classList.add("is-down");
    }
  } catch {
    pill.textContent = "API unavailable";
    pill.classList.add("is-down");
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("neostats.theme", theme);
  $("theme-toggle-label").textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

function initTheme() {
  applyTheme(localStorage.getItem("neostats.theme") || "light");
}

function setUploadStatus(kind, html) {
  const box = $("upload-status");
  box.hidden = false;
  box.className = `status-box is-${kind}`;
  box.innerHTML = html;
}

function initUpload() {
  const input = $("file-input");
  const drop = $("dropzone");
  const label = $("file-label");
  input.addEventListener("change", () => {
    label.textContent = input.files[0] ? input.files[0].name : "PDF, JPG, or PNG";
  });
  ["dragenter", "dragover"].forEach((evt) => {
    drop.addEventListener(evt, (e) => {
      e.preventDefault();
      drop.classList.add("is-drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    drop.addEventListener(evt, (e) => {
      e.preventDefault();
      drop.classList.remove("is-drag");
    });
  });
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      label.textContent = e.dataTransfer.files[0].name;
    }
  });
  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = input.files[0];
    const type = $("document_type").value;
    if (!file) {
      setUploadStatus("err", "Choose a PDF, JPG, or PNG file.");
      return;
    }
    const btn = $("process-btn");
    btn.disabled = true;
    setUploadStatus("loading", "Processing… this uses the live API (OCR/Gemini if the backend calls them).");
    try {
      const result = await window.NeoStatsAPI.processDocument(file, type);
      const href = `#result/${encodeURIComponent(result.document_name)}`;
      setUploadStatus(
        "ok",
        `Saved <strong>${escapeHtml(result.document_name)}</strong> · processing ${escapeHtml(result.processing_status)} · validation ${escapeHtml((result.validation || {}).overall_status || "—")}. <a class="text-link" href="${href}">Open result</a>`
      );
    } catch (err) {
      setUploadStatus("err", `${escapeHtml(err.code || "ERROR")}: ${escapeHtml(err.message)}`);
    } finally {
      btn.disabled = false;
    }
  });
}

function applySidebar(collapsed) {
  $("sidebar").classList.toggle("is-collapsed", collapsed);
  localStorage.setItem("neostats.sidebar", collapsed ? "collapsed" : "expanded");
  $("collapse-btn").setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
}

function initChrome() {
  applySidebar(localStorage.getItem("neostats.sidebar") === "collapsed");
  $("collapse-btn").addEventListener("click", () => {
    applySidebar(!$("sidebar").classList.contains("is-collapsed"));
  });
  $("theme-toggle").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  $("menu-btn").addEventListener("click", () => {
    $("sidebar").classList.add("is-open");
    $("sidebar-backdrop").hidden = false;
  });
  $("sidebar-backdrop").addEventListener("click", () => {
    $("sidebar").classList.remove("is-open");
    $("sidebar-backdrop").hidden = true;
  });
  $("refresh-docs").addEventListener("click", renderDocuments);
  $("back-to-docs").addEventListener("click", () => {
    location.hash = "documents";
  });
  $("api-docs-link").addEventListener("click", (e) => {
    e.preventDefault();
    window.open(window.NeoStatsAPI.apiUrl("/docs"), "_blank", "noopener");
  });
  $("save-settings").addEventListener("click", () => {
    localStorage.setItem(window.NeoStatsAPI.STORAGE_KEY, $("api-base").value.trim().replace(/\/$/, ""));
    $("settings-status").textContent = "Saved.";
    refreshHealth();
  });
  $("ping-health").addEventListener("click", async () => {
    await refreshHealth();
    $("settings-status").textContent = $("health-pill").textContent;
  });
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initChrome();
  initUpload();
  refreshHealth();
  route();
});
