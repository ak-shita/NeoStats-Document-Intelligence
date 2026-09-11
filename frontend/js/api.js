const STORAGE_KEY = "neostats.apiBase";

function apiBase() {
  return (localStorage.getItem(STORAGE_KEY) || window.NEOSTATS_API_BASE || "").replace(/\/$/, "");
}

function apiUrl(path) {
  const base = apiBase();
  return `${base}${path}`;
}

async function parseBody(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { error: { code: "INVALID_RESPONSE", message: text.slice(0, 240) || "Invalid response" } };
  }
}

function apiError(body, status) {
  const err = body && body.error ? body.error : {};
  const error = new Error(err.message || `Request failed (${status})`);
  error.code = err.code || "HTTP_ERROR";
  error.status = status;
  return error;
}

async function getHealth() {
  const response = await fetch(apiUrl("/api/v1/health"));
  return { ok: response.ok, status: response.status, body: await parseBody(response) };
}

async function listDocuments() {
  const response = await fetch(apiUrl("/api/v1/documents"), { cache: "no-store" });
  const body = await parseBody(response);
  if (!response.ok) {
    throw apiError(body, response.status);
  }
  return body;
}

async function getDocument(name) {
  const encoded = encodeURIComponent(name);
  const response = await fetch(apiUrl(`/api/v1/documents/${encoded}`), { cache: "no-store" });
  const body = await parseBody(response);
  if (!response.ok) {
    throw apiError(body, response.status);
  }
  return body;
}

async function processDocument(file, documentType) {
  const form = new FormData();
  form.append("file", file);
  form.append("document_type", documentType);
  const response = await fetch(apiUrl("/api/v1/documents/process"), {
    method: "POST",
    body: form,
  });
  const body = await parseBody(response);
  if (!response.ok) {
    throw apiError(body, response.status);
  }
  return body;
}

window.NeoStatsAPI = {
  apiBase,
  apiUrl,
  getHealth,
  listDocuments,
  getDocument,
  processDocument,
  STORAGE_KEY,
};
