#!/usr/bin/env python3
"""dbs-web — a tiny stdlib-only browser front-end for dbs-vector.

Standalone helper that lives OUTSIDE the packaged codebase (src/ is never
touched). It drives dbs-vector in-process: it imports the project's services,
loads each engine's MLX model lazily on first use, runs hybrid searches, and
can (re)build an engine's index.

Why a browser instead of Tk: rendering happens in the browser, which handles
emoji / astral codepoints natively. That removes the whole class of Tk 8.6
macOS segfaults (Text-widget index ops over surrogate pairs) AND lets us
highlight matches safely — see ui/dbs-ui.py and the tk-macos-astral-segfault
memory for the history.

No external framework: just the stdlib `http.server`. One file, one process,
bound to localhost only.

Run from the repo root with the project venv so `dbs_vector` and its deps
(lancedb / mlx) resolve:

    uv run python ui/dbs-web.py

then open http://127.0.0.1:8765 (it auto-opens unless DBS_WEB_NO_OPEN is set).
Override the port with DBS_WEB_PORT.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --------------------------------------------------------------------------- #
# Bootstrap dbs-vector config (mirror the CLI callback) BEFORE importing the
# heavy service layer. Locate config.yaml relative to this script so the GUI
# works regardless of the current working directory.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config.yaml"

HOST = "127.0.0.1"
PORT = int(os.environ.get("DBS_WEB_PORT", "8765"))


def _bootstrap_settings() -> object:
    """Load + validate config.yaml and populate the global Settings singleton."""
    from dbs_vector.config import _populate_singleton_from, load_settings, settings
    from dbs_vector.logger import configure_logger

    os.environ["DBS_CONFIG_FILE"] = str(CONFIG_FILE)
    new_settings = load_settings(str(CONFIG_FILE), validate=True)
    _populate_singleton_from(new_settings)
    # Quiet the loguru default (DEBUG -> stderr) down to something readable.
    configure_logger(level="INFO", serialize=False)
    return settings


class _State:
    """Process-wide state shared across HTTP worker threads."""

    def __init__(self) -> None:
        self.settings: object = None
        # engine name -> cached EngineDeps (embedder/store/...). Lazy: built on
        # first search for that engine, invalidated after a reindex.
        self.deps_cache: dict[str, object] = {}
        # Serializes model loads + ingestion across concurrent requests. The
        # MLXEmbedder has its own per-model lock, so query execution can run
        # outside this; we only guard cache mutation and the (write) ingest.
        self.lock = threading.Lock()


STATE = _State()


# --------------------------------------------------------------------- core -- #
def _get_deps(engine: str) -> object:
    """Lazily build + cache the embedder/store stack for an engine."""
    with STATE.lock:
        cached = STATE.deps_cache.get(engine)
        if cached is not None:
            return cached
        from dbs_vector.services.bootstrap import build_dependencies

        deps = build_dependencies(engine)
        STATE.deps_cache[engine] = deps
        return deps


def _similarity_str(res: object) -> str:
    from dbs_vector.services.search import retrieved_by_label

    sim = res.similarity  # type: ignore[attr-defined]
    label = retrieved_by_label(res.retrieved_by)  # type: ignore[attr-defined]
    return f"{sim:.3f} ({label})"


def _serialize(res: object) -> dict[str, object]:
    """Flatten a SearchResult / SqlSearchResult into a JSON-ready record.

    Mirrors the Tk version's _summary + _detail_parts: the client gets both the
    one-line list summary and the full detail (header rows + body) in one shot,
    so clicking a result needs no extra round-trip.
    """
    from dbs_vector.services.search import retrieved_by_label

    chunk = res.chunk  # type: ignore[attr-defined]
    sc = _similarity_str(res)
    retrieved_by = retrieved_by_label(res.retrieved_by)  # type: ignore[attr-defined]
    src_start: int | None = None
    if hasattr(chunk, "raw_query"):  # SQL result — no source file
        snippet = (chunk.raw_query or chunk.text)[:90].replace("\n", " ")
        summary = f"{sc}  {chunk.source}  {chunk.execution_time_ms:.0f}ms  {snippet}"
        rows = [
            ("Source / DB", chunk.source),
            ("Similarity", sc),
            ("Retrieved by", retrieved_by),
            ("Calls", chunk.calls),
            ("Exec time (ms)", chunk.execution_time_ms),
            ("Rows sent", chunk.rows_sent),
            ("Rows examined", chunk.rows_examined),
            ("Lock time (s)", chunk.lock_time_sec),
            ("Tables", ", ".join(chunk.tables)),
            ("User", chunk.user),
            ("Host", chunk.host),
            ("Latest ts", chunk.latest_ts),
            ("Content hash", chunk.content_hash),
        ]
        body = chunk.raw_query or chunk.text
        heading = "RAW QUERY" if chunk.raw_query else "NORMALIZED QUERY"
    else:  # document result — line_range maps the chunk back to the file
        snippet = chunk.text[:100].replace("\n", " ")
        summary = f"{sc}  {chunk.source}  {snippet}"
        rows = [
            ("Source", chunk.source),
            ("Similarity", sc),
            ("Retrieved by", retrieved_by),
            ("Content hash", chunk.content_hash),
            ("Node type", chunk.node_type),
            ("Parent scope", chunk.parent_scope),
            ("Line range", chunk.line_range),
        ]
        body = chunk.text
        heading = "TEXT"
        if chunk.line_range:
            head = str(chunk.line_range).split("-", 1)[0].strip()
            if head.isdigit():
                src_start = int(head)
    header = [[label, "" if value is None else str(value)] for label, value in rows]
    return {
        "summary": summary,
        "header": header,
        "heading": heading,
        "body": body,
        "src_start": src_start,
    }


def _engines_payload() -> dict[str, object]:
    meta: dict[str, object] = {}
    for name, cfg in STATE.settings.engines.items():  # type: ignore[attr-defined]
        meta[name] = {
            "chunker_type": cfg.chunker_type,
            "api_base_url": getattr(cfg, "api_base_url", None),
        }
    return {"engines": list(STATE.settings.engines.keys()), "meta": meta}  # type: ignore[attr-defined]


def _handle_search(payload: dict[str, object]) -> dict[str, object]:
    engine = str(payload.get("engine") or "")
    if engine not in STATE.settings.engines:  # type: ignore[attr-defined]
        raise ValueError(f"Unknown engine: {engine!r}")
    query = str(payload.get("query") or "").strip()
    if not query:
        return {"results": []}
    try:
        limit = max(1, int(payload.get("limit") or 20))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        limit = 20

    from dbs_vector.services.search import SearchService

    deps = _get_deps(engine)
    service = SearchService(deps.embedder, deps.store)  # type: ignore[attr-defined]
    response = service.execute_query(query, limit=limit)
    return {
        "results": [_serialize(r) for r in response.results],
        "floor": response.floor,
        "inspected": response.inspected,
        "best_rejected": (
            response.best_rejected.model_dump(mode="json") if response.best_rejected else None
        ),
    }


def _handle_update(payload: dict[str, object]) -> dict[str, object]:
    engine = str(payload.get("engine") or "")
    if engine not in STATE.settings.engines:  # type: ignore[attr-defined]
        raise ValueError(f"Unknown engine: {engine!r}")
    cfg = STATE.settings.engines[engine]  # type: ignore[attr-defined]
    rebuild = bool(payload.get("rebuild"))

    # Resolve the ingestion source from the engine's chunker type. The browser
    # runs on the same machine as this server, so non-api engines take a
    # server-side path typed into the UI (there is no native file dialog).
    chunker_type = cfg.chunker_type
    if chunker_type == "api":
        path = cfg.api_base_url
        if not path:
            raise ValueError(f"Engine '{engine}' has no api_base_url configured.")
    else:
        path = str(payload.get("source") or "").strip()
        if not path:
            raise ValueError(f"A source path is required for engine '{engine}'.")

    url_override = path if path.startswith(("http://", "https://")) else None

    from dbs_vector.services.bootstrap import build_dependencies
    from dbs_vector.services.ingestion import IngestionService

    # Hold the lock across the whole (write) ingest so concurrent requests can't
    # interleave writes or double-load the model.
    with STATE.lock:
        deps = build_dependencies(engine, url_override=url_override)
        service = IngestionService(
            deps.chunker,
            deps.embedder,
            deps.store,
            deps.workflow,
            batch_size=deps.batch_size,
        )
        service.ingest_directory(path, rebuild=rebuild)
        try:
            count: int | None = deps.store.count_matching()
        except Exception:
            count = None
        # Drop the cached read store so the next search reopens fresh rows.
        STATE.deps_cache.pop(engine, None)
    return {"engine": engine, "count": count}


# --------------------------------------------------------------------- HTTP -- #
class Handler(BaseHTTPRequestHandler):
    server_version = "dbs-web"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A002 - stdlib name
        sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: object, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/engines":
            try:
                self._send_json(_engines_payload())
            except Exception as exc:
                self._send_json({"error": str(exc)}, code=500)
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"bad JSON: {exc}"}, code=400)
            return

        handlers = {"/api/search": _handle_search, "/api/update": _handle_update}
        handler = handlers.get(self.path)
        if handler is None:
            self._send_json({"error": "not found"}, code=404)
            return
        try:
            self._send_json(handler(payload))
        except Exception as exc:
            self._send_json({"error": str(exc)}, code=500)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dbs-vector search</title>
<style>
  :root { --bd:#d0d0d7; --bg:#fafafb; --accent:#2563eb; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font: 14px/1.4 -apple-system, Helvetica, Arial, sans-serif; color:#1c1c22;
         display:flex; flex-direction:column; background:var(--bg); }
  .bar { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
         padding:8px 10px; border-bottom:1px solid var(--bd); background:#fff; }
  .bar select, .bar input, .bar button {
         font:inherit; padding:5px 8px; border:1px solid var(--bd); border-radius:6px; background:#fff; }
  #query { flex:1 1 240px; min-width:160px; }
  #source { flex:1 1 220px; min-width:140px; }
  .bar button { cursor:pointer; }
  .bar button:hover:not(:disabled) { border-color:var(--accent); }
  .bar button:disabled { opacity:.5; cursor:default; }
  label.chk { display:flex; align-items:center; gap:4px; user-select:none; }
  main { flex:1; display:flex; min-height:0; }
  #list { flex:2; overflow:auto; border-right:1px solid var(--bd); }
  #detail { flex:3; overflow:auto; padding:12px 14px; }
  .item { padding:6px 10px; border-bottom:1px solid #eee; cursor:pointer;
          font:12px/1.35 Menlo, monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .item:hover { background:#f0f4ff; }
  .item.active { background:#dbe6ff; }
  table.meta { border-collapse:collapse; margin-bottom:10px; }
  table.meta td { padding:2px 10px 2px 0; vertical-align:top; font:12px/1.4 Menlo, monospace; }
  table.meta td.k { color:#6b7280; white-space:nowrap; }
  .heading { color:#6b7280; font:12px Menlo, monospace; margin:6px 0; }
  pre.body { white-space:pre-wrap; word-break:break-word; margin:0;
             font:12px/1.5 Menlo, monospace;
             background:#fffdf5; border-left:3px solid #ffd166; padding:6px 0 6px 10px; }
  mark.match { background:#ffe58a; color:inherit; border-radius:2px; }
  mark.first { background:#ffb347; }
  #status { padding:5px 10px; border-top:1px solid var(--bd); background:#fff;
            font:12px Menlo, monospace; color:#444; }
  #status.err { color:#b00020; }
  .empty { color:#9aa0aa; padding:14px; }
</style>
</head>
<body>
  <div class="bar">
    <select id="engine" title="engine"></select>
    <input id="query" type="text" placeholder="search query…" autocomplete="off">
    <button id="searchBtn">Search</button>
    <label class="chk">limit <input id="limit" type="number" min="1" max="1000" value="20" style="width:64px"></label>
    <span style="width:1px;height:22px;background:var(--bd)"></span>
    <input id="source" type="text" placeholder="server path to ingest…" autocomplete="off">
    <label class="chk"><input id="rebuild" type="checkbox"> Full rebuild</label>
    <button id="updateBtn">Update Index</button>
  </div>
  <main>
    <div id="list"><div class="empty">No results yet.</div></div>
    <div id="detail"><div class="empty">Select a result to see its detail.</div></div>
  </main>
  <div id="status">Loading engines…</div>

<script>
const $ = (id) => document.getElementById(id);
let META = {};            // engine -> {chunker_type, api_base_url}
let RESULTS = [];         // last search results (serialized)
let currentQuery = "";    // query behind RESULTS, used for highlighting
let busy = false;

function setStatus(msg, err=false) {
  const el = $("status");
  el.textContent = msg;
  el.className = err ? "err" : "";
}
function setBusy(b) {
  busy = b;
  $("searchBtn").disabled = b;
  $("updateBtn").disabled = b;
}
function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

// Full phrase first, then each token (len>=2), de-duped case-insensitively;
// longest first so phrase matches win when ranges overlap. Mirrors the Tk
// _highlight_terms, but the browser renders emoji fine so nothing is stripped.
function highlightTerms(query) {
  const out = [], seen = new Set();
  for (const raw of [query.trim(), ...query.split(/\\s+/)]) {
    const t = raw.trim(), k = t.toLowerCase();
    if (t.length >= 2 && !seen.has(k)) { seen.add(k); out.push(t); }
  }
  out.sort((a, b) => b.length - a.length);
  return out;
}

// Returns {html, line} where line is the 1-based line of the first match (or 0).
function highlightBody(text, query) {
  const terms = query ? highlightTerms(query) : [];
  if (!terms.length) return { html: escapeHtml(text), line: 0 };
  const lower = text.toLowerCase();
  const ranges = [];
  for (const term of terms) {
    const t = term.toLowerCase();
    let i = 0;
    while (true) {
      const idx = lower.indexOf(t, i);
      if (idx < 0) break;
      ranges.push([idx, idx + term.length]);
      i = idx + term.length;
    }
  }
  if (!ranges.length) return { html: escapeHtml(text), line: 0 };
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push(r.slice());
  }
  let html = "", pos = 0, first = true;
  for (const [s, e] of merged) {
    html += escapeHtml(text.slice(pos, s));
    const cls = first ? "match first" : "match";
    const id = first ? ' id="first-match"' : "";
    html += `<mark class="${cls}"${id}>` + escapeHtml(text.slice(s, e)) + "</mark>";
    first = false;
    pos = e;
  }
  html += escapeHtml(text.slice(pos));
  const line = text.slice(0, merged[0][0]).split("\\n").length;
  return { html, line };
}

function renderList() {
  const box = $("list");
  box.innerHTML = "";
  if (!RESULTS.length) {
    box.innerHTML = '<div class="empty">No results.</div>';
    return;
  }
  RESULTS.forEach((r, i) => {
    const div = document.createElement("div");
    div.className = "item";
    div.textContent = r.summary;
    div.onclick = () => {
      document.querySelectorAll(".item.active").forEach(e => e.classList.remove("active"));
      div.classList.add("active");
      renderDetail(r);
    };
    box.appendChild(div);
  });
}

function renderDetail(r) {
  const headerHtml = r.header
    .map(([k, v]) => `<tr><td class="k">${escapeHtml(k)}</td><td class="v">${escapeHtml(v)}</td></tr>`)
    .join("");
  const { html, line } = highlightBody(r.body, currentQuery);
  $("detail").innerHTML =
    `<table class="meta">${headerHtml}</table>` +
    `<div class="heading" id="chunk-heading">── ${escapeHtml(r.heading)} ──</div>` +
    `<pre class="body">${html}</pre>`;
  const fm = $("first-match");
  if (fm) {
    // Literal hit: land on the first match, centered.
    fm.scrollIntoView({ block: "center" });
    let note = `Match on chunk line ${line}`;
    if (r.src_start != null) note += `  →  source line ≈ ${r.src_start + line - 1}`;
    setStatus(note);
  } else {
    // No literal hit (e.g. a semantic / vector match): still jump to the chunk
    // itself so a click always lands on the matched content, not the metadata.
    $("chunk-heading").scrollIntoView({ block: "start" });
    setStatus(currentQuery
      ? "Semantic match — no literal term in this chunk; jumped to the chunk."
      : "");
  }
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function doSearch() {
  if (busy) return;
  const query = $("query").value.trim();
  if (!query) return;
  const engine = $("engine").value;
  const limit = parseInt($("limit").value, 10) || 20;
  currentQuery = query;
  setBusy(true);
  setStatus(`Searching '${engine}'…`);
  try {
    const data = await postJSON("/api/search", { engine, query, limit });
    RESULTS = data.results;
    renderList();
    $("detail").innerHTML = RESULTS.length
      ? '<div class="empty">Select a result to see its detail.</div>'
      : '<div class="empty">No results.</div>';
    setStatus(`${RESULTS.length} result(s)`);
  } catch (e) {
    setStatus("Search failed: " + e.message, true);
  } finally {
    setBusy(false);
  }
}

async function doUpdate() {
  if (busy) return;
  const engine = $("engine").value;
  const rebuild = $("rebuild").checked;
  const m = META[engine] || {};
  let source = $("source").value.trim();
  if (m.chunker_type === "api") source = "";  // server uses api_base_url
  if (m.chunker_type !== "api" && !source) {
    setStatus(`A source path is required for '${engine}'.`, true);
    return;
  }
  if (rebuild && !confirm(`Wipe and rebuild the '${engine}' index from scratch?\\n` +
                          "This erases all existing rows for this engine.")) return;
  setBusy(true);
  setStatus(`${rebuild ? "Rebuilding" : "Updating"} '${engine}' index… (loading model, ingesting)`);
  try {
    const data = await postJSON("/api/update", { engine, source, rebuild });
    const tail = data.count != null ? ` — ${data.count} rows total` : "";
    setStatus(`Index updated for '${data.engine}'${tail}`);
  } catch (e) {
    setStatus("Indexing failed: " + e.message, true);
  } finally {
    setBusy(false);
  }
}

// Show/adapt the source field to the selected engine's chunker type.
function syncSourceField() {
  const engine = $("engine").value;
  const m = META[engine] || {};
  const src = $("source");
  if (m.chunker_type === "api") {
    src.style.display = "none";
    src.value = "";
  } else {
    src.style.display = "";
    src.placeholder =
      m.chunker_type === "document" ? "server path to docs folder…" :
      m.chunker_type === "duckdb"   ? "server path to .duckdb slow-log…" :
                                      "server path to source…";
  }
}

async function init() {
  try {
    const res = await fetch("/api/engines");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    META = data.meta || {};
    const sel = $("engine");
    sel.innerHTML = "";
    data.engines.forEach(name => {
      const o = document.createElement("option");
      o.value = o.textContent = name;
      sel.appendChild(o);
    });
    sel.onchange = syncSourceField;
    syncSourceField();
    setStatus("Ready");
  } catch (e) {
    setStatus("Failed to load engines: " + e.message, true);
  }
  $("searchBtn").onclick = doSearch;
  $("updateBtn").onclick = doUpdate;
  $("query").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
}

init();
</script>
</body>
</html>
"""


def main() -> None:
    try:
        STATE.settings = _bootstrap_settings()
    except Exception as exc:
        print(f"Failed to load {CONFIG_FILE}:\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    url = f"http://{HOST}:{PORT}"
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"dbs-web serving at {url}  (Ctrl-C to stop)", file=sys.stderr)
    if not os.environ.get("DBS_WEB_NO_OPEN"):
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…", file=sys.stderr)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
