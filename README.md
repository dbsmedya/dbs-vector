# ⚡️ dbs-vector

Local RAG search and MCP server for your text documents, markdown files and MySQL
slow-query logs — very fast, completely offline, LLM-native and optimized for Mac.

> **Requires Apple Silicon (M-series) + macOS.** Embeddings run on Metal via MLX.
> Other accelerators: see [Roadmap](#-roadmap).

Point it at a directory or a slow-query log, and your LLM gets a search tool over it.

---

## 🚀 Quick start

```bash
uv tool install dbs-vector                    # or: pip install dbs-vector
dbs-vector init                               # asks what to index, writes the config
dbs-vector ingest --type md                   # index it (init prints this exact line)
dbs-vector search "how does dedupe work?"     # ask it something
```

`init` interviews you and writes a `config.yaml` plus a Claude-format `.mcp.json`.
Every field it sets is explained in
[docs/README_CONFIGURATION.md](docs/README_CONFIGURATION.md).

---

## 🤖 What it gives an LLM

**MCP server.** `dbs-vector mcp` exposes one `search_<engine>` tool per configured
engine over stdio, plus `list_engines` for discovery. Works with Claude Code, Claude
Desktop, and any stdio MCP client. → [MCP guide](docs/README_MCP.md)

**Hybrid search you can trust.** Vector and full-text retrieval run together and fuse
into one ranking. Every result carries a true cosine similarity, so an empty result
means *low confidence*, not *nothing indexed*. → [Scoring & calibration](docs/README_CALIBRATE_CORPUS.md)

**Chunk navigation.** A hit is a starting point, not a dead end — the model can read the
chunks on either side of a match to recover context the chunker split apart.
→ [MCP guide](docs/README_MCP.md)

**A live index.** Watched engines re-ingest changed files while the MCP server runs, so
answers follow your edits without a manual re-index. → [Watch guide](docs/README_WATCH.md)

**Slow-query triage.** Point it at a MySQL slow-query log and one call returns the
costliest query fingerprints ranked by `calls × execution_time`, each with a
paste-ready exemplar for `EXPLAIN`. → [SQL guide](docs/README_SQL.md)

**Your data never leaves the machine.** No dependency on an external embedding API: the
model you choose is pulled once from Hugging Face and installed locally, and every
embedding after that runs on your own GPU — with Hugging Face telemetry disabled in code.
No third-party service is contacted. Pair it with a local LLM and the whole chain stays
on-premise, which is the point when the corpus is production SQL; verbatim queries
(`raw_query`, often PII) stay hidden unless you opt in with `--allow-raw-queries`.
→ [MCP guide](docs/README_MCP.md)

**Bundled skills.** Two read-only Claude skills ship under `skills/`:
`find-impacting-queries` (triage → index recommendation) and `query-rewrite`, its
handoff partner for when no index will help.
→ [Skill workflow](docs/README_MCP.md#diagnostic-skills-find-impacting-queries--query-rewrite)

---

## ⚡ Why it's fast

**Metal, not CPU.** Embeddings run on the Mac GPU through MLX, and tensors cross into
NumPy through unified memory instead of being rebuilt value by value.
→ [Architecture](docs/README_ARCHITECTURE.md)

**Arrow end to end, Rust underneath.** Chunks stream to LanceDB as Arrow record batches;
vector indexing and full-text search are Rust, not Python.
→ [Architecture](docs/README_ARCHITECTURE.md)

**Sized to your GPU.** dbs-vector reads your Metal memory budget and derives chunk and
batch settings that fit it — and when a config doesn't fit, it tells you which numbers
would. → [Profiles & memory](docs/README_PROFILES.md)

---

## 🔧 Make it yours

**What it can index.** Plain text, Markdown, MySQL slow-query logs (JSON or DuckDB), and
any paginated HTTP API. Markdown is parsed with `markdown-it-py` and split along real
document structure — headings, lists, tables, fenced code — so a chunk ends where a
section does and a code block is never cut in half.
→ [Chunking](docs/README_DOCS.md) · [SQL](docs/README_SQL.md) · [DuckDB](docs/README_duckdb.md) · [API](docs/README_REMOTE_SQL_API.md)

**Swap the embedding model.** Models are declared, not hardcoded. Two ship today — a
2K-context English-first model and a 32K multilingual one — and adding a third is a
single registration. → [Embedding models](docs/README_EMBEDDINGS.md)

**Model-specific workflows.** Instruction-tuned models want one prompt for documents,
another for queries, and another again for clustering. That is configuration, not code.
→ [Workflows](docs/README_WORKFLOW.md)

**Configure everything else.** Engines, chunk sizes, batch sizes, ignore patterns, watch
behaviour and result filtering all live in `config.yaml`. `init` hands you a working
one; the guide explains every field. → [Configuration guide](docs/README_CONFIGURATION.md)

---

## 🗺 Roadmap

**Other accelerators.** Embedding sits behind a single `IEmbedder` protocol, so CUDA or
CPU support means writing one new embedder that returns NumPy arrays — no change to
ingestion, storage, the CLI, or MCP. Not implemented; no CUDA hardware on hand.

Longer-range plans (AST-aware chunking, context assembly) are in
[docs/README.md](docs/README.md).

---

## 🛠 Development

```bash
git clone https://github.com/dbsmedya/dbs-vector.git && cd dbs-vector
uv sync
uv run poe check     # format, lint, typecheck, test
```

---

## 📄 Status & license

In development for six months and in daily use across several organizations. Licensed
[GPL-3.0-or-later](LICENSE.md), which disclaims warranty and liability — you run it at
your own risk. Issues and pull requests are welcome.
