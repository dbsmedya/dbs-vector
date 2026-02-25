# dbs-vector: Arrow-Native Codebase Search

`dbs-vector` is a production-grade, local, high-performance Retrieval-Augmented Generation (RAG) search engine built specifically for Apple Silicon. 

It indexes text, code, and database logs into a **LanceDB (Apache Arrow)** store using **MLX** for batched GPU inference. It utilizes **Unified Memory Architecture (UMA)** to bypass Python-object serialization bottlenecks, and performs native Rust-level **Hybrid Search** (Vector + Full-Text Search via Tantivy).

---

## 1. Architectural Philosophy (SOLID & Clean Architecture)

This project is built using strict Clean Architecture principles and a Configuration-Driven Registry Pattern. It enforces strict separation of concerns through Dependency Inversion.

### Directory Structure
```text
src/dbs_vector/
├── config.py              # YAML Configuration Loader (Pydantic validated)
├── cli.py                 # Typer Entrypoint (DI Factory)
├── api/                   # FastAPI Search Service (Async offloading)
│
├── core/                  # Domain Layer (Strictly logic & Interfaces)
│   ├── models.py          # Pydantic data schemas (Chunk, SqlChunk, SearchResult)
│   ├── ports.py           # Protocol Definitions (IEmbedder, IVectorStore, IStoreMapper)
│   └── registry.py        # Component Registry (OCP-compliant dynamic lookup)
│
├── infrastructure/        # Infrastructure Adapters (Concrete Implementations)
│   ├── chunking/          
│   │   ├── document.py    # Robust markdown & fallback text parser
│   │   └── sql.py         # JSON SQL log parser
│   ├── embeddings/
│   │   └── mlx_engine.py  # Unified Memory MLX model wrapper
│   └── storage/
│       ├── lancedb_engine.py # Generic Arrow-native LanceDB backend
│       └── mappers.py     # Domain-to-Arrow Adapters (Document & SQL)
│
└── services/              # Application Orchestration
    ├── ingestion.py       # Manages batching -> deduplication -> storage stream
    └── search.py          # Manages embedding queries -> generic store search
```

### Key Principles Applied:
*   **Dependency Inversion (DIP):** The `IngestionService` and `SearchService` do not import `mlx` or `lancedb`. They only depend on `ports.py`.
*   **Open/Closed (OCP):** The system utilizes a **Registry Pattern**. Adding a new engine (e.g., `libcst` for AST) simply requires registering a new Chunker/Mapper and adding a block to `config.yaml`. No modification to core services or CLI logic is needed.
*   **Single Responsibility (SRP):** Database serialization is offloaded to specialized `Mapper` classes, keeping the `LanceDBStore` focused purely on storage and retrieval.

---

## 2. Toolchain & Quality Gates

*   **Dependency Manager:** `uv`
*   **Task Runner:** `poethepoet` (Execute via `uv run poe <task>`)
*   **Linter & Formatter:** `ruff`
*   **Type Checker:** `mypy` (Strict mode)
*   **Testing:** `pytest` (Unit and Integration)

### Common Commands
```bash
# Run all quality gates (Format, Lint, Typecheck, Test)
uv run poe check

# Ingest data using a specific engine (from config.yaml)
uv run dbs-vector ingest "docs/" --type md
uv run dbs-vector ingest "slow_queries.json" --type sql

# Perform a Hybrid Search
uv run dbs-vector search "What is MLX?" --type md
uv run dbs-vector search "SELECT * FROM users" --type sql --min-time 1000
```

---

## 3. Engineering Highlights

1.  **Generic Storage Adapters:** `LanceDBStore` is entirely data-agnostic. It accepts an `IStoreMapper` adapter that defines the schema and serialization logic, allowing it to store Markdown and SQL side-by-side in separate tables without internal branching.
2.  **Arrow-Native Pipeline:** Data is streamed directly to disk via `pyarrow.RecordBatch`, bypassing Python's slow object creation and utilizing Apple's Unified Memory for zero-copy tensor extraction.
3.  **Deduplication & Delta-Updates:** `IngestionService` uses content hashing (SHA-256) to skip files that haven't changed, significantly reducing GPU compute costs during re-ingestion.
4.  **Async API Offloading:** The FastAPI server uses `asyncio.to_thread` to handle blocking MLX inference, ensuring the web loop remains responsive during heavy searches.
5.  **Polymorphic Retrieval:** `SearchService` handles different result models (e.g., `SearchResult` vs `SqlSearchResult`) dynamically through the mapper pattern.

---

## 4. Roadmap

### Phase 2: Structural AST Parsing (LibCST Integration)
*   **Action:** Build `infrastructure/chunking/libcst_parser.py` and `AstMapper`.
*   **Goal:** Parse `.py` files into an Abstract Syntax Tree (AST). Extract atomic nodes (Functions, Classes).
*   **Integration:** Register in `registry.py` and add to `config.yaml`.

### Phase 3: The Context Assembler
*   **Action:** Create a `ContextAssemblerService`.
*   **Goal:** Use the `parent_scope` metadata defined in the schema to fetch enclosing classes or imports for a matched function chunk, providing complete context to the LLM.

### Phase 4: Managed Cloud Scaling
*   **Action:** Implement hardware-agnostic embedders.
*   **Goal:** Extend the `IEmbedder` port to support CUDA/TPU engines for deployment in Linux cloud environments.
