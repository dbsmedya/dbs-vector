# Production RAG Pipeline: MLX, LanceDB, and Apache Arrow on Apple Silicon

This document serves as a high-level reference and architectural deep dive into building a **production-grade Local RAG (Retrieval-Augmented Generation)** engine specifically optimized for Apple Silicon. 

While benchmark demos often tout "zero-copy" pipelines, production reality requires explicit memory management, indexing strategies, metadata filtering, and an understanding of exactly where serialization bottlenecks *actually* occur. This architecture leverages Apple's Unified Memory Architecture alongside native Apache Arrow pipelines (via LanceDB and PyArrow) to achieve maximum theoretical throughput.

---

## 1. The Bottleneck: Deconstructing "Zero-Copy" Myths

In a traditional local AI pipeline (e.g., PyTorch + ChromaDB + Pandas), data is forced through expensive transformations:
1.  **The `.tolist()` Catastrophe:** To insert a tensor into a SQLite-backed database, the C/C++ tensor must be converted into standard Python data types. Calling `.tolist()` on a 10,000x384 matrix forces the CPU to iterate and instantiate 3.8 million Python `float` objects.
2.  **Dict Comprehensions:** Even when using vector databases, building ingestion payloads via `[{"vector": v, ...} for v in vectors]` recreates massive Python dictionary overheads, defeating the purpose of Arrow-native storage.
3.  **The LLM Boundary:** Claims of "end-to-end zero copy" are technically false the moment you extract text from a Polars DataFrame to feed it into an LLM prompt (e.g., `results["text"].to_list()`). You are inevitably re-entering Python-object land to format strings.

**The Solution:** We restrict Python entirely to control flow. The dense data (vectors and text arrays) remains exclusively within MLX (Compute), NumPy (Transfer), and Apache Arrow/Polars (Storage/Retrieval) until the absolute final mile (Prompt Assembly).

---

## 2. Core Architectural Components & SOLID Design

The system is built upon a strict **Configuration-Driven Registry Pattern** designed to adhere to the Open/Closed Principle (OCP) and Single Responsibility Principle (SRP).

### A. The Adapter/Mapper Pattern (SRP)
Instead of hardcoding database schemas or branching logic (`if is_sql: ... else: ...`) into the storage layer, `LanceDBStore` delegates entirely to `IStoreMapper` implementations (`DocumentMapper`, `SqlMapper`). 
*   **Why this matters:** The database engine only knows how to append Arrow arrays and execute searches. It has zero knowledge of the domain objects (`SqlChunk`, `Chunk`), making it infinitely reusable.

### B. Dynamic Engine Registry (OCP)
The entire application is driven by `config.yaml`. The CLI and API factory (`_build_dependencies`) dynamically instantiate the correct Embedder, Mapper, and Chunker based on the `--type` flag (e.g., `md` or `sql`) by looking them up in `ComponentRegistry`.
*   **Why this matters:** Adding a new engine (like an AST parser) requires *zero* modifications to the core orchestration code. You simply write the new parser, register it, and add its YAML configuration.

### C. The Unified Memory Bridge (Compute to Transfer)
```python
embeds_mlx = model(inputs)
vectors_np = np.array(embeds_mlx).astype(np.float32)
```
*   **The Reality of `np.array()`:** This is *not* a true zero-copy operation. MLX and NumPy maintain separate memory allocators and reference counting semantics. When `np.array()` is called, MLX evaluates the lazy computation graph on the GPU and materializes the result. NumPy then performs a fast `memcpy` to own the buffer. Because of Apple's Unified Memory (UMA), this is a highly optimized memory map of contiguous floats within shared RAM, rather than a slow transfer across a PCIe bus, but a copy does occur.
*   **The Upcast:** Chaining `.astype(np.float32)` creates a second copy to upcast the hardware-optimized `bf16`/`fp16` results to standard 32-bit floats required by the PyArrow schema. Because this only happens on the small, final output matrices (e.g., `64x768`), the cost is in microseconds.
*   **Safety Assertions:** We strictly validate shapes `if query_vector.shape != (self._dimension,): raise ValueError()` to prevent silent sequence mismatches from the model.

### D. Streaming Arrow-Native Ingestion (The Python Bypass)
```python
arrow_batch = pa.RecordBatch.from_arrays([
    pa.array(ids),
    pa.FixedSizeListArray.from_arrays(vectors_np.ravel(), list_size=VECTOR_DIM),
    ...
], schema=schema)
table.add(arrow_batch)
```
*   **Why this matters:** By using `pyarrow.RecordBatch` in the Mappers, we pass the raw, flattened C-contiguous NumPy buffer directly into LanceDB's Rust engine, completely bypassing Python iteration.
*   **Streaming:** Instead of accumulating 100,000 vectors in RAM, we stream chunks dynamically using `itertools.islice`. Peak memory utilization remains flat regardless of codebase size.

---

## 3. Production Readiness & Features (Phase 1)

### A. Dual-Engine Support (Docs vs. SQL)
The architecture supports specialized parallel pipelines:
*   **MD Engine:** Uses semantic models (MiniLM) and a robust `markdown-it-py` chunker that guarantees code fences remain atomic.
*   **SQL Engine:** Uses code-native models (StarEncoder) to cluster parsed execution logs, finding identical logic hidden behind structural query differences.

### B. Deduplication & Delta-Updates
Before running expensive GPU embeddings, `IngestionService` queries the database for all existing `content_hash` values. Any chunk that hasn't changed is skipped entirely, preventing index bloat and accelerating re-ingestion.

### C. Hybrid Search & Rust-Level Pushdown
LanceDB simultaneously executes an Approximate Nearest Neighbor (ANN) vector search (enforced `metric="cosine"`) alongside a native **Tantivy** Full-Text Search (FTS). Metadata filters (like `--min-time` for SQL) are pushed down to the Rust layer *before* the vector scan.

### D. Async API Concurrency
The FastAPI server exposes both engines over HTTP. Because MLX inference is synchronous and locks the GPU, the API utilizes `asyncio.to_thread` and a strict `threading.Lock` within `MLXEmbedder` to serialize inference while keeping the web event loop responsive to concurrent requests (like health checks).

---

## 4. Next Steps (Phases 2-4)

*   **Phase 2 - AST Chunking (LibCST):** Implement `AstMapper` and `AstChunker` to parse `.py` files into structural nodes (Functions, Classes). Register them in `config.yaml` to instantly enable semantic code-tree search.
*   **Phase 3 - The Context Assembler:** Intercept the raw `SearchResult`. Use the `parent_scope` metadata to query the database *again* to fetch parent classes or module imports, providing complete, self-contained context to the LLM.
*   **Phase 4 - Cloud Hardware (CUDA/TPU):** Implement a `CudaEmbedder` utilizing PyTorch/Transformers that implements the `IEmbedder` protocol. The core architecture guarantees zero code changes are required to migrate off Apple Silicon to Linux GPU clusters.