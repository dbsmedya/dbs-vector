# Markdown Document Engine

The Document engine (`--type md`) in `dbs-vector` is designed to ingest, chunk, and semantically search through prose and code documentation. It is the core engine for building a Retrieval-Augmented Generation (RAG) codebase assistant.

## Overview
This engine utilizes a specialized parsing strategy to ensure that context sent to the LLM is logically coherent. It natively understands Markdown syntax, prioritizing the integrity of code blocks over strict character limits.

---

## The Parsing Strategy (`DocumentChunker`)

Unlike naive splitters that blindly cut text every 1,000 characters (often slicing a function in half), the `DocumentChunker` utilizes `markdown-it-py` to construct a semantic representation of the document.

### 1. Atomic Code Fences
When the parser encounters a markdown code fence (e.g., ````python ... ````), it treats the entire block as a single, **atomic** unit. 
*   **Benefit:** A 150-line function will never be split across two separate database chunks. The LLM will always receive the complete, unbroken logic.

### 2. Prose Accumulation
For standard prose (paragraphs, lists, blockquotes), the chunker accumulates text until it approaches the `chunk_max_chars` limit defined in your `config.yaml` (default: 1000 characters).
*   **Benefit:** Keeps context dense and reduces the number of small, fragmented vectors in the database.

### 3. Plain Text Fallback
If a `.txt` file is ingested instead of `.md`, the engine safely falls back to splitting by double-newlines (`

`), ensuring broad compatibility with raw logs or unformatted notes.

---

## Batching & Memory Management

Embedding models (like `all-MiniLM-L6-v2`) require significant GPU memory to process text into vectors. To handle massive codebases (e.g., thousands of files) without crashing, `dbs-vector` employs a **Streaming Batch Architecture**.

1.  **Lazy Generation:** The `DocumentChunker` uses Python generators (`yield`) to extract chunks one by one, meaning the entire codebase is never loaded into RAM simultaneously.
2.  **Configurable Batching:** The `IngestionService` groups these chunks into strict batches (controlled by `batch_size` in `config.yaml`, default: 64). 
3.  **GPU Offloading:** Only one batch is sent to the Apple MLX GPU at a time. 
4.  **Zero-Copy Storage:** The resulting tensors are instantly mapped to PyArrow `RecordBatch` arrays and flushed to disk via LanceDB.

This architecture ensures a flat, predictable memory profile regardless of the repository size.

---

## Indexing & Hybrid Search

Once the documents are ingested, the engine generates two powerful indices to guarantee ultra-fast retrieval.

### 1. Vector Indexing (IVF-PQ)
LanceDB automatically builds an **Inverted File with Product Quantization (IVF-PQ)** index on the `vector` column. 
*   **Dynamic Scaling:** To prevent poor recall on small datasets, `dbs-vector` dynamically calculates the number of IVF partitions based on the total number of chunks (`sqrt(N)`).
*   **Cosine Similarity:** The engine explicitly enforces `metric="cosine"` to match the training objective of modern embedding models.

### 2. Full-Text Search (FTS)
A native **Tantivy** Full-Text index is built on the raw text column. 

When you execute a search:
```bash
uv run dbs-vector search "Unified Memory Architecture" --type md
```
The engine performs a **Hybrid Search**. It simultaneously looks for the semantic meaning of the query (Vector) AND exact keyword matches (FTS), returning the most mathematically relevant code snippets to feed to your LLM.
