# EmbeddingGemma Workflows in dbs-vector

`dbs-vector` utilizes `embeddinggemma` (specifically `mlx-community/embeddinggemma-300m-bf16`) as its core embedding model. `embeddinggemma` is an instruction-tuned model, meaning its behavior and the vectors it produces change significantly based on the "task" or "workflow" provided as a text prefix before the actual content.

This document explains the two primary workflows implemented in `dbs-vector`, how they are configured, and how they bridge the gap between ingestion and search.

## Why Workflows Matter

Standard embedding models treat all text equally. Instruction-tuned models like `embeddinggemma` require explicit string prefixes to optimize their internal representations for the task at hand. 

By defining distinct prefixes for **passages** (the data being ingested) and **queries** (the user's search input), you can tune the model for:
1. **Asymmetric Search:** Finding long documents that answer short questions.
2. **Symmetric Clustering/Similarity:** Finding items that are semantically identical or highly similar to each other.

In `dbs-vector`, these prefixes are configured per engine in `config.yaml` and are stored in LanceDB under a dedicated `workflow` column to track the exact representation space used.

---

## Workflow 1: Asymmetric Search (`md` Engine)

This workflow is used for general prose, code documentation, and markdown files. It is designed to match a short user query (a question or keyword) against longer, detailed chunks of text.

### Configuration (`config.yaml`)
```yaml
engines:
  md:
    model_name: "mlx-community/embeddinggemma-300m-bf16"
    passage_prefix: "title: none | text: "
    query_prefix: "task: search result | query: "
    workflow: "md_search"
```

### How it works during Ingestion
When you run `uv run dbs-vector ingest ./docs --type md`, the `IngestionService` uses the `MLXEmbedder`.
1. The text is chunked into smaller pieces.
2. The `MLXEmbedder` takes each chunk and prepends the `passage_prefix`. 
   - *Example:* `"title: none | text: This document explains the architecture of the vector store."`
3. The prepended text is passed to the MLX model to generate the vector.
4. The vector, the original text (without the prefix), and the `workflow` string (`"md_search"`) are stored in LanceDB.

### How it works during Search
When you run `uv run dbs-vector search "how does storage work?" --type md`, the `SearchService` processes the query.
1. The `MLXEmbedder` prepends the `query_prefix` to the user's input.
   - *Example:* `"task: search result | query: how does storage work?"`
2. The model generates a vector optimized to find documents matching that specific task.
3. LanceDB performs a cosine similarity search against the vectors stored during ingestion.

---

## Workflow 2: Symmetric Clustering (`sql` Engine)

This workflow is used for structured data like SQL Slow Query Logs. The goal is not to answer a question, but to find other queries that are structurally or semantically similar (e.g., finding all variations of a heavy `JOIN` query).

### Configuration (`config.yaml`)
```yaml
engines:
  sql:
    model_name: "mlx-community/embeddinggemma-300m-bf16"
    passage_prefix: "task: clustering | query: "
    query_prefix: "task: clustering | query: "
    workflow: "sql_clustering"
```

### How it works during Ingestion
When you run `uv run dbs-vector ingest ./slow_queries.json --type sql`:
1. The SQL logs are parsed into chunks.
2. The `MLXEmbedder` prepends the `passage_prefix`.
   - *Example:* `"task: clustering | query: SELECT * FROM users WHERE age > 30;"`
3. The vector is generated and stored in LanceDB along with the `workflow` string (`"sql_clustering"`).

### How it works during Search
When you run `uv run dbs-vector search "SELECT name FROM users" --type sql`:
1. Because the goal is to find similar queries, the `query_prefix` is identical to the `passage_prefix`.
2. The `MLXEmbedder` prepends the `query_prefix`.
   - *Example:* `"task: clustering | query: SELECT name FROM users"`
3. LanceDB returns the most similar SQL queries from the database.

---

## Under the Hood: The MLXEmbedder Architecture

To efficiently handle multiple workflows that use the same underlying model (e.g., both `md` and `sql` use `embeddinggemma-300m-bf16`), `dbs-vector` employs a centralized model cache.

1. **Shared Model Weights:** When the API server starts (`dbs-vector serve`), the system loads the heavy model weights into Apple Unified Memory exactly once.
2. **Isolated Prefixes:** Even though the `md` and `sql` engines share the exact same model in memory, they instantiate separate `MLXEmbedder` objects. Each embedder instance holds its own `passage_prefix` and `query_prefix` configuration.
3. **Transparent Execution:** Developers do not need to manually append strings. The `embed_batch(texts)` method automatically applies the `passage_prefix`, and the `embed_query(text)` method automatically applies the `query_prefix` before the text reaches the MLX tokenizer.

By separating the **model weights** from the **workflow prefixes**, `dbs-vector` achieves high performance and low memory overhead while fully supporting `embeddinggemma`'s instruction-tuned capabilities.
