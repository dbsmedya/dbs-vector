from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dbs_vector.core.model_registry import ModelContract


class _ProfileShape(Protocol):
    """Two attributes used by the estimator. Both TuningProfile (config) and
    test fakes satisfy it structurally."""

    max_token_length: int
    batch_size: int


# Calibrated empirically against the user's 2025-05 OOM:
#   batch=64, seq=16384, bf16 (2 bytes) → real allocation 41 GB
#   raw: 64 × 16384² × 2 = 34.4 GB. real / raw = 1.19× per-element overhead.
# We use 3.0× because the 41 GB is just the largest single buffer; total
# memory pressure also includes weights, KV cache, and activations.
_PEAK_BUFFER_OVERHEAD = 3.0

# Approximate char-per-token ratio for English+code; used by the recommender.
_CHARS_PER_TOKEN = 2.5

# 90 % of the budget — leave 10 % for OS/MPS overhead, allocator
# fragmentation, and non-attention buffers (weights, KV cache).
_BUDGET_HEADROOM = 0.9

# Minimum useful sequence length. Below this we'd rather raise than
# recommend a profile that can barely embed a sentence.
_MIN_SEQ_LEN = 512

# Half of the model's effective char-window: leaves slack for tokenizer
# expansion (code/Unicode) and keeps at least 2 chunks per max-context window.
_CHUNK_TO_CONTEXT_FRACTION = 0.5


def estimate_peak_buffer_bytes(
    profile: _ProfileShape, contract: "ModelContract"
) -> int:
    """Approximate peak Metal memory pressure during a forward pass.

    Dominated by attention: O(batch × seq² × dtype_bytes), with a 3× safety
    factor for temporaries, weights, and KV cache. Hidden_dim drops out
    because the attention matrix is the long pole, not the activations.

    The dtype is the model's compute dtype (contract.compute_dtype_bytes),
    not the attention_mask cast — the mask is cheap; the attention buffer
    is what blows up.
    """
    return int(
        _PEAK_BUFFER_OVERHEAD
        * profile.batch_size
        * profile.max_token_length ** 2
        * contract.compute_dtype_bytes
    )


def recommend_profile(
    contract: "ModelContract",
    memory_budget_gb: float,
    target_chunker: str = "document",
    target_seq_len: int | None = None,
) -> dict[str, int | bool]:
    """Suggest profile values that fit memory_budget_gb for this contract.

    Strategy (preserves user's intended context length over throughput):
      1. Start from target_seq_len (defaults to contract.model_max_token_length).
         Values above contract.model_max_token_length are silently clamped via
         min(...).
      2. Pick the largest batch_size ≥ 1 that fits at that seq length.
      3. If no batch fits, halve seq_len and retry; set seq_len_reduced=True.
      4. Pick chunk_max_chars from chunker-type heuristic.
         For chunker types "duckdb" and "api", atomic chunks are kept
         (chunk_max_chars=0) so a full SQL query / log entry stays in one
         chunk; document chunkers split on the char window.

    Returns:
        dict with keys: max_token_length, chunk_max_chars, batch_size,
        seq_len_reduced (bool — True if step 3 fired).
    """
    budget = int(memory_budget_gb * 1024 ** 3 * _BUDGET_HEADROOM)
    seq = target_seq_len if target_seq_len is not None else contract.model_max_token_length
    seq = min(seq, contract.model_max_token_length)
    seq_len_reduced = False

    while seq >= _MIN_SEQ_LEN:
        per_sample = int(_PEAK_BUFFER_OVERHEAD * seq ** 2 * contract.compute_dtype_bytes)
        max_batch = budget // per_sample if per_sample > 0 else 0
        if max_batch >= 1:
            chunk = (
                0
                if target_chunker in ("duckdb", "api")
                else int(seq * _CHARS_PER_TOKEN * _CHUNK_TO_CONTEXT_FRACTION)
            )
            return {
                "max_token_length": seq,
                "chunk_max_chars": chunk,
                "batch_size": int(max_batch),
                "seq_len_reduced": seq_len_reduced,
            }
        seq //= 2
        seq_len_reduced = True

    raise ValueError(
        f"No profile fits {memory_budget_gb} GB for model with cap "
        f"{contract.model_max_token_length}. Reduce model or increase budget."
    )
