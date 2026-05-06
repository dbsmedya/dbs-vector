from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import TuningProfile
    from dbs_vector.core.model_registry import ModelContract


# Calibrated empirically against the user's 2025-05 OOM:
#   batch=64, seq=16384, bf16 (2 bytes) → real allocation 41 GB
#   raw: 64 × 16384² × 2 = 34.4 GB. real / raw = 1.19× per-element overhead.
# We use 3.0× because the 41 GB is just the largest single buffer; total
# memory pressure also includes weights, KV cache, and activations.
_PEAK_BUFFER_OVERHEAD = 3.0

# Approximate char-per-token ratio for English+code; used by the recommender.
_CHARS_PER_TOKEN = 2.5


def estimate_peak_buffer_bytes(
    profile: "TuningProfile", contract: "ModelContract"
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
      2. Pick the largest batch_size ≥ 1 that fits at that seq length.
      3. If no batch fits, halve seq_len and retry; set seq_len_reduced=True.
      4. Pick chunk_max_chars from chunker-type heuristic.

    Returns:
        dict with keys: max_token_length, chunk_max_chars, batch_size,
        seq_len_reduced (bool — True if step 3 fired).
    """
    budget = int(memory_budget_gb * 1024 ** 3 * 0.9)
    seq = target_seq_len if target_seq_len is not None else contract.model_max_token_length
    seq = min(seq, contract.model_max_token_length)
    seq_len_reduced = False

    while seq >= 512:
        per_sample = int(_PEAK_BUFFER_OVERHEAD * seq ** 2 * contract.compute_dtype_bytes)
        max_batch = budget // per_sample if per_sample > 0 else 0
        if max_batch >= 1:
            chunk = (
                0
                if target_chunker in ("duckdb", "api")
                else int(seq * _CHARS_PER_TOKEN * 0.5)
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
