from dataclasses import dataclass
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


def estimate_peak_buffer_bytes(profile: _ProfileShape, contract: "ModelContract") -> int:
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
        * profile.max_token_length**2
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
    budget = int(memory_budget_gb * 1024**3 * _BUDGET_HEADROOM)
    seq = target_seq_len if target_seq_len is not None else contract.model_max_token_length
    seq = min(seq, contract.model_max_token_length)
    seq_len_reduced = False

    while seq >= _MIN_SEQ_LEN:
        per_sample = int(_PEAK_BUFFER_OVERHEAD * seq**2 * contract.compute_dtype_bytes)
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


# Tokens reserved between chunk_max_tokens and max_token_length. rule 8 permits
# chunk_max_tokens == max_token_length, but the passage prefix is prepended
# AFTER chunking, so a maximal chunk plus its prefix would truncate.
PREFIX_HEADROOM_TOKENS = 256


@dataclass(frozen=True)
class ProfileTier:
    """A chunk-granularity tier.

    These values are a CORPUS property, not a model property: the three
    document profiles shipping in config.yaml carry three distinct triples
    that do not correlate with their model. This table therefore does NOT
    grow when a model is registered.
    """

    key: str
    label: str
    chunk_target_tokens: int
    chunk_max_tokens: int
    batch_size: int


class ProfileTierRegistry:
    """Open/closed registry of granularity tiers (cf. ModelRegistry)."""

    # Insertion order is meaningful: keys() drives prompt ordering, and
    # sorted() would render as large, medium, small.
    _tiers: dict[str, ProfileTier] = {}

    @classmethod
    def register(cls, tier: ProfileTier) -> None:
        if tier.key in cls._tiers:
            raise ValueError(f"Profile tier '{tier.key}' already registered")
        cls._tiers[tier.key] = tier

    @classmethod
    def get(cls, key: str) -> ProfileTier:
        if key not in cls._tiers:
            raise KeyError(f"Unknown profile tier '{key}'. Known: {cls.keys()}")
        return cls._tiers[key]

    @classmethod
    def keys(cls) -> list[str]:
        """Registration order (smallest first), NOT sorted."""
        return list(cls._tiers)

    @classmethod
    def values(cls) -> list[ProfileTier]:
        return list(cls._tiers.values())


ProfileTierRegistry.register(
    ProfileTier(
        key="small",
        label="Small - precise hits, more chunks per file",
        chunk_target_tokens=512,
        chunk_max_tokens=1024,
        batch_size=64,
    )
)
ProfileTierRegistry.register(
    ProfileTier(
        key="medium",
        label="Medium - balanced (recommended)",
        chunk_target_tokens=768,
        chunk_max_tokens=1536,
        batch_size=16,
    )
)
ProfileTierRegistry.register(
    ProfileTier(
        key="large",
        label="Large - whole sections per chunk",
        chunk_target_tokens=1024,
        chunk_max_tokens=2048,
        batch_size=8,
    )
)


def next_power_of_two(n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power


def _target_seq_len(contract: "ModelContract", tier: ProfileTier) -> int:
    return min(
        contract.model_max_token_length,
        next_power_of_two(tier.chunk_max_tokens + PREFIX_HEADROOM_TOKENS),
    )


def tier_fits(contract: "ModelContract", tier: ProfileTier) -> bool:
    """True when the model's context holds this tier's maximal chunk PLUS its
    prefix headroom.

    Testing `_target_seq_len(...) >= tier.chunk_max_tokens` instead would admit
    a tier with zero headroom: a 2048-cap model at the large tier lands on
    chunk_max_tokens == max_token_length == 2048, so every maximal chunk
    truncates the moment its passage prefix is prepended. Headroom is part of
    fitting, not merely of sizing.
    """
    return contract.model_max_token_length >= tier.chunk_max_tokens + PREFIX_HEADROOM_TOKENS


def fitting_tiers(contract: "ModelContract") -> list[ProfileTier]:
    """Tiers this model can hold, in registration order. May be empty."""
    return [t for t in ProfileTierRegistry.values() if tier_fits(contract, t)]


def derive_document_profile(
    contract: "ModelContract",
    tier_key: str,
    memory_budget_gb: float,
) -> dict[str, int]:
    """Derive the five document-profile knobs from a contract, tier, and budget.

    Model-shaped values come from the contract; corpus-shaped values from the
    tier. No model names appear here, so registering a model needs no change
    to this function or to `init`.

    Returns a plain dict rather than a TuningProfile: core/ must not import
    config. The caller (services layer) constructs the TuningProfile.
    """
    tier = ProfileTierRegistry.get(tier_key)
    if not tier_fits(contract, tier):
        raise ValueError(
            f"Model '{contract.model_name}' has a "
            f"{contract.model_max_token_length}-token context; granularity "
            f"'{tier_key}' needs {tier.chunk_max_tokens} plus "
            f"{PREFIX_HEADROOM_TOKENS} tokens of prefix headroom. Choose "
            f"another model or a smaller granularity."
        )
    seq_len = _target_seq_len(contract, tier)
    recommended = recommend_profile(
        contract,
        memory_budget_gb,
        target_chunker="document",
        target_seq_len=seq_len,
    )
    # POSTCONDITION. recommend_profile HALVES seq_len when no batch fits the
    # budget, so a tight budget can hand back a max_token_length below the
    # tier's chunk_max_tokens - measured: granite-r2/large at 0.02 GB returns
    # 1024 against a 2048 chunk_max, violating rule 8, and 0.03-0.1 GB returns
    # exactly 2048, i.e. zero prefix headroom. Refuse cleanly here rather than
    # emit a profile that fails validation with a confusing message.
    granted = int(recommended["max_token_length"])
    if granted < tier.chunk_max_tokens + PREFIX_HEADROOM_TOKENS:
        raise ValueError(
            f"A {memory_budget_gb:.2f} GB memory budget only supports a "
            f"{granted}-token context, but granularity '{tier_key}' needs "
            f"{tier.chunk_max_tokens + PREFIX_HEADROOM_TOKENS} "
            f"({tier.chunk_max_tokens} plus {PREFIX_HEADROOM_TOKENS} tokens of "
            f"prefix headroom). Choose a smaller granularity, or raise "
            f"system.memory_budget_gb."
        )
    return {
        "max_token_length": granted,
        "chunk_max_chars": int(recommended["chunk_max_chars"]),
        # The tier value is a CAP: recommend_profile returns the largest batch
        # that fits (~785 at seq 2048 on 21 GB), which is an upper bound, not
        # a sane default.
        "batch_size": min(tier.batch_size, int(recommended["batch_size"])),
        "chunk_target_tokens": tier.chunk_target_tokens,
        "chunk_max_tokens": tier.chunk_max_tokens,
    }
