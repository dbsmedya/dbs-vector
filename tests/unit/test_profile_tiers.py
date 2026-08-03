import pytest

from dbs_vector.core.model_registry import ModelContract, ModelRegistry
from dbs_vector.core.profile_math import (
    PREFIX_HEADROOM_TOKENS,
    ProfileTierRegistry,
    derive_document_profile,
    fitting_tiers,
    next_power_of_two,
    tier_fits,
)

BUDGET = 21.0  # fixed, so tests never depend on the host's Metal budget


def test_next_power_of_two():
    assert next_power_of_two(1) == 1
    assert next_power_of_two(1280) == 2048
    assert next_power_of_two(2048) == 2048
    assert next_power_of_two(2049) == 4096


def test_registry_keys_are_in_size_order_not_alphabetical():
    """Alphabetical would render the prompt as large, medium, small."""
    assert ProfileTierRegistry.keys() == ["small", "medium", "large"]


def test_tier_rows_match_the_shipped_profiles():
    assert ProfileTierRegistry.get("small").chunk_target_tokens == 512
    assert ProfileTierRegistry.get("small").chunk_max_tokens == 1024
    assert ProfileTierRegistry.get("small").batch_size == 64
    assert ProfileTierRegistry.get("medium").chunk_target_tokens == 768
    assert ProfileTierRegistry.get("medium").chunk_max_tokens == 1536
    assert ProfileTierRegistry.get("medium").batch_size == 16
    assert ProfileTierRegistry.get("large").chunk_target_tokens == 1024
    assert ProfileTierRegistry.get("large").chunk_max_tokens == 2048
    assert ProfileTierRegistry.get("large").batch_size == 8


def test_get_unknown_tier_raises():
    with pytest.raises(KeyError, match="Unknown profile tier 'huge'"):
        ProfileTierRegistry.get("huge")


def test_derive_reproduces_shipped_gemma_md():
    """gemma-bf16 + small must reproduce the shipped `gemma-md` numbers."""
    contract = ModelRegistry.get("gemma-bf16")
    profile = derive_document_profile(contract, "small", BUDGET)
    assert profile["max_token_length"] == 2048
    assert profile["batch_size"] == 64
    assert profile["chunk_target_tokens"] == 512
    assert profile["chunk_max_tokens"] == 1024


def test_derive_reproduces_shipped_granite_md_medium():
    """granite-r2 + medium must reproduce the shipped `granite-md-medium` numbers."""
    contract = ModelRegistry.get("granite-r2")
    profile = derive_document_profile(contract, "medium", BUDGET)
    assert profile["max_token_length"] == 2048
    assert profile["batch_size"] == 16
    assert profile["chunk_target_tokens"] == 768
    assert profile["chunk_max_tokens"] == 1536


def test_derive_clamps_max_token_length_to_model_cap():
    """granite's large tier wants 4096 and its 32768 cap allows it; gemma's
    2048 cap clamps the medium tier's 1792 preference up to 2048."""
    assert (
        derive_document_profile(ModelRegistry.get("granite-r2"), "large", BUDGET)[
            "max_token_length"
        ]
        == 4096
    )
    assert (
        derive_document_profile(ModelRegistry.get("gemma-bf16"), "medium", BUDGET)[
            "max_token_length"
        ]
        == 2048
    )


def test_gemma_cannot_hold_the_large_tier():
    """2048 cap < 2048 chunk_max + 256 headroom. Admitting this tier would
    produce chunk_max_tokens == max_token_length and truncate every maximal
    chunk once its prefix is prepended."""
    contract = ModelRegistry.get("gemma-bf16")
    assert [t.key for t in fitting_tiers(contract)] == ["small", "medium"]
    with pytest.raises(ValueError, match="prefix headroom"):
        derive_document_profile(contract, "large", BUDGET)


def test_derive_leaves_prefix_headroom_for_every_fitting_tier():
    """chunk_max_tokens must stay strictly below max_token_length: the passage
    prefix is prepended after chunking and would otherwise truncate."""
    for model_key in ModelRegistry.keys():
        contract = ModelRegistry.get(model_key)
        for tier in fitting_tiers(contract):
            profile = derive_document_profile(contract, tier.key, BUDGET)
            assert profile["chunk_max_tokens"] < profile["max_token_length"], (
                f"{model_key}/{tier.key} has zero prefix headroom"
            )


def test_batch_size_is_a_cap_not_a_target():
    """recommend_profile returns the LARGEST batch that fits (~785 at seq 2048
    on a 21 GB budget). The tier value must win."""
    contract = ModelRegistry.get("granite-r2")
    profile = derive_document_profile(contract, "medium", BUDGET)
    assert profile["batch_size"] == 16


def test_batch_size_downshifts_when_budget_is_tight():
    contract = ModelRegistry.get("granite-r2")
    profile = derive_document_profile(contract, "small", memory_budget_gb=1.0)
    assert profile["batch_size"] < 64


@pytest.mark.parametrize("budget", [0.1, 0.05, 0.02, 0.01])
def test_a_budget_too_tight_for_the_tier_is_refused_not_silently_shrunk(budget):
    """recommend_profile halves seq_len when no batch fits. Measured for
    granite-r2/large: 0.03-0.1 GB returns exactly 2048 (zero headroom) and
    <=0.02 GB returns 1024 (below chunk_max_tokens, a rule-8 violation).
    Both must refuse rather than emit a profile that fails validation."""
    contract = ModelRegistry.get("granite-r2")
    with pytest.raises(ValueError, match="prefix headroom"):
        derive_document_profile(contract, "large", memory_budget_gb=budget)


def test_a_generous_budget_still_grants_the_large_tier():
    contract = ModelRegistry.get("granite-r2")
    profile = derive_document_profile(contract, "large", memory_budget_gb=1.0)
    assert profile["max_token_length"] == 4096


def test_open_closed_unregistered_model_needs_no_init_change(temp_model_registry):
    """A model that does not exist in this codebase derives valid profiles."""
    temp_model_registry.register(
        "qwen3-embed-8b",
        ModelContract(
            model_name="mlx-community/qwen3-embedding-8b",
            vector_dimension=4096,
            model_max_token_length=40960,
            attention_mask_dtype=None,
            compute_dtype_bytes=2,
        ),
    )
    contract = temp_model_registry.get("qwen3-embed-8b")
    assert [t.key for t in fitting_tiers(contract)] == ["small", "medium", "large"]
    for key in ProfileTierRegistry.keys():
        profile = derive_document_profile(contract, key, BUDGET)
        assert profile["chunk_max_tokens"] < profile["max_token_length"]
        assert profile["batch_size"] >= 1


def test_tiny_context_model_fits_no_tier(temp_model_registry):
    temp_model_registry.register(
        "tiny-512",
        ModelContract(
            model_name="fake/tiny",
            vector_dimension=384,
            model_max_token_length=512,
            attention_mask_dtype=None,
        ),
    )
    contract = temp_model_registry.get("tiny-512")
    assert fitting_tiers(contract) == []
    assert tier_fits(contract, ProfileTierRegistry.get("small")) is False


def test_derive_refuses_a_tier_the_model_cannot_hold(temp_model_registry):
    temp_model_registry.register(
        "tiny-512b",
        ModelContract(
            model_name="fake/tiny",
            vector_dimension=384,
            model_max_token_length=512,
            attention_mask_dtype=None,
        ),
    )
    contract = temp_model_registry.get("tiny-512b")
    with pytest.raises(ValueError, match="512-token context"):
        derive_document_profile(contract, "small", BUDGET)


def test_fitting_tiers_filters_partially_capable_models(temp_model_registry):
    """A 4096-cap model holds small and medium and large; a 2048-cap one drops
    large. This filtering is what prompt 3 renders, so it must be exact."""
    temp_model_registry.register(
        "mid-2048",
        ModelContract(
            model_name="fake/mid",
            vector_dimension=768,
            model_max_token_length=2048,
            attention_mask_dtype=None,
        ),
    )
    temp_model_registry.register(
        "big-4096",
        ModelContract(
            model_name="fake/big",
            vector_dimension=768,
            model_max_token_length=4096,
            attention_mask_dtype=None,
        ),
    )
    assert [t.key for t in fitting_tiers(temp_model_registry.get("mid-2048"))] == [
        "small",
        "medium",
    ]
    assert [t.key for t in fitting_tiers(temp_model_registry.get("big-4096"))] == [
        "small",
        "medium",
        "large",
    ]


def test_prefix_headroom_constant_is_documented():
    assert PREFIX_HEADROOM_TOKENS == 256
