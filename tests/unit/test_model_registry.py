import pytest

from dbs_vector.core.model_registry import ModelContract, ModelRegistry


def test_get_returns_gemma_contract():
    contract = ModelRegistry.get("gemma-bf16")
    assert contract.model_name == "mlx-community/embeddinggemma-300m-bf16"
    assert contract.vector_dimension == 768
    assert contract.model_max_token_length == 2048
    assert contract.attention_mask_dtype == "float16"
    assert contract.compute_dtype_bytes == 2


def test_get_returns_granite_contract():
    contract = ModelRegistry.get("granite-r2")
    assert contract.model_name == "ibm-granite/granite-embedding-311m-multilingual-r2"
    assert contract.vector_dimension == 768
    assert contract.model_max_token_length == 32768
    assert contract.attention_mask_dtype is None
    assert contract.compute_dtype_bytes == 2


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown model contract 'nonexistent'"):
        ModelRegistry.get("nonexistent")


def test_get_unknown_lists_known_keys():
    with pytest.raises(KeyError, match=r"Known: \['gemma-bf16', 'granite-r2'\]"):
        ModelRegistry.get("nonexistent")


def test_register_duplicate_raises_valueerror():
    duplicate = ModelContract(
        model_name="x",
        vector_dimension=1,
        model_max_token_length=1,
        attention_mask_dtype=None,
        compute_dtype_bytes=2,
    )
    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry.register("gemma-bf16", duplicate)


def test_keys_returns_sorted():
    keys = ModelRegistry.keys()
    assert "gemma-bf16" in keys
    assert "granite-r2" in keys
    assert keys == sorted(keys)
