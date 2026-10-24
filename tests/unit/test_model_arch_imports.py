"""Each built-in model's mlx-embeddings architecture module must import.

mlx-embeddings resolves a model's architecture with a dynamic import and reports
any ImportError as "Model type ... not supported", so a missing transitive
dependency only surfaces when the model loads. mlx-embeddings 0.1.0 imports
mlx_lm from gemma3_text without declaring it; mlx-lm used to arrive through
mlx-vlm 0.5 and silently vanished with mlx-vlm 0.7.
"""

import importlib

import pytest

from dbs_vector.core.model_registry import ModelRegistry

# Registry key -> mlx-embeddings architecture module (the model's config.json
# `model_type`). Add an entry whenever a built-in model is registered.
_ARCH_MODULES = {
    "gemma-bf16": "gemma3_text",
    "granite-r2": "modernbert",
}


@pytest.mark.parametrize("key", sorted(_ARCH_MODULES))
def test_architecture_module_imports(key: str) -> None:
    ModelRegistry.get(key)
    importlib.import_module(f"mlx_embeddings.models.{_ARCH_MODULES[key]}")
