from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContract:
    """Immutable model contract. All fields are properties of the model itself,
    not of any particular deployment that uses the model."""

    model_name: str
    vector_dimension: int
    model_max_token_length: int
    attention_mask_dtype: str | None
    compute_dtype_bytes: int = 2

    # Prefix defaults are a property of the MODEL: an instruction-tuned
    # encoder needs them, a symmetric bi-encoder does not. Carrying them here
    # (rather than in a lookup keyed by model name) is what lets `init` skip
    # the prefix prompt automatically for models that need none, and what
    # keeps a newly registered model from silently getting no prefixes.
    # Engines may still override both in config.yaml — the same model is used
    # with different prefixes by different workflows.
    default_passage_prefix: str = ""
    default_query_prefix: str = ""


class ModelRegistry:
    """Open/closed registry of model contracts. Adding a model = register() call."""

    _models: dict[str, ModelContract] = {}

    @classmethod
    def register(cls, key: str, contract: ModelContract) -> None:
        if key in cls._models:
            raise ValueError(f"Model contract '{key}' already registered")
        cls._models[key] = contract

    @classmethod
    def get(cls, key: str) -> ModelContract:
        if key not in cls._models:
            known = sorted(cls._models)
            raise KeyError(f"Unknown model contract '{key}'. Known: {known}")
        return cls._models[key]

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._models)


# Built-in registrations
ModelRegistry.register(
    "gemma-bf16",
    ModelContract(
        model_name="mlx-community/embeddinggemma-300m-bf16",
        vector_dimension=768,
        model_max_token_length=2048,
        attention_mask_dtype="float16",
        compute_dtype_bytes=2,
        default_passage_prefix="title: none | text: ",
        default_query_prefix="task: search result | query: ",
    ),
)

ModelRegistry.register(
    "granite-r2",
    ModelContract(
        model_name="ibm-granite/granite-embedding-311m-multilingual-r2",
        vector_dimension=768,
        model_max_token_length=32768,
        attention_mask_dtype=None,
        compute_dtype_bytes=2,
    ),
)
