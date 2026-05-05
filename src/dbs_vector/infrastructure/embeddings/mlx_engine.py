import threading
from typing import Any

import mlx.core as mx
import numpy as np
from loguru import logger
from mlx_embeddings.utils import load
from numpy.typing import NDArray

_MODEL_CACHE: dict[str, tuple[Any, Any, threading.Lock]] = {}

_MASK_DTYPE_MAP: dict[str, Any] = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


class MLXEmbedder:
    """
    Concrete implementation of IEmbedder using Apple MLX.
    Forces lazy tensor evaluation and returns contiguous NumPy arrays via Unified Memory.
    """

    def __init__(
        self,
        model_name: str,
        max_token_length: int,
        dimension: int,
        passage_prefix: str = "",
        query_prefix: str = "",
        attention_mask_dtype: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._max_token_length = max_token_length
        self._dimension = dimension
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix
        self._attention_mask_dtype = attention_mask_dtype

        global _MODEL_CACHE
        if model_name not in _MODEL_CACHE:
            logger.info("Loading MLX model: {}", model_name)
            _MODEL_CACHE[model_name] = (*load(model_name), threading.Lock())
        else:
            logger.debug("Using cached MLX model: {}", model_name)

        self.model: Any
        self.tokenizer: Any
        self.model, self.tokenizer, self._lock = _MODEL_CACHE[model_name]

    @property
    def dimension(self) -> int:
        return self._dimension

    def _execute_mlx(self, texts: list[str]) -> NDArray[np.float32]:
        """Internal helper to tokenize, run the MLX model, and extract the tensor."""
        with self._lock:
            # Pre-tokenize without truncation/padding to detect over-budget inputs.
            # Cost: one extra fast tokenizer pass; negligible vs. the model forward.
            no_trunc = self.tokenizer._tokenizer(
                texts,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            lengths = [len(ids) for ids in no_trunc["input_ids"]]
            max_len = max(lengths) if lengths else 0
            if max_len > self._max_token_length:
                over_count = sum(1 for n in lengths if n > self._max_token_length)
                logger.warning(
                    "Truncating {}/{} inputs above max_token_length={} for model '{}' "
                    "(longest observed: {} tokens, includes task prefix). "
                    "Consider raising max_token_length or lowering chunk_max_chars.",
                    over_count,
                    len(texts),
                    self._max_token_length,
                    self._model_name,
                    max_len,
                )

            # Existing tokenizer call — performs truncation+padding for the model.
            inputs = self.tokenizer._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_token_length,
                return_tensors="mlx",
            )

            try:
                if self._attention_mask_dtype and "attention_mask" in inputs:
                    if self._attention_mask_dtype not in _MASK_DTYPE_MAP:
                        raise ValueError(
                            f"Unsupported attention_mask_dtype '{self._attention_mask_dtype}'. "
                            f"Allowed: {list(_MASK_DTYPE_MAP)}"
                        )
                    inputs["attention_mask"] = inputs["attention_mask"].astype(
                        _MASK_DTYPE_MAP[self._attention_mask_dtype]
                    )

                outputs = self.model(
                    inputs["input_ids"], attention_mask=inputs.get("attention_mask")
                )
                embeds_mlx = (
                    outputs.text_embeds
                    if hasattr(outputs, "text_embeds")
                    else outputs["text_embeds"]
                )
                # np.array(...) forces MLX lazy evaluation. Type-promotion errors can
                # surface here rather than at the model() call.
                vectors_np: NDArray[np.float32] = np.array(embeds_mlx).astype(np.float32)
            except Exception as e:
                # "promote" matches "Cannot promote types: …" — the MLX type-promotion
                # error string. No more-specific exception class is available from MLX.
                if "promote" in str(e).lower():
                    raise RuntimeError(
                        f"MLX type-promotion error while running model '{self._model_name}'. "
                        f"This usually means the model requires the attention_mask cast "
                        f"to a specific dtype. Set `attention_mask_dtype` in this engine's "
                        f'block in config.yaml — try "float16" (common for bf16 models '
                        f'like embeddinggemma) or "bfloat16". Original error: {e}'
                    ) from e
                raise
        return vectors_np

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Embeds a batch of texts safely, prepending the passage prefix for asymmetric models."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        prefixed_texts = [f"{self._passage_prefix}{text}" for text in texts]

        try:
            vectors = self._execute_mlx(prefixed_texts)
            return vectors
        except Exception as e:
            logger.error("Error embedding batch: {}", e)
            raise

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embeds a single query safely, prepending the query prefix for asymmetric models."""
        if not text.strip():
            raise ValueError("Query text cannot be empty.")

        prefixed_text = f"{self._query_prefix}{text}"
        vectors = self._execute_mlx([prefixed_text])
        query_vector: NDArray[np.float32] = vectors[0]

        # Critical structural guarantee
        if query_vector.shape != (self._dimension,):
            raise ValueError(f"Expected ({self._dimension},), got {query_vector.shape}")

        return query_vector
