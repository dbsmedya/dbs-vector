import threading
from typing import Any

import numpy as np
from loguru import logger
from mlx_embeddings.utils import load
from numpy.typing import NDArray

_MODEL_CACHE: dict[str, tuple[Any, Any, threading.Lock]] = {}


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
    ) -> None:
        self._model_name = model_name
        self._max_token_length = max_token_length
        self._dimension = dimension
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix

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
        import mlx.core as mx

        with self._lock:
            # We call the underlying transformers tokenizer directly to obtain the attention_mask.
            # Some models (like Gemma bf16) require the mask to be cast to bfloat16 to avoid
            # type promotion errors during inference.
            inputs = self.tokenizer._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_token_length,
                return_tensors="mlx",
            )

            if hasattr(inputs, "attention_mask"):
                inputs["attention_mask"] = inputs["attention_mask"].astype(mx.float16)

            # We pass the input_ids as the first positional argument and attention_mask as a keyword.
            outputs = self.model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))

            if hasattr(outputs, "text_embeds"):
                embeds_mlx = outputs.text_embeds
            else:
                embeds_mlx = outputs["text_embeds"]

            # Unified Memory mapping (Forces MLX Lazy Evaluation)
            # Note: This involves a memcpy within Unified Memory due to different allocators.
            vectors_np: NDArray[np.float32] = np.array(embeds_mlx).astype(np.float32)
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
