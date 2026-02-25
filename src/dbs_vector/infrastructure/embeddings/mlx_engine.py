import threading
from typing import Any

import numpy as np
from mlx_embeddings.utils import load
from numpy.typing import NDArray


class MLXEmbedder:
    """
    Concrete implementation of IEmbedder using Apple MLX.
    Forces lazy tensor evaluation and returns contiguous NumPy arrays via Unified Memory.
    """

    def __init__(self, model_name: str, max_token_length: int = 512, dimension: int = 384) -> None:
        self._model_name = model_name
        self._max_token_length = max_token_length
        self._dimension = dimension
        self._lock = threading.Lock()

        print(f"Loading MLX model: {model_name}...")
        self.model: Any
        self.tokenizer: Any
        self.model, self.tokenizer = load(model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _execute_mlx(self, texts: list[str]) -> NDArray[np.float32]:
        """Internal helper to tokenize, run the MLX model, and extract the tensor."""
        with self._lock:
            inputs = self.tokenizer.encode(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_token_length,
                return_tensors="mlx",
            )
            outputs = self.model(inputs)

            if hasattr(outputs, "text_embeds"):
                embeds_mlx = outputs.text_embeds
            else:
                embeds_mlx = outputs["text_embeds"]

            # Unified Memory mapping (Forces MLX Lazy Evaluation)
            vectors_np: NDArray[np.float32] = np.array(embeds_mlx).astype(np.float32)
        return vectors_np

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Embeds a batch of texts safely, handling failures per-batch."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        try:
            vectors = self._execute_mlx(texts)
            return vectors
        except Exception as e:
            print(f"Error embedding batch: {e}")
            raise

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embeds a single query safely, enforcing exact tensor shapes."""
        if not text.strip():
            raise ValueError("Query text cannot be empty.")

        vectors = self._execute_mlx([text])
        query_vector: NDArray[np.float32] = vectors[0]

        # Critical structural guarantee
        if query_vector.shape != (self._dimension,):
            raise ValueError(f"Expected ({self._dimension},), got {query_vector.shape}")

        return query_vector
