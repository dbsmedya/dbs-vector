"""Unit tests for MLXEmbedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import dbs_vector.infrastructure.embeddings.mlx_engine as mlx_engine_module
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the global model cache before each test to prevent mock leakage."""
    mlx_engine_module._MODEL_CACHE.clear()


@pytest.fixture
def mock_load():
    """Mock the mlx_embeddings.utils.load function."""
    with patch("dbs_vector.infrastructure.embeddings.mlx_engine.load") as mock:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        # Mock the internal transformers tokenizer
        mock_tokenizer._tokenizer = MagicMock()
        mock.return_value = (mock_model, mock_tokenizer)
        yield mock, mock_model, mock_tokenizer


@pytest.fixture
def embedder(mock_load):
    """Create an MLXEmbedder with mocked dependencies."""
    _, mock_model, mock_tokenizer = mock_load
    emb = MLXEmbedder(
        model_name="test-model",
        max_token_length=128,
        dimension=384,
    )
    return emb


class TestInit:
    """Tests for MLXEmbedder initialization."""

    def test_init_loads_model(self, mock_load):
        """Test that initialization loads the model and tokenizer."""
        mock_load_func, mock_model, mock_tokenizer = mock_load

        emb = MLXEmbedder(
            model_name="my-model",
            max_token_length=256,
            dimension=512,
        )

        mock_load_func.assert_called_once_with("my-model")
        assert emb.model is mock_model
        assert emb.tokenizer is mock_tokenizer
        assert emb._model_name == "my-model"
        assert emb._max_token_length == 256
        assert emb._dimension == 512

    def test_dimension_property(self, embedder):
        """Test that dimension property returns correct value."""
        assert embedder.dimension == 384


class TestExecuteMlx:
    """Tests for the _execute_mlx internal method."""

    def test_execute_mlx_with_object_output(self, embedder, mock_load):
        """Test _execute_mlx when model outputs object with text_embeds attribute."""
        _, mock_model, mock_tokenizer = mock_load

        # Setup mock outputs as object with attribute
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_model.return_value = mock_outputs

        # Setup tokenizer side_effect: first call is the pre-check, second is the real tokenize.
        mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [list(range(3))]},  # pre-check
            mock_inputs,  # real tokenize
        ]

        result = embedder._execute_mlx(["test text"])

        # Verify both tokenizer calls happened with the right arguments.
        assert mock_tokenizer._tokenizer.call_count == 2

        # Pre-check call (first): truncation=False, padding=False, add_special_tokens=True
        pre_check_call = mock_tokenizer._tokenizer.call_args_list[0]
        assert pre_check_call.args == (["test text"],)
        assert pre_check_call.kwargs == {
            "padding": False,
            "truncation": False,
            "add_special_tokens": True,
        }

        # Real tokenize call (second): truncation=True, padding=True, max_length, return_tensors
        real_call = mock_tokenizer._tokenizer.call_args_list[1]
        assert real_call.args == (["test text"],)
        assert real_call.kwargs == {
            "padding": True,
            "truncation": True,
            "max_length": 128,
            "return_tensors": "mlx",
        }

        mock_model.assert_called_once()

        # Verify result
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

    def test_execute_mlx_with_dict_output(self, embedder, mock_load):
        """Test _execute_mlx when model outputs dict with text_embeds key."""
        _, mock_model, mock_tokenizer = mock_load

        # Setup mock outputs as dict
        mock_outputs = {"text_embeds": np.array([[0.4, 0.5, 0.6]], dtype=np.float32)}
        mock_model.return_value = mock_outputs

        mock_inputs = {"input_ids": MagicMock()}
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [list(range(3))]},  # pre-check
            mock_inputs,  # real tokenize
        ]

        result = embedder._execute_mlx(["another text"])

        np.testing.assert_array_equal(result, np.array([[0.4, 0.5, 0.6]], dtype=np.float32))

    def test_execute_mlx_thread_safety(self, embedder, mock_load):
        """Test that _execute_mlx uses threading lock."""
        _, mock_model, mock_tokenizer = mock_load

        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1]], dtype=np.float32)
        mock_model.return_value = mock_outputs

        mock_inputs = {"input_ids": MagicMock()}
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [list(range(3))]},  # pre-check
            mock_inputs,  # real tokenize
        ]

        # Mock the lock to verify it's used
        mock_lock = MagicMock()
        embedder._lock = mock_lock

        embedder._execute_mlx(["test"])

        # Verify lock was used
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestEmbedBatch:
    """Tests for the embed_batch method."""

    def test_embed_batch_empty_list(self, embedder):
        """Test embed_batch with empty list returns empty array."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            result = embedder.embed_batch([])

            mock_execute.assert_not_called()
            assert isinstance(result, np.ndarray)
            assert result.shape == (0, 384)
            assert result.dtype == np.float32

    def test_embed_batch_single_text(self, embedder):
        """Test embed_batch with single text."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            mock_execute.return_value = np.array([[0.1] * 384], dtype=np.float32)

            result = embedder.embed_batch(["hello world"])

            mock_execute.assert_called_once_with(["hello world"])
            assert result.shape == (1, 384)

    def test_embed_batch_multiple_texts(self, embedder):
        """Test embed_batch with multiple texts."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            mock_vectors = np.random.rand(3, 384).astype(np.float32)
            mock_execute.return_value = mock_vectors

            result = embedder.embed_batch(["text1", "text2", "text3"])

            mock_execute.assert_called_once_with(["text1", "text2", "text3"])
            np.testing.assert_array_equal(result, mock_vectors)

    def test_embed_batch_error_handling(self, embedder):
        """Test that embed_batch raises exception on error."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            mock_execute.side_effect = RuntimeError("MLX error")

            with pytest.raises(RuntimeError, match="MLX error"):
                embedder.embed_batch(["test"])

    def test_embed_batch_with_prefix(self, mock_load):
        """Test that passage_prefix is prepended correctly."""
        _, _, _ = mock_load
        emb = MLXEmbedder(
            model_name="test",
            max_token_length=128,
            dimension=384,
            passage_prefix="passage: ",
        )

        with patch.object(emb, "_execute_mlx") as mock_execute:
            mock_execute.return_value = np.random.rand(2, 384).astype(np.float32)
            emb.embed_batch(["t1", "t2"])
            mock_execute.assert_called_once_with(["passage: t1", "passage: t2"])


class TestEmbedQuery:
    """Tests for the embed_query method."""

    def test_embed_query_basic(self, embedder):
        """Test embed_query with valid text."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            mock_vector = np.random.rand(1, 384).astype(np.float32)
            mock_execute.return_value = mock_vector

            result = embedder.embed_query("search query")

            mock_execute.assert_called_once_with(["search query"])
            assert result.shape == (384,)
            np.testing.assert_array_equal(result, mock_vector[0])

    def test_embed_query_empty_string_raises(self, embedder):
        """Test that embed_query raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            embedder.embed_query("")

    def test_embed_query_whitespace_only_raises(self, embedder):
        """Test that embed_query raises ValueError for whitespace-only string."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            embedder.embed_query("   \t\n  ")

    def test_embed_query_wrong_shape_raises(self, embedder):
        """Test that embed_query raises ValueError if shape doesn't match dimension."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            # Return wrong shape (385 instead of 384)
            mock_execute.return_value = np.random.rand(1, 385).astype(np.float32)

            with pytest.raises(ValueError, match="Expected \\(384,\\), got"):
                embedder.embed_query("test query")

    def test_embed_query_shape_validation_exact(self, embedder):
        """Test that embed_query validates exact shape match."""
        with patch.object(embedder, "_execute_mlx") as mock_execute:
            # Return shape (384,) directly - should fail because we expect [0] indexing
            mock_execute.return_value = np.random.rand(1, 384).astype(np.float32)

            result = embedder.embed_query("test")

            # Result should be 1D array with correct shape
            assert result.shape == (384,)

    def test_embed_query_with_prefix(self, mock_load):
        """Test that query_prefix is prepended correctly."""
        _, _, _ = mock_load
        emb = MLXEmbedder(
            model_name="test",
            max_token_length=128,
            dimension=384,
            query_prefix="query: ",
        )

        with patch.object(emb, "_execute_mlx") as mock_execute:
            mock_execute.return_value = np.random.rand(1, 384).astype(np.float32)
            emb.embed_query("search")
            mock_execute.assert_called_once_with(["query: search"])


class TestIntegrationScenarios:
    """Integration-style tests with mocked MLX."""

    def test_end_to_end_batch_embedding(self, mock_load):
        """Test full batch embedding workflow."""
        _, mock_model, mock_tokenizer = mock_load

        # Create embedder
        embedder = MLXEmbedder(
            model_name="integration-model",
            max_token_length=512,
            dimension=768,
        )

        # Setup realistic mock behavior
        def mock_encode(texts, **kwargs):
            return {"input_ids": np.array([[1, 2, 3]] * len(texts))}

        def mock_forward(inputs, **kwargs):
            # Return embeddings matching batch size and dimension
            batch_size = len(inputs) if isinstance(inputs, np.ndarray) else 1
            mock_embeds = np.random.rand(batch_size, 768).astype(np.float32)
            outputs = MagicMock()
            outputs.text_embeds = mock_embeds
            return outputs

        mock_tokenizer._tokenizer.side_effect = mock_encode
        mock_model.side_effect = mock_forward

        # Test batch embedding
        texts = ["query one", "query two", "query three"]
        result = embedder.embed_batch(texts)

        assert result.shape == (3, 768)
        assert result.dtype == np.float32

    def test_end_to_end_query_embedding(self, mock_load):
        """Test full single query embedding workflow."""
        _, mock_model, mock_tokenizer = mock_load

        embedder = MLXEmbedder(
            model_name="query-model",
            max_token_length=256,
            dimension=384,
        )

        # Setup mock
        expected_vector = np.random.rand(384).astype(np.float32)
        outputs = MagicMock()
        outputs.text_embeds = expected_vector.reshape(1, 384)
        mock_model.return_value = outputs

        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [list(range(3))]},  # pre-check
            {"input_ids": np.array([[1, 2, 3]])},  # real tokenize
        ]

        result = embedder.embed_query("what is the meaning of life?")

        np.testing.assert_array_equal(result, expected_vector)


class TestTruncationAlarm:
    """Tests for the truncation alarm in _execute_mlx."""

    def test_truncation_warning_emitted_when_over_budget(self, embedder, mock_load, caplog):
        """Warning fires when at least one input exceeds max_token_length."""
        _, mock_model, mock_tokenizer = mock_load

        # First tokenizer call (no truncation) returns long input_ids; second call (with
        # truncation) returns the trimmed version. _execute_mlx makes both calls.
        long_ids = list(range(200))  # 200 tokens — exceeds embedder's 128 limit
        short_ids = list(range(50))  # 50 tokens — within limit
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [long_ids, short_ids]},  # pre-check pass
            {"input_ids": MagicMock(), "attention_mask": MagicMock()},  # real tokenize
        ]
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        mock_model.return_value = mock_outputs

        with caplog.at_level("WARNING"):
            embedder._execute_mlx(["long text", "short text"])

        assert any("Truncating 1/2" in rec.message for rec in caplog.records)
        assert any("max_token_length=128" in rec.message for rec in caplog.records)
        assert any("test-model" in rec.message for rec in caplog.records)
        assert any("longest observed: 200" in rec.message for rec in caplog.records)

    def test_no_warning_when_all_inputs_under_budget(self, embedder, mock_load, caplog):
        """No warning when every input is within max_token_length."""
        _, mock_model, mock_tokenizer = mock_load

        short_ids_a = list(range(10))
        short_ids_b = list(range(20))
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [short_ids_a, short_ids_b]},  # pre-check pass
            {"input_ids": MagicMock(), "attention_mask": MagicMock()},  # real tokenize
        ]
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1], [0.2]], dtype=np.float32)
        mock_model.return_value = mock_outputs

        with caplog.at_level("WARNING"):
            embedder._execute_mlx(["a", "b"])

        assert not any("Truncating" in rec.message for rec in caplog.records)

    def test_no_warning_when_input_at_exact_limit(self, embedder, mock_load, caplog):
        """Predicate is strict >; an input of exactly max_token_length must not warn."""
        _, mock_model, mock_tokenizer = mock_load

        at_limit_ids = list(range(128))  # exactly equals embedder's max_token_length=128
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [at_limit_ids]},
            {"input_ids": MagicMock(), "attention_mask": MagicMock()},
        ]
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1]], dtype=np.float32)
        mock_model.return_value = mock_outputs

        with caplog.at_level("WARNING"):
            embedder._execute_mlx(["x"])

        assert not any("Truncating" in rec.message for rec in caplog.records)


class TestAttentionMaskDtype:
    """Tests for config-driven attention-mask dtype + promotion-error mapping."""

    def _setup_tokenizer(self, mock_tokenizer, attention_mask_value):
        """Helper: tokenizer returns short ids on the pre-check, then the real inputs."""
        mock_tokenizer._tokenizer.side_effect = [
            {"input_ids": [list(range(5))]},
            {"input_ids": MagicMock(), "attention_mask": attention_mask_value},
        ]

    def test_dtype_none_does_not_cast(self, mock_load):
        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="granite",
            max_token_length=128,
            dimension=384,
            attention_mask_dtype=None,
        )
        mask = MagicMock()
        self._setup_tokenizer(mock_tokenizer, mask)
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1] * 384], dtype=np.float32)
        mock_model.return_value = mock_outputs

        emb._execute_mlx(["x"])

        mask.astype.assert_not_called()

    def test_dtype_float16_casts_mask(self, mock_load):
        import mlx.core as mx

        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="gemma",
            max_token_length=128,
            dimension=384,
            attention_mask_dtype="float16",
        )
        mask = MagicMock()
        mask.astype.return_value = MagicMock()
        self._setup_tokenizer(mock_tokenizer, mask)
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = np.array([[0.1] * 384], dtype=np.float32)
        mock_model.return_value = mock_outputs

        emb._execute_mlx(["x"])

        mask.astype.assert_called_once_with(mx.float16)

    def test_invalid_dtype_raises_value_error(self, mock_load):
        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="weird",
            max_token_length=128,
            dimension=384,
            attention_mask_dtype="int8",
        )
        self._setup_tokenizer(mock_tokenizer, MagicMock())

        with pytest.raises(ValueError, match="Unsupported attention_mask_dtype 'int8'"):
            emb._execute_mlx(["x"])

        mock_model.assert_not_called()

    def test_promote_error_in_forward_remapped_to_config_hint(self, mock_load):
        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="gemma",
            max_token_length=128,
            dimension=384,
            attention_mask_dtype=None,
        )
        self._setup_tokenizer(mock_tokenizer, MagicMock())
        mock_model.side_effect = ValueError("Cannot promote types: bf16, int32")

        with pytest.raises(RuntimeError, match="attention_mask_dtype"):
            emb._execute_mlx(["x"])

    def test_promote_error_during_materialization_remapped(self, mock_load):
        """Lazy MLX evaluation can defer the promotion error to np.array(...)."""
        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="gemma",
            max_token_length=128,
            dimension=384,
            attention_mask_dtype=None,
        )
        self._setup_tokenizer(mock_tokenizer, MagicMock())

        # Forward "succeeds" but returns an MLX-like object whose materialization fails.
        # Mirror the NumPy 2.x __array__(dtype, copy) signature so the mock doesn't
        # raise a DeprecationWarning before the type-promotion error we're testing.
        class _LazyEmbeds:
            def __array__(self, dtype=None, copy=None):
                raise ValueError("Cannot promote types: bf16, int32")

        mock_outputs = MagicMock()
        mock_outputs.text_embeds = _LazyEmbeds()
        mock_model.return_value = mock_outputs

        with pytest.raises(RuntimeError, match="attention_mask_dtype"):
            emb._execute_mlx(["x"])

    def test_bfloat16_text_embeds_converted_without_buffer_error(self, mock_load):
        """Regression: granite-r2 returns bf16 text_embeds; NumPy 2.x can't bridge
        bf16 via the buffer protocol, so the embedder must cast to fp32 in MLX
        space first. Without the cast this raises:
        'Item size 2 for PEP 3118 buffer format string B does not match the
        dtype B item size 1.'"""
        import mlx.core as mx

        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="granite",
            max_token_length=128,
            dimension=3,
            attention_mask_dtype=None,
        )
        self._setup_tokenizer(mock_tokenizer, MagicMock())
        mock_outputs = MagicMock()
        mock_outputs.text_embeds = mx.array([[1.5, 2.5, 3.5]], dtype=mx.bfloat16)
        mock_model.return_value = mock_outputs

        result = emb._execute_mlx(["x"])

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, [[1.5, 2.5, 3.5]], rtol=1e-2)

    def test_unrelated_error_passes_through(self, mock_load):
        _, mock_model, mock_tokenizer = mock_load
        emb = MLXEmbedder(
            model_name="m",
            max_token_length=128,
            dimension=384,
            attention_mask_dtype=None,
        )
        self._setup_tokenizer(mock_tokenizer, MagicMock())
        mock_model.side_effect = RuntimeError("disk full")

        with pytest.raises(RuntimeError, match="disk full"):
            emb._execute_mlx(["x"])


class TestCountTokens:
    """Tests for the count_tokens method."""

    def test_count_tokens_returns_input_id_length(self, mock_load):
        _, _, mock_tokenizer = mock_load
        mock_tokenizer._tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        embedder = MLXEmbedder(model_name="m", max_token_length=2048, dimension=768)
        assert embedder.count_tokens("hello world") == 5
        # add_special_tokens must be requested so the count matches embedding time
        _, kwargs = mock_tokenizer._tokenizer.call_args
        assert kwargs.get("add_special_tokens") is True
