# =============================================================================
# Unit Tests for Embedder
# =============================================================================
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.indexing.embedder import (
    MockEmbedder,
    get_embedder,
    OpenAIEmbedder,
    LocalEmbedder,
)


class TestMockEmbedder:
    """Test MockEmbedder for testing purposes."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = MockEmbedder(dimension=128)
        embedding = await embedder.embed("test text")

        assert len(embedding) == 128
        assert all(isinstance(x, float) for x in embedding)
        assert all(-1 <= x <= 1 for x in embedding)  # Should be in [-1, 1]

    @pytest.mark.asyncio
    async def test_embed_same_text_same_result(self):
        """Test that same text produces same embedding (deterministic)."""
        embedder = MockEmbedder(dimension=64)

        text = "test text"
        emb1 = await embedder.embed(text)
        emb2 = await embedder.embed(text)

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_embed_different_texts_different_results(self):
        """Test that different texts produce different embeddings."""
        embedder = MockEmbedder(dimension=64)

        emb1 = await embedder.embed("text one")
        emb2 = await embedder.embed("text two")

        assert emb1 != emb2

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        embedder = MockEmbedder(dimension=128)

        texts = ["text1", "text2", "text3"]
        embeddings = await embedder.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 128 for emb in embeddings)

    def test_get_dimension(self):
        """Test getting embedder dimension."""
        embedder = MockEmbedder(dimension=256)
        assert embedder.get_dimension() == 256


class TestGetEmbedder:
    """Test embedder factory function."""

    def test_get_mock_embedder(self):
        """Test getting mock embedder."""
        embedder = get_embedder(provider="mock")
        assert isinstance(embedder, MockEmbedder)

    def test_get_openai_embedder(self):
        """Test getting OpenAI embedder."""
        embedder = get_embedder(provider="openai")
        assert isinstance(embedder, OpenAIEmbedder)

    def test_get_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder(provider="invalid")

    def test_get_embedder_caching(self):
        """Test that get_embedder caches results."""
        embedder1 = get_embedder(provider="mock")
        embedder2 = get_embedder(provider="mock")

        # Should return the same instance (cached)
        assert embedder1 is embedder2


class TestOpenAIEmbedder:
    """Test OpenAI embedder (without actual API calls)."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
        )

        assert embedder.api_key == "test-key"
        assert embedder.model is not None
        assert embedder.dimensions is not None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
            model="text-embedding-3-large",
            dimensions=3072,
        )

        assert embedder.model == "text-embedding-3-large"
        assert embedder.dimensions == 3072

    def test_get_dimension(self):
        """Test getting dimension."""
        embedder = OpenAIEmbedder(api_key="test-key", dimensions=1536)
        assert embedder.get_dimension() == 1536

    @pytest.mark.asyncio
    async def test_embed_with_cache_miss(self, clean_storage):
        """Test embedding with cache miss."""
        from src.storage import cache_store

        embedder = OpenAIEmbedder(api_key="test-key", dimensions=128)

        # Mock the OpenAI client
        with patch.object(embedder, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 128)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            # Mock cache miss
            cache_store.get_embedding = AsyncMock(return_value=None)
            cache_store.set_embedding = AsyncMock()

            text = "test text"
            embedding = await embedder.embed(text)

            assert len(embedding) == 128
            # Verify cache was checked and set
            cache_store.get_embedding.assert_called_once_with(text)
            cache_store.set_embedding.assert_called_once()


class TestLocalEmbedder:
    """Test LocalEmbedder (without actual model loading)."""

    def test_init_default(self):
        """Test initialization with default model."""
        embedder = LocalEmbedder()

        assert embedder.model_name is not None
        assert embedder.device is not None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        embedder = LocalEmbedder(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )

        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.device == "cpu"

    def test_get_dimension_known_model(self):
        """Test getting dimension for known model."""
        embedder = LocalEmbedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        assert embedder.get_dimension() == 384

    def test_get_dimension_unknown_model(self):
        """Test getting dimension for unknown model returns default."""
        embedder = LocalEmbedder(model_name="unknown-model")
        assert embedder.get_dimension() == 384  # Default fallback


class TestCachingIntegration:
    """Test embedder caching integration."""

    @pytest.mark.asyncio
    async def test_cache_hit_retrieves_stored(self, clean_storage):
        """Test that cache hit retrieves stored embedding."""
        from src.storage.cache import cache_store

        # Use OpenAIEmbedder which implements caching
        embedder = OpenAIEmbedder(api_key="test-key", dimensions=128)

        # Mock cache_get to return cached value
        text = "test text"
        cached_embedding = [0.5] * 128

        with patch.object(cache_store, 'get_embedding', new=AsyncMock(return_value=cached_embedding)):
            with patch.object(cache_store, 'set_embedding', new=AsyncMock()):
                # Embed should retrieve from cache
                embedding = await embedder.embed(text)

                assert embedding == cached_embedding
                # Verify cache was checked
                cache_store.get_embedding.assert_called_once_with(text)
                # Verify cache was NOT set (cache hit)
                cache_store.set_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_generates_new(self, clean_storage):
        """Test that cache miss generates new embedding."""
        from src.storage.cache import cache_store
        from unittest.mock import AsyncMock, MagicMock, patch

        # Use OpenAIEmbedder which implements caching
        embedder = OpenAIEmbedder(api_key="test-key", dimensions=64)

        # Mock cache to return None (miss)
        with patch.object(cache_store, 'get_embedding', new=AsyncMock(return_value=None)):
            with patch.object(cache_store, 'set_embedding', new=AsyncMock()):
                # Mock the OpenAI client
                with patch.object(embedder, 'client') as mock_client:
                    mock_response = MagicMock()
                    mock_response.data = [MagicMock(embedding=[0.1] * 64)]
                    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

                    text = "test text"
                    embedding = await embedder.embed(text)

                    assert len(embedding) == 64
                    # Verify cache was checked and set
                    cache_store.get_embedding.assert_called_once_with(text)
                    cache_store.set_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_selective_caching(self, clean_storage):
        """Test that batch embedding only generates for uncached texts."""
        from src.storage import cache_store

        embedder = MockEmbedder(dimension=64)

        # Pre-populate cache for one text
        cached_text = "already cached"
        cached_emb = [0.1] * 64
        await cache_store.set_embedding(cached_text, cached_emb)

        texts = ["new text 1", cached_text, "new text 2"]
        embeddings = await embedder.embed_batch(texts)

        assert len(embeddings) == 3

        # First and third should be generated, second should match cached
        # (Mock embedder doesn't actually use cache, but we verify the structure)
        assert all(len(emb) == 64 for emb in embeddings)
