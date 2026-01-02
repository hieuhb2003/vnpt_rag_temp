# =============================================================================
# Embedding Service - Multi-Provider Embedding with Caching
# =============================================================================
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config.settings import get_settings
from src.storage.cache import cache_store
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass

    async def close(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding service with Redis caching.

    Supports:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key (from settings if None)
            model: Model name (from settings if None)
            dimensions: Embedding dimensions (from settings if None)
        """
        super().__init__()
        from openai import AsyncOpenAI

        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            "Initialized OpenAIEmbedder",
            model=self.model,
            dimensions=self.dimensions
        )

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        cached = await cache_store.get_embedding(text)
        if cached:
            logger.debug("Embedding cache hit", text_length=len(text))
            return cached

        # Generate embedding
        logger.debug("Embedding cache miss, generating", text_length=len(text))
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions if "3-" in self.model else None
            )
            embedding = response.data[0].embedding

            # Cache result
            await cache_store.set_embedding(text, embedding)
            logger.debug("Generated and cached embedding", dimension=len(embedding))

            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing and caching.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(
            "Batch embedding started",
            text_count=len(texts),
            batch_size=batch_size
        )

        # Check cache for all texts
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = await cache_store.get_embedding(text)
            if cached:
                embeddings[i] = cached
            else:
                uncached_indices.append(i)

        logger.info(
            "Cache check complete",
            cached=len(texts) - len(uncached_indices),
            uncached=len(uncached_indices)
        )

        # Process uncached texts in batches
        if uncached_indices:
            for offset in range(0, len(uncached_indices), batch_size):
                batch_indices = uncached_indices[offset:offset + batch_size]
                batch_texts = [texts[i] for i in batch_indices]

                try:
                    # Generate embeddings for batch
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        dimensions=self.dimensions if "3-" in self.model else None
                    )

                    # Update embeddings and cache
                    for idx, result in zip(batch_indices, response.data):
                        embedding = result.embedding
                        embeddings[idx] = embedding
                        await cache_store.set_embedding(batch_texts[batch_indices.index(idx)], embedding)

                    logger.debug(
                        "Batch embedded successfully",
                        batch_size=len(batch_indices),
                        offset=offset
                    )

                except Exception as e:
                    logger.error(
                        "Failed to embed batch",
                        offset=offset,
                        batch_size=len(batch_indices),
                        error=str(e)
                    )
                    # Fill with None for failed embeddings
                    for idx in batch_indices:
                        if embeddings[idx] is None:
                            embeddings[idx] = []

        # Convert None to empty list
        result = [emb if emb else [] for emb in embeddings]
        logger.info(
            "Batch embedding complete",
            total=len(texts),
            successful=sum(1 for e in result if e)
        )

        return result

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.dimensions


class LocalEmbedder(BaseEmbedder):
    """
    Local embedding service using sentence-transformers.

    Supports any model from the sentence-transformers library:
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, multilingual)
    - all-MiniLM-L6-v2 (384 dimensions, English)
    - paraphrase-multilingual-mpnet-base-v2 (768 dimensions, multilingual)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize local embedder.

        Args:
            model_name: Model name from sentence-transformers
            device: Device to run on (cpu, cuda, etc.)
        """
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or "paraphrase-multilingual-MiniLM-L12-v2"
        self.device = device or "cpu"

        # Load model in executor to avoid blocking
        def load_model():
            return SentenceTransformer(self.model_name, device=self.device)

        self.model = asyncio.get_event_loop().run_in_executor(
            self._executor,
            load_model
        )

        logger.info(
            "Initialized LocalEmbedder",
            model=self.model_name,
            device=self.device
        )

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        cached = await cache_store.get_embedding(text)
        if cached:
            logger.debug("Embedding cache hit", text_length=len(text))
            return cached

        # Generate embedding
        logger.debug("Embedding cache miss, generating", text_length=len(text))
        try:
            # Wait for model to be loaded
            model = await self.model if isinstance(self.model, asyncio.Future) else self.model

            # Run encoding in executor
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                lambda: model.encode(text, convert_to_numpy=True).tolist()
            )

            # Cache result
            await cache_store.set_embedding(text, embedding)
            logger.debug("Generated and cached embedding", dimension=len(embedding))

            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing and caching.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(
            "Batch embedding started",
            text_count=len(texts),
            batch_size=batch_size
        )

        # Check cache for all texts
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = await cache_store.get_embedding(text)
            if cached:
                embeddings[i] = cached
            else:
                uncached_indices.append(i)

        logger.info(
            "Cache check complete",
            cached=len(texts) - len(uncached_indices),
            uncached=len(uncached_indices)
        )

        # Process uncached texts in batches
        if uncached_indices:
            # Wait for model to be loaded
            model = await self.model if isinstance(self.model, asyncio.Future) else self.model

            for offset in range(0, len(uncached_indices), batch_size):
                batch_indices = uncached_indices[offset:offset + batch_size]
                batch_texts = [texts[i] for i in batch_indices]

                try:
                    # Run encoding in executor
                    loop = asyncio.get_event_loop()
                    batch_embeddings = await loop.run_in_executor(
                        self._executor,
                        lambda: model.encode(batch_texts, convert_to_numpy=True).tolist()
                    )

                    # Update embeddings and cache
                    for idx, embedding in zip(batch_indices, batch_embeddings):
                        embeddings[idx] = embedding
                        await cache_store.set_embedding(batch_texts[batch_indices.index(idx)], embedding)

                    logger.debug(
                        "Batch embedded successfully",
                        batch_size=len(batch_indices),
                        offset=offset
                    )

                except Exception as e:
                    logger.error(
                        "Failed to embed batch",
                        offset=offset,
                        batch_size=len(batch_indices),
                        error=str(e)
                    )
                    # Fill with None for failed embeddings
                    for idx in batch_indices:
                        if embeddings[idx] is None:
                            embeddings[idx] = []

        # Convert None to empty list
        result = [emb if emb else [] for emb in embeddings]
        logger.info(
            "Batch embedding complete",
            total=len(texts),
            successful=sum(1 for e in result if e)
        )

        return result

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Common dimensions for popular models
        dimensions_map = {
            # Multilingual models (good for Vietnamese + English)
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "paraphrase-multilingual-mpnet-base-v2": 768,
            "intfloat/multilingual-e5-large": 1024,
            "intfloat/multilingual-e5-base": 768,
            "intfloat/multilingual-e5-small": 384,
            # English-only models
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            # BAAI models (good for QA, English)
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "bge-base-en-v1.5": 768,
            "bge-small-en-v1.5": 384,
            # E5 models (high quality)
            "intfloat/e5-large-v2": 1024,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-small-v2": 384,
        }
        return dimensions_map.get(self.model_name, 384)


class MockEmbedder(BaseEmbedder):
    """
    Mock embedder for testing purposes.

    Generates deterministic pseudo-random embeddings based on text hash.
    """

    def __init__(self, dimension: int = 1536):
        """
        Initialize mock embedder.

        Args:
            dimension: Dimension of embedding vectors
        """
        super().__init__()
        self.dimension = dimension
        logger.warning("Using MockEmbedder - not for production use")

    async def embed(self, text: str) -> List[float]:
        """Generate deterministic mock embedding."""
        import hashlib

        # Create deterministic hash-based embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Expand to dimension
        embedding = []
        for i in range(self.dimension):
            val = (hash_bytes[i % len(hash_bytes)] / 255.0) * 2 - 1  # Scale to [-1, 1]
            embedding.append(val)

        return embedding

    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.dimension


@lru_cache(maxsize=1)
def get_embedder(
    provider: Optional[str] = None,
    **kwargs
) -> BaseEmbedder:
    """
    Get embedder instance based on provider.

    Cached singleton pattern ensures only one instance per provider.

    Args:
        provider: Embedding provider (openai, local, mock)
        **kwargs: Additional arguments for embedder initialization

    Returns:
        BaseEmbedder instance
    """
    provider = provider or settings.embedding_provider

    logger.info("Creating embedder", provider=provider)

    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    elif provider == "local":
        return LocalEmbedder(**kwargs)
    elif provider == "mock":
        return MockEmbedder(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported providers: openai, local, mock"
        )


# Singleton instance for default provider
embedder = get_embedder()
