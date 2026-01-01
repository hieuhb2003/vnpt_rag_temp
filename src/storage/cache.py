# =============================================================================
# Cache Store - Redis Caching
# =============================================================================
import json
import hashlib
from typing import Optional, Any, List, Dict
from uuid import UUID
import numpy as np

from redis.asyncio import Redis, ConnectionPool
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CacheStore:
    """Async Redis cache store with embedding, semantic, and retrieval caching."""

    # Cache key prefixes
    PREFIX_EMBEDDING = "emb:"
    PREFIX_SEMANTIC = "sem:"
    PREFIX_RETRIEVAL = "ret:"
    PREFIX_VECTOR = "vec:"

    def __init__(self):
        self.redis: Optional[Redis] = None
        self.pool: Optional[ConnectionPool] = None

    async def connect(self):
        """Initialize Redis connection pool."""
        self.pool = ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=0,
            decode_responses=False,  # Handle bytes manually for binary data
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self.redis = Redis(connection_pool=self.pool)
        # Test connection
        await self.redis.ping()
        logger.info(
            "Connected to Redis",
            host=settings.redis_host,
            port=settings.redis_port,
        )

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            if self.pool:
                await self.pool.disconnect()
            logger.info("Disconnected from Redis")

    # =========================================================================
    # Embedding Cache
    # =========================================================================

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._embedding_key(text)
        cached = await self.redis.get(key)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to decode cached embedding", key=key)
        return None

    async def set_embedding(self, text: str, embedding: List[float], ttl: Optional[int] = None):
        """Cache embedding for text with TTL."""
        key = self._embedding_key(text)
        ttl = ttl or settings.cache_embedding_ttl
        try:
            await self.redis.setex(key, ttl, json.dumps(embedding))
            logger.debug("Cached embedding", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("Failed to cache embedding", key=key, error=str(e))

    # =========================================================================
    # Semantic Cache
    # =========================================================================

    async def get_semantic_cache(
        self,
        query: str,
        query_vector: List[float],
        threshold: float = 0.85,
    ) -> Optional[str]:
        """
        Get semantically similar cached query result.
        Uses simple cosine similarity on stored vectors.
        """
        # Get all semantic cache keys
        pattern = f"{self.PREFIX_SEMANTIC}*"
        keys = []
        async for key in self.redis.scan_iter(match=pattern.encode()):
            keys.append(key.decode() if isinstance(key, bytes) else key)

        if not keys:
            return None

        # Calculate similarity with each cached query
        best_match = None
        best_score = 0

        for key in keys:
            # Get stored vector
            vec_key = f"{self.PREFIX_VECTOR}{key[len(self.PREFIX_SEMANTIC):]}"
            cached_vec_bytes = await self.redis.get(vec_key)
            if not cached_vec_bytes:
                continue

            try:
                cached_vec = json.loads(cached_vec_bytes)
                similarity = self._cosine_similarity(query_vector, cached_vec)

                if similarity >= threshold and similarity > best_score:
                    best_score = similarity
                    best_match = key
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning("Failed to process cached vector", key=key, error=str(e))
                continue

        if best_match:
            # Get the cached response
            cached_response = await self.redis.get(best_match)
            if cached_response:
                try:
                    response = json.loads(cached_response)
                    logger.info(
                        "Semantic cache hit",
                        key=best_match,
                        similarity=best_score,
                    )
                    return response
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to decode cached response", key=best_match)

        return None

    async def set_semantic_cache(
        self,
        query: str,
        query_vector: List[float],
        response: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """Cache query with vector for semantic matching."""
        key = self._hash(query)
        sem_key = f"{self.PREFIX_SEMANTIC}{key}"
        vec_key = f"{self.PREFIX_VECTOR}{key}"
        ttl = ttl or settings.cache_semantic_ttl

        try:
            # Store response
            await self.redis.setex(sem_key, ttl, json.dumps(response))
            # Store vector
            await self.redis.setex(vec_key, ttl, json.dumps(query_vector))
            logger.debug("Cached semantic query", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("Failed to cache semantic query", key=key, error=str(e))

    # =========================================================================
    # Retrieval Cache
    # =========================================================================

    async def get_retrieval_cache(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached retrieval results."""
        key = self._retrieval_key(query, top_k, filters)
        cached = await self.redis.get(key)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to decode cached retrieval", key=key)
        return None

    async def set_retrieval_cache(
        self,
        query: str,
        top_k: int,
        results: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ):
        """Cache retrieval results with TTL."""
        key = self._retrieval_key(query, top_k, filters)
        ttl = ttl or settings.cache_retrieval_ttl
        try:
            await self.redis.setex(key, ttl, json.dumps(results))
            logger.debug("Cached retrieval", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("Failed to cache retrieval", key=key, error=str(e))

    # =========================================================================
    # Generic Cache Operations
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cached = await self.redis.get(key)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                return cached
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        try:
            await self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning("Failed to set cache", key=key, error=str(e))

    async def delete(self, key: str):
        """Delete key from cache."""
        await self.redis.delete(key)

    async def delete_pattern(self, pattern: str):
        """Delete keys matching pattern."""
        keys = []
        async for key in self.redis.scan_iter(match=pattern.encode()):
            keys.append(key)
        if keys:
            await self.redis.delete(*keys)
            logger.info("Deleted cache keys", pattern=pattern, count=len(keys))

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.redis.exists(key) > 0

    async def expire(self, key: str, ttl: int):
        """Set TTL for existing key."""
        await self.redis.expire(key, ttl)

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        return await self.redis.ttl(key)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _hash(self, text: str) -> str:
        """Create hash key from text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _embedding_key(self, text: str) -> str:
        """Create cache key for embedding."""
        hash_part = self._hash(text)
        return f"{self.PREFIX_EMBEDDING}{hash_part}"

    def _retrieval_key(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> str:
        """Create cache key for retrieval."""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        key_str = f"{query}:{top_k}:{filter_str}"
        hash_part = self._hash(key_str)
        return f"{self.PREFIX_RETRIEVAL}{hash_part}"

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            arr1 = np.array(vec1)
            arr2 = np.array(vec2)

            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except (ValueError, TypeError):
            return 0.0

    async def clear_all(self):
        """Clear all cache (use with caution)."""
        patterns = [
            f"{self.PREFIX_EMBEDDING}*",
            f"{self.PREFIX_SEMANTIC}*",
            f"{self.PREFIX_VECTOR}*",
            f"{self.PREFIX_RETRIEVAL}*",
        ]
        for pattern in patterns:
            await self.delete_pattern(pattern)
        logger.info("Cleared all cache")

    async def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = {}
        patterns = {
            "embeddings": f"{self.PREFIX_EMBEDDING}*",
            "semantic_queries": f"{self.PREFIX_SEMANTIC}*",
            "vectors": f"{self.PREFIX_VECTOR}*",
            "retrievals": f"{self.PREFIX_RETRIEVAL}*",
        }
        for name, pattern in patterns.items():
            count = 0
            async for _ in self.redis.scan_iter(match=pattern.encode()):
                count += 1
            stats[name] = count
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health and return statistics."""
        try:
            await self.redis.ping()
            stats = await self.get_stats()
            return {
                "status": "healthy",
                **stats,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Singleton instance
cache_store = CacheStore()
