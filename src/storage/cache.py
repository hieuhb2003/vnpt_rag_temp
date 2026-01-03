# =============================================================================
# Cache Store - Multi-Level Redis Caching
# =============================================================================
import json
import hashlib
import asyncio
from typing import Optional, Any, List, Dict, Callable, Awaitable
from uuid import UUID
from collections import OrderedDict
import numpy as np

from redis.asyncio import Redis, ConnectionPool
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# Multi-Level Cache Manager
# =============================================================================

class CacheManager:
    """
    Multi-level cache manager with warm-up and monitoring.

    Levels:
    - L1: In-memory LRU cache (fastest, smallest, ~1000 items)
    - L2: Redis semantic cache (fast, medium, TTL 3600s)
    - L3: Redis embedding cache (medium, large, TTL 3600s)
    """

    def __init__(self, l1_max_size: int = 1000):
        # L1: In-memory LRU cache using OrderedDict
        self.l1_cache: OrderedDict[str, Any] = OrderedDict()
        self.l1_max_size = l1_max_size
        self.l1_lock = asyncio.Lock()

        # Statistics tracking
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "l1_evictions": 0,
        }
        self.stats_lock = asyncio.Lock()

    async def get_with_fallback(
        self,
        key: str,
        fetch_func: Callable[[], Awaitable[Any]],
        cache_level: str = "all",
        ttl: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Get value with multi-level cache fallback.

        Flow:
        1. Check L1 (memory) - fastest
        2. Check L2/L3 (Redis) - populate L1 on hit
        3. Call fetch_func if miss - populate all levels
        4. Return value

        Args:
            key: Cache key
            fetch_func: Async function to fetch value on cache miss
            cache_level: Which levels to check ("l1", "l2", "all")
            ttl: TTL for Redis cache (default from settings)

        Returns:
            Cached or fetched value
        """
        # L1 check (memory)
        async with self.l1_lock:
            if key in self.l1_cache:
                self.stats["l1_hits"] += 1
                # Update LRU - move to end
                self.l1_cache.move_to_end(key)
                logger.debug("L1 cache hit", key=key)
                return self.l1_cache[key]

        # L2/L3 check (Redis)
        if cache_level in ("l2", "all"):
            cached = await cache_store.redis.get(key)
            if cached:
                try:
                    value = json.loads(cached)
                    async with self.stats_lock:
                        self.stats["l2_hits"] += 1
                    # Promote to L1
                    await self._set_l1(key, value)
                    logger.debug("L2 cache hit", key=key)
                    return value
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Failed to decode cached value", key=key, error=str(e))

        # Miss - fetch and cache
        async with self.stats_lock:
            self.stats["misses"] += 1

        try:
            value = await fetch_func()

            if value is not None:
                # Populate L1
                await self._set_l1(key, value)

                # Populate L2/L3 (Redis)
                if cache_level in ("l2", "all"):
                    ttl = ttl or settings.cache_semantic_ttl
                    try:
                        await cache_store.redis.setex(
                            key,
                            ttl,
                            json.dumps(value)
                        )
                        logger.debug("Cached value", key=key, ttl=ttl)
                    except Exception as e:
                        logger.warning("Failed to cache in Redis", key=key, error=str(e))

            return value

        except Exception as e:
            logger.error("Failed to fetch value", key=key, error=str(e))
            return None

    async def _set_l1(self, key: str, value: Any):
        """Set L1 cache with LRU eviction."""
        async with self.l1_lock:
            # Evict oldest if at capacity
            if len(self.l1_cache) >= self.l1_max_size:
                oldest_key, _ = self.l1_cache.popitem(last=False)
                async with self.stats_lock:
                    self.stats["l1_evictions"] += 1
                logger.debug("L1 eviction", key=oldest_key)

            # Set new value
            self.l1_cache[key] = value
            logger.debug("L1 cache set", key=key, size=len(self.l1_cache))

    async def invalidate_l1(self, key: str):
        """Invalidate key from L1 cache."""
        async with self.l1_lock:
            if key in self.l1_cache:
                del self.l1_cache[key]
                logger.debug("L1 cache invalidated", key=key)

    async def invalidate_all_l1(self):
        """Clear all L1 cache."""
        async with self.l1_lock:
            self.l1_cache.clear()
            logger.info("L1 cache cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hit rates, cache sizes, and evictions
        """
        async with self.stats_lock:
            total_hits = (
                self.stats["l1_hits"] +
                self.stats["l2_hits"] +
                self.stats["l3_hits"]
            )
            total_requests = total_hits + self.stats["misses"]
            hit_rate = total_hits / total_requests if total_requests > 0 else 0

            l1_hit_rate = (
                self.stats["l1_hits"] / total_requests
                if total_requests > 0 else 0
            )
            l2_hit_rate = (
                self.stats["l2_hits"] / total_requests
                if total_requests > 0 else 0
            )

            return {
                **self.stats,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "l1_hit_rate": l1_hit_rate,
                "l2_hit_rate": l2_hit_rate,
                "l1_size": len(self.l1_cache),
                "l1_capacity": self.l1_max_size,
                "l1_utilization": len(self.l1_cache) / self.l1_max_size,
            }

    async def reset_stats(self):
        """Reset cache statistics."""
        async with self.stats_lock:
            self.stats = {
                "l1_hits": 0,
                "l2_hits": 0,
                "l3_hits": 0,
                "misses": 0,
                "l1_evictions": 0,
            }
        logger.info("Cache stats reset")

    async def warm_l1(self, items: Dict[str, Any]):
        """
        Warm L1 cache with pre-computed values.

        Args:
            items: Dictionary of key -> value pairs to cache
        """
        async with self.l1_lock:
            for key, value in items.items():
                # Evict if necessary
                if len(self.l1_cache) >= self.l1_max_size:
                    self.l1_cache.popitem(last=False)
                    self.stats["l1_evictions"] += 1

                self.l1_cache[key] = value

        logger.info("L1 cache warmed", count=len(items))


# Singleton cache manager instance
cache_manager = CacheManager(l1_max_size=1000)


# =============================================================================
# Redis Cache Store
# =============================================================================

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
