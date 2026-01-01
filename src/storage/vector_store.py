# =============================================================================
# Vector Store - Qdrant Client
# =============================================================================
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint,
    CreateCollection,
    UpdateCollection,
)
from typing import Optional, Any, List, Dict
from uuid import UUID
import numpy as np

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """Async Qdrant vector store with multi-collection support."""

    COLLECTIONS = {
        "documents": "doc_embeddings",
        "sections": "section_embeddings",
        "chunks": "chunk_embeddings",
    }

    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None

    async def connect(self):
        """Initialize async Qdrant client."""
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
        # Test connection
        collections = await self.client.get_collections()
        logger.info(
            "Connected to Qdrant",
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            existing_collections=len(collections.collections),
        )

    async def disconnect(self):
        """Close Qdrant connection."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Qdrant")

    async def initialize_collections(self):
        """Create collections if they don't exist."""
        # Get list of existing collections
        try:
            collections_response = await self.client.get_collections()
            existing_collections = {c.name for c in collections_response.collections}
        except Exception as e:
            logger.warning("Failed to get collections", error=str(e))
            existing_collections = set()

        for name, collection_name in self.COLLECTIONS.items():
            if collection_name not in existing_collections:
                try:
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=settings.embedding_dimensions,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info("Created collection", collection=collection_name, size=settings.embedding_dimensions)
                except Exception as e:
                    # Collection might have been created by another process
                    if "already exists" not in str(e):
                        logger.warning("Failed to create collection", collection=collection_name, error=str(e))
            else:
                logger.debug("Collection already exists", collection=collection_name)

    async def upsert(
        self,
        collection: str,
        id: UUID,
        vector: List[float],
        payload: Dict[str, Any],
    ):
        """Upsert a single vector."""
        collection_name = self.COLLECTIONS[collection]
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(id),
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.debug("Upserted point", collection=collection_name, id=str(id))

    async def upsert_batch(
        self,
        collection: str,
        points: List[Dict[str, Any]],  # [{id, vector, payload}, ...]
    ):
        """Batch upsert vectors."""
        collection_name = self.COLLECTIONS[collection]
        qdrant_points = [
            PointStruct(
                id=str(p["id"]),
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points
        ]
        await self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
        )
        logger.info("Upserted batch", collection=collection_name, count=len(points))

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredPoint]:
        """Search for similar vectors."""
        collection_name = self.COLLECTIONS[collection]

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Match any in list
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        logger.debug("Search completed", collection=collection_name, results=len(results))
        return results

    async def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.7,  # weight for vector search
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and full-text search.
        Uses Qdrant's built-in hybrid search if available,
        otherwise falls back to RRF fusion.
        """
        # Vector search
        vector_results = await self.search(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k * 2,  # Get more for fusion
            filters=filters,
        )

        # For now, return vector results
        # TODO: Implement BM25 search and RRF fusion when full-text index is set up

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in vector_results[:top_k]
        ]

    async def delete(self, collection: str, ids: List[UUID]):
        """Delete vectors by IDs."""
        collection_name = self.COLLECTIONS[collection]
        await self.client.delete(
            collection_name=collection_name,
            points_selector=[str(id) for id in ids],
        )
        logger.info("Deleted points", collection=collection_name, count=len(ids))

    async def count(self, collection: str) -> int:
        """Count vectors in a collection."""
        collection_name = self.COLLECTIONS[collection]
        count_info = await self.client.count(collection_name)
        return count_info.count

    async def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """Get information about a collection."""
        collection_name = self.COLLECTIONS[collection]
        info = await self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
        }

    async def clear_collection(self, collection: str):
        """Delete all vectors from a collection."""
        collection_name = self.COLLECTIONS[collection]
        # Get all points and delete them
        count_info = await self.client.count(collection_name)
        if count_info.count > 0:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=[{"is_empty": False}],
            )
            logger.info("Cleared collection", collection=collection_name, deleted=count_info.count)


# Singleton instance
vector_store = VectorStore()
