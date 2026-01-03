# =============================================================================
# Cache Warmer - Pre-populate caches with frequently accessed data
# =============================================================================
import asyncio
from typing import Optional, List

from src.storage import metadata_store, cache_store, cache_manager
from src.indexing.embedder import get_embedder
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Common Queries for Cache Warming
# =============================================================================

# Common FAQ queries to warm - based on typical HR/Support queries
COMMON_QUERIES = [
    # Vietnamese queries
    "Làm sao để reset password?",
    "Quy trình hoàn tiền như thế nào?",
    "Thời gian giao hàng bao lâu?",
    "Cách liên hệ hỗ trợ khách hàng?",
    "Chính sách đổi trả hàng?",
    "Làm sao để theo dõi đơn hàng?",
    "Phương thức thanh toán nào được hỗ trợ?",
    "Làm sao để cập nhật thông tin tài khoản?",
    "Số ngày nghỉ phép trong năm?",
    "Quy định về lương tháng 13?",
    "Thủ tục xin nghỉ phép?",
    "Cách tính tiền tăng ca?",
    "Chính sách bảo hiểm xã hội?",
    "Quy định về giờ làm việc?",
    "Thời gian thử việc là bao lâu?",
    "Lương cơ bản được tính như thế nào?",
    "Quy định về ngày nghỉ lễ?",
    "Cách đăng ký bảo hiểm y tế?",
    "Thủ tục xin thôi việc?",
    "Chính sách đào tạo nhân viên?",

    # English queries
    "How to reset password?",
    "What is the refund process?",
    "How long does shipping take?",
    "How to contact customer support?",
    "What is the return policy?",
    "How to track my order?",
    "What payment methods are supported?",
    "How to update account information?",
    "How many annual leave days?",
    "What is the 13th month salary policy?",
    "How to request leave?",
    "How is overtime calculated?",
    "Social insurance policy?",
    "Working hours policy?",
    "How long is probation?",
    "How is base salary calculated?",
    "Public holiday policy?",
    "How to enroll for health insurance?",
    "Resignation procedure?",
    "Employee training policy?",
]


# =============================================================================
# Cache Warmer Class
# =============================================================================

class CacheWarmer:
    """
    Pre-populate caches with frequently accessed data.

    Run on startup or periodically to improve cold-start performance.
    """

    def __init__(self):
        self.embedder = None
        self._warming = False

    async def warm_document_embeddings(self, limit: int = 100) -> int:
        """
        Pre-embed most accessed document summaries.

        Args:
            limit: Maximum number of documents to warm

        Returns:
            Number of documents warmed
        """
        logger.info("Starting document embedding cache warm-up...", limit=limit)

        try:
            # Get recent documents (would use access tracking if available)
            # For now, get documents by creation date
            from sqlalchemy import select, desc
            from src.storage.models import DocumentORM

            async with metadata_store.session() as session:
                stmt = (
                    select(DocumentORM)
                    .where(DocumentORM.status == "indexed")
                    .order_by(desc(DocumentORM.created_at))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                docs = result.scalars().all()

            if not docs:
                logger.info("No documents found for warming")
                return 0

            # Initialize embedder
            if not self.embedder:
                self.embedder = get_embedder()

            # Prepare texts for batch embedding
            texts = []
            doc_ids = []
            for doc in docs:
                text = doc.summary or doc.title or ""
                if text:
                    texts.append(text)
                    doc_ids.append(str(doc.id))

            if not texts:
                logger.info("No valid texts found for warming")
                return 0

            # Batch embed (will be cached automatically)
            logger.info(f"Embedding {len(texts)} document texts...")
            embeddings = await self.embedder.embed_batch(texts)

            # Store in cache explicitly
            for i, (doc_id, text, embedding) in enumerate(zip(doc_ids, texts, embeddings)):
                await cache_store.set_embedding(
                    f"doc_summary:{doc_id}",
                    embedding,
                    ttl=7200  # 2 hours
                )

            logger.info(
                "Document embedding warm-up complete",
                count=len(embeddings),
                cached_in_l1=await self._get_l1_count()
            )
            return len(embeddings)

        except Exception as e:
            logger.error("Failed to warm document embeddings", error=str(e))
            return 0

    async def warm_common_queries(
        self,
        queries: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> int:
        """
        Pre-process common queries.

        Args:
            queries: List of queries to warm (default: COMMON_QUERIES)
            batch_size: Batch size for embedding

        Returns:
            Number of queries warmed
        """
        if queries is None:
            queries = COMMON_QUERIES

        logger.info(f"Warming {len(queries)} common queries...")

        try:
            if not self.embedder:
                self.embedder = get_embedder()

            # Process in batches
            warmed = 0
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]

                # Embed batch (will be cached automatically)
                await self.embedder.embed_batch(batch)

                warmed += len(batch)
                logger.debug(f"Warmed {warmed}/{len(queries)} queries")

                # Small delay to avoid rate limiting
                if i + batch_size < len(queries):
                    await asyncio.sleep(0.1)

            logger.info(
                "Common query warm-up complete",
                count=warmed,
                cached_in_l1=await self._get_l1_count()
            )
            return warmed

        except Exception as e:
            logger.error("Failed to warm common queries", error=str(e))
            return 0

    async def warm_section_summaries(self, limit: int = 500) -> int:
        """
        Pre-embed section summaries for navigation.

        Args:
            limit: Maximum number of sections to warm

        Returns:
            Number of sections warmed
        """
        logger.info("Warming section summaries...", limit=limit)

        try:
            # Get sections with summaries
            from sqlalchemy import select, desc
            from src.storage.models import SectionORM

            async with metadata_store.session() as session:
                stmt = (
                    select(SectionORM)
                    .where(SectionORM.level <= 2)  # Only main sections
                    .order_by(desc(SectionORM.created_at))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                sections = result.scalars().all()

            if not sections:
                logger.info("No sections found for warming")
                return 0

            if not self.embedder:
                self.embedder = get_embedder()

            # Prepare summaries
            texts = []
            section_ids = []
            for section in sections:
                # Use heading + first part of content
                text = f"{section.heading}: {section.content[:200]}"
                texts.append(text)
                section_ids.append(str(section.id))

            if not texts:
                logger.info("No valid section texts found")
                return 0

            # Batch embed
            logger.info(f"Embedding {len(texts)} section texts...")
            embeddings = await self.embedder.embed_batch(texts)

            # Store in cache
            for i, (section_id, text, embedding) in enumerate(zip(section_ids, texts, embeddings)):
                await cache_store.set_embedding(
                    f"section:{section_id}",
                    embedding,
                    ttl=7200
                )

            logger.info(
                "Section summary warm-up complete",
                count=len(embeddings),
                cached_in_l1=await self._get_l1_count()
            )
            return len(embeddings)

        except Exception as e:
            logger.error("Failed to warm section summaries", error=str(e))
            return 0

    async def warm_vector_search_cache(
        self,
        queries: Optional[List[str]] = None,
        top_k: int = 10
    ) -> int:
        """
        Warm hybrid search results for common queries.

        This performs actual searches and caches the results.

        Args:
            queries: List of queries to search for
            top_k: Number of results per query

        Returns:
            Number of queries cached
        """
        if queries is None:
            queries = COMMON_QUERIES[:20]  # Use first 20 for search warming

        logger.info(f"Warming vector search cache for {len(queries)} queries...")

        try:
            from src.tools.hybrid_search import hybrid_search_tool

            warmed = 0
            for query in queries:
                try:
                    # Perform search (results will be cached)
                    result = await hybrid_search_tool.ainvoke({
                        "query": query,
                        "collection": "chunks",
                        "top_k": top_k,
                        "use_cache": False  # Force search, then cache manually
                    })

                    # Manually cache the retrieval result
                    await cache_store.set_retrieval_cache(
                        query=query,
                        top_k=top_k,
                        results={
                            "results": [r.model_dump() for r in result.results],
                            "total": result.total_results,
                        },
                        ttl=1800  # 30 minutes
                    )

                    warmed += 1

                except Exception as e:
                    logger.warning("Failed to warm search for query", query=query, error=str(e))

            logger.info("Vector search cache warm-up complete", count=warmed)
            return warmed

        except Exception as e:
            logger.error("Failed to warm vector search cache", error=str(e))
            return 0

    async def warm_all(self, doc_limit: int = 100, section_limit: int = 500) -> dict:
        """
        Run all cache warming strategies.

        Args:
            doc_limit: Limit for document warming
            section_limit: Limit for section warming

        Returns:
            Dictionary with warming results
        """
        if self._warming:
            logger.warning("Cache warming already in progress")
            return {"status": "already_running"}

        self._warming = True
        start_time = asyncio.get_event_loop().time()

        logger.info("Starting full cache warm-up...")

        results = {
            "documents": 0,
            "queries": 0,
            "sections": 0,
            "searches": 0,
            "total_time_ms": 0,
        }

        try:
            # Step 1: Warm document embeddings
            results["documents"] = await self.warm_document_embeddings(limit=doc_limit)
            await asyncio.sleep(0.5)

            # Step 2: Warm section summaries
            results["sections"] = await self.warm_section_summaries(limit=section_limit)
            await asyncio.sleep(0.5)

            # Step 3: Warm common queries
            results["queries"] = await self.warm_common_queries()

            # Step 4: Warm search results
            results["searches"] = await self.warm_vector_search_cache()

            # Get cache stats
            stats = await cache_manager.get_stats()

            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
            results["total_time_ms"] = elapsed
            results["cache_stats"] = stats

            logger.info(
                "Full cache warm-up complete",
                **results,
                l1_size=stats.get("l1_size", 0),
                hit_rate=stats.get("hit_rate", 0),
            )

            return results

        finally:
            self._warming = False

    async def _get_l1_count(self) -> int:
        """Get current L1 cache size."""
        stats = await cache_manager.get_stats()
        return stats.get("l1_size", 0)

    def is_warming(self) -> bool:
        """Check if warming is in progress."""
        return self._warming


# Singleton instance
cache_warmer = CacheWarmer()


# =============================================================================
# Convenience Functions
# =============================================================================

async def warm_startup_cache():
    """
    Warm caches on application startup.

    Call this in the lifespan() function of the FastAPI app.
    """
    logger.info("Starting startup cache warm-up...")

    # Run warming in background
    asyncio.create_task(cache_warmer.warm_all(
        doc_limit=50,    # Smaller limit for startup
        section_limit=200
    ))

    logger.info("Startup cache warm-up initiated in background")


async def warm_periodic_cache(interval_minutes: int = 60):
    """
    Periodically warm caches.

    Args:
        interval_minutes: Minutes between warm-up cycles
    """
    while True:
        try:
            await asyncio.sleep(interval_minutes * 60)
            logger.info("Running periodic cache warm-up...")
            await cache_warmer.warm_all(
                doc_limit=100,
                section_limit=500
            )
        except asyncio.CancelledError:
            logger.info("Periodic cache warming cancelled")
            break
        except Exception as e:
            logger.error("Periodic cache warming failed", error=str(e))
