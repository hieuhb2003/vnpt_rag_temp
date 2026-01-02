# =============================================================================
# Hybrid Search Tool - Combine vector and keyword search with caching
# =============================================================================
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.indexing.embedder import get_embedder
from src.storage.vector_store import vector_store
from src.storage.cache import cache_store
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# Input/Output Models
# =============================================================================

class SearchFilters(BaseModel):
    """Filters for search operations."""

    category: Optional[str] = Field(None, description="Lọc theo danh mục tài liệu")
    tags: Optional[List[str]] = Field(None, description="Lọc theo tags")
    language: Optional[str] = Field(None, description="Lọc theo ngôn ngữ (vi, en)")
    document_id: Optional[str] = Field(None, description="Lọc theo ID tài liệu cụ thể")
    date_from: Optional[str] = Field(None, description="Lọc tài liệu từ ngày")
    date_to: Optional[str] = Field(None, description="Lọc tài liệu đến ngày")


class SearchResult(BaseModel):
    """Single search result."""

    id: str = Field(..., description="ID của kết quả (chunk/section/document)")
    type: str = Field(..., description="Loại kết quả (chunk, section, document)")
    content: str = Field(..., description="Nội dung kết quả")
    score: float = Field(..., ge=0.0, le=1.0, description="Độ tương đồng")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung")

    # Document reference
    document_id: Optional[str] = Field(None, description="ID tài liệu chứa kết quả")
    document_title: Optional[str] = Field(None, description="Tiêu đề tài liệu")

    # Section reference (if applicable)
    section_id: Optional[str] = Field(None, description="ID section chứa chunk")
    section_heading: Optional[str] = Field(None, description="Tiêu đề section")


class HybridSearchOutput(BaseModel):
    """Output from hybrid search tool."""

    query: str = Field(..., description="Câu hỏi tìm kiếm")
    results: List[SearchResult] = Field(..., description="Danh sách kết quả")
    total_results: int = Field(..., description="Tổng số kết quả tìm được")
    search_method: str = Field(..., description="Phương pháp tìm kiếm (vector, keyword, hybrid)")
    cache_hit: bool = Field(default=False, description="Kết quả từ cache")
    search_time_ms: float = Field(..., description="Thời gian tìm kiếm (ms)")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Bộ lọc đã áp dụng")


# =============================================================================
# Hybrid Search Tool
# =============================================================================

@tool
async def hybrid_search_tool(
    query: str,
    collection: str = "chunks",
    top_k: int = 10,
    alpha: float = 0.7,
    filters: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    min_score: float = 0.5
) -> HybridSearchOutput:
    """
    Tìm kiếm kết hợp vector và keyword với caching.

    Công cụ này kết hợp tìm kiếm vector (semantic) và keyword (BM25)
    để trả về kết quả chính xác nhất:
    - Vector search: Tìm kiếm dựa trên ngữ cảnh ý nghĩa
    - Keyword search: Tìm kiếm dựa trên từ khóa chính xác
    - Fusion: Kết hợp cả hai phương pháp với tỷ lệ alpha

    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm
        collection: Bộ sưu tập (documents, sections, chunks)
        top_k: Số lượng kết quả trả về
        alpha: Tỷ lệ vector search (0=keyword only, 1=vector only, 0.7=balanced)
        filters: Bộ lọc kết quả (category, tags, language, etc.)
        use_cache: Có sử dụng cache không
        min_score: Độ tương đồng tối thiểu

    Returns:
        HybridSearchOutput: Kết quả tìm kiếm

    Example:
        >>> result = await hybrid_search_tool(
        ...     query="ngày nghỉ phép hàng năm",
        ...     collection="chunks",
        ...     top_k=5,
        ...     alpha=0.7
        ... )
        >>> print(f"Found {result.total_results} results")
    """
    import time

    start_time = time.time()

    try:
        logger.info(
            "Hybrid search started",
            query=query[:100],
            collection=collection,
            top_k=top_k,
            alpha=alpha
        )

        # Build cache key
        cache_key = None
        if use_cache:
            cache_key_parts = {
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "alpha": alpha,
                "min_score": min_score,
                "filters": filters
            }

        # Check retrieval cache
        if use_cache:
            cached = await cache_store.get_retrieval_cache(
                query=query,
                top_k=top_k,
                filters=filters
            )
            if cached:
                logger.info("Retrieval cache hit", query=query[:50])
                results = [SearchResult(**r) for r in cached.get("results", [])]
                return HybridSearchOutput(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_method=cached.get("search_method", "cached"),
                    cache_hit=True,
                    search_time_ms=(time.time() - start_time) * 1000,
                    filters_applied=filters
                )

        # Get embedder
        embedder = get_embedder()

        # Generate query embedding
        query_vector = await embedder.embed(query)

        # Perform hybrid search
        search_results = await vector_store.hybrid_search(
            collection=collection,
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            alpha=alpha,
            filters=filters
        )

        # Convert to SearchResult format
        results = []
        for r in search_results:
            payload = r.get("payload", {})
            score = r.get("score", 0.0)

            # Filter by minimum score
            if score < min_score:
                continue

            result = SearchResult(
                id=r.get("id", ""),
                type=collection,
                content=payload.get("content", "")[:500],  # Truncate if too long
                score=score,
                metadata=payload.get("metadata", {}),
                document_id=payload.get("document_id"),
                document_title=payload.get("document_title"),
                section_id=payload.get("section_id"),
                section_heading=payload.get("section_heading")
            )
            results.append(result)

        search_time_ms = (time.time() - start_time) * 1000

        # Cache results
        if use_cache and results:
            await cache_store.set_retrieval_cache(
                query=query,
                top_k=top_k,
                results={
                    "results": [r.model_dump() for r in results],
                    "search_method": "hybrid",
                    "total": len(results)
                },
                filters=filters
            )

        logger.info(
            "Hybrid search completed",
            query=query[:50],
            results=len(results),
            time_ms=search_time_ms
        )

        return HybridSearchOutput(
            query=query,
            results=results,
            total_results=len(results),
            search_method="hybrid",
            cache_hit=False,
            search_time_ms=search_time_ms,
            filters_applied=filters
        )

    except Exception as e:
        logger.error(
            "Hybrid search failed",
            error=str(e),
            query=query[:100]
        )
        search_time_ms = (time.time() - start_time) * 1000

        return HybridSearchOutput(
            query=query,
            results=[],
            total_results=0,
            search_method="hybrid",
            cache_hit=False,
            search_time_ms=search_time_ms,
            filters_applied=filters
        )


def hybrid_search_tool_sync(
    query: str,
    collection: str = "chunks",
    top_k: int = 10,
    alpha: float = 0.7,
    filters: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    min_score: float = 0.5
) -> HybridSearchOutput:
    """Synchronous wrapper for hybrid_search_tool."""
    import asyncio
    return asyncio.run(hybrid_search_tool.ainvoke({
        "query": query,
        "collection": collection,
        "top_k": top_k,
        "alpha": alpha,
        "filters": filters,
        "use_cache": use_cache,
        "min_score": min_score
    }))


# =============================================================================
# Helper Functions
# =============================================================================

async def search_multiple_collections(
    query: str,
    collections: List[str] = ["documents", "sections", "chunks"],
    top_k: int = 10,
    alpha: float = 0.7,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, HybridSearchOutput]:
    """
    Search across multiple collections simultaneously.

    Args:
        query: Search query
        collections: List of collections to search
        top_k: Number of results per collection
        alpha: Vector search weight
        filters: Search filters

    Returns:
        Dictionary mapping collection names to search results
    """
    import asyncio

    tasks = [
        hybrid_search_tool.ainvoke({
            "query": query,
            "collection": col,
            "top_k": top_k,
            "alpha": alpha,
            "filters": filters
        })
        for col in collections
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for col, result in zip(collections, results):
        if isinstance(result, Exception):
            logger.error("Search failed for collection", collection=col, error=str(result))
            output[col] = HybridSearchOutput(
                query=query,
                results=[],
                total_results=0,
                search_method="hybrid",
                cache_hit=False,
                search_time_ms=0
            )
        else:
            output[col] = result

    return output


def fuse_results(
    results_list: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple result lists.

    Args:
        results_list: List of search result lists to fuse
        k: RRF constant (default 60)

    Returns:
        Fused and sorted list of results
    """
    # Calculate RRF scores
    scores = {}

    for results in results_list:
        for rank, result in enumerate(results, 1):
            if result.id not in scores:
                scores[result.id] = {
                    "result": result,
                    "rrf_score": 0.0
                }
            # RRF formula: 1/(k + rank)
            scores[result.id]["rrf_score"] += 1.0 / (k + rank)

    # Sort by RRF score
    fused = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    # Update scores and return
    return [
        SearchResult(
            **item["result"].model_dump(),
            score=item["rrf_score"]
        )
        for item in fused
    ]


# =============================================================================
# Testing
# =============================================================================

async def test_hybrid_search():
    """Test hybrid search with sample queries."""
    test_queries = [
        "ngày nghỉ phép hàng năm",
        "quy trình đăng ký bảo hiểm",
        "lương tháng 13"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = await hybrid_search_tool.ainvoke({
            "query": query,
            "collection": "chunks",
            "top_k": 5,
            "alpha": 0.7
        })

        print(f"\nResults: {result.total_results}")
        print(f"Method: {result.search_method}")
        print(f"Time: {result.search_time_ms:.2f}ms")
        print(f"Cache hit: {result.cache_hit}")

        for i, r in enumerate(result.results[:3], 1):
            print(f"\n[{i}] Score: {r.score:.3f}")
            print(f"    Content: {r.content[:100]}...")
            if r.document_title:
                print(f"    Document: {r.document_title}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hybrid_search())
