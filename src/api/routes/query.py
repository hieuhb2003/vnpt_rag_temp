# =============================================================================
# Query Route - Process natural language queries
# =============================================================================
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from uuid import UUID

from src.agents.orchestrator import orchestrator
from src.storage.cache import cache_store
from src.indexing.embedder import get_embedder
from src.models.query import QueryRequest
from src.models.response import QueryResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


class QueryInput(BaseModel):
    """Input for query processing."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters for retrieval")
    use_cache: bool = Field(default=True, description="Whether to use semantic cache")


class QueryOutput(BaseModel):
    """Output from query processing."""
    query_id: str
    answer: str
    citations: list[dict]
    metadata: dict
    verification: dict
    processing_time_ms: float
    cached: bool


@router.post("", response_model=QueryOutput)
async def process_query(
    input: QueryInput,
    background_tasks: BackgroundTasks,
):
    """
    Process a natural language query against the knowledge base.

    Flow:
    1. Check semantic cache (if enabled)
    2. Process through multi-agent orchestrator
    3. Cache response in background
    """
    logger.info(
        "Query received",
        extra={"query": input.query[:100], "use_cache": input.use_cache}
    )

    try:
        # ================================================================
        # Step 1: Check semantic cache if enabled
        # ================================================================
        query_embedding = None
        if input.use_cache:
            embedder = get_embedder()
            query_embedding = await embedder.embed(input.query)

            cached_response = await cache_store.get_semantic_cache(
                query=input.query,
                query_vector=query_embedding,
                threshold=0.85
            )

            if cached_response:
                logger.info("Cache hit for query", extra={"query": input.query[:50]})
                return QueryOutput(
                    query_id=cached_response.get("query_id", ""),
                    answer=cached_response.get("answer", ""),
                    citations=cached_response.get("citations", []),
                    metadata=cached_response.get("metadata", {}),
                    verification=cached_response.get("verification", {}),
                    processing_time_ms=cached_response.get("processing_time_ms", 0),
                    cached=True
                )

        # ================================================================
        # Step 2: Process through orchestrator
        # ================================================================
        response = await orchestrator.process_query(
            query=input.query,
            thread_id=input.conversation_id
        )

        output = QueryOutput(
            query_id=str(response["query_id"]),
            answer=response["answer"],
            citations=response.get("citations", []),
            metadata=response.get("metadata", {}),
            verification=response.get("verification", {}),
            processing_time_ms=response.get("processing_time_ms", 0),
            cached=False
        )

        # ================================================================
        # Step 3: Cache in background
        # ================================================================
        if input.use_cache and query_embedding:
            background_tasks.add_task(
                cache_store.set_semantic_cache,
                input.query,
                query_embedding,
                output.model_dump()
            )

        logger.info(
            "Query processed successfully",
            extra={
                "query_id": output.query_id,
                "cached": output.cached,
                "processing_time_ms": output.processing_time_ms
            }
        )

        return output

    except Exception as e:
        logger.error(
            f"Query processing failed: {e}",
            extra={"query": input.query[:100]},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/health")
async def query_health():
    """Health check for query endpoint."""
    return {"status": "healthy", "service": "query"}
