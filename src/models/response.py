# =============================================================================
# Response Models
# =============================================================================
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Dict, List
from uuid import UUID
from datetime import datetime
from enum import Enum


class Citation(BaseModel):
    """Source citation for answer attribution."""

    document_id: UUID = Field(..., description="Document unique identifier")
    document_title: str = Field(..., description="Document title")
    section_path: Optional[str] = Field(None, description="Section path (e.g., '1.2.3')")
    chunk_id: Optional[UUID] = Field(None, description="Chunk unique identifier")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    snippet: str = Field(..., description="Text snippet from source")
    position: Optional[int] = Field(None, ge=0, description="Position in document")
    url: Optional[str] = Field(None, description="URL to document (if available)")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "document_title": "Employee Handbook 2024",
                "section_path": "1.2",
                "relevance_score": 0.95,
                "snippet": "Employees are entitled to 20 days of paid vacation per year..."
            }
        }
    )


class VerificationTier(str, Enum):
    """Verification tier type."""

    TIER_1 = "tier_1"  # Fast semantic similarity check
    TIER_2 = "tier_2"  # LLM-based groundedness evaluation


class ClaimEvidence(BaseModel):
    """Evidence for a claim in groundedness verification."""

    claim: str = Field(..., description="Text claim being verified")
    is_supported: bool = Field(..., description="Whether claim is supported by sources")
    supporting_citations: List[int] = Field(
        default_factory=list,
        description="Indices of supporting citations"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in verification")


class VerificationResult(BaseModel):
    """Result of answer groundedness verification."""

    is_grounded: bool = Field(..., description="Whether answer is grounded in sources")
    tier: int = Field(..., ge=1, le=2, description="Verification tier used (1 or 2)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of unsupported claims found"
    )
    claim_evidence: List[ClaimEvidence] = Field(
        default_factory=list,
        description="Detailed evidence for each claim"
    )
    tier_1_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score from Tier 1"
    )
    reasoning: Optional[str] = Field(None, description="Explanation for verification result")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "is_grounded": True,
                "tier": 1,
                "confidence": 0.92,
                "unsupported_claims": [],
                "tier_1_score": 0.92
            }
        }
    )


class ProcessingStep(str, Enum):
    """Types of processing steps."""

    QUERY_REWRITE = "query_rewrite"
    QUERY_DECOMPOSITION = "query_decomposition"
    RETRIEVAL = "retrieval"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"


class ProcessingDetail(BaseModel):
    """Details of a processing step."""

    step: ProcessingStep = Field(..., description="Type of processing step")
    duration_ms: float = Field(..., ge=0.0, description="Step duration in milliseconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional step-specific metadata"
    )
    success: bool = Field(..., description="Whether step completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class QueryResponse(BaseModel):
    """Complete response to a user query."""

    query_id: UUID = Field(..., description="Unique query identifier")
    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source citations for the answer"
    )
    verification: VerificationResult = Field(..., description="Groundedness verification result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )
    processing_time_ms: float = Field(..., ge=0.0, description="Total processing time")
    cached: bool = Field(default=False, description="Whether response was served from cache")
    language: str = Field(default="vi", description="Response language code")

    # Query tracking
    original_query: str = Field(..., description="Original user query")
    rewritten_query: Optional[str] = Field(None, description="Rewritten query if applicable")

    # Processing details
    processing_steps: List[ProcessingDetail] = Field(
        default_factory=list,
        description="Details of each processing step"
    )

    # Retrieval info
    retrieval_count: int = Field(default=0, ge=0, description="Number of documents retrieved")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Filters used in retrieval")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "answer": "According to the Employee Handbook, you are entitled to 20 days of paid vacation per year.",
                "citations": [
                    {
                        "document_id": "123e4567-e89b-12d3-a456-426614174000",
                        "document_title": "Employee Handbook 2024",
                        "relevance_score": 0.95,
                        "snippet": "Employees are entitled to 20 days of paid vacation..."
                    }
                ],
                "verification": {
                    "is_grounded": True,
                    "tier": 1,
                    "confidence": 0.92,
                    "unsupported_claims": []
                },
                "processing_time_ms": 1250.5,
                "cached": False,
                "original_query": "How many vacation days do I get?"
            }
        }
    )


class RetrievalResult(BaseModel):
    """Result from a single retrieval operation."""

    chunk_id: UUID = Field(..., description="Chunk unique identifier")
    document_id: UUID = Field(..., description="Document unique identifier")
    section_id: Optional[UUID] = Field(None, description="Section unique identifier")
    content: str = Field(..., description="Chunk content text")
    score: float = Field(..., ge=0.0, le=1.0, description="Retrieval relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")

    # Document info
    document_title: str = Field(..., description="Document title")
    section_path: Optional[str] = Field(None, description="Section path")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "chunk_id": "123e4567-e89b-12d3-a456-426614174001",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Employees are entitled to 20 days of paid vacation per year...",
                "score": 0.95,
                "metadata": {"heading": "Vacation Policy", "level": 2},
                "document_title": "Employee Handbook 2024",
                "section_path": "1.2"
            }
        }
    )


class RetrievalBatch(BaseModel):
    """Batch of retrieval results with metadata."""

    results: List[RetrievalResult] = Field(..., description="Retrieval results")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    query_embedding: Optional[List[float]] = Field(None, description="Query embedding used")
    search_mode: str = Field(..., description="Search mode used (vector/keyword/hybrid)")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Filters applied")
    retrieval_time_ms: float = Field(..., ge=0.0, description="Time taken for retrieval")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [],
                "total_results": 10,
                "search_mode": "hybrid",
                "retrieval_time_ms": 150.5
            }
        }
    )


class HealthStatus(str, Enum):
    """Health status for components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for a system component."""

    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message or error")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check time")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )


class SystemHealth(BaseModel):
    """Overall system health status."""

    status: HealthStatus = Field(..., description="Overall system status")
    version: str = Field(..., description="System version")
    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Health status of individual components"
    )
    uptime_seconds: float = Field(..., ge=0.0, description="System uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )
