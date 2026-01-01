# =============================================================================
# Query Models
# =============================================================================
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Dict, List
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Classification of query types for routing."""

    FACTOID = "factoid"  # Factual questions (Who, What, Where, When)
    PROCEDURAL = "procedural"  # How-to questions
    COMPARATIVE = "comparative"  # Compare X vs Y
    AGGREGATION = "aggregation"  # Aggregate information from multiple sources
    DEFINITIONAL = "definitional"  # What is X
    DIAGNOSTIC = "diagnostic"  # Why is this happening
    PREDICTIVE = "predictive"  # What will happen
    OPINION = "opinion"  # Subjective questions


class SearchMode(str, Enum):
    """Search mode selection."""

    VECTOR = "vector"  # Pure semantic search
    KEYWORD = "keyword"  # Pure keyword/BM25 search
    HYBRID = "hybrid"  # Combined vector + keyword search


class QueryRequest(BaseModel):
    """User query request with conversation context."""

    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous conversation turns for context"
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters for document retrieval (e.g., category, date range)")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to retrieve")
    search_mode: SearchMode = Field(default=SearchMode.HYBRID, description="Search mode to use")
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")
    include_citations: bool = Field(default=True, description="Whether to include source citations")
    language: str = Field(default="vi", description="Query language code (ISO 639-1)")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "query": "How many days of vacation am I entitled to?",
                "conversation_history": [],
                "filters": {"category": "hr"},
                "top_k": 5,
                "search_mode": "hybrid",
                "include_citations": True
            }
        }
    )


class RewrittenQuery(BaseModel):
    """Query after rewriting for better retrieval."""

    original: str = Field(..., description="Original user query")
    rewritten: str = Field(..., description="Rewritten query for retrieval")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    query_type: QueryType = Field(..., description="Classified query type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for classification")
    reasoning: Optional[str] = Field(None, description="Explanation for the rewrite")
    expansions: List[str] = Field(default_factory=list, description="Query expansions for retrieval")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "original": "vacation days",
                "rewritten": "How many paid vacation days are employees entitled to per year?",
                "keywords": ["vacation", "paid leave", "annual leave", "days off"],
                "query_type": "factoid",
                "confidence": 0.95
            }
        }
    )


class SubQuery(BaseModel):
    """Individual sub-query from decomposition."""

    id: int = Field(..., ge=0, description="Sub-query identifier")
    query: str = Field(..., description="Sub-query text")
    query_type: QueryType = Field(..., description="Type of this sub-query")
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of sub-query IDs this depends on"
    )
    status: str = Field(default="pending", description="Execution status (pending, in_progress, completed)")

    model_config = ConfigDict(use_enum_values=True)


class DecomposedQuery(BaseModel):
    """Query decomposed into multiple sub-queries."""

    original_query: str = Field(..., description="Original user query")
    sub_queries: List[SubQuery] = Field(..., description="Decomposed sub-queries")
    dependencies: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="Dependency graph: sub_query_id -> [dependent_on_ids]"
    )
    expected_answer_types: List[str] = Field(
        default_factory=list,
        description="Expected answer types (e.g., ['number', 'list', 'boolean'])"
    )
    execution_order: List[int] = Field(
        default_factory=list,
        description="Order in which to execute sub-queries"
    )
    requires_aggregation: bool = Field(
        default=False,
        description="Whether results need to be aggregated across sub-queries"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_query": "Compare vacation policies for full-time and part-time employees",
                "sub_queries": [
                    {
                        "id": 0,
                        "query": "What is the vacation policy for full-time employees?",
                        "query_type": "factoid",
                        "dependencies": []
                    },
                    {
                        "id": 1,
                        "query": "What is the vacation policy for part-time employees?",
                        "query_type": "factoid",
                        "dependencies": []
                    }
                ],
                "dependencies": {},
                "expected_answer_types": ["comparison"],
                "execution_order": [0, 1],
                "requires_aggregation": True
            }
        }
    )


class QueryContext(BaseModel):
    """Additional context for query processing."""

    user_location: Optional[str] = Field(None, description="User location for localized answers")
    user_role: Optional[str] = Field(None, description="User role for permission-aware answers")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")
    device_type: Optional[str] = Field(None, description="Client device type for formatting")
    previous_queries: List[str] = Field(
        default_factory=list,
        description="Recent queries from the same session"
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )


class QueryMetadata(BaseModel):
    """Metadata for query tracking and analytics."""

    query_id: UUID = Field(default_factory=uuid4, description="Unique query identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    query_hash: str = Field(..., description="Hash of query text for caching")
    retrieval_count: int = Field(default=0, ge=0, description="Number of retrieval attempts")
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )
