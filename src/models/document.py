# =============================================================================
# Document Models
# =============================================================================
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Dict, List
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import json


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata with support for custom fields."""

    category: Optional[str] = Field(None, description="Document category (e.g., 'technical', 'legal', 'hr')")
    tags: List[str] = Field(default_factory=list, description="Tags for document classification")
    author: Optional[str] = Field(None, description="Document author or creator")
    language: str = Field(default="vi", description="Document language code (ISO 639-1)")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    source: Optional[str] = Field(None, description="Document source system or location")
    document_date: Optional[datetime] = Field(None, description="Document creation/publication date")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        use_enum_values=True,
    )


class Document(BaseModel):
    """Core document model for RAG system."""

    id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    file_path: str = Field(..., max_length=1000, description="Original file path or MinIO object name")
    file_type: str = Field(..., max_length=50, description="File extension (e.g., 'pdf', 'docx', 'md')")
    summary: Optional[str] = Field(None, description="Auto-generated document summary")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    tree_structure: Optional[Dict[str, Any]] = Field(None, description="Hierarchical section tree structure")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    version: int = Field(default=1, ge=1, description="Document version for tracking changes")
    is_active: bool = Field(default=True, description="Whether document is active for retrieval")

    # Qdrant references
    qdrant_document_id: Optional[str] = Field(None, description="Qdrant point ID for document embedding")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        use_enum_values=True,
        from_attributes=True,
        json_schema_extra={
            "example": {
                "title": "Employee Handbook 2024",
                "file_path": "documents/hr/handbook_2024.pdf",
                "file_type": "pdf",
                "summary": "Company policies and procedures",
                "metadata": {
                    "category": "hr",
                    "tags": ["policy", "onboarding"],
                    "language": "vi"
                }
            }
        }
    )


class Section(BaseModel):
    """Document section with hierarchical structure support."""

    id: UUID = Field(default_factory=uuid4, description="Unique section identifier")
    document_id: UUID = Field(..., description="Parent document ID")
    parent_section_id: Optional[UUID] = Field(None, description="Parent section ID for nested sections")
    heading: str = Field(..., min_length=1, max_length=500, description="Section heading text")
    level: int = Field(..., ge=1, le=6, description="Heading level (1=H1, 2=H2, etc.)")
    section_path: str = Field(..., max_length=1000, description="Section path (e.g., '1.2.3')")
    summary: Optional[str] = Field(None, description="Auto-generated section summary")
    content: Optional[str] = Field(None, description="Section content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Section metadata")
    position: int = Field(default=0, ge=0, description="Position within document")

    # Qdrant references
    qdrant_section_id: Optional[str] = Field(None, description="Qdrant point ID for section embedding")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "heading": "Vacation Policy",
                "level": 2,
                "section_path": "1.2",
                "summary": "Annual leave entitlements",
                "position": 1
            }
        }
    )


class Chunk(BaseModel):
    """Content chunk for precise retrieval."""

    id: UUID = Field(default_factory=uuid4, description="Unique chunk identifier")
    document_id: UUID = Field(..., description="Parent document ID")
    section_id: Optional[UUID] = Field(None, description="Parent section ID if applicable")
    content: str = Field(..., min_length=1, description="Chunk text content")
    token_count: int = Field(..., ge=0, description="Estimated token count")
    position: int = Field(..., ge=0, description="Position within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    # Qdrant references
    qdrant_chunk_id: Optional[str] = Field(None, description="Qdrant point ID for chunk embedding")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Employees are entitled to 20 days of paid vacation per year...",
                "token_count": 15,
                "position": 0,
                "metadata": {"heading": "Vacation Policy"}
            }
        }
    )


class RelationType(str, Enum):
    """Types of relationships between documents."""

    RELATED = "related"
    PREREQUISITE = "prerequisite"
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"
    SEE_ALSO = "see_also"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"


class CrossReference(BaseModel):
    """Cross-reference relationships between documents."""

    id: UUID = Field(default_factory=uuid4, description="Unique reference identifier")
    source_doc_id: UUID = Field(..., description="Source document ID")
    target_doc_id: UUID = Field(..., description="Target document ID")
    relation_type: RelationType = Field(..., description="Type of relationship")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional relationship metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Reference creation timestamp")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        use_enum_values=True,
        from_attributes=True,
        json_schema_extra={
            "example": {
                "source_doc_id": "123e4567-e89b-12d3-a456-426614174000",
                "target_doc_id": "223e4567-e89b-12d3-a456-426614174001",
                "relation_type": "see_also",
                "metadata": {"reason": "Both documents cover similar topics"}
            }
        }
    )
