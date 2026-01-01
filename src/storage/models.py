# =============================================================================
# SQLAlchemy ORM Models
# =============================================================================
from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
    Boolean,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base, backref
from sqlalchemy.sql import func
import uuid
from enum import Enum

Base = declarative_base()


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentORM(Base):
    """SQLAlchemy ORM model for documents table."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False, unique=True)
    file_type = Column(String(50), nullable=False)
    summary = Column(Text)
    meta = Column("metadata", JSONB, default=dict)  # renamed to 'meta' in code, 'metadata' in DB
    tree_structure = Column(JSONB)
    qdrant_document_id = Column(String(255))
    embedding_status = Column(String(50), default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)

    # Relationships
    sections = relationship(
        "SectionORM",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="SectionORM.position",
    )
    chunks = relationship(
        "ChunkORM",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkORM.position",
    )
    source_references = relationship(
        "CrossReferenceORM",
        foreign_keys="CrossReferenceORM.source_doc_id",
        back_populates="source_document",
        cascade="all, delete-orphan",
    )
    target_references = relationship(
        "CrossReferenceORM",
        foreign_keys="CrossReferenceORM.target_doc_id",
        back_populates="target_document",
        cascade="all, delete-orphan",
    )


class SectionORM(Base):
    """SQLAlchemy ORM model for sections table."""

    __tablename__ = "sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_section_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sections.id", ondelete="CASCADE"),
        index=True,
    )
    heading = Column(String(500), nullable=False)
    level = Column(Integer, nullable=False)
    section_path = Column(String(100))
    summary = Column(Text)
    content = Column(Text)
    meta = Column("metadata", JSONB, default=dict)  # renamed to 'meta' in code, 'metadata' in DB
    qdrant_section_id = Column(String(255))
    position = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Constraints
    __table_args__ = (CheckConstraint("level BETWEEN 1 AND 6", name="check_level_range"),)

    # Relationships
    document = relationship("DocumentORM", back_populates="sections")
    children = relationship(
        "SectionORM",
        backref=backref("parent", remote_side=[id]),
        foreign_keys=[parent_section_id],
    )
    chunks = relationship("ChunkORM", back_populates="section", cascade="all, delete-orphan")


class ChunkORM(Base):
    """SQLAlchemy ORM model for chunks table."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sections.id", ondelete="CASCADE"),
        index=True,
    )
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    meta = Column("metadata", JSONB, default=dict)  # renamed to 'meta' in code, 'metadata' in DB
    qdrant_chunk_id = Column(String(255))
    position = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("DocumentORM", back_populates="chunks")
    section = relationship("SectionORM", back_populates="chunks")


class CrossReferenceORM(Base):
    """SQLAlchemy ORM model for cross_references table."""

    __tablename__ = "cross_references"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_doc_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_doc_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation_type = Column(String(50), nullable=False)
    meta = Column("metadata", JSONB, default=dict)  # renamed to 'meta' in code, 'metadata' in DB
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    source_document = relationship(
        "DocumentORM",
        foreign_keys=[source_doc_id],
        back_populates="source_references",
    )
    target_document = relationship(
        "DocumentORM",
        foreign_keys=[target_doc_id],
        back_populates="target_references",
    )


class QueryORM(Base):
    """SQLAlchemy ORM model for queries table."""

    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    rewritten_query = Column(Text)
    query_type = Column(String(50))
    user_id = Column(String(255))
    session_id = Column(String(255))
    meta = Column("metadata", JSONB, default=dict)  # renamed to 'meta' in code, 'metadata' in DB
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    results = relationship(
        "QueryResultORM",
        back_populates="query",
        cascade="all, delete-orphan",
    )


class QueryResultORM(Base):
    """SQLAlchemy ORM model for query_results table."""

    __tablename__ = "query_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(
        UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"))
    section_id = Column(UUID(as_uuid=True), ForeignKey("sections.id", ondelete="SET NULL"))
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"))
    score = Column(Integer)
    position = Column(Integer)
    meta = Column("metadata", JSONB, default=dict)  # renamed to 'meta' in code, 'metadata' in DB
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    query = relationship("QueryORM", back_populates="results")
