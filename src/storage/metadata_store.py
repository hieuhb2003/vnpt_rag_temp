# =============================================================================
# Metadata Store - PostgreSQL with Async SQLAlchemy
# =============================================================================
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete, func as sql_func
from sqlalchemy.orm import selectinload
from typing import Optional, Any, List, Dict
from uuid import UUID
from contextlib import asynccontextmanager
from datetime import datetime

from src.config.settings import get_settings
from src.storage.models import (
    Base,
    DocumentORM,
    SectionORM,
    ChunkORM,
    CrossReferenceORM,
    QueryORM,
    QueryResultORM,
    DocumentStatus,
)
from src.models.document import Document, DocumentMetadata, Section, Chunk, CrossReference, RelationType
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MetadataStore:
    """Async PostgreSQL metadata store."""

    def __init__(self):
        self.engine = None
        self.async_session = None

    async def connect(self):
        """Initialize async database connection."""
        self.engine = create_async_engine(
            settings.postgres_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info(
            "Connected to PostgreSQL",
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
        )

    async def disconnect(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Disconnected from PostgreSQL")

    async def init_tables(self):
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")

    @asynccontextmanager
    async def session(self):
        """Get an async session context."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # =============================================================================
    # Document Operations
    # =============================================================================

    async def create_document(self, doc: Document) -> Document:
        """Create a new document."""
        async with self.session() as session:
            db_doc = DocumentORM(
                id=doc.id,
                title=doc.title,
                file_path=doc.file_path,
                file_type=doc.file_type,
                summary=doc.summary,
                meta=doc.metadata.model_dump(),
                tree_structure=doc.tree_structure,
                embedding_status=str(doc.status) if hasattr(doc.status, 'value') else doc.status,
                version=doc.version,
                is_active=doc.is_active,
            )
            session.add(db_doc)
            await session.flush()
            return doc

    async def get_document(self, doc_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(DocumentORM).where(DocumentORM.id == doc_id)
            )
            db_doc = result.scalar_one_or_none()
            if db_doc:
                return self._orm_to_document(db_doc)
            return None

    async def get_document_with_tree(self, doc_id: UUID) -> Optional[Document]:
        """Get document with full section tree."""
        async with self.session() as session:
            result = await session.execute(
                select(DocumentORM)
                .options(selectinload(DocumentORM.sections))
                .where(DocumentORM.id == doc_id)
            )
            db_doc = result.scalar_one_or_none()
            if db_doc:
                doc = self._orm_to_document(db_doc)
                # Build tree structure from sections
                doc.tree_structure = self._build_tree(db_doc.sections)
                return doc
            return None

    async def update_document_status(self, doc_id: UUID, status: str):
        """Update document indexing status."""
        async with self.session() as session:
            await session.execute(
                update(DocumentORM)
                .where(DocumentORM.id == doc_id)
                .values(embedding_status=status)
            )

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> List[Document]:
        """List documents with optional filters."""
        async with self.session() as session:
            query = select(DocumentORM)

            if status:
                query = query.where(DocumentORM.embedding_status == status)
            if file_type:
                query = query.where(DocumentORM.file_type == file_type)

            query = query.order_by(DocumentORM.created_at.desc()).limit(limit).offset(offset)

            result = await session.execute(query)
            return [self._orm_to_document(d) for d in result.scalars().all()]

    async def count_documents(self, status: Optional[str] = None) -> int:
        """Count documents."""
        async with self.session() as session:
            query = sql_func.count(DocumentORM.id)
            if status:
                query = select(query).where(DocumentORM.embedding_status == status)
            else:
                query = select(query)

            result = await session.execute(query)
            return result.scalar()

    # =============================================================================
    # Section Operations
    # =============================================================================

    async def create_sections(self, sections: List[Section]) -> List[Section]:
        """Create sections in batch."""
        async with self.session() as session:
            for section in sections:
                db_section = SectionORM(
                    id=section.id,
                    document_id=section.document_id,
                    parent_section_id=section.parent_section_id,
                    heading=section.heading,
                    level=section.level,
                    section_path=section.section_path,
                    summary=section.summary,
                    content=section.content,
                    metadata=section.metadata,
                    position=section.position,
                )
                session.add(db_section)
            await session.flush()
            return sections

    async def get_sections_by_document(self, doc_id: UUID) -> List[Section]:
        """Get all sections for a document."""
        async with self.session() as session:
            result = await session.execute(
                select(SectionORM)
                .where(SectionORM.document_id == doc_id)
                .order_by(SectionORM.position)
            )
            return [self._orm_to_section(s) for s in result.scalars().all()]

    async def get_section(self, section_id: UUID) -> Optional[Section]:
        """Get section by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(SectionORM).where(SectionORM.id == section_id)
            )
            db_section = result.scalar_one_or_none()
            if db_section:
                return self._orm_to_section(db_section)
            return None

    async def update_section_qdrant_id(self, section_id: UUID, qdrant_id: str):
        """Update section Qdrant ID."""
        async with self.session() as session:
            await session.execute(
                update(SectionORM)
                .where(SectionORM.id == section_id)
                .values(qdrant_section_id=qdrant_id)
            )

    # =============================================================================
    # Chunk Operations
    # =============================================================================

    async def create_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Create chunks in batch."""
        async with self.session() as session:
            for chunk in chunks:
                db_chunk = ChunkORM(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    section_id=chunk.section_id,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    position=chunk.position,
                    metadata=chunk.metadata,
                )
                session.add(db_chunk)
            await session.flush()
            return chunks

    async def get_chunks_by_ids(self, chunk_ids: List[UUID]) -> List[Chunk]:
        """Get chunks by IDs."""
        async with self.session() as session:
            result = await session.execute(
                select(ChunkORM).where(ChunkORM.id.in_(chunk_ids))
            )
            return [self._orm_to_chunk(c) for c in result.scalars().all()]

    async def get_chunks_by_document(self, doc_id: UUID) -> List[Chunk]:
        """Get all chunks for a document."""
        async with self.session() as session:
            result = await session.execute(
                select(ChunkORM)
                .where(ChunkORM.document_id == doc_id)
                .order_by(ChunkORM.position)
            )
            return [self._orm_to_chunk(c) for c in result.scalars().all()]

    async def update_chunk_qdrant_id(self, chunk_id: UUID, qdrant_id: str):
        """Update chunk Qdrant ID."""
        async with self.session() as session:
            await session.execute(
                update(ChunkORM)
                .where(ChunkORM.id == chunk_id)
                .values(qdrant_chunk_id=qdrant_id)
            )

    async def count_chunks(self, document_id: Optional[UUID] = None) -> int:
        """Count chunks."""
        async with self.session() as session:
            query = sql_func.count(ChunkORM.id)
            if document_id:
                query = select(query).where(ChunkORM.document_id == document_id)
            else:
                query = select(query)

            result = await session.execute(query)
            return result.scalar()

    # =============================================================================
    # Cross Reference Operations
    # =============================================================================

    async def create_cross_reference(self, xref: CrossReference) -> CrossReference:
        """Create a cross-reference."""
        async with self.session() as session:
            db_xref = CrossReferenceORM(
                id=xref.id,
                source_doc_id=xref.source_doc_id,
                target_doc_id=xref.target_doc_id,
                relation_type=xref.relation_type.value,
                metadata=xref.metadata,
            )
            session.add(db_xref)
            await session.flush()
            return xref

    async def get_cross_references(
        self, doc_id: UUID, reference_type: Optional[str] = None
    ) -> List[CrossReference]:
        """Get cross-references for a document."""
        async with self.session() as session:
            query = select(CrossReferenceORM).where(
                (CrossReferenceORM.source_doc_id == doc_id) |
                (CrossReferenceORM.target_doc_id == doc_id)
            )

            if reference_type:
                query = query.where(CrossReferenceORM.relation_type == reference_type)

            result = await session.execute(query)
            return [self._orm_to_cross_reference(x) for x in result.scalars().all()]

    # =============================================================================
    # Query Tracking Operations
    # =============================================================================

    async def create_query(
        self,
        query_text: str,
        rewritten_query: Optional[str] = None,
        query_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Create a query record."""
        async with self.session() as session:
            db_query = QueryORM(
                query_text=query_text,
                rewritten_query=rewritten_query,
                query_type=query_type,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            session.add(db_query)
            await session.flush()
            return db_query.id

    async def create_query_result(
        self,
        query_id: UUID,
        document_id: Optional[UUID],
        section_id: Optional[UUID],
        chunk_id: Optional[UUID],
        score: float,
        position: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a query result record."""
        async with self.session() as session:
            db_result = QueryResultORM(
                query_id=query_id,
                document_id=document_id,
                section_id=section_id,
                chunk_id=chunk_id,
                score=int(score * 100),  # Store as integer
                position=position,
                metadata=metadata or {},
            )
            session.add(db_result)

    # =============================================================================
    # Helper Methods - ORM <-> Pydantic Conversion
    # =============================================================================

    def _orm_to_document(self, db_doc: DocumentORM) -> Document:
        """Convert DocumentORM to Document Pydantic model."""
        return Document(
            id=db_doc.id,
            title=db_doc.title,
            file_path=db_doc.file_path,
            file_type=db_doc.file_type,
            summary=db_doc.summary,
            metadata=DocumentMetadata(**db_doc.meta) if db_doc.meta else DocumentMetadata(),
            tree_structure=db_doc.tree_structure,
            status=DocumentStatus(db_doc.embedding_status) if db_doc.embedding_status else DocumentStatus.PENDING,
            created_at=db_doc.created_at,
            updated_at=db_doc.updated_at,
            version=db_doc.version,
            is_active=db_doc.is_active if db_doc.is_active is not None else True,
            qdrant_document_id=db_doc.qdrant_document_id,
        )

    def _orm_to_section(self, db_section: SectionORM) -> Section:
        """Convert SectionORM to Section Pydantic model."""
        return Section(
            id=db_section.id,
            document_id=db_section.document_id,
            parent_section_id=db_section.parent_section_id,
            heading=db_section.heading,
            level=db_section.level,
            section_path=db_section.section_path or "",
            summary=db_section.summary,
            content=db_section.content,
            metadata=db_section.meta or {},
            position=db_section.position or 0,
            qdrant_section_id=db_section.qdrant_section_id,
        )

    def _orm_to_chunk(self, db_chunk: ChunkORM) -> Chunk:
        """Convert ChunkORM to Chunk Pydantic model."""
        return Chunk(
            id=db_chunk.id,
            document_id=db_chunk.document_id,
            section_id=db_chunk.section_id,
            content=db_chunk.content,
            token_count=db_chunk.token_count or 0,
            position=db_chunk.position or 0,
            metadata=db_chunk.meta or {},
            qdrant_chunk_id=db_chunk.qdrant_chunk_id,
        )

    def _orm_to_cross_reference(self, db_xref: CrossReferenceORM) -> CrossReference:
        """Convert CrossReferenceORM to CrossReference Pydantic model."""
        return CrossReference(
            id=db_xref.id,
            source_doc_id=db_xref.source_doc_id,
            target_doc_id=db_xref.target_doc_id,
            relation_type=RelationType(db_xref.relation_type),
            metadata=db_xref.meta or {},
            created_at=db_xref.created_at,
        )

    def _build_tree(self, sections: List[SectionORM]) -> Dict[str, Any]:
        """Build tree structure from flat section list."""
        # Build a map of id -> section
        section_map = {s.id: s for s in sections}

        # Build tree
        tree = {}
        for section in sections:
            section_dict = {
                "id": str(section.id),
                "heading": section.heading,
                "level": section.level,
                "section_path": section.section_path,
                "children": [],
            }

            if section.parent_section_id and section.parent_section_id in section_map:
                # Add as child of parent
                parent = section_map[section.parent_section_id]
                if "children" not in tree.get(str(parent.id), {}):
                    tree[str(parent.id)] = {"children": []}
                if str(parent.id) not in tree:
                    tree[str(parent.id)] = {"children": []}
                tree[str(parent.id)]["children"].append(section_dict)
            else:
                # Top-level section
                if str(section.id) not in tree:
                    tree[str(section.id)] = section_dict
                else:
                    tree[str(section.id)]["children"].append(section_dict)

        return tree

    async def health_check(self) -> Dict[str, Any]:
        """Check database health and return statistics."""
        async with self.session() as session:
            # Count records in each table
            doc_count = await session.execute(sql_func.count(DocumentORM.id))
            section_count = await session.execute(sql_func.count(SectionORM.id))
            chunk_count = await session.execute(sql_func.count(ChunkORM.id))
            xref_count = await session.execute(sql_func.count(CrossReferenceORM.id))
            query_count = await session.execute(sql_func.count(QueryORM.id))

            return {
                "status": "healthy",
                "documents": doc_count.scalar() or 0,
                "sections": section_count.scalar() or 0,
                "chunks": chunk_count.scalar() or 0,
                "cross_references": xref_count.scalar() or 0,
                "queries": query_count.scalar() or 0,
            }


# Singleton instance
metadata_store = MetadataStore()
