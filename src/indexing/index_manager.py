# =============================================================================
# Index Manager - Multi-Level Document Indexing
# =============================================================================
from uuid import UUID
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.storage import vector_store, metadata_store
from src.indexing.embedder import get_embedder
from src.models.document import Document, Section, Chunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndexResult:
    """Result of indexing operation."""
    success: bool
    document_id: UUID
    sections_indexed: int = 0
    chunks_indexed: int = 0
    error: Optional[str] = None


class IndexManager:
    """
    Manages multi-level indexing across vector store.

    Indexes documents at three levels:
    1. Document level - for topic routing and broad queries
    2. Section level - for navigation and contextual retrieval
    3. Chunk level - for precise content retrieval
    """

    def __init__(self):
        self.embedder = None
        self.embedder_provider = None

    async def _get_embedder(self, provider: Optional[str] = None):
        """Lazy load embedder with optional provider override."""
        if self.embedder is None or (provider and self.embedder_provider != provider):
            self.embedder = get_embedder(provider=provider)
            self.embedder_provider = provider
        return self.embedder

    async def index_document(
        self,
        doc_id: UUID,
        parsed,
        sections: List[Section],
        chunks: List[Chunk],
        provider: Optional[str] = None,
    ) -> IndexResult:
        """
        Index a document at all levels (document, section, chunk).

        Args:
            doc_id: Document UUID
            parsed: ParsedDocument from document parser
            sections: List of sections with tree structure
            chunks: List of content chunks
            provider: Embedding provider override (optional)

        Returns:
            IndexResult with counts and status
        """
        embedder = await self._get_embedder(provider)

        try:
            logger.info(
                "Starting document index",
                doc_id=str(doc_id),
                sections=len(sections),
                chunks=len(chunks)
            )

            # ====================================================================
            # Level 1: Document Embedding
            # ====================================================================
            doc_text = f"{parsed.title}\n\n{parsed.raw_content[:1000]}"
            doc_embedding = await embedder.embed(doc_text)

            await vector_store.upsert(
                collection="documents",
                id=doc_id,
                vector=doc_embedding,
                payload={
                    "title": parsed.title,
                    "summary": parsed.raw_content[:500],
                    "language": parsed.language,
                    "metadata": parsed.metadata,
                }
            )
            logger.debug("Indexed document level", doc_id=str(doc_id))

            # ====================================================================
            # Level 2: Section Embeddings
            # ====================================================================
            # Filter sections with content
            sections_with_content = [s for s in sections if s.content]
            if sections_with_content:
                section_texts = [
                    f"{s.heading}\n{s.summary or s.content[:200]}"
                    for s in sections_with_content
                ]
                section_embeddings = await embedder.embed_batch(section_texts)

                section_points = []
                for section, embedding in zip(sections_with_content, section_embeddings):
                    section_points.append({
                        "id": section.id,
                        "vector": embedding,
                        "payload": {
                            "document_id": str(doc_id),
                            "heading": section.heading,
                            "level": section.level,
                            "section_path": section.section_path,
                            "summary": section.summary,
                        }
                    })

                await vector_store.upsert_batch("sections", section_points)
                logger.debug("Indexed sections", count=len(section_points))
            else:
                logger.debug("No sections with content to index")

            # ====================================================================
            # Level 3: Chunk Embeddings
            # ====================================================================
            if chunks:
                chunk_texts = [c.content for c in chunks]
                chunk_embeddings = await embedder.embed_batch(chunk_texts)

                chunk_points = []
                for chunk, embedding in zip(chunks, chunk_embeddings):
                    chunk_points.append({
                        "id": chunk.id,
                        "vector": embedding,
                        "payload": {
                            "document_id": str(doc_id),
                            "section_id": str(chunk.section_id) if chunk.section_id else None,
                            "content": chunk.content,
                            "token_count": chunk.token_count,
                            "heading": chunk.metadata.get("heading", ""),
                            "level": chunk.metadata.get("level", 0),
                            "section_path": chunk.metadata.get("section_path", ""),
                        }
                    })

                await vector_store.upsert_batch("chunks", chunk_points)
                logger.debug("Indexed chunks", count=len(chunk_points))
            else:
                logger.debug("No chunks to index")

            # Update document status
            await metadata_store.update_document_status(doc_id, "indexed")

            logger.info(
                "Document indexing complete",
                doc_id=str(doc_id),
                sections=len(sections_with_content),
                chunks=len(chunks)
            )

            return IndexResult(
                success=True,
                document_id=doc_id,
                sections_indexed=len(sections_with_content),
                chunks_indexed=len(chunks),
            )

        except Exception as e:
            logger.error(
                "Document indexing failed",
                doc_id=str(doc_id),
                error=str(e)
            )
            await metadata_store.update_document_status(doc_id, "failed")

            return IndexResult(
                success=False,
                document_id=doc_id,
                error=str(e)
            )

    async def delete_document_index(self, doc_id: UUID) -> bool:
        """
        Remove all index entries for a document from all collections.

        Args:
            doc_id: Document UUID

        Returns:
            True if successful
        """
        try:
            logger.info("Deleting document index", doc_id=str(doc_id))

            # Delete from documents collection
            await vector_store.delete("documents", [doc_id])

            # Get all sections for this document
            sections = await metadata_store.get_sections_by_document(doc_id)
            if sections:
                section_ids = [s.id for s in sections]
                await vector_store.delete("sections", section_ids)
                logger.debug("Deleted section vectors", count=len(section_ids))

            # Get all chunks for this document
            chunks = await metadata_store.get_chunks_by_document(doc_id)
            if chunks:
                chunk_ids = [c.id for c in chunks]
                await vector_store.delete("chunks", chunk_ids)
                logger.debug("Deleted chunk vectors", count=len(chunk_ids))

            logger.info("Document index deleted", doc_id=str(doc_id))
            return True

        except Exception as e:
            logger.error(
                "Failed to delete document index",
                doc_id=str(doc_id),
                error=str(e)
            )
            return False

    async def reindex_document(
        self,
        doc_id: UUID,
        parsed,
        sections: List[Section],
        chunks: List[Chunk],
    ) -> IndexResult:
        """
        Re-index a document by deleting old index and creating new one.

        Args:
            doc_id: Document UUID
            parsed: ParsedDocument from document parser
            sections: List of sections with tree structure
            chunks: List of content chunks

        Returns:
            IndexResult with counts and status
        """
        logger.info("Re-indexing document", doc_id=str(doc_id))

        # Delete existing index
        await self.delete_document_index(doc_id)

        # Create new index
        return await self.index_document(doc_id, parsed, sections, chunks)

    async def index_batch(
        self,
        documents: List[tuple[UUID, object, List[Section], List[Chunk]]]
    ) -> List[IndexResult]:
        """
        Index multiple documents in batch.

        Args:
            documents: List of (doc_id, parsed, sections, chunks) tuples

        Returns:
            List of IndexResult objects
        """
        logger.info("Batch indexing", count=len(documents))

        results = []
        for i, (doc_id, parsed, sections, chunks) in enumerate(documents):
            logger.info(f"Indexing document {i+1}/{len(documents)}", doc_id=str(doc_id))
            result = await self.index_document(doc_id, parsed, sections, chunks)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(
            "Batch indexing complete",
            total=len(documents),
            successful=successful,
            failed=len(documents) - successful
        )

        return results

    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed content.

        Returns:
            Dictionary with counts from all collections
        """
        stats = {}

        for collection_name in ["documents", "sections", "chunks"]:
            try:
                count = await vector_store.count(collection_name)
                stats[collection_name] = count
            except Exception as e:
                logger.warning("Failed to get collection stats", collection=collection_name, error=str(e))
                stats[collection_name] = 0

        return stats


# Singleton instance
index_manager = IndexManager()
