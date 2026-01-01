# =============================================================================
# Unit Tests for Index Manager
# =============================================================================
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.indexing.index_manager import IndexManager, IndexResult
from src.models.document import Document, Section, Chunk
from src.indexing.document_parser import ParsedDocument


class TestIndexManager:
    """Test IndexManager functionality."""

    @pytest.fixture
    def index_manager(self):
        """Create an IndexManager instance."""
        return IndexManager()

    @pytest.fixture
    def sample_parsed_document(self):
        """Create a sample parsed document."""
        return ParsedDocument(
            title="Test Document",
            language="en",
            sections=[],
            raw_content="# Test\n\nThis is test content." * 20,
            metadata={"source": "test.md"},
        )

    @pytest.fixture
    def sample_sections(self):
        """Create sample sections."""
        doc_id = uuid4()
        return [
            Section(
                id=uuid4(),
                document_id=doc_id,
                heading="Introduction",
                level=1,
                section_path="1",
                content="This is the introduction section with some content.",
                summary="Introduction section summary",
                metadata={},
                position=0,
            ),
            Section(
                id=uuid4(),
                document_id=doc_id,
                heading="Getting Started",
                level=2,
                section_path="1.1",
                content="Getting started with the application.",
                summary=None,
                metadata={},
                position=1,
            ),
            Section(
                id=uuid4(),
                document_id=doc_id,
                heading="Empty Section",
                level=2,
                section_path="1.2",
                content="",  # Empty content - should be filtered
                summary=None,
                metadata={},
                position=2,
            ),
        ]

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        doc_id = uuid4()
        section_id = uuid4()
        return [
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                section_id=section_id,
                content="This is chunk one with test content.",
                token_count=8,
                position=0,
                metadata={"heading": "Introduction", "level": 1, "section_path": "1"},
            ),
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                section_id=section_id,
                content="This is chunk two with more test content.",
                token_count=9,
                position=1,
                metadata={"heading": "Getting Started", "level": 2, "section_path": "1.1"},
            ),
        ]

    def test_init_index_manager(self, index_manager):
        """Test IndexManager initialization."""
        assert index_manager.embedder is None
        assert index_manager.embedder_provider is None

    @pytest.mark.asyncio
    async def test_get_embedder_lazy_load(self, index_manager):
        """Test that embedder is lazy loaded."""
        from src.indexing.embedder import MockEmbedder

        embedder = await index_manager._get_embedder(provider="mock")

        assert embedder is not None
        assert isinstance(embedder, MockEmbedder)
        assert index_manager.embedder is embedder
        assert index_manager.embedder_provider == "mock"

    @pytest.mark.asyncio
    async def test_get_embedder_provider_switch(self, index_manager):
        """Test provider switching."""
        from src.indexing.embedder import MockEmbedder

        # First provider
        embedder1 = await index_manager._get_embedder(provider="mock")
        assert index_manager.embedder_provider == "mock"

        # Switch to different provider
        with patch("src.indexing.index_manager.get_embedder") as mock_get_embedder:
            mock_embedder2 = MagicMock()
            mock_get_embedder.return_value = mock_embedder2

            embedder2 = await index_manager._get_embedder(provider="openai")

            assert embedder2 is mock_embedder2
            assert index_manager.embedder is mock_embedder2

    @pytest.mark.asyncio
    async def test_index_document_success(
        self, index_manager, sample_parsed_document, sample_sections, sample_chunks, clean_storage
    ):
        """Test successful document indexing at all levels."""
        from src.storage import vector_store, metadata_store

        doc_id = uuid4()

        # Mock the storage operations
        vector_store.upsert = AsyncMock()
        vector_store.upsert_batch = AsyncMock()
        metadata_store.update_document_status = AsyncMock()

        result = await index_manager.index_document(
            doc_id, sample_parsed_document, sample_sections, sample_chunks, provider="mock"
        )

        # Verify result
        assert result.success is True
        assert result.document_id == doc_id
        assert result.sections_indexed == 2  # Empty section filtered out
        assert result.chunks_indexed == 2
        assert result.error is None

        # Verify document level was indexed
        vector_store.upsert.assert_called_once()
        doc_call = vector_store.upsert.call_args
        assert doc_call[1]["collection"] == "documents"
        assert doc_call[1]["id"] == doc_id

        # Verify sections were indexed
        vector_store.upsert_batch.assert_called()
        section_call = vector_store.upsert_batch.call_args_list[0]
        assert section_call[0][0] == "sections"
        assert len(section_call[0][1]) == 2  # Empty section filtered

        # Verify chunks were indexed
        chunk_call = vector_store.upsert_batch.call_args_list[1]
        assert chunk_call[0][0] == "chunks"
        assert len(chunk_call[0][1]) == 2

        # Verify status was updated
        metadata_store.update_document_status.assert_called_once_with(doc_id, "indexed")

    @pytest.mark.asyncio
    async def test_index_document_with_provider_override(
        self, index_manager, sample_parsed_document, sample_sections, sample_chunks
    ):
        """Test indexing with provider override."""
        from src.storage import vector_store, metadata_store

        doc_id = uuid4()

        vector_store.upsert = AsyncMock()
        vector_store.upsert_batch = AsyncMock()
        metadata_store.update_document_status = AsyncMock()

        with patch("src.indexing.index_manager.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 128)
            mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 128] * 4)
            mock_get_embedder.return_value = mock_embedder

            result = await index_manager.index_document(
                doc_id, sample_parsed_document, sample_sections, sample_chunks, provider="openai"
            )

            assert result.success is True
            mock_get_embedder.assert_called_once_with(provider="openai")

    @pytest.mark.asyncio
    async def test_index_document_empty_sections_chunks(
        self, index_manager, sample_parsed_document, clean_storage
    ):
        """Test indexing document with no sections or chunks."""
        from src.storage import vector_store, metadata_store

        doc_id = uuid4()

        vector_store.upsert = AsyncMock()
        vector_store.upsert_batch = AsyncMock()
        metadata_store.update_document_status = AsyncMock()

        result = await index_manager.index_document(
            doc_id, sample_parsed_document, [], [], provider="mock"
        )

        assert result.success is True
        assert result.sections_indexed == 0
        assert result.chunks_indexed == 0

        # Document should still be indexed
        vector_store.upsert.assert_called_once()

        # Batch should not be called for empty sections/chunks
        assert vector_store.upsert_batch.call_count == 0

    @pytest.mark.asyncio
    async def test_index_document_error_handling(
        self, index_manager, sample_parsed_document, sample_sections
    ):
        """Test error handling during indexing."""
        from src.storage import vector_store, metadata_store

        doc_id = uuid4()

        # Mock vector store to raise error
        vector_store.upsert = AsyncMock(side_effect=Exception("Vector store error"))
        metadata_store.update_document_status = AsyncMock()

        result = await index_manager.index_document(
            doc_id, sample_parsed_document, sample_sections, []
        )

        assert result.success is False
        assert result.document_id == doc_id
        # Error message should contain information about the failure
        assert result.error is not None and len(result.error) > 0

        # Status should be updated to failed
        metadata_store.update_document_status.assert_called_once_with(doc_id, "failed")

    @pytest.mark.asyncio
    async def test_delete_document_index(self, index_manager):
        """Test deleting document index from all collections."""
        from src.storage import vector_store, metadata_store

        doc_id = uuid4()
        section_id = uuid4()
        chunk_id = uuid4()

        # Mock storage operations
        vector_store.delete = AsyncMock()
        metadata_store.get_sections_by_document = AsyncMock(
            return_value=[Section(
                id=section_id,
                document_id=doc_id,
                heading="Test",
                level=1,
                section_path="1",
                content="Content",
                metadata={},
                position=0,
            )]
        )
        metadata_store.get_chunks_by_document = AsyncMock(
            return_value=[Chunk(
                id=chunk_id,
                document_id=doc_id,
                section_id=section_id,
                content="Chunk content",
                token_count=3,
                position=0,
                metadata={},
            )]
        )

        result = await index_manager.delete_document_index(doc_id)

        assert result is True

        # Verify document was deleted
        vector_store.delete.assert_called()

        # Verify sections were retrieved and deleted
        metadata_store.get_sections_by_document.assert_called_once_with(doc_id)
        delete_calls = vector_store.delete.call_args_list
        assert len(delete_calls) >= 1

    @pytest.mark.asyncio
    async def test_delete_document_index_error_handling(self, index_manager):
        """Test delete error handling."""
        from src.storage import vector_store

        doc_id = uuid4()

        # Mock to raise error
        vector_store.delete = AsyncMock(side_effect=Exception("Delete error"))

        result = await index_manager.delete_document_index(doc_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_reindex_document(
        self, index_manager, sample_parsed_document, sample_sections, sample_chunks
    ):
        """Test re-indexing document."""
        doc_id = uuid4()

        with patch.object(index_manager, "delete_document_index", new=AsyncMock(return_value=True)):
            with patch.object(index_manager, "index_document", new=AsyncMock(
                return_value=IndexResult(success=True, document_id=doc_id, sections_indexed=2, chunks_indexed=2)
            )) as mock_index:
                result = await index_manager.reindex_document(
                    doc_id, sample_parsed_document, sample_sections, sample_chunks
                )

                # Verify delete was called
                index_manager.delete_document_index.assert_called_once_with(doc_id)

                # Verify index was called
                mock_index.assert_called_once_with(doc_id, sample_parsed_document, sample_sections, sample_chunks)

                assert result.success is True

    @pytest.mark.asyncio
    async def test_index_batch(self, index_manager):
        """Test batch indexing multiple documents."""
        docs = []
        for i in range(3):
            doc_id = uuid4()
            parsed = MagicMock(title=f"Doc {i}", raw_content="content", language="en")
            sections = [MagicMock(id=uuid4(), content=f"Section {i}")]
            chunks = [MagicMock(id=uuid4(), content=f"Chunk {i}")]
            docs.append((doc_id, parsed, sections, chunks))

        with patch.object(index_manager, "index_document", new=AsyncMock(
            return_value=IndexResult(success=True, document_id=uuid4(), sections_indexed=1, chunks_indexed=1)
        )) as mock_index:
            results = await index_manager.index_batch(docs)

            assert len(results) == 3
            assert all(r.success for r in results)
            assert mock_index.call_count == 3

    @pytest.mark.asyncio
    async def test_index_batch_partial_failure(self, index_manager):
        """Test batch indexing with some failures."""
        docs = [
            (uuid4(), MagicMock(), [], []),
            (uuid4(), MagicMock(), [], []),
            (uuid4(), MagicMock(), [], []),
        ]

        call_count = [0]

        async def mock_index(*args, **kwargs):
            call_count[0] += 1
            # Second document fails
            if call_count[0] == 2:
                return IndexResult(
                    success=False,
                    document_id=args[0],
                    error="Indexing error"
                )
            return IndexResult(
                success=True,
                document_id=args[0],
                sections_indexed=1,
                chunks_indexed=1
            )

        with patch.object(index_manager, "index_document", new=mock_index):
            results = await index_manager.index_batch(docs)

            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[2].success is True

    @pytest.mark.asyncio
    async def test_get_index_stats(self, index_manager):
        """Test getting index statistics."""
        from src.storage import vector_store

        vector_store.count = AsyncMock(side_effect=lambda x: {"documents": 5, "sections": 20, "chunks": 50}.get(x, 0))

        stats = await index_manager.get_index_stats()

        assert stats["documents"] == 5
        assert stats["sections"] == 20
        assert stats["chunks"] == 50

    @pytest.mark.asyncio
    async def test_get_index_stats_with_errors(self, index_manager):
        """Test stats retrieval with partial errors."""
        from src.storage import vector_store

        async def mock_count(collection):
            if collection == "sections":
                raise Exception("Collection error")
            return {"documents": 5, "chunks": 50}.get(collection, 0)

        vector_store.count = AsyncMock(side_effect=mock_count)

        stats = await index_manager.get_index_stats()

        # Errors should result in 0 count
        assert stats["documents"] == 5
        assert stats["sections"] == 0  # Error returns 0
        assert stats["chunks"] == 50


class TestIndexResult:
    """Test IndexResult dataclass."""

    def test_index_result_success(self):
        """Test successful IndexResult."""
        doc_id = uuid4()
        result = IndexResult(
            success=True,
            document_id=doc_id,
            sections_indexed=5,
            chunks_indexed=20,
        )

        assert result.success is True
        assert result.document_id == doc_id
        assert result.sections_indexed == 5
        assert result.chunks_indexed == 20
        assert result.error is None

    def test_index_result_failure(self):
        """Test failed IndexResult."""
        doc_id = uuid4()
        result = IndexResult(
            success=False,
            document_id=doc_id,
            error="Embedding generation failed"
        )

        assert result.success is False
        assert result.document_id == doc_id
        assert result.error == "Embedding generation failed"
        assert result.sections_indexed == 0
        assert result.chunks_indexed == 0


class TestIndexManagerIntegration:
    """Integration tests for IndexManager with actual storage."""

    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self, clean_storage):
        """Test end-to-end indexing workflow."""
        from src.indexing.index_manager import index_manager
        from src.storage import vector_store, metadata_store
        from src.models.document import Section, Chunk
        from src.indexing.document_parser import ParsedDocument

        doc_id = uuid4()
        section_id = uuid4()
        chunk_id = uuid4()

        parsed = ParsedDocument(
            title="Integration Test Document",
            language="en",
            sections=[],
            raw_content="This is a test document for integration testing.",
            metadata={},
        )

        sections = [
            Section(
                id=section_id,
                document_id=doc_id,
                heading="Test Section",
                level=1,
                section_path="1",
                content="This is test section content.",
                summary="Test section summary",
                metadata={},
                position=0,
            )
        ]

        chunks = [
            Chunk(
                id=chunk_id,
                document_id=doc_id,
                section_id=section_id,
                content="This is chunk content.",
                token_count=5,
                position=0,
                metadata={"heading": "Test Section", "level": 1, "section_path": "1"},
            )
        ]

        # Use mock embedder for testing
        result = await index_manager.index_document(
            doc_id, parsed, sections, chunks, provider="mock"
        )

        assert result.success is True
        assert result.sections_indexed == 1
        assert result.chunks_indexed == 1

        # Verify stats
        stats = await index_manager.get_index_stats()
        assert stats["documents"] >= 1
        assert stats["sections"] >= 1
        assert stats["chunks"] >= 1

    @pytest.mark.asyncio
    async def test_delete_and_reindex(self, clean_storage):
        """Test delete and reindex workflow."""
        from src.indexing.index_manager import index_manager
        from src.models.document import Section, Chunk
        from src.indexing.document_parser import ParsedDocument

        doc_id = uuid4()
        section_id = uuid4()

        parsed = ParsedDocument(
            title="Reindex Test",
            language="en",
            sections=[],
            raw_content="Content for reindex test.",
            metadata={},
        )

        sections = [
            Section(
                id=section_id,
                document_id=doc_id,
                heading="Original",
                level=1,
                section_path="1",
                content="Original content",
                metadata={},
                position=0,
            )
        ]

        chunks = [
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                section_id=section_id,
                content="Original chunk",
                token_count=2,
                position=0,
                metadata={},
            )
        ]

        # Initial index
        result1 = await index_manager.index_document(
            doc_id, parsed, sections, chunks, provider="mock"
        )
        assert result1.success is True

        # Delete
        deleted = await index_manager.delete_document_index(doc_id)
        assert deleted is True

        # Reindex with updated content
        sections[0].heading = "Updated"
        result2 = await index_manager.index_document(
            doc_id, parsed, sections, chunks, provider="mock"
        )
        assert result2.success is True
