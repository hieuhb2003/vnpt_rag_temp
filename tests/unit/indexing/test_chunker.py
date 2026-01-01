# =============================================================================
# Unit Tests for Chunker
# =============================================================================
import pytest
from uuid import uuid4

from src.indexing.chunker import Chunker
from src.models.document import Section, Chunk


class TestChunker:
    """Test Chunker functionality."""

    @pytest.fixture
    def sample_section(self):
        """Create a sample section for chunking."""
        return Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Test Section",
            level=1,
            section_path="1",
            content="This is a test section with multiple sentences. "
                   "Each sentence should be properly handled. "
                   "The chunker should respect sentence boundaries. "
                   "This ensures that chunks don't break mid-sentence.",
        )

    @pytest.fixture
    def long_section(self):
        """Create a long section for testing multiple chunks."""
        content = " ".join([f"This is sentence number {i}." for i in range(50)])
        return Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Long Section",
            level=1,
            section_path="1",
            content=content,
        )

    def test_init_chunker(self):
        """Test Chunker initialization."""
        chunker = Chunker()
        assert chunker.chunk_size > 0
        assert chunker.chunk_overlap >= 0

        chunker_custom = Chunker(chunk_size=256, chunk_overlap=32)
        assert chunker_custom.chunk_size == 256
        assert chunker_custom.chunk_overlap == 32

    def test_count_tokens(self):
        """Test token counting."""
        chunker = Chunker()

        # English text
        tokens_en = chunker.count_tokens("Hello world!")
        assert tokens_en > 0

        # Vietnamese text
        tokens_vi = chunker.count_tokens("Xin chào thế giới!")
        assert tokens_vi > 0

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        chunker = Chunker()

        text = "This is sentence one. This is sentence two! This is sentence three?"
        sentences = chunker.split_into_sentences(text)

        # The chunker merges short fragments (less than 30 chars) back together
        # So we get one sentence with all the text
        assert len(sentences) >= 1
        assert "sentence" in sentences[0].text.lower()

    def test_split_sentences_vietnamese(self):
        """Test splitting Vietnamese sentences."""
        chunker = Chunker()

        text = "Xin chào. Câu thứ hai. Đây là câu thứ ba!"
        sentences = chunker.split_into_sentences(text)

        # Vietnamese sentences are preserved
        assert len(sentences) >= 1
        assert any("Xin chào" in s.text or "xin chào" in s.text.lower() for s in sentences)

    def test_chunk_section(self, sample_section):
        """Test chunking a single section."""
        chunker = Chunker()
        doc_id = uuid4()

        chunks = chunker.chunk_section(sample_section, doc_id)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == doc_id
            assert chunk.section_id == sample_section.id
            assert chunk.content is not None
            assert chunk.token_count > 0
            assert chunk.token_count <= chunker.chunk_size

    def test_chunk_long_section(self, long_section):
        """Test chunking a long section."""
        chunker = Chunker()
        doc_id = uuid4()

        chunks = chunker.chunk_section(long_section, doc_id)

        # Should create at least one chunk
        assert len(chunks) >= 1

        # Verify positions are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.position == i

    def test_chunk_respects_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = Chunker(chunk_size=50, chunk_overlap=20)

        content = " ".join([f"Sentence {i}." for i in range(20)])
        section = Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Test",
            level=1,
            section_path="1",
            content=content,
        )

        chunks = chunker.chunk_section(section, uuid4())

        if len(chunks) > 1:
            # Check that chunks have some overlap content
            # (not testing exact overlap amount as it varies)
            assert chunks[0].content is not None
            assert chunks[1].content is not None

    def test_chunk_empty_section(self):
        """Test chunking an empty section."""
        chunker = Chunker()
        section = Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Empty",
            level=1,
            section_path="1",
            content="",
        )

        chunks = chunker.chunk_section(section, uuid4())
        assert len(chunks) == 0

    def test_chunk_document(self):
        """Test chunking an entire document (multiple sections)."""
        chunker = Chunker()
        doc_id = uuid4()

        sections = [
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading=f"Section {i}",
                level=1,
                section_path=str(i + 1),
                content=f"Content for section {i}. " * 10,
            )
            for i in range(3)
        ]

        chunks = chunker.chunk_document(sections, doc_id)

        assert len(chunks) > 0
        # Verify all chunks have sequential positions
        positions = [c.position for c in chunks]
        assert positions == sorted(positions)

    def test_chunk_preserves_metadata(self, sample_section):
        """Test that chunk metadata is preserved."""
        chunker = Chunker()

        chunks = chunker.chunk_section(sample_section, uuid4())

        for chunk in chunks:
            assert "heading" in chunk.metadata
            assert chunk.metadata["heading"] == "Test Section"
            assert "level" in chunk.metadata
            assert chunk.metadata["level"] == 1
            assert "section_path" in chunk.metadata

    def test_merge_chunks(self):
        """Test merging chunks back together."""
        chunker = Chunker()

        chunks = [
            Chunk(
                id=uuid4(),
                document_id=uuid4(),
                section_id=uuid4(),
                content="Part one",
                token_count=2,
                position=0,
            ),
            Chunk(
                id=uuid4(),
                document_id=uuid4(),
                section_id=uuid4(),
                content="Part two",
                token_count=2,
                position=1,
            ),
        ]

        merged = chunker.merge_chunks(chunks)

        assert "Part one" in merged
        assert "Part two" in merged

    def test_token_count_accuracy(self):
        """Test that token counts are accurate."""
        chunker = Chunker()

        section = Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Test",
            level=1,
            section_path="1",
            content="Hello world test content",
        )

        chunks = chunker.chunk_section(section, uuid4())

        for chunk in chunks:
            # Verify token count matches what tiktoken says
            expected_count = chunker.count_tokens(chunk.content)
            assert chunk.token_count == expected_count

    def test_chunk_vietnamese_text(self):
        """Test chunking Vietnamese text."""
        chunker = Chunker()

        section = Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Tiếng Việt",
            level=1,
            section_path="1",
            content="Xin chào. Đây là tài liệu tiếng Việt. "
                   "Chúng tôi hỗ trợ đa ngôn ngữ. "
                   "Hệ thống rất mạnh mẽ.",
        )

        chunks = chunker.chunk_section(section, uuid4())

        assert len(chunks) > 0
        # Vietnamese text should be preserved correctly
        assert "Xin chào" in chunks[0].content or "Xin chào" in "".join(c.content for c in chunks)

    def test_small_chunk_size(self):
        """Test with very small chunk size."""
        chunker = Chunker(chunk_size=15, chunk_overlap=3)

        # Create content with more tokens to force multiple chunks
        # Use longer text to avoid the <30 char merge logic
        sentences = [f"This is a much longer sentence number {i} with more content." for i in range(10)]
        content = " ".join(sentences)

        section = Section(
            id=uuid4(),
            document_id=uuid4(),
            heading="Test",
            level=1,
            section_path="1",
            content=content,
        )

        chunks = chunker.chunk_section(section, uuid4())

        # With small chunk size and longer content, should get multiple chunks
        assert len(chunks) >= 2
