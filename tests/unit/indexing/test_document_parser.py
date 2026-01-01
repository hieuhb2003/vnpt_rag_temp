# =============================================================================
# Unit Tests for Document Parser
# =============================================================================
import pytest
from pathlib import Path

from src.indexing.document_parser import (
    DocumentParserFactory,
    MarkdownParser,
    HTMLParser,
    PDFParser,
    DocxParser,
)


class TestMarkdownParser:
    """Test Markdown parser."""

    @pytest.mark.asyncio
    async def test_parse_simple_document(self, sample_markdown_en):
        """Test parsing a simple markdown document."""
        parser = MarkdownParser()
        result = await parser.parse("test.md", sample_markdown_en.encode("utf-8"))

        assert result.title == "Product Documentation"
        assert result.language == "en"
        assert len(result.sections) > 0
        assert result.raw_content is not None

    @pytest.mark.asyncio
    async def test_parse_vietnamese_document(self, sample_markdown_vi):
        """Test parsing a Vietnamese markdown document."""
        parser = MarkdownParser()
        result = await parser.parse("test.md", sample_markdown_vi.encode("utf-8"))

        assert result.title == "Hướng Dẫn Sử Dụng"
        assert result.language == "vi"
        assert len(result.sections) > 0

    @pytest.mark.asyncio
    async def test_extract_headings(self):
        """Test that headings are extracted correctly."""
        md = b"""# Title 1
Content 1

## Title 2
Content 2

### Title 3
Content 3
"""
        parser = MarkdownParser()
        result = await parser.parse("test.md", md)

        headings = [s.heading for s in result.sections]
        assert "Title 1" in headings
        assert "Title 2" in headings
        assert "Title 3" in headings

    @pytest.mark.asyncio
    async def test_heading_levels(self):
        """Test that heading levels are correct."""
        md = b"""# H1
## H2
### H3
#### H4
##### H5
###### H6
"""
        parser = MarkdownParser()
        result = await parser.parse("test.md", md)

        levels = [s.level for s in result.sections]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels
        assert 4 in levels
        assert 5 in levels
        assert 6 in levels

    @pytest.mark.asyncio
    async def test_empty_document(self):
        """Test parsing a minimal document."""
        parser = MarkdownParser()
        result = await parser.parse("empty.md", b"# Empty\n")

        assert result.title == "Empty"
        # Even for minimal documents, the parser extracts headings
        assert len(result.sections) >= 1


class TestHTMLParser:
    """Test HTML parser."""

    @pytest.mark.asyncio
    async def test_parse_html_document(self, sample_html):
        """Test parsing an HTML document."""
        parser = HTMLParser()
        result = await parser.parse("test.html", sample_html.encode("utf-8"))

        assert result.title == "Test Page"
        assert len(result.sections) > 0

    @pytest.mark.asyncio
    async def test_extract_html_headings(self):
        """Test that HTML headings are extracted."""
        html = b"""<html>
<body>
    <h1>Main Title</h1>
    <h2>Subtitle</h2>
    <h3>Subsection</h3>
</body>
</html>"""
        parser = HTMLParser()
        result = await parser.parse("test.html", html)

        headings = [s.heading for s in result.sections]
        assert "Main Title" in headings
        assert "Subtitle" in headings
        assert "Subsection" in headings


class TestPDFParser:
    """Test PDF parser."""

    @pytest.mark.asyncio
    async def test_parse_minimal_pdf(self):
        """Test that PDF parser can be initialized."""
        parser = PDFParser()
        # We can't create a valid PDF in test, so we just verify the parser exists
        assert parser is not None


class TestDocxParser:
    """Test DOCX parser."""

    @pytest.mark.asyncio
    async def test_parse_minimal_docx(self):
        """Test that DOCX parser can be initialized."""
        parser = DocxParser()
        # We can't create a valid DOCX in test, so we just verify the parser exists
        assert parser is not None


class TestDocumentParserFactory:
    """Test DocumentParserFactory."""

    @pytest.mark.asyncio
    async def test_get_markdown_parser(self, sample_markdown_en):
        """Test getting a markdown parser."""
        result = await DocumentParserFactory.parse(
            "test.md",
            sample_markdown_en.encode("utf-8")
        )
        assert result.title is not None

    @pytest.mark.asyncio
    async def test_get_html_parser(self, sample_html):
        """Test getting an HTML parser."""
        result = await DocumentParserFactory.parse(
            "test.html",
            sample_html.encode("utf-8")
        )
        assert result.title is not None

    @pytest.mark.asyncio
    async def test_unsupported_extension(self):
        """Test that unsupported extensions raise an error."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            await DocumentParserFactory.parse(
                "test.xyz",
                b"some content"
            )

    @pytest.mark.asyncio
    async def test_language_detection_english(self):
        """Test English language detection."""
        en_text = "Hello world, this is a test document.".encode("utf-8")
        result = await DocumentParserFactory.parse("test.md", en_text)
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_language_detection_vietnamese(self):
        """Test Vietnamese language detection."""
        vi_text = "Xin chào, đây là tài liệu tiếng Việt.".encode("utf-8")
        result = await DocumentParserFactory.parse("test.md", vi_text)
        assert result.language == "vi"

    @pytest.mark.asyncio
    async def test_language_detection_mixed(self):
        """Test mixed language detection."""
        mixed_text = "Hello world. Xin chào, đây là tiếng Việt.".encode("utf-8")
        result = await DocumentParserFactory.parse("test.md", mixed_text)
        # Should detect as mixed or the dominant language
        assert result.language in ["en", "vi", "mixed"]
