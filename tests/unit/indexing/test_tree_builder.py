# =============================================================================
# Unit Tests for Tree Builder
# =============================================================================
import pytest
from uuid import uuid4

from src.indexing.tree_builder import TreeBuilder
from src.models.document import Section


class TestTreeBuilder:
    """Test TreeBuilder functionality."""

    @pytest.fixture
    def sample_sections(self):
        """Create sample sections for testing."""
        doc_id = uuid4()
        return [
            Section(
                id=uuid4(),
                document_id=uuid4(),  # Temp ID
                heading="Introduction",
                level=1,
                section_path="0",
                content="This is the introduction.",
            ),
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading="Getting Started",
                level=2,
                section_path="0",
                content="Getting started guide.",
            ),
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading="Installation",
                level=3,
                section_path="0",
                content="Installation instructions.",
            ),
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading="Advanced Topics",
                level=2,
                section_path="0",
                content="Advanced configuration.",
            ),
        ]

    def test_init_tree_builder(self):
        """Test TreeBuilder initialization."""
        builder = TreeBuilder()
        assert builder.max_tree_depth == 5

        builder_custom = TreeBuilder(max_tree_depth=10)
        assert builder_custom.max_tree_depth == 10

    def test_build_tree_hierarchy(self, sample_sections):
        """Test building hierarchical tree structure."""
        builder = TreeBuilder()
        doc_id = uuid4()

        sections = builder.build_tree(sample_sections, doc_id)

        # Verify document IDs are set
        for section in sections:
            assert section.document_id == doc_id

        # Verify hierarchy
        intro = next(s for s in sections if "Introduction" in s.heading)
        getting_started = next(s for s in sections if "Getting Started" in s.heading)
        installation = next(s for s in sections if "Installation" in s.heading)
        advanced = next(s for s in sections if "Advanced Topics" in s.heading)

        # Introduction should be root (no parent)
        assert intro.parent_section_id is None

        # Getting Started should be child of Introduction
        assert getting_started.parent_section_id == intro.id

        # Installation should be child of Getting Started
        assert installation.parent_section_id == getting_started.id

        # Advanced should be child of Introduction (sibling of Getting Started)
        assert advanced.parent_section_id == intro.id

    def test_section_paths(self, sample_sections):
        """Test that section paths are generated correctly."""
        builder = TreeBuilder()
        doc_id = uuid4()

        sections = builder.build_tree(sample_sections, doc_id)

        # Section paths should be hierarchical
        paths = {s.heading: s.section_path for s in sections}

        # First root section should have path "1"
        intro = next(s for s in sections if s.heading == "Introduction")
        assert intro.section_path == "1"

        # Child of 1 should be "1.1"
        getting_started = next(s for s in sections if s.heading == "Getting Started")
        assert getting_started.section_path == "1.1"

        # Child of 1.1 should be "1.1.1"
        installation = next(s for s in sections if s.heading == "Installation")
        assert installation.section_path == "1.1.1"

    def test_generate_summaries(self):
        """Test generating section summaries."""
        builder = TreeBuilder()

        sections = [
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading="Test Section",
                level=1,
                section_path="1",
                content="This is a longer section content that needs to be summarized.",
            ),
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading="Empty Section",
                level=2,
                section_path="1.1",
                content="",
            ),
        ]

        result = builder.generate_section_summaries(sections, max_length=50)

        # First section should have summary
        test_section = next(s for s in result if s.heading == "Test Section")
        assert test_section.summary is not None
        # Allow for "..." suffix, so actual length may be slightly more
        assert len(test_section.summary) <= 55  # Small buffer for "..."

        # Empty section should have fallback summary
        empty_section = next(s for s in result if s.heading == "Empty Section")
        assert empty_section.summary == "Section: Empty Section"

    def test_validate_tree_valid(self, sample_sections):
        """Test tree validation for valid tree."""
        builder = TreeBuilder()
        doc_id = uuid4()

        sections = builder.build_tree(sample_sections, doc_id)
        is_valid = builder.validate_tree(sections)

        assert is_valid is True

    def test_validate_tree_invalid_parent(self):
        """Test tree validation with invalid parent reference."""
        builder = TreeBuilder()

        sections = [
            Section(
                id=uuid4(),
                document_id=uuid4(),
                heading="Section 1",
                level=1,
                section_path="1",
                content="Content",
                parent_section_id=uuid4(),  # Invalid - doesn't exist
            ),
        ]

        is_valid = builder.validate_tree(sections)
        assert is_valid is False

    def test_get_root_sections(self, sample_sections):
        """Test getting root sections."""
        builder = TreeBuilder()
        doc_id = uuid4()

        sections = builder.build_tree(sample_sections, doc_id)
        roots = builder.get_root_sections(sections)

        # Should have one root (Introduction)
        assert len(roots) == 1
        assert roots[0].heading == "Introduction"

    def test_get_section_depth(self, sample_sections):
        """Test getting section depth."""
        builder = TreeBuilder()
        doc_id = uuid4()

        sections = builder.build_tree(sample_sections, doc_id)

        intro = next(s for s in sections if s.heading == "Introduction")
        getting_started = next(s for s in sections if s.heading == "Getting Started")
        installation = next(s for s in sections if s.heading == "Installation")

        assert builder.get_section_depth(intro) == 1
        assert builder.get_section_depth(getting_started) == 2
        assert builder.get_section_depth(installation) == 3

    def test_max_tree_depth_limit(self):
        """Test that max tree depth is respected."""
        builder = TreeBuilder(max_tree_depth=2)
        doc_id = uuid4()

        # Create sections deeper than max_depth
        sections = [
            Section(id=uuid4(), document_id=uuid4(), heading=f"S{i}", level=i + 1,
                   section_path="0", content=f"Content {i}")
            for i in range(5)  # 5 levels deep, but max_depth is 2
        ]

        result = builder.build_tree(sections, doc_id)

        # Very deep sections should be flattened (no parent)
        deep_sections = [s for s in result if builder.get_section_depth(s) > 2]
        for section in deep_sections:
            # Should have been flattened to root
            assert section.parent_section_id is None

    def test_empty_sections_list(self):
        """Test building tree with empty sections list."""
        builder = TreeBuilder()
        doc_id = uuid4()

        result = builder.build_tree([], doc_id)
        assert result == []
