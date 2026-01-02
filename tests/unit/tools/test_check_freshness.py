# =============================================================================
# Tests for Check Freshness Tool
# =============================================================================
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from src.tools.check_freshness import (
    check_freshness_tool,
    DocumentFreshness,
    FreshnessOutput,
    FRESHNESS_THRESHOLDS,
    FRESHNESS_CATEGORIES,
    _get_document_freshness,
    _get_overall_freshness_category,
    _generate_recommendations,
    get_freshness_summary,
    should_use_document
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_document(days_old=30):
    """Mock document with specific age."""
    now = datetime.utcnow()
    updated = now - timedelta(days=days_old)

    mock = Mock()
    mock.id = "test-doc-001"
    mock.title = "Test Document"
    mock.updated_at = updated
    mock.created_at = updated
    return mock


@pytest.fixture
def mock_metadata_store():
    """Mock metadata store."""
    mock_store = AsyncMock()
    return mock_store


# =============================================================================
# Tests for DocumentFreshness Model
# =============================================================================

class TestDocumentFreshness:
    """Tests for DocumentFreshness model."""

    def test_create_freshness(self):
        """Test creating a valid DocumentFreshness."""
        freshness = DocumentFreshness(
            document_id="doc-001",
            document_title="Test Doc",
            last_updated=datetime.utcnow(),
            days_since_update=30,
            freshness_category="recent",
            freshness_score=0.8,
            is_fresh=True
        )

        assert freshness.document_id == "doc-001"
        assert freshness.freshness_score == 0.8
        assert freshness.is_fresh == True

    def test_validation(self):
        """Test field validation."""
        with pytest.raises(ValueError):
            DocumentFreshness(
                document_id="doc-001",
                document_title="Test",
                last_updated=None,
                days_since_update=None,
                freshness_category="unknown",
                freshness_score=1.5,  # Invalid: > 1.0
                is_fresh=False
            )


# =============================================================================
# Tests for Check Freshness Tool
# =============================================================================

class TestCheckFreshnessTool:
    """Tests for check_freshness_tool."""

    @pytest.mark.asyncio
    async def test_check_freshness_with_documents(self, mock_document, mock_metadata_store):
        """Test checking freshness with valid documents."""
        mock_metadata_store.get_document.return_value = mock_document

        with patch('src.tools.check_freshness.metadata_store', mock_metadata_store):
            result = await check_freshness_tool.ainvoke({
                "query": "Test query",
                "document_ids": ["test-doc-001"],
                "freshness_threshold": 180
            })

            assert len(result.documents) == 1
            assert result.documents[0].document_id == "test-doc-001"
            assert result.documents[0].is_fresh == True  # 30 days old < 180 threshold

    @pytest.mark.asyncio
    async def test_check_freshness_from_sources(self, mock_document, mock_metadata_store):
        """Test checking freshness from sources list."""
        mock_metadata_store.get_document.return_value = mock_document

        sources = [
            {
                "content": "Test content",
                "metadata": {"document_id": "test-doc-001"}
            }
        ]

        with patch('src.tools.check_freshness.metadata_store', mock_metadata_store):
            result = await check_freshness_tool.ainvoke({
                "query": "Test query",
                "sources": sources,
                "freshness_threshold": 180
            })

            assert len(result.documents) == 1

    @pytest.mark.asyncio
    async def test_check_freshness_no_documents(self):
        """Test with no documents."""
        result = await check_freshness_tool.ainvoke({
            "query": "Test query",
            "document_ids": [],
            "freshness_threshold": 180
        })

        assert len(result.documents) == 0
        assert result.average_freshness_score == 0.0

    @pytest.mark.asyncio
    async def test_check_freshness_document_not_found(self, mock_metadata_store):
        """Test when document is not found."""
        mock_metadata_store.get_document.return_value = None

        with patch('src.tools.check_freshness.metadata_store', mock_metadata_store):
            result = await check_freshness_tool.ainvoke({
                "query": "Test query",
                "document_ids": ["nonexistent-doc"],
                "freshness_threshold": 180
            })

            assert len(result.documents) == 1
            assert result.documents[0].freshness_category == "unknown"
            assert result.documents[0].is_fresh == False


# =============================================================================
# Tests for FreshnessOutput Model
# =============================================================================

class TestFreshnessOutput:
    """Tests for FreshnessOutput model."""

    def test_create_output(self):
        """Test creating FreshnessOutput."""
        output = FreshnessOutput(
            query="Test query",
            documents=[],
            overall_freshness="recent",
            average_freshness_score=0.75,
            recommendations=["Recommendation 1"],
            stale_documents=[]
        )

        assert output.query == "Test query"
        assert output.overall_freshness == "recent"
        assert output.average_freshness_score == 0.75


# =============================================================================
# Tests for Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_overall_freshness_category(self):
        """Test overall freshness category calculation."""
        assert _get_overall_freshness_category(0.95) == "very_recent"
        assert _get_overall_freshness_category(0.75) == "recent"
        assert _get_overall_freshness_category(0.55) == "moderate"
        assert _get_overall_freshness_category(0.35) == "stale"
        assert _get_overall_freshness_category(0.15) == "very_stale"

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        fresh_docs = [
            DocumentFreshness(
                document_id="doc-001",
                document_title="Fresh Doc",
                last_updated=datetime.utcnow(),
                days_since_update=10,
                freshness_category="very_recent",
                freshness_score=1.0,
                is_fresh=True
            )
        ]

        recommendations = _generate_recommendations(fresh_docs, 1.0, 180)

        assert len(recommendations) > 0
        assert any("mới" in rec for rec in recommendations)

    def test_generate_recommendations_stale(self):
        """Test recommendations for stale documents."""
        stale_docs = [
            DocumentFreshness(
                document_id="doc-001",
                document_title="Old Doc",
                last_updated=datetime.utcnow() - timedelta(days=400),
                days_since_update=400,
                freshness_category="stale",
                freshness_score=0.3,
                is_fresh=False
            )
        ]

        recommendations = _generate_recommendations(stale_docs, 0.3, 180)

        assert len(recommendations) > 0
        assert any("cũ" in rec for rec in recommendations)

    def test_should_use_document(self):
        """Test should_use_document function."""
        fresh = DocumentFreshness(
            document_id="doc-001",
            document_title="Test",
            last_updated=datetime.utcnow() - timedelta(days=30),
            days_since_update=30,
            freshness_category="recent",
            freshness_score=0.8,
            is_fresh=True
        )

        assert should_use_document(fresh, max_age_days=365) == True
        assert should_use_document(fresh, max_age_days=10) == False

    def test_should_use_document_unknown_age(self):
        """Test with unknown age."""
        unknown = DocumentFreshness(
            document_id="doc-001",
            document_title="Test",
            last_updated=None,
            days_since_update=None,
            freshness_category="unknown",
            freshness_score=0.0,
            is_fresh=False
        )

        # Unknown age should be conservative (return False)
        assert should_use_document(unknown) == False


# =============================================================================
# Tests for Thresholds and Categories
# =============================================================================

class TestThresholds:
    """Tests for freshness thresholds."""

    def test_freshness_thresholds_defined(self):
        """Test that all thresholds are defined."""
        assert "very_recent" in FRESHNESS_THRESHOLDS
        assert "recent" in FRESHNESS_THRESHOLDS
        assert "moderate" in FRESHNESS_THRESHOLDS
        assert "stale" in FRESHNESS_THRESHOLDS
        assert "very_stale" in FRESHNESS_THRESHOLDS

    def test_threshold_values(self):
        """Test threshold values are correct."""
        assert FRESHNESS_THRESHOLDS["very_recent"] == 30
        assert FRESHNESS_THRESHOLDS["recent"] == 90
        assert FRESHNESS_THRESHOLDS["moderate"] == 180
        assert FRESHNESS_THRESHOLDS["stale"] == 365
        assert FRESHNESS_THRESHOLDS["very_stale"] == 730

    def test_freshness_categories_defined(self):
        """Test that all categories are defined."""
        assert "very_recent" in FRESHNESS_CATEGORIES
        assert "recent" in FRESHNESS_CATEGORIES
        assert "moderate" in FRESHNESS_CATEGORIES
        assert "stale" in FRESHNESS_CATEGORIES
        assert "very_stale" in FRESHNESS_CATEGORIES
        assert "unknown" in FRESHNESS_CATEGORIES


# =============================================================================
# Tests for _get_document_freshness
# =============================================================================

class TestGetDocumentFreshness:
    """Tests for _get_document_freshness function."""

    @pytest.mark.asyncio
    async def test_very_recent_document(self, mock_metadata_store):
        """Test freshness calculation for very recent document."""
        now = datetime.utcnow()
        doc = Mock()
        doc.id = "doc-001"
        doc.title = "Recent Doc"
        doc.updated_at = now - timedelta(days=15)
        doc.created_at = now - timedelta(days=15)

        mock_metadata_store.get_document.return_value = doc

        with patch('src.tools.check_freshness.metadata_store', mock_metadata_store):
            freshness = await _get_document_freshness("doc-001", threshold=180)

            assert freshness.freshness_category == "very_recent"
            assert freshness.freshness_score == 1.0
            assert freshness.is_fresh == True
            assert freshness.days_since_update == 15

    @pytest.mark.asyncio
    async def test_stale_document(self, mock_metadata_store):
        """Test freshness calculation for stale document."""
        now = datetime.utcnow()
        doc = Mock()
        doc.id = "doc-001"
        doc.title = "Old Doc"
        doc.updated_at = now - timedelta(days=200)
        doc.created_at = now - timedelta(days=200)

        mock_metadata_store.get_document.return_value = doc

        with patch('src.tools.check_freshness.metadata_store', mock_metadata_store):
            freshness = await _get_document_freshness("doc-001", threshold=180)

            assert freshness.freshness_category == "stale"
            assert freshness.freshness_score == 0.4
            assert freshness.is_fresh == False  # 200 > 180 threshold
