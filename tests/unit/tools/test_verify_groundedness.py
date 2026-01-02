# =============================================================================
# Tests for Verify Groundedness Tool
# =============================================================================
import pytest
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from src.tools.verify_groundedness import (
    verify_groundedness_tool,
    VerificationOutput,
    _tier1_semantic_check,
    _tier2_llm_verification,
    _cosine_similarity,
    format_sources_for_verification
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    mock = AsyncMock()
    # Return orthogonal vectors for low similarity
    # Vector 1: [1, 0, 0, ...] - all 1s in even positions
    # Vector 2: [0, 1, 0, ...] - all 1s in odd positions
    mock.embed.side_effect = lambda text: (
        [1 if i % 2 == 0 else 0 for i in range(384)] if "answer" in text
        else [0 if i % 2 == 0 else 1 for i in range(384)]
    )
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for Tier 2 verification."""
    mock = AsyncMock()
    mock.ainvoke.return_value = {
        "is_grounded": True,
        "confidence": 0.90,
        "llm_assessment": "Answer is well supported by sources",
        "ungrounded_claims": [],
        "reasoning": "All claims in answer are found in sources"
    }
    return mock


@pytest.fixture
def grounded_case():
    """Case where answer is grounded."""
    return {
        "answer": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm.",
        "sources": [
            {
                "content": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm.",
                "metadata": {"document_title": "Quy chế nhân sự"}
            }
        ]
    }


@pytest.fixture
def ungrounded_case():
    """Case where answer is NOT grounded."""
    return {
        "answer": "Nhân viên được 30 ngày nghỉ phép và 5 tháng thưởng.",
        "sources": [
            {
                "content": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm.",
                "metadata": {"document_title": "Quy chế nhân sự"}
            }
        ]
    }


# =============================================================================
# Tests
# =============================================================================

class TestVerifyGroundednessTool:
    """Tests for verify_groundedness_tool."""

    @pytest.mark.asyncio
    async def test_tier1_only_grounded(self, grounded_case, mock_embedder):
        """Test Tier 1 verification only when grounded."""
        with patch('src.tools.verify_groundedness.get_embedder', return_value=mock_embedder):
            result = await verify_groundedness_tool.ainvoke({
                "answer": grounded_case["answer"],
                "sources": grounded_case["sources"],
                "threshold": 0.5,
                "enable_tier2": False  # Disable Tier 2
            })

            # Should use Tier 1
            assert result.tier_used == "tier1"
            assert result.similarity_score is not None

    @pytest.mark.asyncio
    async def test_tier2_verification(self, grounded_case, mock_llm):
        """Test Tier 2 verification is used."""
        # Create embedder with orthogonal vectors for low similarity
        # Vector 1: [1, 0, 1, 0, ...] - for answer
        # Vector 2: [0, 1, 0, 1, ...] - for source (orthogonal)
        call_count = [0]

        async def mock_embed_func(text):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: answer embedding
                return [1.0 if i % 2 == 0 else 0.0 for i in range(384)]
            else:
                # Subsequent calls: source embeddings (orthogonal to answer)
                return [0.0 if i % 2 == 0 else 1.0 for i in range(384)]

        mock_embedder = AsyncMock()
        mock_embedder.embed.side_effect = mock_embed_func

        # Create mock chain
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = {
            "is_grounded": True,
            "confidence": 0.90,
            "llm_assessment": "Answer is well supported by sources",
            "ungrounded_claims": [],
            "reasoning": "All claims in answer are found in sources"
        }

        with patch('src.tools.verify_groundedness.get_embedder', return_value=mock_embedder), \
             patch('src.tools.verify_groundedness._tier2_llm_verification', return_value=mock_chain.ainvoke.return_value):
            result = await verify_groundedness_tool.ainvoke({
                "answer": grounded_case["answer"],
                "sources": grounded_case["sources"],
                "threshold": 0.95,  # High threshold to fail Tier 1
                "enable_tier2": True
            })

            # Should use Tier 2 (similarity ~0 < threshold)
            assert result.tier_used == "tier2", f"Expected tier2, got {result.tier_used}, similarity={result.similarity_score}"
            assert result.llm_assessment is not None
            assert result.is_grounded == True

    @pytest.mark.asyncio
    async def test_empty_sources(self):
        """Test with empty sources."""
        result = await verify_groundedness_tool.ainvoke({
            "answer": "Test answer",
            "sources": [],
            "threshold": 0.75
        })

        assert result.is_grounded == False
        assert result.confidence == 0.0
        assert result.tier_used == "none"

    @pytest.mark.asyncio
    async def test_empty_answer(self, grounded_case):
        """Test with empty answer."""
        result = await verify_groundedness_tool.ainvoke({
            "answer": "",
            "sources": grounded_case["sources"],
            "threshold": 0.75
        })

        assert result.is_grounded == False
        assert result.tier_used == "none"


class TestVerificationOutputModel:
    """Tests for VerificationOutput model."""

    def test_create_verification_output(self):
        """Test creating verification output."""
        output = VerificationOutput(
            is_grounded=True,
            confidence=0.90,
            tier_used="tier2",
            similarity_score=0.85,
            llm_assessment="Well grounded",
            ungrounded_claims=[],
            reasoning="Test reasoning"
        )

        assert output.is_grounded == True
        assert output.confidence == 0.90
        assert output.tier_used == "tier2"

    def test_validation(self):
        """Test field validation."""
        with pytest.raises(ValueError):
            VerificationOutput(
                is_grounded=True,
                confidence=1.5,  # Invalid: > 1.0
                tier_used="tier1"
            )


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # Identical vectors
        sim = _cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(1.0)

        # Orthogonal vectors
        sim = _cosine_similarity(vec1, vec3)
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        sim = _cosine_similarity([], [])
        assert sim == 0.0

    def test_format_sources_for_verification(self):
        """Test formatting sources for verification."""
        sources = [
            {
                "content": "Test content here",
                "metadata": {"document_title": "Test Doc"}
            }
        ]

        formatted = format_sources_for_verification(sources)

        assert "Nguồn 1" in formatted
        assert "Test content here" in formatted
        assert "Test Doc" in formatted

    def test_format_sources_empty(self):
        """Test formatting empty sources."""
        formatted = format_sources_for_verification([])
        assert "Không có nguồn" in formatted


# =============================================================================
# Integration Tests
# =============================================================================

class TestVerificationIntegration:
    """Integration tests for verification workflow."""

    @pytest.mark.asyncio
    async def test_full_verification_workflow(self, grounded_case, mock_embedder, mock_llm):
        """Test complete verification workflow with both tiers."""
        # Setup: Tier 1 passes
        mock_embedder.embed.side_effect = lambda text: [0.9] * 384

        with patch('src.tools.verify_groundedness.get_embedder', return_value=mock_embedder):
            result = await verify_groundedness_tool.ainvoke({
                "answer": grounded_case["answer"],
                "sources": grounded_case["sources"],
                "threshold": 0.75,
                "enable_tier2": True
            })

            # Should pass Tier 1 and skip Tier 2
            assert result.tier_used == "tier1"
            assert result.is_grounded == True
            assert result.confidence >= 0.75

    @pytest.mark.asyncio
    async def test_tier1_fails_tier2_succeeds(self, grounded_case, mock_llm):
        """Test scenario where Tier 1 fails but Tier 2 succeeds."""
        # Setup: Tier 1 fails (low similarity with orthogonal vectors)
        call_count = [0]

        async def mock_embed_func(text):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: answer embedding
                return [1.0 if i % 2 == 0 else 0.0 for i in range(384)]
            else:
                # Subsequent calls: source embeddings (orthogonal to answer)
                return [0.0 if i % 2 == 0 else 1.0 for i in range(384)]

        mock_embedder = AsyncMock()
        mock_embedder.embed.side_effect = mock_embed_func

        # Create mock chain
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = {
            "is_grounded": True,
            "confidence": 0.90,
            "llm_assessment": "Answer is well supported by sources",
            "ungrounded_claims": [],
            "reasoning": "All claims in answer are found in sources"
        }

        with patch('src.tools.verify_groundedness.get_embedder', return_value=mock_embedder), \
             patch('src.tools.verify_groundedness._tier2_llm_verification', return_value=mock_chain.ainvoke.return_value):
            result = await verify_groundedness_tool.ainvoke({
                "answer": grounded_case["answer"],
                "sources": grounded_case["sources"],
                "threshold": 0.8,
                "enable_tier2": True
            })

            # Should use Tier 2
            assert result.tier_used == "tier2", f"Expected tier2, got {result.tier_used}, similarity={result.similarity_score}"
            # Tier 2 succeeds
            assert result.is_grounded == True
            assert result.llm_assessment is not None
