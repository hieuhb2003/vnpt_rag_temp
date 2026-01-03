# =============================================================================
# Unit Tests for Tools
# =============================================================================
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.tools.query_rewriter import (
    query_rewriter_tool,
    QueryRewriterInput,
    QueryRewriterOutput,
    convert_to_rewritten_query,
    format_conversation_context
)
from src.tools.hybrid_search import (
    hybrid_search_tool,
    SearchFilters,
    SearchResult,
    HybridSearchOutput,
    fuse_results
)
from src.tools.verify_groundedness import (
    verify_groundedness_tool,
    VerificationOutput,
    _cosine_similarity,
    format_sources_for_verification
)


# =============================================================================
# Query Rewriter Tests
# =============================================================================

class TestQueryRewriter:
    """Test query rewriter tool."""

    @pytest.mark.asyncio
    async def test_format_conversation_context_empty(self):
        """Test formatting empty conversation history."""
        context = format_conversation_context([])
        assert "Không có lịch sử" in context or "No history" in context

    @pytest.mark.asyncio
    async def test_format_conversation_context_with_messages(self):
        """Test formatting conversation history with messages."""
        history = ["Message 1", "Message 2", "Message 3"]
        context = format_conversation_context(history)
        assert "Message 1" in context
        # Should only include last 5 messages
        assert "Message 3" in context

    @pytest.mark.asyncio
    async def test_format_conversation_context_truncates_to_5(self):
        """Test that conversation context is limited to last 5 messages."""
        history = [f"Message {i}" for i in range(10)]
        context = format_conversation_context(history)
        # Should only have last 5 messages
        assert "Message 9" in context
        # First message should not be in context
        assert "Message 0" not in context

    @pytest.mark.asyncio
    async def test_query_rewriter_basic(self):
        """Test basic query rewriting output model."""
        # Test the output model directly
        result = QueryRewriterOutput(
            original="test query",
            rewritten="improved test query",
            keywords=["test", "query"],
            query_type="factoid",
            confidence=0.9,
            reasoning="test reasoning",
            expansions=["expanded query"]
        )

        assert result.original == "test query"
        assert result.rewritten == "improved test query"
        assert result.keywords == ["test", "query"]
        assert result.query_type == "factoid"
        assert result.confidence == 0.9
        assert result.reasoning == "test reasoning"

    @pytest.mark.asyncio
    async def test_query_rewriter_with_vietnamese(self):
        """Test query rewriting with Vietnamese language."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value={
            "original": "ngày nghỉ phép",
            "rewritten": "số ngày nghỉ phép hàng năm",
            "keywords": ["nghỉ", "phép", "năm"],
            "query_type": "factoid",
            "confidence": 0.95,
            "reasoning": "lý do",
            "expansions": []
        })

        with patch("src.tools.query_rewriter.get_llm", return_value=mock_llm):
            result = await query_rewriter_tool.ainvoke({
                "query": "ngày nghỉ phép",
                "conversation_history": [],
                "language": "vi"
            })

            assert "ngày nghỉ phép" in result.original

    @pytest.mark.asyncio
    async def test_query_rewriter_with_conversation_history(self):
        """Test query rewriting with conversation history - model validation."""
        # Test the output model with conversation history context
        result = QueryRewriterOutput(
            original="lương tháng 13",
            rewritten="lương thưởng tháng 13 cho nhân viên",
            keywords=["lương", "thưởng", "tháng 13"],
            query_type="factoid",
            confidence=0.85,
            reasoning="context from history",
            expansions=[]
        )

        assert result.confidence == 0.85
        assert "lương thưởng" in result.rewritten

    @pytest.mark.asyncio
    async def test_query_rewriter_error_handling(self):
        """Test query rewriter error handling."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        with patch("src.tools.query_rewriter.get_llm", return_value=mock_llm):
            result = await query_rewriter_tool.ainvoke({
                "query": "test query",
                "conversation_history": [],
                "language": "en"
            })

            # Should return original query on error
            assert result.original == "test query"
            assert result.rewritten == "test query"
            assert result.confidence == 0.0

    def test_convert_to_rewritten_query(self):
        """Test converting QueryRewriterOutput to RewrittenQuery."""
        from src.models.query import QueryType, RewrittenQuery

        output = QueryRewriterOutput(
            original="original query",
            rewritten="rewritten query",
            keywords=["key1", "key2"],
            query_type="procedural",
            confidence=0.9,
            reasoning="test",
            expansions=["expansion1"]
        )

        result = convert_to_rewritten_query(output)

        assert isinstance(result, RewrittenQuery)
        assert result.original == "original query"
        assert result.query_type == QueryType.PROCEDURAL

    def test_convert_to_rewritten_query_invalid_type(self):
        """Test converting with invalid query type."""
        from src.models.query import QueryType

        output = QueryRewriterOutput(
            original="original",
            rewritten="rewritten",
            keywords=[],
            query_type="invalid_type",
            confidence=0.5,
            reasoning=None,
            expansions=[]
        )

        result = convert_to_rewritten_query(output)

        # Should default to FACTOID for invalid type
        assert result.query_type == QueryType.FACTOID


# =============================================================================
# Hybrid Search Tests
# =============================================================================

class TestHybridSearch:
    """Test hybrid search tool."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self):
        """Test basic hybrid search."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_vector_store = AsyncMock()
        mock_vector_store.hybrid_search = AsyncMock(return_value=[
            {
                "id": "chunk-1",
                "score": 0.95,
                "payload": {
                    "content": "Test content about holidays",
                    "document_id": str(uuid4()),
                    "document_title": "HR Policy"
                }
            }
        ])

        mock_cache = AsyncMock()
        mock_cache.get_retrieval_cache = AsyncMock(return_value=None)
        mock_cache.set_retrieval_cache = AsyncMock()

        with patch("src.tools.hybrid_search.get_embedder", return_value=mock_embedder), \
             patch("src.tools.hybrid_search.vector_store", mock_vector_store), \
             patch("src.tools.hybrid_search.cache_store", mock_cache):

            result = await hybrid_search_tool.ainvoke({
                "query": "holiday leave",
                "collection": "chunks",
                "top_k": 5,
                "alpha": 0.7
            })

            assert result.query == "holiday leave"
            assert result.total_results == 1
            assert len(result.results) == 1
            assert result.results[0].score == 0.95
            assert result.cache_hit is False

    @pytest.mark.asyncio
    async def test_hybrid_search_with_cache_hit(self):
        """Test hybrid search with cache hit."""
        cached_data = {
            "results": [
                {
                    "id": "chunk-1",
                    "type": "chunk",
                    "content": "Cached content",
                    "score": 0.9,
                    "metadata": {},
                    "document_id": str(uuid4())
                }
            ],
            "search_method": "hybrid",
            "total": 1
        }

        mock_cache = AsyncMock()
        mock_cache.get_retrieval_cache = AsyncMock(return_value=cached_data)

        with patch("src.tools.hybrid_search.cache_store", mock_cache):
            result = await hybrid_search_tool.ainvoke({
                "query": "cached query",
                "collection": "chunks",
                "top_k": 5,
                "use_cache": True
            })

            assert result.cache_hit is True
            assert result.total_results == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self):
        """Test hybrid search with filters."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_vector_store = AsyncMock()
        mock_vector_store.hybrid_search = AsyncMock(return_value=[])

        mock_cache = AsyncMock()
        mock_cache.get_retrieval_cache = AsyncMock(return_value=None)
        mock_cache.set_retrieval_cache = AsyncMock()

        filters = {"category": "hr", "language": "vi"}

        with patch("src.tools.hybrid_search.get_embedder", return_value=mock_embedder), \
             patch("src.tools.hybrid_search.vector_store", mock_vector_store), \
             patch("src.tools.hybrid_search.cache_store", mock_cache):

            result = await hybrid_search_tool.ainvoke({
                "query": "test",
                "collection": "chunks",
                "filters": filters
            })

            assert result.filters_applied == filters

    @pytest.mark.asyncio
    async def test_hybrid_search_min_score_filtering(self):
        """Test hybrid search minimum score filtering."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_vector_store = AsyncMock()
        mock_vector_store.hybrid_search = AsyncMock(return_value=[
            {
                "id": "chunk-1",
                "score": 0.9,  # Above threshold
                "payload": {"content": "High score content", "document_id": str(uuid4())}
            },
            {
                "id": "chunk-2",
                "score": 0.3,  # Below threshold
                "payload": {"content": "Low score content", "document_id": str(uuid4())}
            }
        ])

        mock_cache = AsyncMock()
        mock_cache.get_retrieval_cache = AsyncMock(return_value=None)
        mock_cache.set_retrieval_cache = AsyncMock()

        with patch("src.tools.hybrid_search.get_embedder", return_value=mock_embedder), \
             patch("src.tools.hybrid_search.vector_store", mock_vector_store), \
             patch("src.tools.hybrid_search.cache_store", mock_cache):

            result = await hybrid_search_tool.ainvoke({
                "query": "test",
                "collection": "chunks",
                "min_score": 0.5
            })

            # Only high score result should be included
            assert result.total_results == 1
            assert result.results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self):
        """Test hybrid search error handling."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(side_effect=Exception("Embedding error"))

        mock_cache = AsyncMock()
        mock_cache.get_retrieval_cache = AsyncMock(return_value=None)

        with patch("src.tools.hybrid_search.get_embedder", return_value=mock_embedder), \
             patch("src.tools.hybrid_search.cache_store", mock_cache):

            result = await hybrid_search_tool.ainvoke({
                "query": "test",
                "collection": "chunks"
            })

            # Should return empty results on error
            assert result.total_results == 0
            assert len(result.results) == 0

    def test_fuse_results_basic(self):
        """Test basic result fusion with RRF."""
        results_list1 = [
            SearchResult(id="1", type="chunk", content="A", score=0.9, metadata={}),
            SearchResult(id="2", type="chunk", content="B", score=0.8, metadata={})
        ]
        results_list2 = [
            SearchResult(id="2", type="chunk", content="B", score=0.9, metadata={}),
            SearchResult(id="3", type="chunk", content="C", score=0.7, metadata={})
        ]

        fused = fuse_results([results_list1, results_list2])

        # Should have 3 unique results
        assert len(fused) == 3
        # Item 2 appears in both lists, should have higher RRF score
        item_2 = next((r for r in fused if r.id == "2"), None)
        assert item_2 is not None

    def test_fuse_results_empty(self):
        """Test fusing empty result lists."""
        fused = fuse_results([])
        assert fused == []

    def test_fuse_results_single_list(self):
        """Test fusing single result list."""
        results = [
            SearchResult(id="1", type="chunk", content="A", score=0.9, metadata={}),
            SearchResult(id="2", type="chunk", content="B", score=0.8, metadata={})
        ]

        fused = fuse_results([results])

        assert len(fused) == 2
        # Scores should be updated to RRF scores
        assert all(0 < r.score <= 1 for r in fused)

    def test_fuse_results_with_duplicates(self):
        """Test RRF properly handles duplicates."""
        # Same result in both lists at same rank
        result = SearchResult(id="1", type="chunk", content="A", score=0.9, metadata={})
        results_list1 = [result]
        results_list2 = [result]

        fused = fuse_results([results_list1, results_list2], k=60)

        # Should only have one result with combined RRF score
        assert len(fused) == 1
        assert fused[0].id == "1"
        # RRF score should be higher for duplicate (2 * 1/(60+1))
        assert fused[0].score > 0


# =============================================================================
# Verify Groundedness Tests
# =============================================================================

class TestVerifyGroundedness:
    """Test verify groundedness tool."""

    @pytest.mark.asyncio
    async def test_verify_groundedness_empty_sources(self):
        """Test verification with empty sources."""
        result = await verify_groundedness_tool.ainvoke({
            "answer": "Test answer",
            "sources": [],
            "threshold": 0.75
        })

        assert result.is_grounded is False
        assert result.confidence == 0.0
        assert result.tier_used == "none"

    @pytest.mark.asyncio
    async def test_verify_groundedness_empty_answer(self):
        """Test verification with empty answer."""
        result = await verify_groundedness_tool.ainvoke({
            "answer": "",
            "sources": [{"content": "Source content", "metadata": {}}],
            "threshold": 0.75
        })

        assert result.is_grounded is False
        assert result.tier_used == "none"

    @pytest.mark.asyncio
    async def test_verify_groundedness_tier1_pass(self):
        """Test Tier 1 verification passing."""
        mock_embedder = AsyncMock()
        # Return identical embeddings for high similarity
        mock_embedder.embed = AsyncMock(return_value=[0.5] * 1536)

        with patch("src.tools.verify_groundedness.get_embedder", return_value=mock_embedder), \
             patch("src.tools.verify_groundedness.settings", MagicMock(groundedness_threshold=0.75)):

            result = await verify_groundedness_tool.ainvoke({
                "answer": "Test answer",
                "sources": [{"content": "Test answer", "metadata": {}}],
                "threshold": 0.75,
                "enable_tier2": False
            })

            assert result.is_grounded is True
            assert result.tier_used == "tier1"
            assert result.similarity_score is not None
            assert result.similarity_score >= 0.75

    @pytest.mark.asyncio
    async def test_verify_groundedness_tier2_enabled(self):
        """Test Tier 2 LLM verification."""
        # Create embeddings with very low similarity (orthogonal)
        mock_embedder = AsyncMock()
        # Answer: all positive in first half, zeros in second half
        answer_emb = [0.1] * 768 + [0.0] * 768
        # Source: zeros in first half, all positive in second half (orthogonal)
        source_emb = [0.0] * 768 + [0.1] * 768

        mock_embedder.embed = AsyncMock(side_effect=[
            answer_emb,   # answer embedding
            source_emb    # source embedding (orthogonal - similarity ~0)
        ])

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value={
            "is_grounded": True,
            "confidence": 0.85,
            "llm_assessment": "Answer is well supported",
            "ungrounded_claims": [],
            "reasoning": "All claims are in sources"
        })

        with patch("src.tools.verify_groundedness.get_embedder", return_value=mock_embedder), \
             patch("src.tools.verify_groundedness.get_llm", return_value=mock_llm), \
             patch("src.tools.verify_groundedness.settings", MagicMock(groundedness_threshold=0.75)):

            result = await verify_groundedness_tool.ainvoke({
                "answer": "Test answer",
                "sources": [{"content": "Source content", "metadata": {}}],
                "threshold": 0.5,  # Lower threshold for this test
                "enable_tier2": True
            })

            # With orthogonal vectors, similarity will be near 0, so Tier 2 should be used
            assert result.tier_used == "tier2"
            assert result.llm_assessment is not None

    @pytest.mark.asyncio
    async def test_verify_groundedness_tier2_disabled(self):
        """Test with Tier 2 disabled."""
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(side_effect=[
            [0.1] * 1536,
            [0.9] * 1536
        ])

        with patch("src.tools.verify_groundedness.get_embedder", return_value=mock_embedder), \
             patch("src.tools.verify_groundedness.settings", MagicMock(groundedness_threshold=0.75)):

            result = await verify_groundedness_tool.ainvoke({
                "answer": "Test answer",
                "sources": [{"content": "Source", "metadata": {}}],
                "threshold": 0.75,
                "enable_tier2": False
            })

            # Should return Tier 1 result without calling Tier 2
            assert result.tier_used == "tier1"
            assert result.llm_assessment is None

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        vec = [0.5, 0.5, 0.5, 0.5]
        similarity = _cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]
        similarity = _cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, rel=1e-5)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        vec1 = [1.0, 1.0, 1.0]
        vec2 = [-1.0, -1.0, -1.0]
        similarity = _cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, rel=1e-5)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = _cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_general(self):
        """Test cosine similarity with general vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [2.0, 3.0, 4.0]
        similarity = _cosine_similarity(vec1, vec2)
        # Should be between 0 and 1
        assert 0 <= similarity <= 1

    def test_format_sources_for_verification_empty(self):
        """Test formatting empty sources list."""
        formatted = format_sources_for_verification([])
        assert "Không có nguồn" in formatted or "No sources" in formatted

    def test_format_sources_for_verification_with_sources(self):
        """Test formatting sources for verification."""
        sources = [
            {
                "content": "Content 1",
                "metadata": {"document_title": "Doc 1"}
            },
            {
                "content": "Content 2",
                "metadata": {"document_title": "Doc 2"}
            }
        ]

        formatted = format_sources_for_verification(sources)

        assert "Nguồn 1" in formatted or "Source 1" in formatted
        assert "Content 1" in formatted
        assert "Doc 1" in formatted
        assert "Nguồn 2" in formatted or "Source 2" in formatted

    def test_format_sources_truncates_long_content(self):
        """Test that long content is truncated."""
        sources = [{
            "content": "x" * 1000,  # Long content
            "metadata": {"document_title": "Test"}
        }]

        formatted = format_sources_for_verification(sources)

        # Content should be truncated to 500 chars
        assert len(formatted) < 1000
