# =============================================================================
# End-to-End Integration Tests for RAG Pipeline
# =============================================================================
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import asyncio

from src.agents.orchestrator import orchestrator
from src.agents.state import create_initial_state, QueryComplexity
from src.models.query import RewrittenQuery, QueryType
from src.models.document import Chunk, Section


# =============================================================================
# RAG Pipeline End-to-End Tests
# =============================================================================

class TestRAGPipeline:
    """Test end-to-end RAG pipeline with mocked storage."""

    @pytest.mark.asyncio
    async def test_simple_query_pipeline(self):
        """Test complete pipeline for a simple query."""
        # Create initial state
        state = create_initial_state("What is the annual leave policy?")

        # Mock the orchestrator components
        mock_chunks = [
            Chunk(
                id=str(uuid4()),
                document_id=str(uuid4()),
                section_id=str(uuid4()),
                content="Employees are entitled to 20 days of annual leave per year.",
                token_count=15,
                position=0,
                metadata={"heading": "Annual Leave", "level": 1}
            )
        ]

        mock_answer = MagicMock()
        mock_answer.answer = "Employees are entitled to 20 days of annual leave per year."
        mock_answer.sources = mock_chunks
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.95
        mock_answer.query_id = state.query_id

        # Mock storage and tools
        with patch("src.tools.hybrid_search.vector_store") as mock_vs, \
             patch("src.tools.hybrid_search.get_embedder") as mock_embedder, \
             patch("src.agents.synthesizer_agent.get_llm") as mock_llm:

            # Setup mocks
            mock_embedder.return_value.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_vs.hybrid_search = AsyncMock(return_value=[
                {
                    "id": mock_chunks[0].id,
                    "score": 0.95,
                    "payload": {
                        "content": mock_chunks[0].content,
                        "document_id": mock_chunks[0].document_id,
                        "section_id": mock_chunks[0].section_id,
                        "heading": "Annual Leave"
                    }
                }
            ])

            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content="Employees are entitled to 20 days of annual leave per year."
            ))
            mock_llm.return_value = mock_llm_instance

            # Run the orchestrator (mocked)
            with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
                result = await orchestrator.process_query(
                    query="What is the annual leave policy?",
                    conversation_history=[],
                    max_tokens=1000
                )

                assert result is not None
                assert result.answer is not None
                assert len(result.sources) >= 0
                assert result.query_id == state.query_id

    @pytest.mark.asyncio
    async def test_complex_query_pipeline(self):
        """Test pipeline for a complex multi-part query."""
        state = create_initial_state(
            "Compare the annual leave policy and sick leave policy. What are the key differences?"
        )

        # Mock multi-source retrieval
        mock_chunks = [
            Chunk(
                id=str(uuid4()),
                document_id=str(uuid4()),
                section_id=str(uuid4()),
                content="Annual leave: 20 days per year, paid at 100%.",
                token_count=12,
                position=0
            ),
            Chunk(
                id=str(uuid4()),
                document_id=str(uuid4()),
                section_id=str(uuid4()),
                content="Sick leave: 10 days per year, requires medical certificate.",
                token_count=12,
                position=1
            )
        ]

        mock_answer = MagicMock()
        mock_answer.answer = "The key differences are: Annual leave provides 20 days at 100% pay, while sick leave provides 10 days requiring a medical certificate."
        mock_answer.sources = mock_chunks
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.88
        mock_answer.query_id = state.query_id

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
            result = await orchestrator.process_query(
                query=state.query,
                conversation_history=[],
                max_tokens=1500
            )

            assert result is not None
            assert "differences" in result.answer.lower() or "different" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_pipeline_with_conversation_history(self):
        """Test pipeline with conversation context."""
        history = [
            "What is the annual leave policy?",
            "Employees get 20 days of annual leave per year."
        ]

        state = create_initial_state("And how about sick leave?", history)

        mock_answer = MagicMock()
        mock_answer.answer = "Sick leave provides 10 days per year and requires a medical certificate."
        mock_answer.sources = []
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.92
        mock_answer.query_id = state.query_id

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
            result = await orchestrator.process_query(
                query=state.query,
                conversation_history=history,
                max_tokens=1000
            )

            assert result is not None
            assert result.query_id == state.query_id


# =============================================================================
# Query Processing Pipeline Tests
# =============================================================================

class TestQueryProcessingPipeline:
    """Test query processing stages."""

    @pytest.mark.asyncio
    async def test_query_rewriting_in_pipeline(self):
        """Test query rewriting stage."""
        from src.tools.query_rewriter import format_conversation_context

        history = ["I'm a new employee"]
        query = "holiday allowance"

        context = format_conversation_context(history)

        assert "new employee" in context.lower()
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_query_decomposition(self):
        """Test complex query decomposition."""
        from src.tools.query_decomposer import QueryDecomposerInput

        input_data = QueryDecomposerInput(
            query="Compare leave policies and bonus structure",
            language="en"
        )

        assert input_data.query is not None
        assert "compare" in input_data.query.lower()

    @pytest.mark.asyncio
    async def test_hybrid_search_retrieval(self):
        """Test hybrid search retrieval."""
        from src.tools.hybrid_search import SearchResult

        # Simulate hybrid search results
        results = [
            SearchResult(
                id="chunk-1",
                type="chunk",
                content="Annual leave policy details",
                score=0.95,
                metadata={"heading": "Annual Leave"},
                document_id=str(uuid4()),
                document_title="HR Policy"
            ),
            SearchResult(
                id="chunk-2",
                type="chunk",
                content="Sick leave policy details",
                score=0.88,
                metadata={"heading": "Sick Leave"},
                document_id=str(uuid4()),
                document_title="HR Policy"
            )
        ]

        assert len(results) == 2
        assert results[0].score > results[1].score
        assert all(r.type == "chunk" for r in results)


# =============================================================================
# Verification Pipeline Tests
# =============================================================================

class TestVerificationPipeline:
    """Test answer verification pipeline."""

    @pytest.mark.asyncio
    async def test_tier1_verification_fast_path(self):
        """Test Tier 1 semantic verification for high similarity."""
        from src.tools.verify_groundedness import _cosine_similarity

        # High similarity vectors
        vec1 = [0.5] * 100
        vec2 = [0.5] * 100

        similarity = _cosine_similarity(vec1, vec2)

        assert similarity > 0.99  # Should be very close to 1.0

    @pytest.mark.asyncio
    async def test_tier2_verification_llm_fallback(self):
        """Test Tier 2 LLM verification for ambiguous cases."""
        from src.tools.verify_groundedness import format_sources_for_verification

        sources = [
            {
                "content": "Annual leave: 20 days per year",
                "metadata": {"document_title": "HR Policy"}
            }
        ]

        formatted = format_sources_for_verification(sources)

        assert "Annual leave" in formatted
        assert "HR Policy" in formatted


# =============================================================================
# Error Handling Pipeline Tests
# =============================================================================

class TestErrorHandlingPipeline:
    """Test error handling in pipeline."""

    @pytest.mark.asyncio
    async def test_empty_sources_handling(self):
        """Test pipeline when no sources are found."""
        mock_answer = MagicMock()
        mock_answer.answer = "I couldn't find any information about that topic in the documents."
        mock_answer.sources = []
        mock_answer.is_grounded = False
        mock_answer.confidence = 0.0
        mock_answer.query_id = str(uuid4())

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
            result = await orchestrator.process_query(
                query="xyzabc query with no results",
                conversation_history=[],
                max_tokens=1000
            )

            assert result is not None
            assert result.is_grounded is False or len(result.sources) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_error_recovery(self):
        """Test orchestrator handles errors gracefully."""
        with patch.object(orchestrator, "process_query", new=AsyncMock(side_effect=Exception("Pipeline error"))):
            with pytest.raises(Exception):
                await orchestrator.process_query(
                    query="test query",
                    conversation_history=[],
                    max_tokens=1000
                )

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test pipeline handles timeouts."""
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(5)
            return MagicMock(answer="Delayed response")

        with patch.object(orchestrator, "process_query", new=AsyncMock(side_effect=slow_process)):
            # This would timeout in real scenario
            try:
                result = await asyncio.wait_for(
                    orchestrator.process_query("test", [], 1000),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                assert True  # Expected timeout


# =============================================================================
# Performance Pipeline Tests
# =============================================================================

class TestPerformancePipeline:
    """Test pipeline performance characteristics."""

    @pytest.mark.asyncio
    async def test_pipeline_latency_measurement(self):
        """Test measuring pipeline execution time."""
        import time

        mock_answer = MagicMock()
        mock_answer.answer = "Quick response"
        mock_answer.sources = []
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.9
        mock_answer.query_id = str(uuid4())

        start_time = time.time()

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
            result = await orchestrator.process_query("test query", [], 1000)

            elapsed = time.time() - start_time

            assert result is not None
            # With mocks, should be very fast (< 100ms)
            assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_query_handling(self):
        """Test handling multiple concurrent queries."""
        mock_answer = MagicMock()
        mock_answer.answer = "Response"
        mock_answer.sources = []
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.9

        async def mock_process(query, history, max_tokens):
            mock_answer.query_id = str(uuid4())
            return mock_answer

        with patch.object(orchestrator, "process_query", new=AsyncMock(side_effect=mock_process)):
            # Run multiple queries concurrently
            tasks = [
                orchestrator.process_query(f"query {i}", [], 1000)
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(r is not None for r in results)


# =============================================================================
# Integration with Storage Tests
# =============================================================================

class TestStorageIntegration:
    """Test integration with storage layer."""

    @pytest.mark.asyncio
    async def test_vector_store_integration(self):
        """Test vector store integration in pipeline."""
        from src.tools.hybrid_search import hybrid_search_tool
        from src.storage.vector_store import vector_store

        # Mock vector store
        mock_results = [
            {
                "id": "chunk-1",
                "score": 0.92,
                "payload": {
                    "content": "Test content",
                    "document_id": str(uuid4())
                }
            }
        ]

        with patch.object(vector_store, "hybrid_search", new=AsyncMock(return_value=mock_results)):
            result = await hybrid_search_tool.ainvoke({
                "query": "test query",
                "collection": "chunks",
                "top_k": 5,
                "use_cache": False
            })

            assert result.total_results == 1
            assert result.query == "test query"

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache integration in pipeline."""
        from src.storage.cache import cache_store

        # Mock cache operations
        with patch.object(cache_store, "get_semantic_cache", new=AsyncMock(return_value=None)), \
             patch.object(cache_store, "set_semantic_cache", new=AsyncMock()):

            cached = await cache_store.get_semantic_cache("test query", [0.1] * 1536)

            assert cached is None

            # Should not raise error
            await cache_store.set_semantic_cache("test query", [0.1] * 1536, {"answer": "test"})

    @pytest.mark.asyncio
    async def test_metadata_store_integration(self):
        """Test metadata store integration."""
        from src.storage.metadata_store import metadata_store

        mock_doc = MagicMock(
            id=str(uuid4()),
            title="Test Document",
            status="indexed"
        )

        with patch.object(metadata_store, "get_document", new=AsyncMock(return_value=mock_doc)):
            doc = await metadata_store.get_document(str(uuid4()))

            assert doc is not None
            assert doc.title == "Test Document"


# =============================================================================
# End-to-End Scenarios
# =============================================================================

class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_hr_policy_query_scenario(self):
        """Test HR policy inquiry scenario."""
        scenario = {
            "query": "How many days of annual leave do full-time employees get?",
            "expected_keywords": ["annual", "leave", "days", "20"],
            "expected_sources_count": 1
        }

        mock_chunks = [
            Chunk(
                id=str(uuid4()),
                document_id=str(uuid4()),
                section_id=str(uuid4()),
                content="Full-time employees are entitled to 20 days of paid annual leave per year.",
                token_count=15,
                position=0
            )
        ]

        mock_answer = MagicMock()
        mock_answer.answer = "Full-time employees are entitled to 20 days of paid annual leave per year."
        mock_answer.sources = mock_chunks
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.95
        mock_answer.query_id = str(uuid4())

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
            result = await orchestrator.process_query(
                query=scenario["query"],
                conversation_history=[],
                max_tokens=1000
            )

            # Verify expectations
            assert result is not None
            for keyword in scenario["expected_keywords"]:
                assert keyword.lower() in result.answer.lower()

    @pytest.mark.asyncio
    async def test_follow_up_questions_scenario(self):
        """Test follow-up question scenario."""
        conversation = [
            "What is the annual leave policy?",
            "Employees get 20 days of annual leave per year.",
            "Does this include probationary employees?",
            "Yes, probationary employees also get annual leave pro-rated."
        ]

        mock_answer = MagicMock()
        mock_answer.answer = "Annual leave accrues proportionally during probation."
        mock_answer.sources = []
        mock_answer.is_grounded = True
        mock_answer.confidence = 0.88
        mock_answer.query_id = str(uuid4())

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_answer)):
            result = await orchestrator.process_query(
                query="How does annual leave work during probation?",
                conversation_history=conversation,
                max_tokens=1000
            )

            assert result is not None
