# =============================================================================
# Unit Tests for All Agents
# =============================================================================
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from src.agents.state import AgentState, QueryComplexity, create_initial_state
from src.agents.router_agent import router_agent, RouterAgent
from src.agents.planner_agent import planner_agent, PlannerAgent
from src.agents.retriever_agent import retriever_agent, RetrieverAgent
from src.agents.grader_agent import grader_agent, GraderAgent
from src.agents.synthesizer_agent import synthesizer_agent, SynthesizerAgent
from src.agents.verifier_agent import verifier_agent, VerifierAgent
from src.agents.orchestrator import orchestrator, RAGOrchestrator


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_state():
    """Base agent state for testing."""
    return create_initial_state("Test query")


@pytest.fixture
def mock_rewriter_result():
    """Mock query rewriter result."""
    mock = Mock()
    mock.original = "Test query"
    mock.rewritten = "Test query rewritten"
    mock.query_type = "factual"
    mock.keywords = ["test", "query"]
    mock.confidence = 0.9
    mock.expansions = []
    return mock


@pytest.fixture
def mock_decomposer_result():
    """Mock query decomposer result."""
    mock = Mock()
    mock.original_query = "Test query"
    mock.sub_queries = [
        Mock(id=0, query="Sub query 1", query_type="factual", dependencies=[])
    ]
    mock.dependencies = {}
    mock.expected_answer_types = []
    mock.execution_order = [0]
    mock.requires_aggregation = False
    mock.reasoning = "Test reasoning"
    return mock


@pytest.fixture
def mock_search_result():
    """Mock hybrid search result."""
    mock = AsyncMock()
    mock.results = [
        Mock(
            chunk_id="chunk-1",
            content="Test content 1",
            document_id="doc-1",
            section_id="sec-1",
            score=0.9,
            document_title="Test Doc",
            section_path="/path"
        )
    ]
    mock.cache_hit = False
    mock.total_results = 1
    mock.query_type = "vector"
    return mock


@pytest.fixture
def mock_synthesis_result():
    """Mock synthesis result."""
    mock = Mock()
    mock.answer = "Test answer based on sources"
    mock.citations = [
        Mock(
            section_id="sec-1",
            document_id="doc-1",
            document_title="Test Doc",
            section_heading="Test Section",
            content_snippet="Test snippet",
            relevance_score=0.9
        )
    ]
    mock.confidence = 0.85
    mock.reasoning = "Test reasoning"
    mock.sources_summary = "1 source used"
    mock.language = "vi"
    return mock


@pytest.fixture
def mock_verification_result():
    """Mock verification result."""
    mock = Mock()
    mock.is_grounded = True
    mock.tier_used = "tier1"
    mock.confidence = 0.85
    mock.similarity_score = 0.85
    mock.llm_assessment = "Well grounded"
    mock.ungrounded_claims = []
    mock.reasoning = "Good grounding"
    return mock


# =============================================================================
# RouterAgent Tests
# =============================================================================

class TestRouterAgent:
    """Tests for RouterAgent."""

    @pytest.mark.asyncio
    async def test_simple_query_routing(self, base_state, mock_rewriter_result, mock_decomposer_result):
        """Test routing a simple query."""
        mock_rewrite_tool = AsyncMock()
        mock_rewrite_tool.ainvoke = AsyncMock(return_value=mock_rewriter_result)
        mock_decompose_tool = AsyncMock()
        mock_decompose_tool.ainvoke = AsyncMock(return_value=mock_decomposer_result)

        with patch('src.agents.router_agent.query_rewriter_tool', mock_rewrite_tool), \
             patch('src.agents.router_agent.query_decomposer_tool', mock_decompose_tool):

            result = await router_agent(base_state)

            assert result["current_step"] == "routed"
            assert result["rewritten_query"] == "Test query rewritten"
            assert result["query_type"] == "factual"
            assert result["complexity"] == QueryComplexity.SIMPLE
            assert result["is_decomposed"] is False

    @pytest.mark.asyncio
    async def test_complex_query_routing(self, base_state, mock_rewriter_result):
        """Test routing a complex query."""
        mock_decomposer_result = Mock()
        mock_decomposer_result.original_query = "Test query"
        mock_decomposer_result.sub_queries = [
            Mock(id=0, query="Sub 1", query_type="factual", dependencies=[]),
            Mock(id=1, query="Sub 2", query_type="factual", dependencies=[0])
        ]
        mock_decomposer_result.requires_aggregation = True
        mock_decomposer_result.dependencies = {0: [], 1: [0]}
        mock_decomposer_result.expected_answer_types = []
        mock_decomposer_result.execution_order = [0, 1]
        mock_decomposer_result.reasoning = "Complex"

        mock_rewrite_tool = AsyncMock()
        mock_rewrite_tool.ainvoke = AsyncMock(return_value=mock_rewriter_result)
        mock_decompose_tool = AsyncMock()
        mock_decompose_tool.ainvoke = AsyncMock(return_value=mock_decomposer_result)

        with patch('src.agents.router_agent.query_rewriter_tool', mock_rewrite_tool), \
             patch('src.agents.router_agent.query_decomposer_tool', mock_decompose_tool):

            result = await router_agent(base_state)

            assert result["complexity"] == QueryComplexity.COMPLEX
            assert result["is_decomposed"] is True
            assert len(result["sub_queries"]) == 2

    def test_get_next_step_simple(self, base_state):
        """Test next step routing for simple query."""
        base_state["complexity"] = QueryComplexity.SIMPLE
        base_state["is_decomposed"] = False
        base_state["current_step"] = "routed"

        next_step = router_agent.get_next_step(base_state)
        assert next_step == "retriever"

    def test_get_next_step_complex(self, base_state):
        """Test next step routing for complex query."""
        base_state["complexity"] = QueryComplexity.COMPLEX
        base_state["is_decomposed"] = True
        base_state["current_step"] = "routed"

        next_step = router_agent.get_next_step(base_state)
        assert next_step == "planner"

    def test_get_next_step_error(self, base_state):
        """Test next step routing with error."""
        base_state["error"] = "Test error"
        base_state["current_step"] = "routed"

        next_step = router_agent.get_next_step(base_state)
        assert next_step == "error_handler"


# =============================================================================
# PlannerAgent Tests
# =============================================================================

class TestPlannerAgent:
    """Tests for PlannerAgent."""

    @pytest.mark.asyncio
    async def test_simple_query_plan(self, base_state):
        """Test planning for a simple (non-decomposed) query."""
        base_state["rewritten_query"] = "Simple query"
        base_state["is_decomposed"] = False

        result = await planner_agent(base_state)

        assert result["current_step"] == "planned"
        assert result["execution_plan"] is not None
        assert len(result["execution_plan"]) == 1
        assert result["execution_plan"][0]["collection"] == "chunks"
        assert result["execution_plan"][0]["top_k"] == 20

    @pytest.mark.asyncio
    async def test_complex_query_plan(self, base_state):
        """Test planning for a complex query."""
        base_state["original_query"] = "Complex question"
        base_state["query_type"] = "complex"
        base_state["is_decomposed"] = True
        base_state["sub_queries"] = [
            {"id": 0, "query": "Sub 1", "query_type": "factual", "dependencies": []},
            {"id": 1, "query": "Sub 2", "query_type": "factual", "dependencies": []}
        ]

        with patch('src.agents.planner_agent.get_llm') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = Mock()
            mock_response.content = '{"execution_plan": [{"step": 1, "sub_query_index": 0, "query": "Sub 1", "collection": "chunks", "top_k": 15, "filters": {}, "reasoning": "Test"}], "overall_strategy": "Sequential"}'
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await planner_agent(base_state)

            assert result["current_step"] == "planned"
            assert result["execution_plan"] is not None
            assert len(result["execution_plan"]) >= 1

    def test_create_fallback_plan(self, base_state):
        """Test fallback plan creation."""
        base_state["rewritten_query"] = "Test query"

        plan = planner_agent._create_fallback_plan(base_state)

        assert isinstance(plan, list)
        assert len(plan) >= 1
        assert plan[0]["query"] == "Test query"
        assert plan[0]["collection"] == "chunks"


# =============================================================================
# RetrieverAgent Tests
# =============================================================================

class TestRetrieverAgent:
    """Tests for RetrieverAgent."""

    @pytest.mark.asyncio
    async def test_retrieve_with_plan(self, base_state, mock_search_result):
        """Test retrieval with execution plan."""
        base_state["rewritten_query"] = "Test query"
        base_state["execution_plan"] = [
            {"step": 1, "query": "Test query", "collection": "chunks", "top_k": 10, "filters": {}}
        ]

        mock_search = AsyncMock()
        mock_search.ainvoke = AsyncMock(return_value=mock_search_result)

        with patch('src.agents.retriever_agent.hybrid_search_tool', mock_search):
            result = await retriever_agent(base_state)

            assert result["current_step"] == "retrieved"
            assert len(result["retrieved_chunks"]) == 1
            assert result["retrieved_chunks"][0]["chunk_id"] == "chunk-1"

    @pytest.mark.asyncio
    async def test_retrieve_without_plan(self, base_state, mock_search_result):
        """Test retrieval without execution plan (default)."""
        base_state["rewritten_query"] = "Test query"
        base_state["execution_plan"] = None

        mock_search = AsyncMock()
        mock_search.ainvoke = AsyncMock(return_value=mock_search_result)

        with patch('src.agents.retriever_agent.hybrid_search_tool', mock_search):
            result = await retriever_agent(base_state)

            assert result["current_step"] == "retrieved"
            assert result["execution_plan"] is not None  # Should create default

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, base_state):
        """Test retrieval with no results."""
        base_state["rewritten_query"] = "Test query"
        base_state["execution_plan"] = [
            {"step": 1, "query": "Test query", "collection": "chunks", "top_k": 10, "filters": {}}
        ]

        mock_result = AsyncMock()
        mock_result.results = []
        mock_search = AsyncMock()
        mock_search.ainvoke = AsyncMock(return_value=mock_result)

        with patch('src.agents.retriever_agent.hybrid_search_tool', mock_search):
            result = await retriever_agent(base_state)

            assert result["current_step"] == "retrieved"
            assert len(result["retrieved_chunks"]) == 0


# =============================================================================
# GraderAgent Tests
# =============================================================================

class TestGraderAgent:
    """Tests for GraderAgent."""

    @pytest.mark.asyncio
    async def test_grade_chunks(self, base_state):
        """Test grading retrieved chunks."""
        base_state["retrieved_chunks"] = [
            {"content": "Relevant content", "score": 0.9},
            {"content": "Less relevant", "score": 0.5}
        ]

        with patch('src.agents.grader_agent.get_llm') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = Mock()
            mock_response.content = '{"score": 8, "reason": "Good match"}'
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await grader_agent(base_state)

            assert result["current_step"] == "graded"
            assert len(result["relevance_scores"]) > 0
            assert isinstance(result["needs_more_retrieval"], bool)

    @pytest.mark.asyncio
    async def test_grade_empty_chunks(self, base_state):
        """Test grading with no chunks."""
        base_state["retrieved_chunks"] = []

        result = await grader_agent(base_state)

        assert result["current_step"] == "graded"
        assert result["needs_more_retrieval"] is True

    def test_should_retry_retrieval(self, base_state):
        """Test retry logic."""
        base_state["needs_more_retrieval"] = True
        base_state["retrieval_retries"] = 0

        should_retry = grader_agent.should_retry_retrieval(base_state)
        assert should_retry is True

        # At max retries
        base_state["retrieval_retries"] = 2
        should_retry = grader_agent.should_retry_retrieval(base_state)
        assert should_retry is False


# =============================================================================
# SynthesizerAgent Tests
# =============================================================================

class TestSynthesizerAgent:
    """Tests for SynthesizerAgent."""

    @pytest.mark.asyncio
    async def test_synthesize_answer(self, base_state, mock_synthesis_result):
        """Test answer synthesis with chunks."""
        base_state["retrieved_chunks"] = [
            {
                "content": "Test content",
                "document_id": "doc-1",
                "section_id": "sec-1",
                "score": 0.9,
                "metadata": {
                    "document_title": "Test Doc",
                    "section_heading": "Test Section",
                    "section_path": "/path"
                }
            }
        ]

        mock_synthesize = AsyncMock()
        mock_synthesize.ainvoke = AsyncMock(return_value=mock_synthesis_result)

        with patch('src.agents.synthesizer_agent.synthesize_answer_tool', mock_synthesize):
            result = await synthesizer_agent(base_state)

            assert result["current_step"] == "synthesized"
            assert result["draft_answer"] == "Test answer based on sources"
            assert len(result["citations"]) == 1

    @pytest.mark.asyncio
    async def test_synthesize_no_chunks(self, base_state):
        """Test synthesis with no chunks."""
        base_state["retrieved_chunks"] = []

        result = await synthesizer_agent(base_state)

        assert result["current_step"] == "synthesized"
        assert "không tìm thấy thông tin" in result["draft_answer"].lower()

    @pytest.mark.asyncio
    async def test_synthesize_error_handling(self, base_state):
        """Test synthesis error handling."""
        base_state["retrieved_chunks"] = [{"content": "Test"}]

        with patch('src.agents.synthesizer_agent.synthesize_answer_tool') as mock_synthesize:
            mock_synthesize.ainvoke.side_effect = Exception("Synthesis error")

            result = await synthesizer_agent(base_state)

            assert result["current_step"] == "synthesized"
            assert "khó khăn" in result["draft_answer"].lower() or "lỗi" in result["draft_answer"].lower()


# =============================================================================
# VerifierAgent Tests
# =============================================================================

class TestVerifierAgent:
    """Tests for VerifierAgent."""

    @pytest.mark.asyncio
    async def test_verify_grounded_answer(self, base_state, mock_verification_result):
        """Test verification of grounded answer."""
        base_state["draft_answer"] = "Test answer"
        base_state["retrieved_chunks"] = [
            {"content": "Test content", "document_id": "doc-1"}
        ]

        mock_verify = AsyncMock()
        mock_verify.ainvoke = AsyncMock(return_value=mock_verification_result)
        mock_freshness = AsyncMock()
        mock_freshness.ainvoke = AsyncMock(return_value=Mock(
            documents=[],
            has_stale_documents=False
        ))

        with patch('src.agents.verifier_agent.verify_groundedness_tool', mock_verify), \
             patch('src.agents.verifier_agent.check_freshness_tool', mock_freshness):

            result = await verifier_agent(base_state)

            assert result["current_step"] == "verified"
            assert result["final_answer"] is not None
            assert result["is_grounded"] is True

    @pytest.mark.asyncio
    async def test_verify_no_answer(self, base_state):
        """Test verification with no draft answer."""
        base_state["draft_answer"] = None
        base_state["retrieved_chunks"] = []

        result = await verifier_agent(base_state)

        assert result["current_step"] == "verified"
        assert result["is_grounded"] is False

    @pytest.mark.asyncio
    async def test_verify_ungrounded_answer(self, base_state):
        """Test verification of ungrounded answer."""
        base_state["draft_answer"] = "Unverified answer"
        base_state["retrieved_chunks"] = [{"content": "Test"}]

        mock_verification = Mock()
        mock_verification.is_grounded = False
        mock_verification.tier_used = "tier2"
        mock_verification.confidence = 0.5
        mock_verification.ungrounded_claims = ["Claim 1", "Claim 2"]

        mock_verify = AsyncMock()
        mock_verify.ainvoke = AsyncMock(return_value=mock_verification)
        mock_freshness = AsyncMock()
        mock_freshness.ainvoke = AsyncMock(return_value=Mock(
            documents=[],
            has_stale_documents=False
        ))

        with patch('src.agents.verifier_agent.verify_groundedness_tool', mock_verify), \
             patch('src.agents.verifier_agent.check_freshness_tool', mock_freshness):

            result = await verifier_agent(base_state)

            assert result["is_grounded"] is False
            assert result["should_escalate"] is True
            assert "cần được xác minh" in result["final_answer"]


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestRAGOrchestrator:
    """Tests for RAGOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orch = RAGOrchestrator(enable_tree_navigation=False)

        assert orch.graph is not None
        assert orch.memory is not None
        assert orch.app is not None
        assert orch.enable_tree_navigation is False

    def test_build_graph(self):
        """Test graph building."""
        orch = RAGOrchestrator()
        graph = orch._build_graph()

        assert graph is not None

    def test_route_after_router_simple(self, base_state):
        """Test routing after router for simple query."""
        base_state["complexity"] = QueryComplexity.SIMPLE
        base_state["is_decomposed"] = False
        base_state["error"] = None

        orch = RAGOrchestrator()
        next_step = orch._route_after_router(base_state)

        assert next_step == "retriever"

    def test_route_after_router_complex(self, base_state):
        """Test routing after router for complex query."""
        base_state["complexity"] = QueryComplexity.COMPLEX
        base_state["is_decomposed"] = True
        base_state["error"] = None

        orch = RAGOrchestrator()
        next_step = orch._route_after_router(base_state)

        assert next_step == "planner"

    def test_route_after_router_error(self, base_state):
        """Test routing after router with error."""
        base_state["error"] = "Test error"

        orch = RAGOrchestrator()
        next_step = orch._route_after_router(base_state)

        assert next_step == "error"

    def test_route_after_grader_retry(self, base_state):
        """Test routing after grader with retry needed."""
        base_state["needs_more_retrieval"] = True
        base_state["retrieval_retries"] = 0
        base_state["error"] = None

        orch = RAGOrchestrator()
        next_step = orch._route_after_grader(base_state)

        assert next_step == "retriever"

    def test_route_after_grader_proceed(self, base_state):
        """Test routing after grader proceeding to synthesis."""
        base_state["needs_more_retrieval"] = False
        base_state["retrieval_retries"] = 0
        base_state["error"] = None

        orch = RAGOrchestrator()
        next_step = orch._route_after_grader(base_state)

        assert next_step == "synthesizer"

    @pytest.mark.asyncio
    async def test_error_handler(self, base_state):
        """Test error handler node."""
        base_state["error"] = "Test error"
        base_state["current_step"] = "retrieved"

        orch = RAGOrchestrator()
        result = await orch._error_handler(base_state)

        assert result["current_step"] == "error"
        assert result["should_escalate"] is True
        assert "lỗi" in result["final_answer"].lower()

    def test_build_response(self, base_state):
        """Test response building."""
        base_state["query_id"] = "test-id"
        base_state["final_answer"] = "Test answer"
        base_state["is_grounded"] = True
        base_state["verification_tier"] = 1
        base_state["retrieved_chunks"] = [{"chunk_id": "1"}]
        base_state["retrieval_scores"] = [0.9]
        base_state["citations"] = []
        base_state["unsupported_claims"] = []
        base_state["query_type"] = "factual"
        base_state["complexity"] = "simple"
        base_state["should_escalate"] = False
        base_state["retrieval_retries"] = 0

        orch = RAGOrchestrator()
        response = orch._build_response(base_state, 1000.0)

        assert response["query_id"] == "test-id"
        assert response["answer"] == "Test answer"
        assert response["processing_time_ms"] == 1000.0
        assert response["verification"]["is_grounded"] is True
        assert response["metadata"]["chunks_retrieved"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for agent workflows."""

    @pytest.mark.asyncio
    async def test_simple_query_flow(self):
        """Test complete flow for a simple query."""
        # This test would require mocking all the tools
        # For now, just test the orchestrator can be called
        orch = RAGOrchestrator()

        # Mock the entire pipeline
        with patch('src.agents.router_agent.query_rewriter_tool'), \
             patch('src.agents.router_agent.query_decomposer_tool'), \
             patch('src.agents.retriever_agent.hybrid_search_tool'), \
             patch('src.agents.synthesizer_agent.synthesize_answer_tool'), \
             patch('src.agents.verifier_agent.verify_groundedness_tool'), \
             patch('src.agents.verifier_agent.check_freshness_tool'):

            result = await orch.process_query("Test query")

            # Should get a response even if tools fail
            assert "query_id" in result
            assert "answer" in result
            assert "processing_time_ms" in result

    @pytest.mark.asyncio
    async def test_orchestrator_singleton(self):
        """Test that orchestrator singleton works."""
        from src.agents import orchestrator

        assert orchestrator is not None
        assert isinstance(orchestrator, RAGOrchestrator)
