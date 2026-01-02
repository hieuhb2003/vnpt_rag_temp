# =============================================================================
# Tests for Agent State
# =============================================================================
import pytest
from datetime import datetime
from uuid import UUID

from src.agents.state import (
    QueryComplexity,
    AgentState,
    create_initial_state,
    update_state_step,
    merge_retrieval_results,
    state_to_log_dict
)


# =============================================================================
# Tests for QueryComplexity Enum
# =============================================================================

class TestQueryComplexity:
    """Tests for QueryComplexity enum."""

    def test_enum_values(self):
        """Test that all enum values are defined correctly."""
        assert QueryComplexity.SIMPLE == "simple"
        assert QueryComplexity.MODERATE == "moderate"
        assert QueryComplexity.COMPLEX == "complex"

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert "simple" in QueryComplexity._value2member_map_
        assert "moderate" in QueryComplexity._value2member_map_
        assert "complex" in QueryComplexity._value2member_map_


# =============================================================================
# Tests for create_initial_state
# =============================================================================

class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_state_with_query(self):
        """Test creating initial state with a query."""
        query = "Số ngày nghỉ phép của nhân viên chính thức?"
        state = create_initial_state(query)

        assert state["original_query"] == query
        assert state["query_id"] is not None
        assert isinstance(state["query_id"], str)

    def test_query_id_is_uuid(self):
        """Test that query_id is a valid UUID string."""
        state = create_initial_state("Test query")

        # Should be able to parse as UUID
        uuid = UUID(state["query_id"])
        assert uuid.version == 4  # UUID v4 (random)

    def test_initial_values_are_none_or_empty(self):
        """Test that optional fields start with None/empty values."""
        state = create_initial_state("Test query")

        # Query info - some should be None initially
        assert state["rewritten_query"] is None
        assert state["query_type"] is None
        assert state["complexity"] is None

        # Decomposition - should be empty/False
        assert state["is_decomposed"] is False
        assert state["sub_queries"] == []
        assert state["sub_query_results"] == {}

        # Execution plan
        assert state["execution_plan"] is None

        # Retrieval - should be empty
        assert state["retrieved_chunks"] == []
        assert state["retrieval_scores"] == []

        # Synthesis - should be None
        assert state["draft_answer"] is None
        assert state["final_answer"] is None
        assert state["citations"] == []

        # Verification - should be None/empty
        assert state["is_grounded"] is None
        assert state["verification_tier"] is None
        assert state["unsupported_claims"] == []

        # Grading - should be empty/False
        assert state["relevance_scores"] == []
        assert state["needs_more_retrieval"] is False
        assert state["retrieval_retries"] == 0

        # Flow control
        assert state["current_step"] == "start"
        assert state["error"] is None
        assert state["should_escalate"] is False

        # Metadata
        assert state["start_time"] > 0
        assert isinstance(state["start_time"], float)
        assert state["messages"] == []

    def test_start_time_is_recent(self):
        """Test that start_time is a recent timestamp."""
        before = datetime.utcnow().timestamp()
        state = create_initial_state("Test query")
        after = datetime.utcnow().timestamp()

        assert before <= state["start_time"] <= after

    def test_create_state_with_conversation_history(self):
        """Test creating state with conversation history (parameter exists)."""
        # The function accepts conversation_history parameter
        # even though it's not used in current implementation
        state = create_initial_state("Test query", conversation_history=["previous query"])

        assert state["original_query"] == "Test query"
        assert state["messages"] == []  # Messages start empty


# =============================================================================
# Tests for update_state_step
# =============================================================================

class TestUpdateStateStep:
    """Tests for update_state_step helper function."""

    def test_update_step_only(self):
        """Test updating just the step."""
        state = create_initial_state("Test query")
        assert state["current_step"] == "start"

        updated = update_state_step(state, "routed")
        assert updated["current_step"] == "routed"
        assert updated["error"] is None

    def test_update_step_with_error(self):
        """Test updating step with error message."""
        state = create_initial_state("Test query")

        updated = update_state_step(state, "error", error="Something went wrong")
        assert updated["current_step"] == "error"
        assert updated["error"] == "Something went wrong"

    def test_update_preserves_other_fields(self):
        """Test that updating step preserves other state fields."""
        state = create_initial_state("Test query")
        query_id = state["query_id"]
        original_query = state["original_query"]

        updated = update_state_step(state, "routed")
        assert updated["query_id"] == query_id
        assert updated["original_query"] == original_query


# =============================================================================
# Tests for merge_retrieval_results
# =============================================================================

class TestMergeRetrievalResults:
    """Tests for merge_retrieval_results helper function."""

    def test_merge_empty_state(self):
        """Test merging into empty state."""
        state = create_initial_state("Test query")

        new_chunks = [
            {"chunk_id": "chunk-1", "content": "Test content 1", "score": 0.9},
            {"chunk_id": "chunk-2", "content": "Test content 2", "score": 0.8}
        ]
        new_scores = [0.9, 0.8]

        merged = merge_retrieval_results(state, new_chunks, new_scores)

        assert len(merged["retrieved_chunks"]) == 2
        assert len(merged["retrieval_scores"]) == 2
        assert merged["retrieved_chunks"][0]["chunk_id"] == "chunk-1"

    def test_merge_with_existing_chunks(self):
        """Test merging with existing chunks."""
        state = create_initial_state("Test query")
        state["retrieved_chunks"] = [
            {"chunk_id": "chunk-1", "content": "Test 1", "score": 0.9}
        ]
        state["retrieval_scores"] = [0.9]

        new_chunks = [
            {"chunk_id": "chunk-2", "content": "Test 2", "score": 0.8}
        ]
        new_scores = [0.8]

        merged = merge_retrieval_results(state, new_chunks, new_scores)

        assert len(merged["retrieved_chunks"]) == 2
        assert len(merged["retrieval_scores"]) == 2

    def test_deduplication_by_chunk_id(self):
        """Test that duplicate chunk_ids are deduplicated."""
        state = create_initial_state("Test query")
        state["retrieved_chunks"] = [
            {"chunk_id": "chunk-1", "content": "First", "score": 0.9}
        ]
        state["retrieval_scores"] = [0.9]

        # Add duplicate chunk_id
        new_chunks = [
            {"chunk_id": "chunk-1", "content": "Duplicate", "score": 0.85},
            {"chunk_id": "chunk-2", "content": "New", "score": 0.8}
        ]
        new_scores = [0.85, 0.8]

        merged = merge_retrieval_results(state, new_chunks, new_scores)

        # Should only have 2 unique chunks
        assert len(merged["retrieved_chunks"]) == 2
        chunk_ids = [c["chunk_id"] for c in merged["retrieved_chunks"]]
        assert chunk_ids.count("chunk-1") == 1
        assert chunk_ids.count("chunk-2") == 1

    def test_chunks_without_id_are_kept(self):
        """Test that chunks without chunk_id are not deduplicated."""
        state = create_initial_state("Test query")

        new_chunks = [
            {"content": "No ID 1"},
            {"content": "No ID 2"},
            {"content": "No ID 3"}
        ]
        new_scores = [0.9, 0.8, 0.7]

        merged = merge_retrieval_results(state, new_chunks, new_scores)

        # All should be kept since they don't have IDs
        assert len(merged["retrieved_chunks"]) == 3


# =============================================================================
# Tests for state_to_log_dict
# =============================================================================

class TestStateToLogDict:
    """Tests for state_to_log_dict helper function."""

    def test_log_dict_contains_required_fields(self):
        """Test that log dict contains all required fields."""
        state = create_initial_state("Test query for logging")
        log_dict = state_to_log_dict(state)

        assert "query_id" in log_dict
        assert "query" in log_dict
        assert "current_step" in log_dict
        assert "complexity" in log_dict
        assert "is_decomposed" in log_dict
        assert "chunks_retrieved" in log_dict
        assert "has_answer" in log_dict
        assert "is_verified" in log_dict
        assert "has_error" in log_dict
        assert "elapsed_ms" in log_dict

    def test_long_query_is_truncated(self):
        """Test that long queries are truncated in log dict."""
        state = create_initial_state("a" * 200)
        log_dict = state_to_log_dict(state)

        assert len(log_dict["query"]) <= 103  # 100 + "..."
        assert "..." in log_dict["query"]

    def test_chunks_retrieved_count(self):
        """Test chunks_retrieved reflects actual count."""
        state = create_initial_state("Test query")
        state["retrieved_chunks"] = [
            {"chunk_id": f"chunk-{i}"}
            for i in range(5)
        ]

        log_dict = state_to_log_dict(state)
        assert log_dict["chunks_retrieved"] == 5

    def test_has_answer_reflects_draft_answer(self):
        """Test has_answer is True when draft_answer exists."""
        state = create_initial_state("Test query")

        log_dict = state_to_log_dict(state)
        assert log_dict["has_answer"] is False

        state["draft_answer"] = "This is an answer"
        log_dict = state_to_log_dict(state)
        assert log_dict["has_answer"] is True

    def test_is_verified_reflects_groundedness(self):
        """Test is_verified is True when is_grounded is set."""
        state = create_initial_state("Test query")

        log_dict = state_to_log_dict(state)
        assert log_dict["is_verified"] is False

        state["is_grounded"] = True
        log_dict = state_to_log_dict(state)
        assert log_dict["is_verified"] is True

    def test_has_error_reflects_error_state(self):
        """Test has_error is True when error is set."""
        state = create_initial_state("Test query")

        log_dict = state_to_log_dict(state)
        assert log_dict["has_error"] is False

        state["error"] = "Something went wrong"
        log_dict = state_to_log_dict(state)
        assert log_dict["has_error"] is True

    def test_elapsed_ms_is_calculated(self):
        """Test that elapsed_ms is calculated correctly."""
        state = create_initial_state("Test query")
        log_dict = state_to_log_dict(state)

        # Should be small for a newly created state
        assert log_dict["elapsed_ms"] >= 0
        assert log_dict["elapsed_ms"] < 100  # Less than 100ms

    def test_sensitive_data_not_in_log(self):
        """Test that sensitive data is not exposed in log dict."""
        state = create_initial_state("Test query")
        state["draft_answer"] = "Secret answer"
        state["final_answer"] = "Final secret"

        log_dict = state_to_log_dict(state)

        # Answers should not be in log dict
        assert "Secret answer" not in str(log_dict)
        assert "Final secret" not in str(log_dict)


# =============================================================================
# Integration Tests
# =============================================================================

class TestStateIntegration:
    """Integration tests for state management."""

    def test_full_state_flow(self):
        """Test state flowing through multiple updates."""
        # Start with initial state
        state = create_initial_state("Nhân viên được bao nhiêu ngày nghỉ?")
        assert state["current_step"] == "start"

        # Route the query
        state = update_state_step(state, "routed")
        state["rewritten_query"] = "Số ngày nghỉ phép của nhân viên"
        state["complexity"] = QueryComplexity.SIMPLE
        assert state["current_step"] == "routed"

        # Add retrieval results
        chunks = [
            {"chunk_id": "c1", "content": "Nhân viên được 20 ngày nghỉ", "score": 0.9}
        ]
        state = merge_retrieval_results(state, chunks, [0.9])
        assert len(state["retrieved_chunks"]) == 1

        # Generate answer
        state["draft_answer"] = "Nhân viên chính thức được hưởng 20 ngày nghỉ phép."
        state = update_state_step(state, "synthesized")
        assert state["current_step"] == "synthesized"

        # Verify
        state["is_grounded"] = True
        state["final_answer"] = state["draft_answer"]
        state = update_state_step(state, "verified")
        assert state["current_step"] == "verified"

        # Check log dict
        log = state_to_log_dict(state)
        assert log["current_step"] == "verified"
        assert log["has_answer"] is True
        assert log["is_verified"] is True

    def test_state_with_complex_query(self):
        """Test state handling for complex (decomposed) query."""
        state = create_initial_state("So sánh ngày nghỉ phép giữa nhân viên chính thức và thời vụ?")
        state["complexity"] = QueryComplexity.COMPLEX
        state["is_decomposed"] = True
        state["sub_queries"] = [
            {"id": 1, "query": "Ngày nghỉ nhân viên chính thức"},
            {"id": 2, "query": "Ngày nghỉ nhân viên thời vụ"}
        ]

        # Simulate planner
        state["execution_plan"] = [
            {"step": 1, "sub_query_index": 0, "collection": "chunks", "top_k": 10},
            {"step": 2, "sub_query_index": 1, "collection": "chunks", "top_k": 10}
        ]

        # Simulate retrieval for each sub-query
        state["sub_query_results"] = {
            0: [{"chunk_id": "c1", "content": "Nhân viên chính thức: 20 ngày"}],
            1: [{"chunk_id": "c2", "content": "Nhân viên thời vụ: 12 ngày"}]
        }

        assert len(state["sub_queries"]) == 2
        assert state["is_decomposed"] is True
        assert len(state["execution_plan"]) == 2
