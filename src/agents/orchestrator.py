# =============================================================================
# RAG Orchestrator - LangGraph-based multi-agent orchestration
# =============================================================================
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState, create_initial_state, state_to_log_dict
from src.agents.router_agent import router_agent
from src.agents.planner_agent import planner_agent
from src.agents.retriever_agent import retriever_agent
from src.agents.grader_agent import grader_agent
from src.agents.synthesizer_agent import synthesizer_agent
from src.agents.verifier_agent import verifier_agent
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RAGOrchestrator:
    """
    LangGraph-based orchestrator for multi-agent RAG pipeline.

    Graph structure:
        start → router → [planner] → retriever → grader → [retriever] → synthesizer → verifier → end
                              ↓                                                    ↓
                        error_handler                                          error_handler

    The graph supports:
    - Conditional routing based on query complexity
    - Retrieval retry loop (max 2 retries)
    - Error handling at each step
    - Memory checkpointing for conversation continuity
    """

    def __init__(self, enable_tree_navigation: bool = True):
        """
        Initialize the RAG Orchestrator.

        Args:
            enable_tree_navigation: Whether to enable tree navigation in retriever
        """
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
        self.enable_tree_navigation = enable_tree_navigation

        logger.info("RAG Orchestrator initialized", extra={
            "tree_navigation": enable_tree_navigation
        })

    def _build_graph(self) -> StateGraph:
        """
        Build the agent workflow graph.

        Returns:
            StateGraph configured with all nodes and edges
        """
        # Create state graph
        graph = StateGraph(AgentState)

        # =====================================================================
        # Add nodes
        # =====================================================================
        graph.add_node("router", router_agent)
        graph.add_node("planner", planner_agent)
        graph.add_node("retriever", retriever_agent)
        graph.add_node("grader", grader_agent)
        graph.add_node("synthesizer", synthesizer_agent)
        graph.add_node("verifier", verifier_agent)
        graph.add_node("error_handler", self._error_handler)

        # =====================================================================
        # Set entry point
        # =====================================================================
        graph.set_entry_point("router")

        # =====================================================================
        # Add conditional edges from router
        # =====================================================================
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "planner": "planner",
                "retriever": "retriever",
                "error": "error_handler"
            }
        )

        # =====================================================================
        # Planner always goes to retriever
        # =====================================================================
        graph.add_edge("planner", "retriever")

        # =====================================================================
        # Grader conditionally loops back or continues
        # =====================================================================
        graph.add_conditional_edges(
            "grader",
            self._route_after_grader,
            {
                "retriever": "retriever",
                "synthesizer": "synthesizer",
                "error": "error_handler"
            }
        )

        # =====================================================================
        # Synthesizer goes to verifier
        # =====================================================================
        graph.add_edge("synthesizer", "verifier")

        # =====================================================================
        # Verifier goes to end
        # =====================================================================
        graph.add_edge("verifier", END)

        # =====================================================================
        # Error handler goes to end
        # =====================================================================
        graph.add_edge("error_handler", END)

        logger.info("Agent workflow graph built")

        return graph

    # ========================================================================
    # Routing Functions
    # ========================================================================

    def _route_after_router(self, state: AgentState) -> str:
        """
        Determine next step after router.

        Args:
            state: Current agent state

        Returns:
            Next node name
        """
        if state.get("error"):
            logger.info(
                "Router error, routing to error handler",
                extra={"query_id": state["query_id"]}
            )
            return "error"

        # Complex or decomposed queries go through planner
        if state.get("is_decomposed") or state.get("complexity") == "complex":
            logger.info(
                "Query requires planning",
                extra={
                    "query_id": state["query_id"],
                    "is_decomposed": state.get("is_decomposed"),
                    "complexity": state.get("complexity")
                }
            )
            return "planner"

        # Simple and moderate queries go directly to retrieval
        logger.info(
            "Direct to retrieval",
            extra={"query_id": state["query_id"]}
        )
        return "retriever"

    def _route_after_grader(self, state: AgentState) -> str:
        """
        Determine if we need more retrieval or can synthesize.

        Args:
            state: Current agent state

        Returns:
            Next node name
        """
        if state.get("error"):
            logger.info(
                "Grader error, routing to error handler",
                extra={"query_id": state["query_id"]}
            )
            return "error"

        # Check if we should retry retrieval
        retry_count = state.get("retrieval_retries", 0)
        needs_more = state.get("needs_more_retrieval", False)

        if needs_more and retry_count < 2:
            logger.info(
                f"Retry retrieval (attempt {retry_count + 1})",
                extra={"query_id": state["query_id"]}
            )
            # Increment retry counter
            state["retrieval_retries"] = retry_count + 1
            return "retriever"

        # Otherwise, proceed to synthesis
        logger.info(
            "Proceeding to synthesis",
            extra={
                "query_id": state["query_id"],
                "retries": retry_count
            }
        )
        return "synthesizer"

    # ========================================================================
    # Error Handler
    # ========================================================================

    async def _error_handler(self, state: AgentState) -> AgentState:
        """
        Handle errors gracefully.

        Args:
            state: Current agent state

        Returns:
            Updated state with error response
        """
        error = state.get("error", "Unknown error")
        query_id = state["query_id"]

        logger.error(
            "Error in pipeline, generating graceful response",
            extra={
                "query_id": query_id,
                "error": error,
                "current_step": state.get("current_step", "unknown")
            }
        )

        # Set error response
        state["final_answer"] = (
            "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. "
            "Vui lòng thử lại hoặc liên hệ với bộ phận hỗ trợ để được trợ giúp."
        )
        state["current_step"] = "error"
        state["should_escalate"] = True

        return state

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    async def process_query(
        self,
        query: str,
        thread_id: Optional[str] = None,
        conversation_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.

        This is the main entry point for the RAG system.

        Args:
            query: User's question
            thread_id: Optional thread ID for conversation continuity
            conversation_history: Optional list of previous conversation messages

        Returns:
            Dictionary containing:
                - query_id: Unique identifier for this query
                - answer: The final answer
                - citations: List of citation dictionaries
                - verification: Verification results
                - metadata: Pipeline metadata
                - processing_time_ms: Processing time in milliseconds
                - error: Error message if any
        """
        start_time = datetime.utcnow()

        # Create initial state
        state = create_initial_state(query, conversation_history)

        logger.info(
            "Starting query processing",
            extra={
                "query_id": state["query_id"],
                "query": query[:100],
                "thread_id": thread_id
            }
        )

        try:
            # Run the graph
            config = {"configurable": {"thread_id": thread_id or state["query_id"]}}

            final_state = await self.app.ainvoke(state, config)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Build response
            response = self._build_response(final_state, processing_time)

            logger.info(
                "Query processing complete",
                extra={
                    "query_id": state["query_id"],
                    "processing_time_ms": processing_time,
                    "chunks_retrieved": len(final_state.get("retrieved_chunks", [])),
                    "is_grounded": final_state.get("is_grounded"),
                    "current_step": final_state.get("current_step")
                }
            )

            return response

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.error(
                f"Pipeline failed: {e}",
                extra={
                    "query_id": state["query_id"],
                    "processing_time_ms": processing_time
                },
                exc_info=True
            )

            # Return error response
            return {
                "query_id": state["query_id"],
                "answer": (
                    "Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi của bạn. "
                    "Vui lòng thử lại sau."
                ),
                "citations": [],
                "verification": {
                    "is_grounded": False,
                    "tier": None,
                    "confidence": 0.0,
                    "unsupported_claims": []
                },
                "metadata": {
                    "query_type": None,
                    "complexity": None,
                    "chunks_retrieved": 0,
                    "should_escalate": True,
                    "error": str(e)
                },
                "processing_time_ms": processing_time,
                "error": str(e)
            }

    def _build_response(self, state: AgentState, processing_time: float) -> Dict[str, Any]:
        """
        Build response dictionary from final state.

        Args:
            state: Final agent state
            processing_time: Processing time in milliseconds

        Returns:
            Response dictionary
        """
        return {
            "query_id": state["query_id"],
            "answer": state.get("final_answer") or state.get("draft_answer") or "Không có câu trả lời.",
            "citations": [
                {
                    "section_id": c.get("section_id"),
                    "document_id": c.get("document_id"),
                    "document_title": c.get("document_title"),
                    "section_heading": c.get("section_heading"),
                    "content_snippet": c.get("content_snippet", "")[:200],
                    "relevance_score": c.get("relevance_score", 0.0)
                }
                for c in state.get("citations", [])
            ],
            "verification": {
                "is_grounded": state.get("is_grounded", False),
                "tier": state.get("verification_tier"),
                "confidence": (
                    sum(state.get("retrieval_scores", [0])) / len(state.get("retrieval_scores", [1]))
                    if state.get("retrieval_scores")
                    else 0.0
                ),
                "unsupported_claims": state.get("unsupported_claims", [])
            },
            "metadata": {
                "query_type": state.get("query_type"),
                "complexity": state.get("complexity"),
                "chunks_retrieved": len(state.get("retrieved_chunks", [])),
                "should_escalate": state.get("should_escalate", False),
                "retrieval_retries": state.get("retrieval_retries", 0)
            },
            "processing_time_ms": processing_time
        }

    # ========================================================================
    # Sync Wrapper
    # ========================================================================

    def process_query_sync(
        self,
        query: str,
        thread_id: Optional[str] = None,
        conversation_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_query.

        Args:
            query: User's question
            thread_id: Optional thread ID for conversation continuity
            conversation_history: Optional list of previous conversation messages

        Returns:
            Response dictionary
        """
        return asyncio.run(self.process_query(query, thread_id, conversation_history))


# =============================================================================
# Singleton Instance
# =============================================================================

orchestrator = RAGOrchestrator(enable_tree_navigation=True)
