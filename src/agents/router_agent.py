# =============================================================================
# Router Agent - Routes queries based on complexity and type
# =============================================================================
from typing import Dict, Any

from src.agents.state import AgentState, QueryComplexity, update_state_step
from src.tools.query_rewriter import query_rewriter_tool
from src.tools.query_decomposer import query_decomposer_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RouterAgent:
    """
    Routes queries based on complexity and type.

    Responsibilities:
    - Rewrite query for better retrieval
    - Classify query complexity
    - Determine if decomposition is needed
    - Route to appropriate processing path
    """

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process query through routing logic.

        Args:
            state: Current agent state

        Returns:
            Updated state with routing decisions
        """
        logger.info(
            "Router processing query",
            extra={
                "query_id": state["query_id"],
                "query": state["original_query"][:100]
            }
        )

        try:
            # ================================================================
            # Step 1: Rewrite query for better retrieval
            # ================================================================
            rewrite_result = await query_rewriter_tool.ainvoke({
                "query": state["original_query"],
                "conversation_history": [],  # Could be passed in from conversation
                "language": "vi"
            })

            state["rewritten_query"] = rewrite_result.rewritten
            state["query_type"] = rewrite_result.query_type

            logger.info(
                "Query rewritten",
                extra={
                    "query_id": state["query_id"],
                    "original": state["original_query"][:50],
                    "rewritten": state["rewritten_query"][:50],
                    "query_type": state["query_type"]
                }
            )

            # ================================================================
            # Step 2: Check if decomposition is needed
            # ================================================================
            decompose_result = await query_decomposer_tool.ainvoke({
                "query": state["rewritten_query"],
                "max_sub_queries": 5,
                "language": "vi"
            })

            # Determine if decomposition is needed
            # Consider as "decomposed" if requires_aggregation OR has more than 1 sub_query
            has_multiple_sub_queries = len(decompose_result.sub_queries) > 1
            is_complex = decompose_result.requires_aggregation or has_multiple_sub_queries

            if is_complex:
                state["complexity"] = QueryComplexity.COMPLEX
                state["is_decomposed"] = True
                state["sub_queries"] = [
                    {
                        "id": sq.id,
                        "query": sq.query,
                        "query_type": sq.query_type,
                        "dependencies": sq.dependencies
                    }
                    for sq in decompose_result.sub_queries
                ]
                state["sub_query_results"] = {
                    sq.id: []
                    for sq in decompose_result.sub_queries
                }

                logger.info(
                    "Query decomposed",
                    extra={
                        "query_id": state["query_id"],
                        "num_sub_queries": len(decompose_result.sub_queries),
                        "requires_aggregation": decompose_result.requires_aggregation
                    }
                )

            elif state["query_type"] in ["comparative", "aggregation", "procedural"]:
                state["complexity"] = QueryComplexity.MODERATE
                state["is_decomposed"] = False

                logger.info(
                    "Query classified as moderate",
                    extra={
                        "query_id": state["query_id"],
                        "query_type": state["query_type"]
                    }
                )

            else:
                state["complexity"] = QueryComplexity.SIMPLE
                state["is_decomposed"] = False

                logger.info(
                    "Query classified as simple",
                    extra={
                        "query_id": state["query_id"],
                        "query_type": state["query_type"]
                    }
                )

            # ================================================================
            # Step 3: Update state and return
            # ================================================================
            state = update_state_step(state, "routed")

            logger.info(
                "Query routed successfully",
                extra={
                    "query_id": state["query_id"],
                    "complexity": state["complexity"],
                    "is_decomposed": state["is_decomposed"],
                    "current_step": state["current_step"]
                }
            )

        except Exception as e:
            logger.error(
                f"Router error: {e}",
                extra={"query_id": state["query_id"]},
                exc_info=True
            )
            state["error"] = str(e)
            state = update_state_step(state, "error", error=str(e))

        return state

    def get_next_step(self, state: AgentState) -> str:
        """
        Determine next step based on routing result.

        Args:
            state: Current agent state

        Returns:
            Next step name: "planner", "retriever", or "error_handler"
        """
        if state.get("error"):
            return "error_handler"

        if state["is_decomposed"]:
            return "planner"
        elif state["complexity"] == QueryComplexity.COMPLEX:
            return "planner"
        else:
            # Simple and moderate queries go directly to retrieval
            return "retriever"


# Singleton instance
router_agent = RouterAgent()
