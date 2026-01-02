# =============================================================================
# Retriever Agent - Executes retrieval based on plan
# =============================================================================
from typing import Dict, Any, List

from src.agents.state import AgentState, update_state_step, merge_retrieval_results
from src.tools.hybrid_search import hybrid_search_tool
from src.tools.tree_navigator import tree_navigator_tool
from src.tools.section_retriever import section_retriever_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RetrieverAgent:
    """
    Executes retrieval based on plan.

    Responsibilities:
    - Execute hybrid search (vector + keyword)
    - Navigate document trees for context expansion
    - Aggregate results from multiple searches
    - Handle retrieval failures gracefully
    """

    def __init__(self, enable_tree_navigation: bool = True):
        """
        Initialize RetrieverAgent.

        Args:
            enable_tree_navigation: Whether to use tree navigation for context
        """
        self.enable_tree_navigation = enable_tree_navigation

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Execute retrieval based on execution plan.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved chunks
        """
        logger.info(
            "Retriever executing search",
            extra={
                "query_id": state["query_id"],
                "current_chunks": len(state["retrieved_chunks"])
            }
        )

        # Get query to use
        query = state["rewritten_query"] or state["original_query"]

        # Get execution plan or create default
        plan = state.get("execution_plan")
        if not plan:
            plan = [{
                "step": 1,
                "query": query,
                "collection": "chunks",
                "top_k": 20,
                "filters": {}
            }]
            state["execution_plan"] = plan

        try:
            # ================================================================
            # Execute each step in the plan
            # ================================================================
            for step in plan:
                step_query = step.get("query", query)
                collection = step.get("collection", "chunks")
                top_k = step.get("top_k", 20)
                filters = step.get("filters")

                logger.info(
                    f"Executing step {step.get('step', 1)}",
                    extra={
                        "query_id": state["query_id"],
                        "collection": collection,
                        "top_k": top_k
                    }
                )

                # Execute hybrid search
                search_result = await hybrid_search_tool.ainvoke({
                    "query": step_query,
                    "collection": collection,
                    "top_k": top_k,
                    "filters": filters if filters else None,
                    "use_cache": True
                })

                # Convert results to dict format
                new_chunks = []
                new_scores = []
                for result in search_result.results:
                    chunk_dict = {
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "document_id": str(result.document_id) if result.document_id else None,
                        "section_id": str(result.section_id) if result.section_id else None,
                        "score": result.score,
                        "metadata": {
                            "document_title": result.document_title,
                            "section_path": result.section_path,
                            "step": step.get("step", 1)
                        }
                    }
                    new_chunks.append(chunk_dict)
                    new_scores.append(result.score)

                # Merge results
                state = merge_retrieval_results(state, new_chunks, new_scores)

                logger.info(
                    f"Step {step.get('step', 1)} complete",
                    extra={
                        "query_id": state["query_id"],
                        "results": len(new_chunks),
                        "total_chunks": len(state["retrieved_chunks"])
                    }
                )

                # ================================================================
                # Optional: Tree navigation for context expansion
                # ================================================================
                if self.enable_tree_navigation and collection == "chunks" and search_result.results:
                    await self._expand_with_tree_context(
                        state,
                        search_result.results[:3]  # Top 3 results
                    )

            # ================================================================
            # Post-processing: Sort and limit
            # ================================================================
            if state["retrieved_chunks"]:
                # Sort by score descending
                sorted_pairs = sorted(
                    zip(state["retrieved_chunks"], state["retrieval_scores"]),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Keep top 30 chunks
                top_chunks = sorted_pairs[:30]

                state["retrieved_chunks"] = [c for c, s in top_chunks]
                state["retrieval_scores"] = [s for c, s in top_chunks]

            state = update_state_step(state, "retrieved")

            logger.info(
                "Retrieval complete",
                extra={
                    "query_id": state["query_id"],
                    "chunks_found": len(state["retrieved_chunks"]),
                    "avg_score": sum(state["retrieval_scores"]) / len(state["retrieval_scores"])
                    if state["retrieval_scores"] else 0
                }
            )

        except Exception as e:
            logger.error(
                f"Retriever error: {e}",
                extra={"query_id": state["query_id"]},
                exc_info=True
            )
            state["error"] = str(e)
            state = update_state_step(state, "error", error=str(e))

        return state

    async def _expand_with_tree_context(
        self,
        state: AgentState,
        top_results: List[Any]
    ) -> None:
        """
        Expand context using tree navigation.

        Args:
            state: Current agent state (modified in place)
            top_results: Top search results to expand
        """
        expansion_count = 0
        max_expansions = 3  # Limit expansions to avoid too many calls

        for result in top_results:
            if expansion_count >= max_expansions:
                break

            if not result.section_id:
                continue

            try:
                # Navigate to sibling sections for additional context
                nav_result = await tree_navigator_tool.ainvoke({
                    "section_id": str(result.section_id),
                    "direction": "siblings",
                    "max_depth": 1,
                    "include_content": False
                })

                # Get content from a few sibling sections
                for sibling in nav_result.related_sections[:2]:
                    if expansion_count >= max_expansions:
                        break

                    try:
                        section_content = await section_retriever_tool.ainvoke({
                            "section_id": str(sibling.section_id),
                            "max_tokens": 500,
                            "include_subsections": False
                        })

                        # Add as additional context with discounted score
                        if section_content.content:
                            chunk_dict = {
                                "chunk_id": sibling.section_id,
                                "content": section_content.content,
                                "document_id": str(sibling.document_id) if sibling.document_id else None,
                                "section_id": str(sibling.section_id),
                                "score": result.score * 0.75,  # Discount for context
                                "metadata": {
                                    "document_title": sibling.document_title,
                                    "section_path": sibling.section_path,
                                    "source": "tree_navigation"
                                }
                            }

                            state["retrieved_chunks"].append(chunk_dict)
                            state["retrieval_scores"].append(result.score * 0.75)
                            expansion_count += 1

                    except Exception as e:
                        logger.debug(
                            f"Failed to retrieve sibling section: {e}",
                            extra={"query_id": state["query_id"]}
                        )
                        continue

            except Exception as e:
                logger.debug(
                    f"Tree navigation failed for section {result.section_id}: {e}",
                    extra={"query_id": state["query_id"]}
                )
                continue

        if expansion_count > 0:
            logger.info(
                "Context expanded with tree navigation",
                extra={
                    "query_id": state["query_id"],
                    "expansions": expansion_count
                }
            )


# Singleton instance
retriever_agent = RetrieverAgent(enable_tree_navigation=True)
