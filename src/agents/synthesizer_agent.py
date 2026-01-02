# =============================================================================
# Synthesizer Agent - Synthesizes final answer from retrieved content
# =============================================================================
from typing import Dict, Any, List

from src.agents.state import AgentState, update_state_step
from src.tools.synthesize_answer import synthesize_answer_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SynthesizerAgent:
    """
    Synthesizes final answer from retrieved content.

    Responsibilities:
    - Generate coherent answer from chunks
    - Include proper citations
    - Format response appropriately based on query type
    - Handle insufficient information gracefully
    """

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Synthesize answer from retrieved chunks.

        Args:
            state: Current agent state

        Returns:
            Updated state with draft_answer and citations
        """
        logger.info(
            "Synthesizer generating answer",
            extra={
                "query_id": state["query_id"],
                "chunk_count": len(state["retrieved_chunks"])
            }
        )

        # Handle no chunks case
        if not state["retrieved_chunks"]:
            state["draft_answer"] = (
                "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn "
                "trong cơ sở kiến thức hiện tại. Vui lòng thử đặt câu hỏi khác hoặc "
                "liên hệ với bộ phận hỗ trợ để được trợ giúp."
            )
            state["citations"] = []
            state = update_state_step(state, "synthesized")

            logger.warning(
                "No chunks available for synthesis",
                extra={"query_id": state["query_id"]}
            )
            return state

        try:
            # ================================================================
            # Prepare sources for synthesis
            # ================================================================
            sources = []
            for chunk in state["retrieved_chunks"][:10]:  # Top 10 chunks
                sources.append({
                    "content": chunk.get("content", ""),
                    "metadata": {
                        "document_id": chunk.get("document_id", ""),
                        "document_title": chunk.get("metadata", {}).get("document_title", "Unknown"),
                        "section_id": chunk.get("section_id", ""),
                        "section_heading": chunk.get("metadata", {}).get("section_heading", ""),
                        "section_path": chunk.get("metadata", {}).get("section_path", ""),
                        "score": chunk.get("score", 0.0)
                    }
                })

            # ================================================================
            # Determine response format based on query type
            # ================================================================
            query_type = state.get("query_type", "factual")
            complexity = state.get("complexity", "simple")

            if query_type == "procedural":
                response_format = "step_by_step"
            elif complexity == "complex" or state.get("is_decomposed"):
                response_format = "detailed"
            elif query_type in ["comparative", "aggregation"]:
                response_format = "structured"
            else:
                response_format = "concise"

            # ================================================================
            # Generate answer
            # ================================================================
            result = await synthesize_answer_tool.ainvoke({
                "query": state["original_query"],
                "sources": sources,
                "max_citations": 5,
                "language": "vi"
            })

            state["draft_answer"] = result.answer
            state["citations"] = [
                {
                    "section_id": c.section_id,
                    "document_id": c.document_id,
                    "document_title": c.document_title,
                    "section_heading": c.section_heading,
                    "content_snippet": c.content_snippet,
                    "relevance_score": c.relevance_score
                }
                for c in result.citations
            ]

            state = update_state_step(state, "synthesized")

            logger.info(
                "Answer synthesized",
                extra={
                    "query_id": state["query_id"],
                    "answer_length": len(result.answer),
                    "citation_count": len(result.citations),
                    "confidence": result.confidence,
                    "response_format": response_format
                }
            )

            # Log reasoning if available
            if result.reasoning:
                logger.debug(
                    f"Synthesis reasoning: {result.reasoning}",
                    extra={"query_id": state["query_id"]}
                )

        except Exception as e:
            logger.error(
                f"Synthesizer error: {e}",
                extra={"query_id": state["query_id"]},
                exc_info=True
            )
            # Graceful degradation
            state["draft_answer"] = (
                "Xin lỗi, tôi gặp khó khăn khi tạo câu trả lời. "
                "Tuy nhiên, tôi đã tìm thấy một số thông tin liên quan. "
                "Vui lòng thử lại hoặc liên hệ hỗ trợ."
            )
            state["citations"] = []
            state = update_state_step(state, "synthesized")

        return state

    def _format_answer_with_freshness_warning(
        self,
        answer: str,
        has_stale: bool,
        freshness_info: str
    ) -> str:
        """
        Add freshness warning to answer if needed.

        Args:
            answer: The generated answer
            has_stale: Whether stale documents were used
            freshness_info: Freshness information string

        Returns:
            Answer with potential warning appended
        """
        if has_stale and freshness_info:
            return f"{answer}\n\n⚠️ {freshness_info}"
        return answer


# Singleton instance
synthesizer_agent = SynthesizerAgent()
