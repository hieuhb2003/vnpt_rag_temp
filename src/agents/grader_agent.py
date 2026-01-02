# =============================================================================
# Grader Agent - Grades relevance of retrieved chunks
# =============================================================================
import json
from typing import Dict, Any

from src.agents.state import AgentState, update_state_step
from src.utils.llm import get_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Grading Prompt (Vietnamese)
# =============================================================================

GRADING_PROMPT = """Đánh giá mức độ liên quan của đoạn văn với câu hỏi.

**Câu hỏi:**
{query}

**Đoạn văn {index}:**
```
{content}
```

**Tiêu chí đánh giá:**
- **0-3 điểm**: Không liên quan - Không chứa thông tin liên quan đến câu hỏi
- **4-6 điểm**: Có liên quan một phần - Chứa một số thông tin nhưng không đầy đủ
- **7-9 điểm**: Rất liên quan - Chứa thông tin tốt để trả lời câu hỏi
- **10 điểm**: Trả lời trực tiếp - Trực tiếp trả lời câu hỏi một cách đầy đủ

**Định dạng JSON trả về:**
```json
{{
    "score": 7,
    "reason": "Lý do cho điểm số"
}}
```

Hãy đánh giá và trả về kết quả."""


class GraderAgent:
    """
    Grades relevance of retrieved chunks.

    Responsibilities:
    - Score each chunk's relevance to query (0-10 scale)
    - Identify if more retrieval is needed
    - Filter out irrelevant chunks
    - Determine if escalation is needed
    """

    def __init__(self, grade_limit: int = 10, relevance_threshold: float = 4.0):
        """
        Initialize GraderAgent.

        Args:
            grade_limit: Maximum number of chunks to grade (to avoid too many LLM calls)
            relevance_threshold: Minimum score to keep chunk (0-10 scale)
        """
        self.grade_limit = grade_limit
        self.relevance_threshold = relevance_threshold

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Grade retrieved chunks and determine if more retrieval is needed.

        Args:
            state: Current agent state

        Returns:
            Updated state with relevance scores and filtered chunks
        """
        logger.info(
            "Grader evaluating chunks",
            extra={
                "query_id": state["query_id"],
                "chunk_count": len(state["retrieved_chunks"])
            }
        )

        # Handle empty chunks
        if not state["retrieved_chunks"]:
            state["needs_more_retrieval"] = True
            state = update_state_step(state, "graded")

            logger.warning(
                "No chunks to grade",
                extra={"query_id": state["query_id"]}
            )
            return state

        try:
            # ================================================================
            # Grade top chunks (limit to avoid too many LLM calls)
            # ================================================================
            chunks_to_grade = state["retrieved_chunks"][:self.grade_limit]
            scores = []
            filtered_chunks = []

            query = state["rewritten_query"] or state["original_query"]

            for i, chunk in enumerate(chunks_to_grade):
                content = chunk.get("content", "")[:800]  # Limit content length

                try:
                    # Grade using LLM
                    score = await self._grade_chunk(query, content, i + 1)
                    scores.append(score)

                    # Keep chunks with score >= threshold
                    if score >= self.relevance_threshold:
                        chunk["relevance_score"] = score
                        filtered_chunks.append(chunk)

                    logger.debug(
                        f"Chunk {i + 1} graded: {score}/10",
                        extra={"query_id": state["query_id"]}
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to grade chunk {i + 1}: {e}",
                        extra={"query_id": state["query_id"]}
                    )
                    # Default middle score on error
                    scores.append(5.0)
                    if 5.0 >= self.relevance_threshold:
                        chunk["relevance_score"] = 5.0
                        filtered_chunks.append(chunk)

            # ================================================================
            # Update state with scores
            # ================================================================
            state["relevance_scores"] = scores
            state["retrieved_chunks"] = filtered_chunks

            # ================================================================
            # Determine if more retrieval is needed
            # ================================================================
            high_relevance_count = sum(1 for s in scores if s >= 7.0)

            # Need more retrieval if:
            # 1. Fewer than 2 high-relevance chunks AND fewer than 3 total chunks
            # 2. Average score is too low
            if scores:
                avg_score = sum(scores) / len(scores)
                if high_relevance_count < 2 and len(filtered_chunks) < 3:
                    state["needs_more_retrieval"] = True
                elif avg_score < 5.0:
                    state["needs_more_retrieval"] = True
                else:
                    state["needs_more_retrieval"] = False
            else:
                state["needs_more_retrieval"] = True

            state = update_state_step(state, "graded")

            logger.info(
                "Grading complete",
                extra={
                    "query_id": state["query_id"],
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "high_relevance": high_relevance_count,
                    "kept_chunks": len(filtered_chunks),
                    "needs_more_retrieval": state["needs_more_retrieval"]
                }
            )

        except Exception as e:
            logger.error(
                f"Grader error: {e}",
                extra={"query_id": state["query_id"]},
                exc_info=True
            )
            # Don't block on error - proceed with what we have
            state["needs_more_retrieval"] = False
            state = update_state_step(state, "graded")

        return state

    async def _grade_chunk(self, query: str, content: str, index: int) -> float:
        """
        Grade a single chunk's relevance to the query.

        Args:
            query: The search query
            content: Chunk content
            index: Chunk index for prompt

        Returns:
            Relevance score (0-10)
        """
        llm = get_llm(temperature=0.1, max_tokens=512)

        prompt = GRADING_PROMPT.format(
            query=query,
            index=index,
            content=content
        )

        response = await llm.ainvoke(prompt)
        response_text = response.content

        # Parse JSON response
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)
            score = float(result.get("score", 5.0))
            return max(0.0, min(10.0, score))  # Clamp to [0, 10]

        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning(f"Failed to parse grading response, using default score")
            return 5.0

    def should_retry_retrieval(self, state: AgentState) -> bool:
        """
        Check if we should try retrieval again.

        Args:
            state: Current agent state

        Returns:
            True if should retry, False otherwise
        """
        # Check retry count
        retry_count = state.get("retrieval_retries", 0)

        # Max 2 retries
        if state.get("needs_more_retrieval") and retry_count < 2:
            return True

        return False


# Singleton instance
grader_agent = GraderAgent(grade_limit=10, relevance_threshold=4.0)
