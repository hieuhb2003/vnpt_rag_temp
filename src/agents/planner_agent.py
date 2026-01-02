# =============================================================================
# Planner Agent - Creates execution plan for complex queries
# =============================================================================
import json
from typing import Dict, Any, List

from src.agents.state import AgentState, update_state_step
from src.utils.llm import get_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Planning Prompt (Vietnamese)
# =============================================================================

PLANNING_PROMPT = """Bạn là chuyên gia lập kế hoạch tìm kiếm thông tin.

**Câu hỏi gốc:**
{query}

**Loại câu hỏi:**
{query_type}

**Các câu hỏi con đã xác định:**
{sub_queries}

**Nhiệm vụ của bạn:**
Hãy tạo kế hoạch thực thi tối ưu để tìm kiếm thông tin cho từng câu hỏi con.

**Nguyên tắc lập kế hoạch:**
1. Xác định thứ tự thực hiện các câu hỏi con (tùy thuộc vào dependencies)
2. Chọn collection phù hợp: documents (tổng quan), sections (mục lục), chunks (chi tiết)
3. Đặt top_k phù hợp (5-20 cho documents, 10-30 cho sections, 20-50 cho chunks)
4. Thêm filters nếu cần (document_type, date_range, etc.)

**Định dạng JSON trả về:**
```json
{{
    "execution_plan": [
        {{
            "step": 1,
            "sub_query_index": 0,
            "sub_query": "Câu hỏi con cụ thể",
            "collection": "chunks",
            "top_k": 20,
            "filters": {{}},
            "reasoning": "Lý do chọn collection và top_k"
        }}
    ],
    "overall_strategy": "Giải thích chiến lược tổng thể"
}}
```

Hãy phân tích và trả về kết quả."""


class PlannerAgent:
    """
    Creates execution plan for complex queries.

    Responsibilities:
    - Analyze query dependencies
    - Determine search strategy per sub-query
    - Set retrieval parameters
    - Order execution steps
    """

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Create execution plan based on query complexity.

        Args:
            state: Current agent state

        Returns:
            Updated state with execution plan
        """
        logger.info(
            "Planner creating execution plan",
            extra={"query_id": state["query_id"]}
        )

        try:
            # ================================================================
            # Case 1: Simple/Non-decomposed query - create default plan
            # ================================================================
            if not state["is_decomposed"]:
                state["execution_plan"] = [{
                    "step": 1,
                    "query": state["rewritten_query"] or state["original_query"],
                    "collection": "chunks",
                    "top_k": 20,
                    "filters": {},
                    "reasoning": "Default plan for simple query"
                }]
                state = update_state_step(state, "planned")

                logger.info(
                    "Default execution plan created",
                    extra={
                        "query_id": state["query_id"],
                        "steps": len(state["execution_plan"])
                    }
                )
                return state

            # ================================================================
            # Case 2: Complex decomposed query - use LLM to create plan
            # ================================================================
            llm = get_llm(temperature=0.1, max_tokens=2048)

            # Format sub-queries for prompt
            sub_queries_str = "\n".join([
                f"- [{i}] {sq['query']}"
                f" (type: {sq.get('query_type', 'unknown')})"
                f" (depends on: {sq.get('dependencies', [])})"
                for i, sq in enumerate(state["sub_queries"])
            ])

            # Create prompt
            prompt = PLANNING_PROMPT.format(
                query=state["original_query"],
                query_type=state.get("query_type", "unknown"),
                sub_queries=sub_queries_str
            )

            # Get LLM response
            response = await llm.ainvoke(prompt)
            response_text = response.content

            # Parse JSON response
            try:
                # Try to extract JSON from response
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

                plan_data = json.loads(json_text)

                # Validate and format plan
                execution_plan = []
                for step_data in plan_data.get("execution_plan", []):
                    execution_plan.append({
                        "step": step_data.get("step", 1),
                        "sub_query_index": step_data.get("sub_query_index", 0),
                        "query": step_data.get("sub_query", state["rewritten_query"]),
                        "collection": step_data.get("collection", "chunks"),
                        "top_k": step_data.get("top_k", 20),
                        "filters": step_data.get("filters", {}),
                        "reasoning": step_data.get("reasoning", "")
                    })

                state["execution_plan"] = execution_plan

                logger.info(
                    "LLM execution plan created",
                    extra={
                        "query_id": state["query_id"],
                        "steps": len(execution_plan),
                        "strategy": plan_data.get("overall_strategy", "")[:100]
                    }
                )

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse LLM plan, using fallback: {e}",
                    extra={"query_id": state["query_id"]}
                )
                # Fallback to simple sequential plan
                state["execution_plan"] = self._create_fallback_plan(state)
            except Exception as e:
                logger.warning(
                    f"Error creating LLM plan, using fallback: {e}",
                    extra={"query_id": state["query_id"]}
                )
                state["execution_plan"] = self._create_fallback_plan(state)

            state = update_state_step(state, "planned")

        except Exception as e:
            logger.error(
                f"Planner error: {e}",
                extra={"query_id": state["query_id"]},
                exc_info=True
            )
            # Fallback to simple plan on error
            state["execution_plan"] = self._create_fallback_plan(state)
            state = update_state_step(state, "planned")

        return state

    def _create_fallback_plan(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Create a simple fallback execution plan.

        Args:
            state: Current agent state

        Returns:
            Simple execution plan
        """
        if state["is_decomposed"] and state["sub_queries"]:
            # Create sequential plan for each sub-query
            return [
                {
                    "step": i + 1,
                    "sub_query_index": i,
                    "query": sq["query"],
                    "collection": "chunks",
                    "top_k": 15,
                    "filters": {},
                    "reasoning": f"Plan for sub-query {i + 1}"
                }
                for i, sq in enumerate(state["sub_queries"])
            ]
        else:
            # Single step plan
            return [{
                "step": 1,
                "query": state["rewritten_query"] or state["original_query"],
                "collection": "chunks",
                "top_k": 20,
                "filters": {},
                "reasoning": "Default fallback plan"
            }]


# Singleton instance
planner_agent = PlannerAgent()
