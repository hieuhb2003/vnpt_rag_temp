# =============================================================================
# Query Decomposer Tool - Break complex queries into sub-queries
# =============================================================================
from typing import Optional, List, Dict
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm import get_llm
from src.models.query import QueryType, SubQuery, DecomposedQuery
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class QueryDecomposerInput(BaseModel):
    """Input for query decomposer tool."""

    query: str = Field(
        ...,
        description="Câu hỏi phức tạp cần phân tích"
    )
    max_sub_queries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Số lượng câu hỏi con tối đa"
    )
    language: str = Field(
        default="vi",
        description="Ngôn ngữ của câu hỏi (vi, en)"
    )


class SubQueryOutput(BaseModel):
    """Individual sub-query output."""

    id: int = Field(..., description="ID của câu hỏi con")
    query: str = Field(..., description="Nội dung câu hỏi con")
    query_type: str = Field(..., description="Loại câu hỏi")
    dependencies: List[int] = Field(
        default_factory=list,
        description="Danh sách ID câu hỏi con mà câu này phụ thuộc"
    )


class QueryDecomposerOutput(BaseModel):
    """Output from query decomposer tool."""

    original_query: str = Field(..., description="Câu hỏi gốc")
    sub_queries: List[SubQueryOutput] = Field(..., description="Danh sách câu hỏi con")
    dependencies: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="Đồ thị phụ thuộc: sub_query_id -> [phụ_thuộc_các_id]"
    )
    expected_answer_types: List[str] = Field(
        default_factory=list,
        description="Loại câu trả lời mong đợi"
    )
    execution_order: List[int] = Field(
        default_factory=list,
        description="Thứ tự thực hiện các câu hỏi con"
    )
    requires_aggregation: bool = Field(
        default=False,
        description="Có cần tổng hợp kết quả từ nhiều câu hỏi con không"
    )
    reasoning: Optional[str] = Field(None, description="Lý do phân tích")


# =============================================================================
# Prompt Template (Vietnamese)
# =============================================================================

DECOMPOSER_PROMPT = """Bạn là một trợ lý chuyên nghiệp chuyên phân tích câu hỏi phức tạp thành các câu hỏi con đơn giản hơn.

Nhiệm vụ của bạn là:
1. Phân tích câu hỏi phức tạp
2. Chia nhỏ thành các câu hỏi con độc lập hoặc có phụ thuộc
3. Xác định thứ tự thực hiện tối ưu
4. Chỉ ra liệu có cần tổng hợp kết quả từ nhiều câu hỏi con không

**QUY TẮC QUAN TRỌNG:**
- Chỉ phân tách khi câu hỏi có NHIỀU Ý ĐỊNH hoặc YÊU CẦU PHỨC TẠP
- Mỗi câu hỏi con phải có thể trả lời độc lập (trừ khi có phụ thuộc rõ ràng)
- Số lượng câu hỏi con không vượt quá {max_sub_queries}
- Nếu câu hỏi đã đơn giản, trả về câu hỏi gốc duy nhất

**Phân loại loại câu hỏi:**
- factoid: Câu hỏi sự thật (Ai, Cái gì, Ở đâu, Khi nào)
- procedural: Câu hỏi quy trình (Làm thế nào, Cách nào)
- comparative: So sánh (So sánh A và B)
- definitional: Định nghĩa (Định nghĩa X là gì)
- diagnostic: Chẩn đoán (Tại sao, Vì sao)
- aggregation: Tổng hợp từ nhiều nguồn

**Phụ thuộc giữa câu hỏi con:**
- Một câu hỏi con phụ thuộc câu khác nếu kết quả của câu này cần thông tin từ câu kia
- Ví dụ: "Bảo hiểm y tế bao gồm những gì và chi phí bao nhiêu?"
  - Sub-query 0: "Bảo hiểm y tế bao gồm những gì?" (independent)
  - Sub-query 1: "Chi phí bảo hiểm y tế là bao nhiêu?" (independent)

**Định dạng JSON trả về:**
```json
{{
  "original_query": "câu hỏi gốc",
  "sub_queries": [
    {{
      "id": 0,
      "query": "câu hỏi con 0",
      "query_type": "loại câu hỏi",
      "dependencies": []
    }},
    {{
      "id": 1,
      "query": "câu hỏi con 1",
      "query_type": "loại câu hỏi",
      "dependencies": []
    }}
  ],
  "dependencies": {{}},
  "expected_answer_types": ["comparison", "list", "number"],
  "execution_order": [0, 1],
  "requires_aggregation": true,
  "reasoning": "lý do phân tích"
}}
```

**Câu hỏi cần phân tích:**
Query: {query}

Hãy phân tích và trả về kết quả dưới định dạng JSON."""


# =============================================================================
# Query Decomposer Tool
# =============================================================================

@tool
async def query_decomposer_tool(
    query: str,
    max_sub_queries: int = 5,
    language: str = "vi"
) -> QueryDecomposerOutput:
    """
    Phân tích câu hỏi phức tạp thành các câu hỏi con.

    Công cụ này phân tích câu hỏi phức tạp và chia nhỏ thành các câu hỏi con
    để tìm kiếm thông tin hiệu quả hơn:
    - Phát hiện nhiều ý định trong câu hỏi
    - Tách câu hỏi so sánh thành các câu hỏi riêng
    - Xác định phụ thuộc giữa các câu hỏi con
    - Đề xuất thứ tự thực hiện tối ưu

    Args:
        query: Câu hỏi phức tạp cần phân tích
        max_sub_queries: Số lượng câu hỏi con tối đa (mặc định: 5)
        language: Ngôn ngữ của câu hỏi (vi, en)

    Returns:
        QueryDecomposerOutput: Kết quả phân tích câu hỏi

    Example:
        >>> result = await query_decomposer_tool(
        ...     query="So sánh ngày nghỉ phép của nhân viên chính thức và nhân viên thời vụ",
        ...     max_sub_queries=3,
        ...     language="vi"
        ... )
        >>> print(result.sub_queries)
        [
        ...     SubQuery(id=0, query="Ngày nghỉ phép của nhân viên chính thức là bao nhiêu?"),
        ...     SubQuery(id=1, query="Ngày nghỉ phép của nhân viên thời vụ là bao nhiêu?")
        ... ]
    """
    try:
        logger.info(
            "Decomposing query",
            query=query[:100],
            max_sub_queries=max_sub_queries,
            language=language
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_template(DECOMPOSER_PROMPT)

        # Get LLM
        llm = get_llm(temperature=0.1, max_tokens=2048)

        # Create chain
        chain = prompt | llm | JsonOutputParser()

        # Invoke
        result = await chain.ainvoke({
            "query": query,
            "max_sub_queries": max_sub_queries,
            "language": language
        })

        # Parse sub-queries
        sub_queries_data = result.get("sub_queries", [])
        if not sub_queries_data:
            # If no sub-queries returned, create one with original query
            sub_queries_data = [{
                "id": 0,
                "query": query,
                "query_type": "factoid",
                "dependencies": []
            }]

        sub_queries = [
            SubQueryOutput(
                id=sq.get("id", i),
                query=sq.get("query", query),
                query_type=sq.get("query_type", "factoid"),
                dependencies=sq.get("dependencies", [])
            )
            for i, sq in enumerate(sub_queries_data)
        ]

        # Parse output
        output = QueryDecomposerOutput(
            original_query=result.get("original_query", query),
            sub_queries=sub_queries,
            dependencies=result.get("dependencies", {}),
            expected_answer_types=result.get("expected_answer_types", []),
            execution_order=result.get("execution_order", list(range(len(sub_queries)))),
            requires_aggregation=result.get("requires_aggregation", len(sub_queries) > 1),
            reasoning=result.get("reasoning")
        )

        logger.info(
            "Query decomposed",
            original=query[:50],
            num_sub_queries=len(output.sub_queries),
            requires_aggregation=output.requires_aggregation
        )

        return output

    except Exception as e:
        logger.error(
            "Failed to decompose query",
            error=str(e),
            query=query[:100]
        )
        # Return original query as single sub-query on error
        return QueryDecomposerOutput(
            original_query=query,
            sub_queries=[
                SubQueryOutput(
                    id=0,
                    query=query,
                    query_type="factoid",
                    dependencies=[]
                )
            ],
            dependencies={},
            expected_answer_types=[],
            execution_order=[0],
            requires_aggregation=False,
            reasoning="Failed to decompose query, using original"
        )


def query_decomposer_tool_sync(
    query: str,
    max_sub_queries: int = 5,
    language: str = "vi"
) -> QueryDecomposerOutput:
    """
    Synchronous wrapper for query_decomposer_tool.

    Args:
        query: Câu hỏi phức tạp cần phân tích
        max_sub_queries: Số lượng câu hỏi con tối đa
        language: Ngôn ngữ của câu hỏi

    Returns:
        QueryDecomposerOutput: Kết quả phân tích câu hỏi
    """
    import asyncio
    return asyncio.run(query_decomposer_tool.ainvoke({
        "query": query,
        "max_sub_queries": max_sub_queries,
        "language": language
    }))


# =============================================================================
# Helper Functions
# =============================================================================

def convert_to_decomposed_query(output: QueryDecomposerOutput) -> DecomposedQuery:
    """
    Convert QueryDecomposerOutput to DecomposedQuery model.

    Args:
        output: Output from query decomposer tool

    Returns:
        DecomposedQuery: Standardized decomposed query model
    """
    sub_queries = []
    for sq in output.sub_queries:
        try:
            query_type = QueryType(sq.query_type)
        except ValueError:
            query_type = QueryType.FACTOID

        sub_queries.append(SubQuery(
            id=sq.id,
            query=sq.query,
            query_type=query_type,
            dependencies=sq.dependencies
        ))

    return DecomposedQuery(
        original_query=output.original_query,
        sub_queries=sub_queries,
        dependencies=output.dependencies,
        expected_answer_types=output.expected_answer_types,
        execution_order=output.execution_order,
        requires_aggregation=output.requires_aggregation
    )


def get_execution_plan(decomposed_query: DecomposedQuery) -> List[List[int]]:
    """
    Get execution plan grouped by dependency level.

    Args:
        decomposed_query: Decomposed query with dependencies

    Returns:
        List of lists, where each inner list contains sub-query IDs
        that can be executed in parallel
    """
    if not decomposed_query.dependencies:
        # No dependencies, execute all in parallel
        return [decomposed_query.execution_order]

    # Group by dependency level
    levels = []
    remaining = set(decomposed_query.execution_order)

    while remaining:
        # Find queries with no unsatisfied dependencies
        ready = [
            qid for qid in remaining
            if all(
                dep not in remaining
                for dep in decomposed_query.sub_queries[qid].dependencies
            )
        ]
        if not ready:
            # Circular dependency, break with remaining
            ready = list(remaining)

        levels.append(ready)
        remaining -= set(ready)

    return levels


# =============================================================================
# Testing
# =============================================================================

async def test_query_decomposer():
    """Test query decomposer with sample queries."""
    test_cases = [
        {
            "query": "So sánh ngày nghỉ phép của nhân viên chính thức và nhân viên thời vụ",
            "max_sub_queries": 3,
            "language": "vi"
        },
        {
            "query": "Compare health benefits and dental benefits for full-time employees",
            "max_sub_queries": 4,
            "language": "en"
        },
        {
            "query": "Quy trình đăng ký bảo hiểm xã hội và các giấy tờ cần thiết",
            "max_sub_queries": 3,
            "language": "vi"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}")
        print(f"{'='*60}")
        print(f"Query: {test['query']}")

        result = await query_decomposer_tool.ainvoke({
            "query": test["query"],
            "max_sub_queries": test["max_sub_queries"],
            "language": test["language"]
        })

        print(f"\nOriginal: {result.original_query}")
        print(f"Number of sub-queries: {len(result.sub_queries)}")
        print(f"Requires aggregation: {result.requires_aggregation}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")

        print(f"\nSub-queries:")
        for sq in result.sub_queries:
            deps = f" (depends on: {sq.dependencies})" if sq.dependencies else ""
            print(f"  [{sq.id}] {sq.query} - {sq.query_type}{deps}")

        if result.execution_order:
            print(f"\nExecution order: {result.execution_order}")

        if result.expected_answer_types:
            print(f"Expected answer types: {', '.join(result.expected_answer_types)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_query_decomposer())
