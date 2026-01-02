# =============================================================================
# Query Rewriter Tool - Improve queries for better retrieval
# =============================================================================
from typing import Optional, List
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm import get_llm
from src.models.query import QueryType, RewrittenQuery
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class QueryRewriterInput(BaseModel):
    """Input for query rewriter tool."""

    query: str = Field(
        ...,
        description="Câu hỏi gốc của người dùng cần viết lại"
    )
    conversation_history: List[str] = Field(
        default_factory=list,
        description="Lịch sử hội thoại để hiểu ngữ cảnh"
    )
    language: str = Field(
        default="vi",
        description="Ngôn ngữ của câu hỏi (vi, en)"
    )


class QueryRewriterOutput(BaseModel):
    """Output from query rewriter tool."""

    original: str = Field(..., description="Câu hỏi gốc")
    rewritten: str = Field(..., description="Câu hỏi đã viết lại")
    keywords: List[str] = Field(..., description="Từ khóa quan trọng")
    query_type: str = Field(..., description="Loại câu hỏi")
    confidence: float = Field(..., description="Độ tự tin")
    reasoning: Optional[str] = Field(None, description="Lý do viết lại")
    expansions: List[str] = Field(..., description="Các câu hỏi mở rộng")


# =============================================================================
# Prompt Template (Vietnamese)
# =============================================================================

REWRITER_PROMPT = """Bạn là một trợ lý chuyên nghiệp chuyên cải thiện câu hỏi để tìm kiếm thông tin hiệu quả hơn.

Nhiệm vụ của bạn là phân tích và viết lại câu hỏi của người dùng để:
1. Làm rõ ý định của người dùng
2. Thêm ngữ cảnh còn thiếu
3. Sử dụng từ ngữ chuyên ngành phù hợp
4. Cải thiện khả năng tìm kiếm thông tin

**QUY TẮC QUAN TRỌNG:**
- Chỉ viết lại khi CẦN THIẾT để làm rõ hoặc cải thiện tìm kiếm
- Nếu câu hỏi đã rõ ràng, giữ nguyên với confidence cao
- Luôn giữ nguyên ý định gốc của người dùng
- Sử dụng ngôn ngữ {language} cho câu hỏi viết lại

**Phân loại loại câu hỏi:**
- factoid: Câu hỏi sự thật (Ai, Cái gì, Ở đâu, Khi nào)
- procedural: Câu hỏi quy trình (Làm thế nào, Cách nào)
- comparative: So sánh (So sánh A và B)
- definitional: Định nghĩa (Định nghĩa X là gì)
- diagnostic: Chẩn đoán (Tại sao, Vì sao)
- aggregation: Tổng hợp từ nhiều nguồn

**Định dạng JSON trả về:**
```json
{{
  "original": "câu hỏi gốc",
  "rewritten": "câu hỏi đã viết lại hoặc giữ nguyên nếu đã tốt",
  "keywords": ["từ khóa 1", "từ khóa 2"],
  "query_type": "loại câu hỏi",
  "confidence": 0.95,
  "reasoning": "lý do viết lại hoặc giữ nguyên",
  "expansions": ["câu hỏi mở rộng 1", "câu hỏi mở rộng 2"]
}}
```

**Câu hỏi cần xử lý:**
Query: {query}

{conversation_context}

Hãy phân tích và trả về kết quả dưới định dạng JSON."""


def format_conversation_context(history: List[str]) -> str:
    """Format conversation history for prompt."""
    if not history:
        return "**Ngữ cảnh hội thoại:** Không có lịch sử hội thoại trước đó."

    context = "**Ngữ cảnh hội thoại trước đó:**\n"
    for i, msg in enumerate(history[-5:], 1):  # Only last 5 messages
        context += f"{i}. {msg}\n"
    return context


# =============================================================================
# Query Rewriter Tool
# =============================================================================

@tool
async def query_rewriter_tool(
    query: str,
    conversation_history: Optional[List[str]] = None,
    language: str = "vi"
) -> QueryRewriterOutput:
    """
    Viết lại câu hỏi để cải thiện kết quả tìm kiếm.

    Công cụ này phân tích câu hỏi của người dùng và viết lại để:
    - Làm rõ ý định của người dùng
    - Thêm ngữ cảnh còn thiếu từ hội thoại
    - Sử dụng từ ngữ phù hợp để tìm kiếm tốt hơn
    - Trích xuất từ khóa quan trọng
    - Phân loại loại câu hỏi

    Args:
        query: Câu hỏi gốc của người dùng
        conversation_history: Lịch sử hội thoại để hiểu ngữ cảnh
        language: Ngôn ngữ của câu hỏi (vi, en)

    Returns:
        QueryRewriterOutput: Kết quả viết lại câu hỏi

    Example:
        >>> result = await query_rewriter_tool(
        ...     query="ngày nghỉ phép",
        ...     conversation_history=["Tôi là nhân viên mới"],
        ...     language="vi"
        ... )
        >>> print(result.rewritten)
        "Số ngày nghỉ phép hàng năm dành cho nhân viên mới là bao nhiêu?"
    """
    try:
        logger.info(
            "Rewriting query",
            query=query[:100],
            language=language,
            history_len=len(conversation_history or [])
        )

        # Format conversation context
        context = format_conversation_context(conversation_history or [])

        # Create prompt
        prompt = ChatPromptTemplate.from_template(REWRITER_PROMPT)

        # Get LLM
        llm = get_llm(temperature=0.1, max_tokens=1024)

        # Create chain
        chain = prompt | llm | JsonOutputParser()

        # Invoke
        result = await chain.ainvoke({
            "query": query,
            "conversation_context": context,
            "language": language
        })

        # Parse result
        output = QueryRewriterOutput(
            original=result.get("original", query),
            rewritten=result.get("rewritten", query),
            keywords=result.get("keywords", []),
            query_type=result.get("query_type", "factoid"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning"),
            expansions=result.get("expansions", [])
        )

        logger.info(
            "Query rewritten",
            original=query[:50],
            rewritten=output.rewritten[:50],
            query_type=output.query_type,
            confidence=output.confidence
        )

        return output

    except Exception as e:
        logger.error(
            "Failed to rewrite query",
            error=str(e),
            query=query[:100]
        )
        # Return original query on error
        return QueryRewriterOutput(
            original=query,
            rewritten=query,
            keywords=[],
            query_type="factoid",
            confidence=0.0,
            reasoning="Failed to rewrite query, using original",
            expansions=[]
        )


def query_rewriter_tool_sync(
    query: str,
    conversation_history: Optional[List[str]] = None,
    language: str = "vi"
) -> QueryRewriterOutput:
    """
    Synchronous wrapper for query_rewriter_tool.

    Args:
        query: Câu hỏi gốc của người dùng
        conversation_history: Lịch sử hội thoại
        language: Ngôn ngữ của câu hỏi

    Returns:
        QueryRewriterOutput: Kết quả viết lại câu hỏi
    """
    import asyncio
    return asyncio.run(query_rewriter_tool.ainvoke({
        "query": query,
        "conversation_history": conversation_history or [],
        "language": language
    }))


# =============================================================================
# Helper Functions
# =============================================================================

def convert_to_rewritten_query(output: QueryRewriterOutput) -> RewrittenQuery:
    """
    Convert QueryRewriterOutput to RewrittenQuery model.

    Args:
        output: Output from query rewriter tool

    Returns:
        RewrittenQuery: Standardized rewritten query model
    """
    try:
        query_type = QueryType(output.query_type)
    except ValueError:
        query_type = QueryType.FACTOID

    return RewrittenQuery(
        original=output.original,
        rewritten=output.rewritten,
        keywords=output.keywords,
        query_type=query_type,
        confidence=output.confidence,
        reasoning=output.reasoning,
        expansions=output.expansions
    )


# =============================================================================
# Testing
# =============================================================================

async def test_query_rewriter():
    """Test query rewriter with sample queries."""
    test_cases = [
        {
            "query": "ngày nghỉ phép",
            "conversation_history": ["Tôi là nhân viên mới"],
            "language": "vi"
        },
        {
            "query": "sick leave",
            "conversation_history": [],
            "language": "en"
        },
        {
            "query": "làm sao để đăng ký bảo hiểm",
            "conversation_history": [],
            "language": "vi"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}")
        print(f"{'='*60}")
        print(f"Query: {test['query']}")

        result = await query_rewriter_tool.ainvoke({
            "query": test["query"],
            "conversation_history": test["conversation_history"],
            "language": test["language"]
        })

        print(f"\nOriginal: {result.original}")
        print(f"Rewritten: {result.rewritten}")
        print(f"Query Type: {result.query_type}")
        print(f"Confidence: {result.confidence}")
        print(f"Keywords: {', '.join(result.keywords)}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
        if result.expansions:
            print(f"Expansions:")
            for exp in result.expansions:
                print(f"  - {exp}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_query_rewriter())
