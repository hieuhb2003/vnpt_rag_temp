# =============================================================================
# Synthesize Answer Tool - Generate answers with citations
# =============================================================================
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm import get_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class Citation(BaseModel):
    """Single citation reference."""

    section_id: str = Field(..., description="ID của section được trích dẫn")
    document_id: str = Field(..., description="ID tài liệu")
    document_title: str = Field(..., description="Tiêu đề tài liệu")
    section_heading: str = Field(..., description="Tiêu đề section")
    content_snippet: str = Field(..., description="Đoạn nội dung được trích dẫn")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Độ liên quan")


class SynthesizeOutput(BaseModel):
    """Output from synthesize answer tool."""

    answer: str = Field(..., description="Câu trả lời được tạo ra")
    citations: List[Citation] = Field(..., description="Danh sách trích dẫn")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy của câu trả lời")
    reasoning: Optional[str] = Field(None, description="Lý do hoặc giải thích thêm")
    sources_summary: str = Field(..., description="Tóm tắt các nguồn sử dụng")
    language: str = Field(..., description="Ngôn ngữ của câu trả lời")


# =============================================================================
# Prompt Template (Vietnamese)
# =============================================================================

SYNTHESIS_PROMPT = """Bạn là một trợ lý chuyên nghiệp chuyên tổng hợp thông tin từ nhiều nguồn để trả lời câu hỏi.

Nhiệm vụ của bạn là:
1. Đọc và phân tích các đoạn văn bản từ tài liệu
2. Tìm ra thông tin liên quan để trả lời câu hỏi
3. Tổng hợp thành câu trả lời mạch lạc, chính xác
4. Trích dẫn rõ ràng từng nguồn thông tin đã sử dụng

**NGUYÊN TẮC QUAN TRỌNG:**
- Chỉ sử dụng thông tin TỪ CÁC NGUỒN ĐƯỢC CUNG CẤP
- KHÔNG tạo ra thông tin không có trong nguồn
- Nếu thông tin không đủ, hãy nói rõ là "Thông tin có sẵn không đủ để trả lời đầy đủ"
- Trích dẫn mỗi điểm thông tin quan trọng
- Sử dụng ngôn ngữ {language} cho câu trả lời

**Câu hỏi:**
{query}

**Các nguồn thông tin:**
{sources}

**Định dạng JSON trả về:**
```json
{{
  "answer": "Câu trả lời chi tiết dựa trên các nguồn",
  "citations": [
    {{
      "section_id": "ID của section",
      "document_id": "ID tài liệu",
      "document_title": "Tiêu đề tài liệu",
      "section_heading": "Tiêu đề section",
      "content_snippet": "Đoạn văn trích dẫn",
      "relevance_score": 0.95
    }}
  ],
  "confidence": 0.85,
  "reasoning": "Giải thích cách bạn tổng hợp thông tin",
  "sources_summary": "Tóm tắt ngắn gọn các nguồn đã sử dụng",
  "language": "{language}"
}}
```

Hãy phân tích và tổng hợp câu trả lời."""


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources for prompt."""
    if not sources:
        return "Không có nguồn thông tin nào được cung cấp."

    formatted = []
    for i, source in enumerate(sources, 1):
        content = source.get("content", "")
        metadata = source.get("metadata", {})

        formatted.append(f"""--- Nguồn {i} ---
Tài liệu: {metadata.get('document_title', 'Unknown')}
Section: {metadata.get('section_heading', 'Unknown')}
Nội dung:
{content}
""")

    return "\n".join(formatted)


# =============================================================================
# Synthesize Answer Tool
# =============================================================================

@tool
async def synthesize_answer_tool(
    query: str,
    sources: List[Dict[str, Any]],
    language: str = "vi",
    max_citations: int = 5
) -> SynthesizeOutput:
    """
    Tạo câu trả lời với trích dẫn từ các nguồn.

    Công cụ này tổng hợp thông tin từ nhiều nguồn để tạo câu trả lời:
    - Phân tích và hiểu câu hỏi
    - Trích xuất thông tin liên quan từ các nguồn
    - Tổng hợp thành câu trả lời mạch lạc
    - Tạo trích dẫn cho từng điểm thông tin

    Args:
        query: Câu hỏi cần trả lời
        sources: Danh sách nguồn thông tin (từ search/retrieval)
        language: Ngôn ngữ câu trả lời (vi, en)
        max_citations: Số trích dẫn tối đa

    Returns:
        SynthesizeOutput: Câu trả lời với trích dẫn

    Example:
        >>> sources = [{"content": "Employees get 20 days...", "metadata": {...}}]
        >>> result = await synthesize_answer_tool.ainvoke({
        ...     "query": "Số ngày nghỉ phép?",
        ...     "sources": sources,
        ...     "language": "vi"
        ... })
        >>> print(result.answer)
    """
    try:
        logger.info(
            "Synthesizing answer",
            query=query[:100],
            num_sources=len(sources),
            language=language
        )

        if not sources:
            return SynthesizeOutput(
                answer="Không có nguồn thông tin nào để tổng hợp câu trả lời.",
                citations=[],
                confidence=0.0,
                reasoning="Không có dữ liệu",
                sources_summary="Không có nguồn",
                language=language
            )

        # Format sources for prompt
        sources_text = format_sources(sources)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)

        # Get LLM
        llm = get_llm(temperature=0.3, max_tokens=2048)

        # Create chain
        chain = prompt | llm | JsonOutputParser()

        # Invoke
        result = await chain.ainvoke({
            "query": query,
            "sources": sources_text,
            "language": language
        })

        # Parse citations
        citations_data = result.get("citations", [])
        if max_citations and len(citations_data) > max_citations:
            citations_data = citations_data[:max_citations]

        citations = [
            Citation(
                section_id=c.get("section_id", ""),
                document_id=c.get("document_id", ""),
                document_title=c.get("document_title", ""),
                section_heading=c.get("section_heading", ""),
                content_snippet=c.get("content_snippet", ""),
                relevance_score=c.get("relevance_score", 0.5)
            )
            for c in citations_data
        ]

        # Parse output
        output = SynthesizeOutput(
            answer=result.get("answer", ""),
            citations=citations,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning"),
            sources_summary=result.get("sources_summary", ""),
            language=result.get("language", language)
        )

        logger.info(
            "Answer synthesized",
            query=query[:50],
            answer_length=len(output.answer),
            num_citations=len(output.citations),
            confidence=output.confidence
        )

        return output

    except Exception as e:
        logger.error(
            "Failed to synthesize answer",
            error=str(e),
            query=query[:100]
        )
        # Return fallback on error
        return SynthesizeOutput(
            answer="Xin lỗi, có lỗi xảy ra khi tổng hợp câu trả lời. Vui lòng thử lại.",
            citations=[],
            confidence=0.0,
            reasoning=f"Lỗi hệ thống: {str(e)}",
            sources_summary=f"Đã cố gắng sử dụng {len(sources)} nguồn",
            language=language
        )


def synthesize_answer_tool_sync(
    query: str,
    sources: List[Dict[str, Any]],
    language: str = "vi",
    max_citations: int = 5
) -> SynthesizeOutput:
    """Synchronous wrapper for synthesize_answer_tool."""
    import asyncio
    return asyncio.run(synthesize_answer_tool.ainvoke({
        "query": query,
        "sources": sources,
        "language": language,
        "max_citations": max_citations
    }))


# =============================================================================
# Helper Functions
# =============================================================================

def format_answer_with_citations(output: SynthesizeOutput) -> str:
    """
    Format answer with inline citations.

    Args:
        output: SynthesizeOutput

    Returns:
        Formatted answer with citations
    """
    result = output.answer + "\n\n"

    if output.citations:
        result += "**Tài liệu tham khảo:**\n\n"
        for i, citation in enumerate(output.citations, 1):
            result += f"{i}. {citation.document_title} - {citation.section_heading}\n"
            result += f"   *{citation.content_snippet[:100]}...*\n\n"

    if output.reasoning:
        result += f"**Ghi chú:** {output.reasoning}\n"

    return result


async def synthesize_with_verification(
    query: str,
    sources: List[Dict[str, Any]],
    language: str = "vi",
    verify_groundedness: bool = True
) -> SynthesizeOutput:
    """
    Synthesize answer with optional groundedness verification.

    Args:
        query: User question
        sources: Information sources
        language: Answer language
        verify_groundedness: Whether to verify answer groundedness

    Returns:
        SynthesizeOutput with verification results
    """
    # Synthesize answer
    output = await synthesize_answer_tool.ainvoke({
        "query": query,
        "sources": sources,
        "language": language
    })

    # Verify groundedness if requested
    if verify_groundedness:
        from src.tools.verify_groundedness import verify_groundedness_tool

        verification = await verify_groundedness_tool.ainvoke({
            "answer": output.answer,
            "sources": sources,
            "language": language
        })

        # Add verification info to reasoning
        if verification.reasoning:
            if output.reasoning:
                output.reasoning += f"\n\nKiểm tra groundedness: {verification.reasoning}"
            else:
                output.reasoning = f"Kiểm tra groundedness: {verification.reasoning}"

        # Update confidence based on verification
        output.confidence = min(output.confidence, verification.confidence)

    return output


# =============================================================================
# Testing
# =============================================================================

async def test_synthesize_answer():
    """Test synthesize answer with sample data."""
    sample_sources = [
        {
            "content": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm. "
                      "Ngày nghỉ phép không sử dụng sẽ được cộng dồn tối đa 5 ngày sang năm sau.",
            "metadata": {
                "document_title": "Quy định về ngày nghỉ phép",
                "section_heading": "1. Ngày nghỉ phép hàng năm",
                "section_id": "sec-001",
                "document_id": "doc-001"
            }
        },
        {
            "content": "Nhân viên thời vụ được hưởng 12 ngày nghỉ phép có lương mỗi năm. "
                      "Ngày nghỉ phép không được cộng dồn.",
            "metadata": {
                "document_title": "Quy định về ngày nghỉ phép",
                "section_heading": "2. Nhân viên thời vụ",
                "section_id": "sec-002",
                "document_id": "doc-001"
            }
        }
    ]

    print("Testing synthesize answer...")
    print(f"Query: Số ngày nghỉ phép của nhân viên?")
    print(f"Sources: {len(sample_sources)}\n")

    result = await synthesize_answer_tool.ainvoke({
        "query": "Số ngày nghỉ phép của nhân viên chính thức và thời vụ?",
        "sources": sample_sources,
        "language": "vi"
    })

    print(f"Answer: {result.answer}")
    print(f"\nConfidence: {result.confidence}")
    print(f"Citations: {len(result.citations)}")
    print(f"Sources summary: {result.sources_summary}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_synthesize_answer())
