# =============================================================================
# Verify Groundedness Tool - Two-tier verification of answer groundedness
# =============================================================================
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import numpy as np

from src.utils.llm import get_llm
from src.indexing.embedder import get_embedder
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# Input/Output Models
# =============================================================================

class VerificationOutput(BaseModel):
    """Output from groundedness verification tool."""

    is_grounded: bool = Field(..., description="Câu trả lời có được grounded không")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy của kết quả")
    tier_used: str = Field(..., description="Tier đã sử dụng (tier1, tier2)")
    similarity_score: Optional[float] = Field(None, description="Điểm tương đồng (Tier 1)")
    llm_assessment: Optional[str] = Field(None, description="Đánh giá của LLM (Tier 2)")
    ungrounded_claims: List[str] = Field(
        default_factory=list,
        description="Các phát biểu không có grounding"
    )
    reasoning: Optional[str] = Field(None, description="Lý do chi tiết")


# =============================================================================
# Prompt Template (Vietnamese) - Tier 2 LLM Verification
# =============================================================================

VERIFICATION_PROMPT = """Bạn là một chuyên gia kiểm tra tính chính xác của thông tin.

Nhiệm vụ của bạn là xác minh xem câu trả lời có được hỗ trợ bởi các nguồn thông tin hay không.

**TIÊU CHÍ ĐÁNH GIÁ:**
1. **GROUNDED**: Thông tin được trình bày có trong nguồn và đúng ngữ cảnh
2. **PARTIALLY GROUNDED**: Một phần thông tin có trong nguồn, nhưng bị sai lệch hoặc thiếu ngữ cảnh
3. **NOT GROUNDED**: Thông tin không có trong nguồn hoặc được bịa ra

**Câu trả lời cần kiểm tra:**
{answer}

**Các nguồn thông tin:**
{sources}

**Định dạng JSON trả về:**
```json
{{
  "is_grounded": true,
  "confidence": 0.90,
  "llm_assessment": "Đánh giá chi tiết về độ chính xác",
  "ungrounded_claims": ["claim 1", "claim 2"],
  "reasoning": "Giải thích chi tiết"
}}
```

Hãy phân tích và trả về kết quả."""


def format_sources_for_verification(sources: List[Dict[str, Any]]) -> str:
    """Format sources for verification prompt."""
    if not sources:
        return "Không có nguồn thông tin."

    formatted = []
    for i, source in enumerate(sources, 1):
        content = source.get("content", "")[:500]  # Limit length
        metadata = source.get("metadata", {})
        formatted.append(
            f"Nguồn {i}: {content}\n"
            f"(Từ: {metadata.get('document_title', 'Unknown')})"
        )
    return "\n\n".join(formatted)


# =============================================================================
# Groundedness Verification Tool
# =============================================================================

@tool
async def verify_groundedness_tool(
    answer: str,
    sources: List[Dict[str, Any]],
    threshold: float = 0.75,
    enable_tier2: bool = True,
    language: str = "vi"
) -> VerificationOutput:
    """
    Kiểm tra tính groundedness của câu trả lời với TWO-TIER logic.

    Công cụ này sử dụng phương pháp TWO-TIER để kiểm tra:
    - **Tier 1**: Semantic similarity check (nhanh)
      * So sánh embedding của câu trả lời và các nguồn
      * Tính toán điểm tương đồng cosine
      * Nhanh, ít tốn kém

    - **Tier 2**: LLM-based verification (nếu Tier 1 không chắc chắn)
      * Sử dụng LLM để phân tích chi tiết
      * Phát hiện các phát biểu không có grounding
      * Chính xác hơn nhưng chậm hơn

    Args:
        answer: Câu trả lời cần kiểm tra
        sources: Danh sách nguồn thông tin
        threshold: Ngưỡng điểm để coi là grounded
        enable_tier2: Có bật Tier 2 verification không
        language: Ngôn ngữ (vi, en)

    Returns:
        VerificationOutput: Kết quả kiểm tra

    Example:
        >>> result = await verify_groundedness_tool.ainvoke({
        ...     "answer": "Nhân viên được 20 ngày nghỉ phép",
        ...     "sources": sources,
        ...     "threshold": 0.75
        ... })
        >>> print(f"Is grounded: {result.is_grounded}")
    """
    try:
        logger.info(
            "Groundedness verification started",
            answer_length=len(answer),
            num_sources=len(sources),
            threshold=threshold
        )

        if not sources:
            return VerificationOutput(
                is_grounded=False,
                confidence=0.0,
                tier_used="none",
                ungrounded_claims=[answer],
                reasoning="Không có nguồn thông tin để xác minh"
            )

        if not answer or not answer.strip():
            return VerificationOutput(
                is_grounded=False,
                confidence=0.0,
                tier_used="none",
                ungrounded_claims=[],
                reasoning="Câu trả lời rỗng"
            )

        # =========================================================================
        # Tier 1: Semantic Similarity Check
        # =========================================================================

        tier1_result = await _tier1_semantic_check(answer, sources)

        logger.info(
            "Tier 1 verification completed",
            similarity=tier1_result["similarity"],
            grounded=tier1_result["is_grounded"]
        )

        # If Tier 1 is confident enough, return early
        if tier1_result["is_grounded"] and tier1_result["similarity"] >= threshold:
            logger.info("Tier 1 verification passed, skipping Tier 2")
            return VerificationOutput(
                is_grounded=True,
                confidence=tier1_result["similarity"],
                tier_used="tier1",
                similarity_score=tier1_result["similarity"],
                reasoning=f"Cosine similarity {tier1_result['similarity']:.2f} >= {threshold}"
            )

        # If Tier 1 clearly fails and Tier 2 is disabled
        if not enable_tier2:
            logger.info("Tier 2 disabled, returning Tier 1 result")
            return VerificationOutput(
                is_grounded=tier1_result["is_grounded"],
                confidence=tier1_result["similarity"],
                tier_used="tier1",
                similarity_score=tier1_result["similarity"],
                reasoning=f"Cosine similarity {tier1_result['similarity']:.2f} below threshold"
            )

        # =========================================================================
        # Tier 2: LLM-based Verification
        # =========================================================================

        logger.info("Proceeding to Tier 2 LLM verification")

        tier2_result = await _tier2_llm_verification(answer, sources, language)

        logger.info(
            "Tier 2 verification completed",
            is_grounded=tier2_result["is_grounded"],
            confidence=tier2_result["confidence"]
        )

        return VerificationOutput(
            is_grounded=tier2_result["is_grounded"],
            confidence=tier2_result["confidence"],
            tier_used="tier2",
            similarity_score=tier1_result["similarity"],
            llm_assessment=tier2_result["llm_assessment"],
            ungrounded_claims=tier2_result.get("ungrounded_claims", []),
            reasoning=tier2_result.get("reasoning", "")
        )

    except Exception as e:
        logger.error(
            "Groundedness verification failed",
            error=str(e),
            answer_length=len(answer)
        )
        return VerificationOutput(
            is_grounded=False,
            confidence=0.0,
            tier_used="error",
            ungrounded_claims=[],
            reasoning=f"Lỗi hệ thống: {str(e)}"
        )


def verify_groundedness_tool_sync(
    answer: str,
    sources: List[Dict[str, Any]],
    threshold: float = 0.75,
    enable_tier2: bool = True,
    language: str = "vi"
) -> VerificationOutput:
    """Synchronous wrapper for verify_groundedness_tool."""
    import asyncio
    return asyncio.run(verify_groundedness_tool.ainvoke({
        "answer": answer,
        "sources": sources,
        "threshold": threshold,
        "enable_tier2": enable_tier2,
        "language": language
    }))


# =============================================================================
# Tier Implementations
# =============================================================================

async def _tier1_semantic_check(
    answer: str,
    sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Tier 1: Fast semantic similarity check.

    Compares answer embedding with source embeddings using cosine similarity.
    """
    try:
        embedder = get_embedder()

        # Embed answer
        answer_embedding = await embedder.embed(answer)

        # Embed each source and calculate similarity
        similarities = []
        for source in sources:
            content = source.get("content", "")
            if not content:
                continue

            source_embedding = await embedder.embed(content)
            similarity = _cosine_similarity(answer_embedding, source_embedding)
            similarities.append(similarity)

        if not similarities:
            return {
                "is_grounded": False,
                "similarity": 0.0
            }

        # Use maximum similarity
        max_similarity = max(similarities)

        # Determine if grounded based on settings
        threshold = settings.groundedness_threshold if hasattr(settings, 'groundedness_threshold') else 0.75

        return {
            "is_grounded": max_similarity >= threshold,
            "similarity": max_similarity
        }

    except Exception as e:
        logger.warning("Tier 1 semantic check failed", error=str(e))
        return {
            "is_grounded": False,
            "similarity": 0.0
        }


async def _tier2_llm_verification(
    answer: str,
    sources: List[Dict[str, Any]],
    language: str
) -> Dict[str, Any]:
    """
    Tier 2: LLM-based verification.

    Uses LLM to analyze answer and sources in detail.
    """
    try:
        # Format sources
        sources_text = format_sources_for_verification(sources)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(VERIFICATION_PROMPT)

        # Get LLM
        llm = get_llm(temperature=0.1, max_tokens=1024)

        # Create chain
        chain = prompt | llm | JsonOutputParser()

        # Invoke
        result = await chain.ainvoke({
            "answer": answer,
            "sources": sources_text
        })

        return {
            "is_grounded": result.get("is_grounded", False),
            "confidence": result.get("confidence", 0.5),
            "llm_assessment": result.get("llm_assessment", ""),
            "ungrounded_claims": result.get("ungrounded_claims", []),
            "reasoning": result.get("reasoning", "")
        }

    except Exception as e:
        logger.warning("Tier 2 LLM verification failed", error=str(e))
        return {
            "is_grounded": False,
            "confidence": 0.0,
            "llm_assessment": f"Lỗi: {str(e)}",
            "ungrounded_claims": [],
            "reasoning": ""
        }


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
    except (ValueError, TypeError):
        return 0.0


# =============================================================================
# Testing
# =============================================================================

async def test_verify_groundedness():
    """Test groundedness verification with sample data."""
    test_cases = [
        {
            "answer": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm.",
            "sources": [
                {
                    "content": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm.",
                    "metadata": {"document_title": "Quy chế nhân sự"}
                }
            ],
            "expected": True
        },
        {
            "answer": "Nhân viên được 30 ngày nghỉ phép và 5 tháng thưởng.",
            "sources": [
                {
                    "content": "Nhân viên chính thức được hưởng 20 ngày nghỉ phép có lương mỗi năm.",
                    "metadata": {"document_title": "Quy chế nhân sự"}
                }
            ],
            "expected": False
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}")
        print(f"{'='*60}")
        print(f"Answer: {test['answer']}")
        print(f"Expected: {'Grounded' if test['expected'] else 'Not Grounded'}")

        result = await verify_groundedness_tool.ainvoke({
            "answer": test["answer"],
            "sources": test["sources"],
            "threshold": 0.75,
            "enable_tier2": True
        })

        print(f"\nResult: {'✓ Grounded' if result.is_grounded else '✗ Not Grounded'}")
        print(f"Tier: {result.tier_used}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.similarity_score:
            print(f"Similarity: {result.similarity_score:.2f}")
        if result.llm_assessment:
            print(f"LLM Assessment: {result.llm_assessment}")
        if result.ungrounded_claims:
            print(f"Ungrounded Claims: {result.ungrounded_claims}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_verify_groundedness())
