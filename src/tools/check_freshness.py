# =============================================================================
# Check Freshness Tool - Check document freshness and currency
# =============================================================================
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.storage.metadata_store import metadata_store
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Freshness Thresholds Configuration
# =============================================================================

FRESHNESS_THRESHOLDS = {
    "very_recent": 30,      # 30 days - very recent
    "recent": 90,           # 90 days - recent
    "moderate": 180,        # 180 days - moderately fresh
    "stale": 365,          # 365 days - stale
    "very_stale": 730,     # 730 days - very stale
}

FRESHNESS_CATEGORIES = {
    "very_recent": "Rất mới - Cập nhật trong 30 ngày qua",
    "recent": "Mới - Cập nhật trong 90 ngày qua",
    "moderate": "Khá mới - Cập nhật trong 6 tháng qua",
    "stale": "Cũ - Cập nhật hơn 1 năm trước",
    "very_stale": "Rất cũ - Cập nhật hơn 2 năm trước",
    "unknown": "Không xác định"
}


# =============================================================================
# Input/Output Models
# =============================================================================

class DocumentFreshness(BaseModel):
    """Freshness information for a single document."""

    document_id: str = Field(..., description="ID tài liệu")
    document_title: str = Field(..., description="Tiêu đề tài liệu")
    last_updated: Optional[datetime] = Field(None, description="Ngày cập nhật cuối cùng")
    days_since_update: Optional[int] = Field(None, description="Số ngày kể từ lần cập nhật")
    freshness_category: str = Field(..., description="Danh mục freshness")
    freshness_score: float = Field(..., ge=0.0, le=1.0, description="Độ mới (0=old, 1=new)")
    is_fresh: bool = Field(..., description="Có được coi là fresh không")


class FreshnessOutput(BaseModel):
    """Output from check freshness tool."""

    query: str = Field(..., description="Câu hỏi cần kiểm tra freshness")
    documents: List[DocumentFreshness] = Field(..., description="Danh sách freshness của từng tài liệu")
    overall_freshness: str = Field(..., description="Freshness tổng thể")
    average_freshness_score: float = Field(..., description="Điểm freshness trung bình")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Khuyến nghị dựa trên freshness"
    )
    stale_documents: List[str] = Field(
        default_factory=list,
        description="Danh sách tài liệu bị stale"
    )


# =============================================================================
# Check Freshness Tool
# =============================================================================

@tool
async def check_freshness_tool(
    query: str,
    document_ids: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    freshness_threshold: int = 180
) -> FreshnessOutput:
    """
    Kiểm tra độ mới (freshness) của tài liệu.

    Công cụ này kiểm tra độ mới của tài liệu để đảm bảo thông tin còn актуал:
    - Kiểm tra ngày cập nhật cuối cùng của tài liệu
    - Phân loại theo mức độ fresh (very_recent, recent, moderate, stale, very_stale)
    - Tính điểm freshness từ 0 đến 1
    - Đưa ra khuyến nghị về việc sử dụng tài liệu

    Args:
        query: Câu hỏi cần kiểm tra
        document_ids: Danh sách ID tài liệu (ưu tiên hơn sources)
        sources: Danh sách nguồn có document_id
        freshness_threshold: Ngưỡng để coi là fresh (ngày)

    Returns:
        FreshnessOutput: Kết quả kiểm tra freshness

    Example:
        >>> result = await check_freshness_tool.ainvoke({
        ...     "query": "Quy định nghỉ phép 2024",
        ...     "document_ids": ["doc-001", "doc-002"],
        ...     "freshness_threshold": 180
        ... })
        >>> print(f"Overall freshness: {result.overall_freshness}")
    """
    try:
        logger.info(
            "Freshness check started",
            query=query[:100],
            num_documents=len(document_ids) if document_ids else len(sources) if sources else 0
        )

        # Extract document IDs
        if document_ids:
            doc_ids = document_ids
        elif sources:
            doc_ids = list(set([
                s.get("metadata", {}).get("document_id")
                for s in sources
                if s.get("metadata", {}).get("document_id")
            ]))
        else:
            return FreshnessOutput(
                query=query,
                documents=[],
                overall_freshness="unknown",
                average_freshness_score=0.0,
                recommendations=["Không có tài liệu nào để kiểm tra"],
                stale_documents=[]
            )

        # Get freshness for each document
        documents = []
        total_score = 0.0
        stale_docs = []

        for doc_id in doc_ids:
            freshness = await _get_document_freshness(doc_id, freshness_threshold)
            documents.append(freshness)
            total_score += freshness.freshness_score

            if not freshness.is_fresh:
                stale_docs.append(freshness.document_title)

        # Calculate overall metrics
        avg_score = total_score / len(documents) if documents else 0.0
        overall_category = _get_overall_freshness_category(avg_score)

        # Generate recommendations
        recommendations = _generate_recommendations(documents, avg_score, freshness_threshold)

        logger.info(
            "Freshness check completed",
            num_documents=len(documents),
            avg_score=avg_score,
            overall_category=overall_category
        )

        return FreshnessOutput(
            query=query,
            documents=documents,
            overall_freshness=overall_category,
            average_freshness_score=avg_score,
            recommendations=recommendations,
            stale_documents=stale_docs
        )

    except Exception as e:
        logger.error(
            "Freshness check failed",
            error=str(e),
            query=query[:100]
        )
        return FreshnessOutput(
            query=query,
            documents=[],
            overall_freshness="error",
            average_freshness_score=0.0,
            recommendations=[f"Lỗi khi kiểm tra freshness: {str(e)}"],
            stale_documents=[]
        )


def check_freshness_tool_sync(
    query: str,
    document_ids: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    freshness_threshold: int = 180
) -> FreshnessOutput:
    """Synchronous wrapper for check_freshness_tool."""
    import asyncio
    return asyncio.run(check_freshness_tool.ainvoke({
        "query": query,
        "document_ids": document_ids,
        "sources": sources,
        "freshness_threshold": freshness_threshold
    }))


# =============================================================================
# Helper Functions
# =============================================================================

async def _get_document_freshness(
    document_id: str,
    threshold: int
) -> DocumentFreshness:
    """Get freshness info for a single document."""
    try:
        # Get document from metadata store
        doc = await metadata_store.get_document(document_id)

        if not doc:
            return DocumentFreshness(
                document_id=document_id,
                document_title="Unknown Document",
                last_updated=None,
                days_since_update=None,
                freshness_category="unknown",
                freshness_score=0.0,
                is_fresh=False
            )

        # Calculate days since update
        last_updated = doc.updated_at if doc.updated_at else doc.created_at
        days_since = (datetime.utcnow() - last_updated).days if last_updated else None

        # Determine freshness category
        if days_since is None:
            category = "unknown"
            score = 0.0
        elif days_since <= FRESHNESS_THRESHOLDS["very_recent"]:
            category = "very_recent"
            score = 1.0
        elif days_since <= FRESHNESS_THRESHOLDS["recent"]:
            category = "recent"
            score = 0.8
        elif days_since <= FRESHNESS_THRESHOLDS["moderate"]:
            category = "moderate"
            score = 0.6
        elif days_since <= FRESHNESS_THRESHOLDS["stale"]:
            category = "stale"
            score = 0.4
        else:
            category = "very_stale"
            score = 0.2

        is_fresh = days_since <= threshold if days_since is not None else False

        return DocumentFreshness(
            document_id=document_id,
            document_title=doc.title,
            last_updated=last_updated,
            days_since_update=days_since,
            freshness_category=category,
            freshness_score=score,
            is_fresh=is_fresh
        )

    except Exception as e:
        logger.warning("Failed to get document freshness", document_id=document_id, error=str(e))
        return DocumentFreshness(
            document_id=document_id,
            document_title="Error Loading Document",
            last_updated=None,
            days_since_update=None,
            freshness_category="unknown",
            freshness_score=0.0,
            is_fresh=False
        )


def _get_overall_freshness_category(avg_score: float) -> str:
    """Get overall freshness category from average score."""
    if avg_score >= 0.9:
        return "very_recent"
    elif avg_score >= 0.7:
        return "recent"
    elif avg_score >= 0.5:
        return "moderate"
    elif avg_score >= 0.3:
        return "stale"
    else:
        return "very_stale"


def _generate_recommendations(
    documents: List[DocumentFreshness],
    avg_score: float,
    threshold: int
) -> List[str]:
    """Generate recommendations based on freshness analysis."""
    recommendations = []

    if not documents:
        return ["Không có tài liệu để đánh giá"]

    # Overall recommendation
    if avg_score >= 0.7:
        recommendations.append("Thông tin từ các tài liệu này là tương đối mới và актуал.")
    elif avg_score >= 0.5:
        recommendations.append("Thông tin từ các tài liệu này ở mức độ chấp nhận được.")
    else:
        recommendations.append(f"Cảnh báo: Các tài liệu này đã cũ (trung bình {threshold}+ ngày).")

    # Stale documents warning
    stale_count = sum(1 for d in documents if not d.is_fresh)
    if stale_count > 0:
        recommendations.append(
            f"{stale_count} tài liệu đã cũ. Nên kiểm tra xem các quy định có còn hiệu lực không."
        )

    # Very stale warning
    very_stale = [d for d in documents if d.freshness_category == "very_stale"]
    if very_stale:
        titles = ", ".join([d.document_title[:30] for d in very_stale[:3]])
        recommendations.append(
            f"Cảnh báo: Các tài liệu sau rất cũ ({titles}...). "
            "Cần cân nhắc tìm thông tin mới hơn."
        )

    return recommendations


async def get_freshness_summary(document_ids: List[str]) -> Dict[str, Any]:
    """
    Get freshness summary for multiple documents.

    Args:
        document_ids: List of document IDs

    Returns:
        Summary dict with counts by category
    """
    result = await check_freshness_tool.ainvoke({
        "query": "Freshness summary",
        "document_ids": document_ids
    })

    # Count by category
    category_counts = {}
    for doc in result.documents:
        category = doc.freshness_category
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "total_documents": len(result.documents),
        "category_counts": category_counts,
        "average_score": result.average_freshness_score,
        "stale_count": sum(1 for d in result.documents if not d.is_fresh),
        "overall_freshness": result.overall_freshness
    }


def should_use_document(freshness: DocumentFreshness, max_age_days: int = 365) -> bool:
    """
    Determine if a document should be used based on freshness.

    Args:
        freshness: DocumentFreshness object
        max_age_days: Maximum acceptable age in days

    Returns:
        True if document should be used, False otherwise
    """
    if not freshness.days_since_update:
        return False  # Unknown age, be conservative

    return freshness.days_since_update <= max_age_days


# =============================================================================
# Testing
# =============================================================================

async def test_check_freshness():
    """Test freshness check with sample data."""
    print("Testing check freshness...")

    # This would require actual documents in the database
    # For now, showing the structure

    result = await check_freshness_tool.ainvoke({
        "query": "Quy định nghỉ phép",
        "document_ids": ["test-doc-1", "test-doc-2"],
        "freshness_threshold": 180
    })

    print(f"\nQuery: {result.query}")
    print(f"Overall freshness: {result.overall_freshness}")
    print(f"Average score: {result.average_freshness_score:.2f}")
    print(f"Documents checked: {len(result.documents)}")

    for doc in result.documents:
        print(f"\n- {doc.document_title}")
        print(f"  Category: {doc.freshness_category}")
        print(f"  Days since update: {doc.days_since_update}")
        print(f"  Is fresh: {doc.is_fresh}")

    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_check_freshness())
