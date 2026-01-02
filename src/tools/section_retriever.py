# =============================================================================
# Section Retriever Tool - Get full section content with token limit handling
# =============================================================================
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.storage.metadata_store import metadata_store
from src.storage.document_store import document_store
from src.models.document import Section
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic: ~4 characters per token for Vietnamese/English.
    For more accuracy, consider using tiktoken.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Approximate: 1 token ≈ 4 characters for Vietnamese/English mixed text
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Input text
        max_tokens: Maximum tokens allowed
        add_ellipsis: Whether to add "..." at the end

    Returns:
        Truncated text
    """
    if not text:
        return text

    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text

    # Truncate to fit
    target_chars = max_tokens * 4
    truncated = text[:target_chars]

    if add_ellipsis:
        truncated += "..."

    return truncated


# =============================================================================
# Input/Output Models
# =============================================================================

class SectionContent(BaseModel):
    """Full content of a section with metadata."""

    section_id: str = Field(..., description="ID của section")
    document_id: str = Field(..., description="ID tài liệu")
    heading: str = Field(..., description="Tiêu đề section")
    level: int = Field(..., description="Cấp độ heading")
    section_path: str = Field(..., description="Đường dẫn section")
    content: str = Field(..., description="Nội dung đầy đủ của section")
    content_token_count: int = Field(..., description="Số token của nội dung")
    is_truncated: bool = Field(default=False, description="Nội dung đã bị cắt ngắn không")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung")

    # Hierarchy info
    parent_section_id: Optional[str] = Field(None, description="ID section cha")
    parent_heading: Optional[str] = Field(None, description="Tiêu đề section cha")
    child_count: int = Field(default=0, description="Số section con")

    # Document info
    document_title: Optional[str] = Field(None, description="Tiêu đề tài liệu")


class SectionRetrieverOutput(BaseModel):
    """Output from section retriever tool."""

    sections: List[SectionContent] = Field(..., description="Danh sách section content")
    total_sections: int = Field(..., description="Tổng số section")
    total_tokens: int = Field(..., description="Tổng số token của tất cả section")
    any_truncated: bool = Field(default=False, description="Có section nào bị cắt ngắn không")


# =============================================================================
# Section Retriever Tool
# =============================================================================

@tool
async def section_retriever_tool(
    section_ids: Optional[List[str]] = None,
    section_id: Optional[str] = None,
    max_tokens_per_section: int = 4000,
    include_metadata: bool = True
) -> SectionRetrieverOutput:
    """
    Lấy nội dung đầy đủ của section với xử lý giới hạn token.

    Công cụ này truy xuất nội dung đầy đủ của các section từ database:
    - Lấy nội dung gốc không bị cắt ngắn từ database
    - Tính toán số lượng token ước lượng
    - Cắt ngắn nội dung nếu vượt quá giới hạn token
    - Bao gồm thông tin metadata và phân cấp

    Args:
        section_ids: Danh sách ID section cần lấy (ưu tiên hơn section_id)
        section_id: ID section đơn lẻ (không dùng nếu section_ids có)
        max_tokens_per_section: Số token tối đa cho mỗi section
        include_metadata: Có bao gồm metadata không

    Returns:
        SectionRetrieverOutput: Nội dung các section

    Example:
        >>> result = await section_retriever_tool.ainvoke({
        ...     "section_ids": ["sec-1", "sec-2"],
        ...     "max_tokens_per_section": 2000
        ... })
        >>> for section in result.sections:
        ...     print(f"{section.heading}: {section.content[:100]}...")
    """
    try:
        # Normalize input
        if section_ids:
            ids = section_ids
        elif section_id:
            ids = [section_id]
        else:
            raise ValueError("Either section_ids or section_id must be provided")

        logger.info(
            "Section retrieval started",
            section_count=len(ids),
            max_tokens=max_tokens_per_section
        )

        sections = []
        total_tokens = 0
        any_truncated = False

        for sec_id in ids:
            # Get section from metadata store
            section = await metadata_store.get_section(sec_id)

            if not section:
                logger.warning("Section not found", section_id=sec_id)
                continue

            # Get content
            content = section.content or ""

            # Estimate tokens
            token_count = estimate_tokens(content)
            is_truncated = False

            # Truncate if needed
            if token_count > max_tokens_per_section:
                content = truncate_to_tokens(content, max_tokens_per_section)
                token_count = max_tokens_per_section
                is_truncated = True
                any_truncated = True

            # Get parent info
            parent_heading = None
            if section.parent_section_id:
                parent = await metadata_store.get_section(section.parent_section_id)
                if parent:
                    parent_heading = parent.heading

            # Get child count
            children = await metadata_store.get_sections_by_parent(sec_id)
            child_count = len(children)

            # Get document title
            document_title = None
            if section.document_id:
                doc = await metadata_store.get_document(section.document_id)
                if doc:
                    document_title = doc.title

            # Create section content
            section_content = SectionContent(
                section_id=str(section.id),
                document_id=str(section.document_id),
                heading=section.heading,
                level=section.level,
                section_path=section.section_path,
                content=content,
                content_token_count=token_count,
                is_truncated=is_truncated,
                metadata=section.metadata if include_metadata else {},
                parent_section_id=str(section.parent_section_id) if section.parent_section_id else None,
                parent_heading=parent_heading,
                child_count=child_count,
                document_title=document_title
            )

            sections.append(section_content)
            total_tokens += token_count

        logger.info(
            "Section retrieval completed",
            sections_found=len(sections),
            total_tokens=total_tokens,
            any_truncated=any_truncated
        )

        return SectionRetrieverOutput(
            sections=sections,
            total_sections=len(sections),
            total_tokens=total_tokens,
            any_truncated=any_truncated
        )

    except Exception as e:
        logger.error(
            "Section retrieval failed",
            error=str(e),
            section_ids=section_ids or [section_id]
        )
        raise


def section_retriever_tool_sync(
    section_ids: Optional[List[str]] = None,
    section_id: Optional[str] = None,
    max_tokens_per_section: int = 4000,
    include_metadata: bool = True
) -> SectionRetrieverOutput:
    """Synchronous wrapper for section_retriever_tool."""
    import asyncio
    return asyncio.run(section_retriever_tool.ainvoke({
        "section_ids": section_ids,
        "section_id": section_id,
        "max_tokens_per_section": max_tokens_per_section,
        "include_metadata": include_metadata
    }))


# =============================================================================
# Helper Functions
# =============================================================================

async def get_full_section_text(
    section_id: str,
    include_children: bool = False,
    max_total_tokens: int = 8000
) -> str:
    """
    Get full section text with optional children.

    Args:
        section_id: Section ID
        include_children: Include child sections
        max_total_tokens: Maximum total tokens

    Returns:
        Combined section text
    """
    # Get main section
    result = await section_retriever_tool.ainvoke({
        "section_id": section_id,
        "max_tokens_per_section": max_total_tokens
    })

    if not result.sections:
        return ""

    if not include_children:
        return result.sections[0].content

    # Build full text with children
    main_section = result.sections[0]
    full_text = f"# {main_section.heading}\n\n{main_section.content}\n\n"

    # Get children
    from src.tools.tree_navigator import _get_child_sections

    children = await _get_child_sections(
        section_id,
        main_section.document_id,
        max_depth=1
    )

    for child in children:
        child_result = await section_retriever_tool.ainvoke({
            "section_id": child.id,
            "max_tokens_per_section": 2000
        })

        if child_result.sections:
            child_section = child_result.sections[0]
            full_text += f"## {child_section.heading}\n\n{child_section.content}\n\n"

    # Truncate if needed
    current_tokens = estimate_tokens(full_text)
    if current_tokens > max_total_tokens:
        full_text = truncate_to_tokens(full_text, max_total_tokens)

    return full_text


async def get_sections_by_document(
    document_id: str,
    max_tokens_per_section: int = 2000,
    max_sections: int = 50
) -> SectionRetrieverOutput:
    """
    Get all sections for a document.

    Args:
        document_id: Document ID
        max_tokens_per_section: Max tokens per section
        max_sections: Maximum number of sections

    Returns:
        Section contents
    """
    # Get all sections for document
    all_sections = await metadata_store.get_sections_by_document(document_id)

    # Limit by max_sections
    section_ids = [str(s.id) for s in all_sections[:max_sections]]

    return await section_retriever_tool.ainvoke({
        "section_ids": section_ids,
        "max_tokens_per_section": max_tokens_per_section
    })


async def find_relevant_sections(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5
) -> SectionRetrieverOutput:
    """
    Find sections relevant to a query using vector search.

    Args:
        query: Search query
        document_id: Optional document filter
        top_k: Number of sections to return

    Returns:
        Relevant section contents
    """
    # First do hybrid search to find relevant sections
    from src.tools.hybrid_search import hybrid_search_tool

    search_result = await hybrid_search_tool.ainvoke({
        "query": query,
        "collection": "sections",
        "top_k": top_k,
        "filters": {"document_id": document_id} if document_id else None
    })

    # Extract section IDs from search results
    section_ids = [r.id for r in search_result.results]

    # Get full section contents
    return await section_retriever_tool.ainvoke({
        "section_ids": section_ids,
        "max_tokens_per_section": 2000
    })


# =============================================================================
# Testing
# =============================================================================

async def test_section_retriever():
    """Test section retriever with sample section IDs."""
    test_section_ids = ["test-section-1", "test-section-2"]

    print(f"Testing section retriever with {len(test_section_ids)} sections")

    result = await section_retriever_tool.ainvoke({
        "section_ids": test_section_ids,
        "max_tokens_per_section": 1000
    })

    print(f"\nResults: {result.total_sections} sections")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Any truncated: {result.any_truncated}")

    for section in result.sections:
        print(f"\n--- {section.heading} ---")
        print(f"Path: {section.section_path}")
        print(f"Tokens: {section.content_token_count}")
        if section.is_truncated:
            print("(Content truncated)")
        print(f"Content preview: {section.content[:150]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_section_retriever())
