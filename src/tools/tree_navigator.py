# =============================================================================
# Tree Navigator Tool - Navigate document tree structure
# =============================================================================
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.storage.metadata_store import metadata_store
from src.models.document import Section
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class NavigationDirection(str):
    """Navigation directions in document tree."""

    CHILDREN = "children"  # Get child sections
    SIBLINGS = "siblings"  # Get sibling sections
    PARENT = "parent"  # Get parent section
    PATH_TO_ROOT = "path_to_root"  # Get path from section to root
    ALL = "all"  # Get all related sections


class SectionInfo(BaseModel):
    """Information about a section."""

    id: str = Field(..., description="ID của section")
    document_id: str = Field(..., description="ID tài liệu chứa section")
    heading: str = Field(..., description="Tiêu đề section")
    level: int = Field(..., description="Cấp độ heading (1-6)")
    section_path: str = Field(..., description="Đường dẫn section (ví dụ: 1.2.3)")
    summary: Optional[str] = Field(None, description="Tóm tắt section")
    position: int = Field(..., description="Vị trí trong tài liệu")
    parent_section_id: Optional[str] = Field(None, description="ID section cha")
    has_children: bool = Field(default=False, description="Có section con không")


class TreeNavigatorOutput(BaseModel):
    """Output from tree navigator tool."""

    section_id: str = Field(..., description="ID section gốc")
    direction: str = Field(..., description="Hướng điều hướng")
    sections: List[SectionInfo] = Field(..., description="Danh sách section tìm được")
    total_sections: int = Field(..., description="Tổng số section")
    path: Optional[List[str]] = Field(None, description="Đường dẫn từ root đến section (nếu có)")
    document_id: str = Field(..., description="ID tài liệu")
    document_title: Optional[str] = Field(None, description="Tiêu đề tài liệu")


# =============================================================================
# Tree Navigator Tool
# =============================================================================

@tool
async def tree_navigator_tool(
    section_id: str,
    direction: str = "children",
    max_depth: int = 1,
    include_content: bool = False
) -> TreeNavigatorOutput:
    """
    Điều hướng trong cấu trúc cây tài liệu.

    Công cụ này cho phép điều hướng trong cấu trúc phân cấp của tài liệu:
    - children: Lấy danh sách section con trực tiếp
    - siblings: Lấy các section cùng cấp (anh chị em)
    - parent: Lấy section cha
    - path_to_root: Lấy đường dẫn từ section đến root
    - all: Lấy tất cả các section liên quan

    Args:
        section_id: ID của section hiện tại
        direction: Hướng điều hướng (children, siblings, parent, path_to_root, all)
        max_depth: Độ sâu tối đa cho điều hướng con (chỉ dùng với 'children')
        include_content: Có bao gồm nội dung section không

    Returns:
        TreeNavigatorOutput: Kết quả điều hướng

    Example:
        >>> result = await tree_navigator_tool(
        ...     section_id="abc-123",
        ...     direction="children",
        ...     max_depth=2
        ... )
        >>> print(f"Found {result.total_sections} child sections")
    """
    try:
        logger.info(
            "Tree navigation started",
            section_id=section_id,
            direction=direction,
            max_depth=max_depth
        )

        # Get the current section
        current_section = await metadata_store.get_section(section_id)
        if not current_section:
            logger.warning("Section not found", section_id=section_id)
            raise ValueError(f"Section {section_id} not found")

        sections = []
        path = None
        document_title = None

        if direction == "children":
            # Get child sections
            sections = await _get_child_sections(
                section_id,
                current_section.document_id,
                max_depth=max_depth
            )

        elif direction == "siblings":
            # Get sibling sections
            if current_section.parent_section_id:
                sections = await _get_child_sections(
                    current_section.parent_section_id,
                    current_section.document_id,
                    max_depth=1
                )
                # Filter out the current section
                sections = [s for s in sections if s.id != section_id]
            else:
                # No parent means no siblings
                sections = []

        elif direction == "parent":
            # Get parent section
            if current_section.parent_section_id:
                parent = await metadata_store.get_section(current_section.parent_section_id)
                if parent:
                    sections = [_section_to_info(parent)]
            else:
                sections = []

        elif direction == "path_to_root":
            # Get path to root
            path_sections = []
            current = current_section
            while current:
                path_sections.insert(0, _section_to_info(current))
                if current.parent_section_id:
                    current = await metadata_store.get_section(current.parent_section_id)
                else:
                    break

            sections = path_sections
            path = [s.heading for s in path_sections]

        elif direction == "all":
            # Get all related sections
            all_sections = []

            # Add path to root
            current = current_section
            while current:
                all_sections.append(_section_to_info(current))
                if current.parent_section_id:
                    current = await metadata_store.get_section(current.parent_section_id)
                else:
                    break

            # Add children (direct only)
            children = await _get_child_sections(
                section_id,
                current_section.document_id,
                max_depth=1
            )
            all_sections.extend(children)

            # Add siblings
            if current_section.parent_section_id:
                siblings = await _get_child_sections(
                    current_section.parent_section_id,
                    current_section.document_id,
                    max_depth=1
                )
                for s in siblings:
                    if s.id != section_id and s.id not in [a.id for a in all_sections]:
                        all_sections.append(s)

            sections = all_sections

        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Get document title
        if current_section.document_id:
            doc = await metadata_store.get_document(current_section.document_id)
            if doc:
                document_title = doc.title

        logger.info(
            "Tree navigation completed",
            section_id=section_id,
            direction=direction,
            sections_found=len(sections)
        )

        return TreeNavigatorOutput(
            section_id=section_id,
            direction=direction,
            sections=sections,
            total_sections=len(sections),
            path=path,
            document_id=str(current_section.document_id),
            document_title=document_title
        )

    except Exception as e:
        logger.error(
            "Tree navigation failed",
            error=str(e),
            section_id=section_id,
            direction=direction
        )
        raise


def tree_navigator_tool_sync(
    section_id: str,
    direction: str = "children",
    max_depth: int = 1,
    include_content: bool = False
) -> TreeNavigatorOutput:
    """Synchronous wrapper for tree_navigator_tool."""
    import asyncio
    return asyncio.run(tree_navigator_tool.ainvoke({
        "section_id": section_id,
        "direction": direction,
        "max_depth": max_depth,
        "include_content": include_content
    }))


# =============================================================================
# Helper Functions
# =============================================================================

async def _get_child_sections(
    section_id: str,
    document_id: str,
    max_depth: int = 1
) -> List[SectionInfo]:
    """
    Get child sections recursively up to max_depth.

    Args:
        section_id: Parent section ID
        document_id: Document ID
        max_depth: Maximum depth to traverse

    Returns:
        List of child section info
    """
    children = await metadata_store.get_sections_by_parent(section_id)

    if max_depth <= 1:
        return [_section_to_info(child) for child in children]

    # Recursively get grandchildren
    all_children = []
    for child in children:
        child_info = _section_to_info(child)
        all_children.append(child_info)

        # Get grandchildren
        grandchildren = await _get_child_sections(
            str(child.id),
            document_id,
            max_depth=max_depth - 1
        )
        all_children.extend(grandchildren)

    return all_children


def _section_to_info(section: Section) -> SectionInfo:
    """Convert Section model to SectionInfo."""
    return SectionInfo(
        id=str(section.id),
        document_id=str(section.document_id),
        heading=section.heading,
        level=section.level,
        section_path=section.section_path,
        summary=section.summary,
        position=section.position,
        parent_section_id=str(section.parent_section_id) if section.parent_section_id else None,
        has_children=False  # TODO: Check if has children
    )


async def get_section_tree(
    document_id: str,
    max_levels: int = 3
) -> List[Dict[str, Any]]:
    """
    Get complete section tree for a document.

    Args:
        document_id: Document ID
        max_levels: Maximum levels to include

    Returns:
        Nested tree structure
    """
    # Get root sections (no parent)
    all_sections = await metadata_store.get_sections_by_document(document_id)
    root_sections = [s for s in all_sections if not s.parent_section_id]

    async def build_tree(section: Section, current_level: int) -> Dict[str, Any]:
        if current_level > max_levels:
            return None

        children = await metadata_store.get_sections_by_parent(str(section.id))

        return {
            "id": str(section.id),
            "heading": section.heading,
            "level": section.level,
            "path": section.section_path,
            "children": [
                await build_tree(child, current_level + 1)
                for child in children
            ] if current_level < max_levels else []
        }

    tree = []
    for root in root_sections:
        node = await build_tree(root, 1)
        if node:
            tree.append(node)

    return tree


async def find_section_by_path(
    document_id: str,
    path: str
) -> Optional[SectionInfo]:
    """
    Find a section by its path.

    Args:
        document_id: Document ID
        path: Section path (e.g., "1.2.3")

    Returns:
        SectionInfo if found, None otherwise
    """
    all_sections = await metadata_store.get_sections_by_document(document_id)

    for section in all_sections:
        if section.section_path == path:
            return _section_to_info(section)

    return None


async def get_breadcrumbs(
    section_id: str
) -> List[SectionInfo]:
    """
    Get breadcrumb path for a section.

    Args:
        section_id: Section ID

    Returns:
        List of sections from root to the given section
    """
    result = await tree_navigator_tool.ainvoke({
        "section_id": section_id,
        "direction": "path_to_root"
    })

    return result.sections


# =============================================================================
# Testing
# =============================================================================

async def test_tree_navigator():
    """Test tree navigator with sample section IDs."""
    # This would require actual data in the database
    test_section_id = "test-section-id"

    print(f"Testing tree navigator with section: {test_section_id}")

    # Test children navigation
    print("\n--- Children ---")
    try:
        result = await tree_navigator_tool.ainvoke({
            "section_id": test_section_id,
            "direction": "children",
            "max_depth": 2
        })
        print(f"Found {result.total_sections} children")
        for section in result.sections:
            print(f"  - {section.heading} (Level {section.level})")
    except Exception as e:
        print(f"Error: {e}")

    # Test path to root
    print("\n--- Path to Root ---")
    try:
        result = await tree_navigator_tool.ainvoke({
            "section_id": test_section_id,
            "direction": "path_to_root"
        })
        print(f"Path: {' -> '.join(result.path) if result.path else 'N/A'}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tree_navigator())
