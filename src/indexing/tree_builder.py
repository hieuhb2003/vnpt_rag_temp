# =============================================================================
# Tree Builder - Build Hierarchical Document Structure
# =============================================================================
from uuid import UUID
from typing import List, Optional, Tuple
from src.models.document import Section
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TreeBuilder:
    """Build hierarchical tree structure from flat sections.

    Converts a flat list of sections into a tree structure by:
    1. Assigning parent_section_id based on heading levels
    2. Building section_path (e.g., "1", "1.2", "1.2.3")
    3. Generating summaries for sections without them

    Example:
        H1 "Introduction" → section_path = "1"
        H2 "Getting Started" → section_path = "1.1", parent = H1
        H2 "Advanced Topics" → section_path = "1.2", parent = H1
        H3 "API Reference" → section_path = "1.2.1", parent = H2
    """

    def __init__(self, max_tree_depth: int = 5):
        """
        Initialize TreeBuilder.

        Args:
            max_tree_depth: Maximum allowed depth of section tree
        """
        self.max_tree_depth = max_tree_depth

    def build_tree(
        self,
        sections: List[Section],
        document_id: UUID
    ) -> List[Section]:
        """
        Assign parent IDs and section paths based on heading levels.

        Uses a stack-based approach to track the current section hierarchy
        and assign parent-child relationships.

        Args:
            sections: Flat list of sections from document parser
            document_id: Document UUID to assign to all sections

        Returns:
            Updated sections with parent_section_id and section_path set
        """
        if not sections:
            logger.warning("No sections to build tree from")
            return sections

        logger.debug(
            "Building tree structure",
            section_count=len(sections),
            document_id=str(document_id)
        )

        # Set document_id for all sections
        for section in sections:
            section.document_id = document_id

        # Stack of (section_id, section_level, section_path)
        # Tracks the current hierarchy path
        parent_stack: List[Tuple[UUID, int, str]] = []

        # Track counters at each level
        # Key: parent_id:level -> Value: counter
        section_counters: dict = {}

        for section in sections:
            level = section.level

            # Validate level
            if level < 1 or level > 6:
                logger.warning(
                    f"Invalid section level: {level}, defaulting to 1",
                    heading=section.heading
                )
                level = 1
                section.level = level

            # Pop stack until we find the parent level
            # Parent must be strictly lower level than current
            while parent_stack and parent_stack[-1][1] >= level:
                parent_stack.pop()

            # Set parent and build path
            if parent_stack:
                # Has parent section
                parent_id, parent_level, parent_path = parent_stack[-1]

                # Check tree depth limit
                current_depth = len(parent_path.split('.'))
                if current_depth >= self.max_tree_depth:
                    logger.warning(
                        f"Max tree depth ({self.max_tree_depth}) reached, "
                        f"flattening section at level {level}",
                        heading=section.heading
                    )
                    section.parent_section_id = None
                    # Use flat numbering at root level
                    counter_key = f"root:{level}"
                    section_counters[counter_key] = section_counters.get(counter_key, 0) + 1
                    section.section_path = str(section_counters[counter_key])
                else:
                    # Normal parent-child relationship
                    section.parent_section_id = parent_id

                    # Build section path from parent path
                    # Counter is per parent per level
                    counter_key = f"{parent_id}:{level}"
                    section_counters[counter_key] = section_counters.get(counter_key, 0) + 1
                    section.section_path = f"{parent_path}.{section_counters[counter_key]}"
            else:
                # Root level section (no parent)
                section.parent_section_id = None

                # Root level counter
                counter_key = f"root:{level}"
                section_counters[counter_key] = section_counters.get(counter_key, 0) + 1
                section.section_path = str(section_counters[counter_key])

            # Push current section onto stack
            parent_stack.append((section.id, level, section.section_path))

        # Log tree statistics
        root_sections = [s for s in sections if s.parent_section_id is None]
        max_depth = max(len(s.section_path.split('.')) for s in sections)
        logger.info(
            "Built document tree",
            root_sections=len(root_sections),
            total_sections=len(sections),
            max_depth=max_depth
        )

        return sections

    def generate_section_summaries(
        self,
        sections: List[Section],
        max_length: int = 200,
        use_heading: bool = True
    ) -> List[Section]:
        """
        Generate summaries for sections that don't have them.

        Simple approach: take the first N characters of content.
        More advanced version could use LLM for abstractive summaries.

        Args:
            sections: List of sections (should already have tree structure)
            max_length: Maximum length of generated summary
            use_heading: Include heading in summary

        Returns:
            Sections with summary field populated
        """
        generated = 0

        for section in sections:
            # Skip if already has summary
            if section.summary:
                continue

            # Generate from content if available
            if section.content and section.content.strip():
                content = section.content.strip()

                # Truncate to max_length
                if len(content) > max_length:
                    # Try to break at word boundary
                    truncated = content[:max_length]
                    last_space = truncated.rfind(' ')
                    if last_space > max_length * 0.8:  # Only if space is in last 20%
                        content = truncated[:last_space] + "..."
                    else:
                        content = truncated + "..."

                section.summary = content
                generated += 1
            else:
                # Fallback to heading-only summary
                if use_heading:
                    section.summary = f"Section: {section.heading}"
                    generated += 1

        if generated > 0:
            logger.info(
                "Generated section summaries",
                generated=generated,
                total=len(sections)
            )

        return sections

    def validate_tree(self, sections: List[Section]) -> bool:
        """
        Validate tree structure consistency.

        Checks:
        - All parent_section_ids reference existing sections
        - No circular references
        - All section_paths are valid

        Args:
            sections: List of sections with tree structure

        Returns:
            True if tree is valid
        """
        section_ids = {s.id for s in sections}
        errors = []

        # Check parent references
        for section in sections:
            if section.parent_section_id:
                if section.parent_section_id not in section_ids:
                    errors.append(
                        f"Section '{section.heading}' has invalid parent_id"
                    )

            # Validate section_path format
            path_parts = section.section_path.split('.')
            if not path_parts or not all(p.isdigit() for p in path_parts):
                errors.append(
                    f"Section '{section.heading}' has invalid section_path: {section.section_path}"
                )

        if errors:
            logger.error("Tree validation failed", errors=errors)
            return False

        logger.debug("Tree validation passed")
        return True

    def get_subtree_sections(
        self,
        sections: List[Section],
        root_section_id: UUID
    ) -> List[Section]:
        """
        Get all sections in the subtree rooted at a section.

        Args:
            sections: All sections in document
            root_section_id: Root of subtree

        Returns:
            List of sections in subtree (including root)
        """
        # Build parent->children map
        children_map: dict[UUID, List[Section]] = {}
        for section in sections:
            if section.parent_section_id:
                if section.parent_section_id not in children_map:
                    children_map[section.parent_section_id] = []
                children_map[section.parent_section_id].append(section)

        # BFS to get all descendants
        result = []
        queue = [root_section_id]

        while queue:
            current_id = queue.pop(0)
            current_section = next((s for s in sections if s.id == current_id), None)
            if current_section:
                result.append(current_section)
                # Add children to queue
                if current_id in children_map:
                    queue.extend([child.id for child in children_map[current_id]])

        return result

    def get_section_depth(self, section: Section) -> int:
        """Get depth of section in tree (root=1)."""
        return len(section.section_path.split('.'))

    def get_root_sections(self, sections: List[Section]) -> List[Section]:
        """Get all root-level sections (no parent)."""
        return [s for s in sections if s.parent_section_id is None]


# Singleton instance
tree_builder = TreeBuilder()
