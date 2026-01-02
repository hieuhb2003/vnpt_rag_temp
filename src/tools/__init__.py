# =============================================================================
# Tools Package - Query Processing and Retrieval Tools
# =============================================================================

from src.tools.query_rewriter import (
    query_rewriter_tool,
    query_rewriter_tool_sync,
    QueryRewriterInput,
    QueryRewriterOutput,
    convert_to_rewritten_query,
    REWRITER_PROMPT
)

from src.tools.query_decomposer import (
    query_decomposer_tool,
    query_decomposer_tool_sync,
    QueryDecomposerInput,
    QueryDecomposerOutput,
    SubQueryOutput,
    convert_to_decomposed_query,
    get_execution_plan,
    DECOMPOSER_PROMPT
)

from src.tools.hybrid_search import (
    hybrid_search_tool,
    hybrid_search_tool_sync,
    SearchFilters,
    SearchResult,
    HybridSearchOutput,
    search_multiple_collections,
    fuse_results
)

from src.tools.tree_navigator import (
    tree_navigator_tool,
    tree_navigator_tool_sync,
    SectionInfo,
    TreeNavigatorOutput,
    get_section_tree,
    find_section_by_path,
    get_breadcrumbs
)

from src.tools.section_retriever import (
    section_retriever_tool,
    section_retriever_tool_sync,
    SectionContent,
    SectionRetrieverOutput,
    estimate_tokens,
    truncate_to_tokens,
    get_full_section_text,
    get_sections_by_document,
    find_relevant_sections
)

from src.tools.synthesize_answer import (
    synthesize_answer_tool,
    synthesize_answer_tool_sync,
    Citation,
    SynthesizeOutput,
    format_answer_with_citations,
    synthesize_with_verification
)

from src.tools.verify_groundedness import (
    verify_groundedness_tool,
    verify_groundedness_tool_sync,
    VerificationOutput
)

from src.tools.check_freshness import (
    check_freshness_tool,
    check_freshness_tool_sync,
    DocumentFreshness,
    FreshnessOutput,
    FRESHNESS_THRESHOLDS,
    get_freshness_summary,
    should_use_document
)

__all__ = [
    # Query Rewriter
    "query_rewriter_tool",
    "query_rewriter_tool_sync",
    "QueryRewriterInput",
    "QueryRewriterOutput",
    "convert_to_rewritten_query",
    "REWRITER_PROMPT",

    # Query Decomposer
    "query_decomposer_tool",
    "query_decomposer_tool_sync",
    "QueryDecomposerInput",
    "QueryDecomposerOutput",
    "SubQueryOutput",
    "convert_to_decomposed_query",
    "get_execution_plan",
    "DECOMPOSER_PROMPT",

    # Hybrid Search
    "hybrid_search_tool",
    "hybrid_search_tool_sync",
    "SearchFilters",
    "SearchResult",
    "HybridSearchOutput",
    "search_multiple_collections",
    "fuse_results",

    # Tree Navigator
    "tree_navigator_tool",
    "tree_navigator_tool_sync",
    "SectionInfo",
    "TreeNavigatorOutput",
    "get_section_tree",
    "find_section_by_path",
    "get_breadcrumbs",

    # Section Retriever
    "section_retriever_tool",
    "section_retriever_tool_sync",
    "SectionContent",
    "SectionRetrieverOutput",
    "estimate_tokens",
    "truncate_to_tokens",
    "get_full_section_text",
    "get_sections_by_document",
    "find_relevant_sections",

    # Synthesize Answer
    "synthesize_answer_tool",
    "synthesize_answer_tool_sync",
    "Citation",
    "SynthesizeOutput",
    "format_answer_with_citations",
    "synthesize_with_verification",

    # Verify Groundedness
    "verify_groundedness_tool",
    "verify_groundedness_tool_sync",
    "VerificationOutput",

    # Check Freshness
    "check_freshness_tool",
    "check_freshness_tool_sync",
    "DocumentFreshness",
    "FreshnessOutput",
    "FRESHNESS_THRESHOLDS",
    "get_freshness_summary",
    "should_use_document",
]
