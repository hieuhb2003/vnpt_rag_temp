# =============================================================================
# Test Tool Structure - Validate tools are properly defined
# =============================================================================
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all tools can be imported."""
    print("Testing imports...")

    try:
        from src.tools.query_rewriter import (
            query_rewriter_tool,
            QueryRewriterInput,
            QueryRewriterOutput,
            REWRITER_PROMPT
        )
        print("  ✓ Query rewriter imports successful")

        from src.tools.query_decomposer import (
            query_decomposer_tool,
            QueryDecomposerInput,
            QueryDecomposerOutput,
            SubQueryOutput,
            DECOMPOSER_PROMPT
        )
        print("  ✓ Query decomposer imports successful")

        from src.tools import (
            query_rewriter_tool,
            query_decomposer_tool,
            QueryRewriterInput,
            QueryDecomposerInput,
            REWRITER_PROMPT,
            DECOMPOSER_PROMPT
        )
        print("  ✓ Package __init__ exports successful")

        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_structure():
    """Test tool structure and metadata."""
    print("\nTesting tool structure...")

    try:
        from src.tools.query_rewriter import query_rewriter_tool
        from src.tools.query_decomposer import query_decomposer_tool

        # Check query_rewriter_tool
        print(f"\n  Query Rewriter Tool:")
        print(f"    - Name: {query_rewriter_tool.name}")
        print(f"    - Description: {query_rewriter_tool.description[:100]}...")
        print(f"    - Args schema: {query_rewriter_tool.args_schema}")

        # Check query_decomposer_tool
        print(f"\n  Query Decomposer Tool:")
        print(f"    - Name: {query_decomposer_tool.name}")
        print(f"    - Description: {query_decomposer_tool.description[:100]}...")
        print(f"    - Args schema: {query_decomposer_tool.args_schema}")

        return True
    except Exception as e:
        print(f"  ✗ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompts():
    """Test that prompts are defined."""
    print("\nTesting prompts...")

    try:
        from src.tools.query_rewriter import REWRITER_PROMPT
        from src.tools.query_decomposer import DECOMPOSER_PROMPT

        print(f"\n  REWRITER_PROMPT:")
        print(f"    - Length: {len(REWRITER_PROMPT)} chars")
        print(f"    - Contains Vietnamese: {'{' in REWRITER_PROMPT and '}' in REWRITER_PROMPT}")
        print(f"    - Has query placeholder: {'{query}' in REWRITER_PROMPT}")

        print(f"\n  DECOMPOSER_PROMPT:")
        print(f"    - Length: {len(DECOMPOSER_PROMPT)} chars")
        print(f"    - Contains Vietnamese: {'{' in DECOMPOSER_PROMPT and '}' in DECOMPOSER_PROMPT}")
        print(f"    - Has query placeholder: {'{query}' in DECOMPOSER_PROMPT}")

        return True
    except Exception as e:
        print(f"  ✗ Prompt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pydantic_models():
    """Test Pydantic models."""
    print("\nTesting Pydantic models...")

    try:
        from src.tools.query_rewriter import QueryRewriterInput, QueryRewriterOutput
        from src.tools.query_decomposer import QueryDecomposerInput, QueryDecomposerOutput, SubQueryOutput

        # Test QueryRewriterInput
        rewriter_input = QueryRewriterInput(
            query="test query",
            language="vi"
        )
        print(f"  ✓ QueryRewriterInput created: {rewriter_input.query}")

        # Test QueryRewriterOutput
        rewriter_output = QueryRewriterOutput(
            original="original",
            rewritten="rewritten",
            keywords=["key1", "key2"],
            query_type="factoid",
            confidence=0.9,
            reasoning="test",
            expansions=["exp1"]
        )
        print(f"  ✓ QueryRewriterOutput created: {rewriter_output.rewritten}")

        # Test QueryDecomposerInput
        decomposer_input = QueryDecomposerInput(
            query="test query",
            max_sub_queries=3,
            language="vi"
        )
        print(f"  ✓ QueryDecomposerInput created: {decomposer_input.query}")

        # Test SubQueryOutput
        sub_query = SubQueryOutput(
            id=0,
            query="sub query",
            query_type="factoid",
            dependencies=[]
        )
        print(f"  ✓ SubQueryOutput created: {sub_query.query}")

        # Test QueryDecomposerOutput
        decomposer_output = QueryDecomposerOutput(
            original_query="original",
            sub_queries=[sub_query],
            dependencies={},
            expected_answer_types=[],
            execution_order=[0],
            requires_aggregation=False
        )
        print(f"  ✓ QueryDecomposerOutput created: {len(decomposer_output.sub_queries)} sub-queries")

        return True
    except Exception as e:
        print(f"  ✗ Pydantic model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_helper_functions():
    """Test helper functions."""
    print("\nTesting helper functions...")

    try:
        from src.tools.query_rewriter import format_conversation_context, convert_to_rewritten_query
        from src.tools.query_decomposer import get_execution_plan

        # Test format_conversation_context
        context = format_conversation_context(["msg1", "msg2"])
        print(f"  ✓ format_conversation_context: {len(context)} chars")

        # Test convert_to_rewritten_query
        from src.tools.query_rewriter import QueryRewriterOutput
        output = QueryRewriterOutput(
            original="original",
            rewritten="rewritten",
            keywords=[],
            query_type="factoid",
            confidence=0.9,
            expansions=[]
        )
        rewritten = convert_to_rewritten_query(output)
        print(f"  ✓ convert_to_rewritten_query: {rewritten.original}")

        # Test get_execution_plan
        from src.models.query import DecomposedQuery, SubQuery, QueryType
        decomposed = DecomposedQuery(
            original_query="test",
            sub_queries=[
                SubQuery(id=0, query="q0", query_type=QueryType.FACTOID),
                SubQuery(id=1, query="q1", query_type=QueryType.FACTOID)
            ],
            execution_order=[0, 1]
        )
        plan = get_execution_plan(decomposed)
        print(f"  ✓ get_execution_plan: {len(plan)} levels")

        return True
    except Exception as e:
        print(f"  ✗ Helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Tool Structure Validation Tests")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Tool Structure", test_tool_structure),
        ("Prompts", test_prompts),
        ("Pydantic Models", test_pydantic_models),
        ("Helper Functions", test_helper_functions),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("="*60)
        return 0
    else:
        print("✗ Some tests failed")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
