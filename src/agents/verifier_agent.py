# =============================================================================
# Verifier Agent - Verifies answer quality and groundedness
# =============================================================================
from typing import Dict, Any, List

from src.agents.state import AgentState, update_state_step
from src.tools.verify_groundedness import verify_groundedness_tool
from src.tools.check_freshness import check_freshness_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VerifierAgent:
    """
    Verifies answer quality and groundedness.

    Responsibilities:
    - Check answer is grounded in sources (two-tier verification)
    - Verify document freshness
    - Flag unsupported claims
    - Decide if escalation is needed
    - Set final_answer with warnings if applicable
    """

    def __init__(
        self,
        groundedness_threshold: float = 0.75,
        enable_tier2_verification: bool = True,
        freshness_threshold_days: int = 180
    ):
        """
        Initialize VerifierAgent.

        Args:
            groundedness_threshold: Minimum confidence for groundedness
            enable_tier2_verification: Whether to enable Tier 2 LLM verification
            freshness_threshold_days: Days threshold for considering documents stale
        """
        self.groundedness_threshold = groundedness_threshold
        self.enable_tier2_verification = enable_tier2_verification
        self.freshness_threshold_days = freshness_threshold_days

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Verify answer groundedness and freshness.

        Args:
            state: Current agent state

        Returns:
            Updated state with final_answer and verification results
        """
        logger.info(
            "Verifier checking answer",
            extra={"query_id": state["query_id"]}
        )

        # Handle no answer case
        if not state.get("draft_answer"):
            state["final_answer"] = state.get("draft_answer", "Không có câu trả lời.")
            state["is_grounded"] = False
            state["verification_tier"] = None
            state = update_state_step(state, "verified")

            logger.warning(
                "No draft answer to verify",
                extra={"query_id": state["query_id"]}
            )
            return state

        try:
            # ================================================================
            # Step 1: Verify groundedness (two-tier verification)
            # ================================================================
            sources = []
            for chunk in state["retrieved_chunks"][:10]:
                sources.append({
                    "content": chunk.get("content", ""),
                    "metadata": {
                        "document_id": chunk.get("document_id", ""),
                        "document_title": chunk.get("metadata", {}).get("document_title", "Unknown")
                    }
                })

            verification = await verify_groundedness_tool.ainvoke({
                "answer": state["draft_answer"],
                "sources": sources,
                "threshold": self.groundedness_threshold,
                "enable_tier2": self.enable_tier2_verification,
                "language": "vi"
            })

            state["is_grounded"] = verification.is_grounded
            state["verification_tier"] = 1 if verification.tier_used == "tier1" else 2
            state["unsupported_claims"] = verification.ungrounded_claims

            logger.info(
                f"Groundedness verification complete (Tier {state['verification_tier']})",
                extra={
                    "query_id": state["query_id"],
                    "is_grounded": verification.is_grounded,
                    "tier_used": verification.tier_used,
                    "confidence": verification.confidence
                }
            )

            # ================================================================
            # Step 2: Check freshness of source documents
            # ================================================================
            doc_ids = list(set(
                chunk.get("document_id")
                for chunk in state["retrieved_chunks"]
                if chunk.get("document_id")
            ))

            freshness_warning = None
            if doc_ids:
                try:
                    freshness = await check_freshness_tool.ainvoke({
                        "document_ids": doc_ids,
                        "freshness_threshold": self.freshness_threshold_days
                    })

                    # Check if any stale documents were used
                    stale_docs = [
                        doc for doc in freshness.documents
                        if not doc.is_fresh
                    ]

                    if stale_docs:
                        stale_info = ", ".join([
                            f"{doc.document_title} ({doc.days_since_update} ngày)"
                            for doc in stale_docs[:3]
                        ])
                        freshness_warning = (
                            f"Lưu ý: Một số thông tin có thể đã cũ. "
                            f"Tài liệu được sử dụng: {stale_info}"
                        )

                        logger.info(
                            "Stale documents detected",
                            extra={
                                "query_id": state["query_id"],
                                "stale_count": len(stale_docs)
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Freshness check failed: {e}",
                        extra={"query_id": state["query_id"]}
                    )

            # ================================================================
            # Step 3: Finalize answer with warnings if needed
            # ================================================================
            final_answer = state["draft_answer"]

            # Add freshness warning if applicable
            if freshness_warning:
                final_answer = f"{final_answer}\n\n⚠️ {freshness_warning}"

            # Handle ungrounded answer
            if not verification.is_grounded:
                if verification.ungrounded_claims:
                    claims_preview = ", ".join(verification.ungrounded_claims[:2])
                    if len(verification.ungrounded_claims) > 2:
                        claims_preview += f", và {len(verification.ungrounded_claims) - 2} thông tin khác"

                    final_answer = (
                        f"{final_answer}\n\n"
                        f"⚠️ Lưu ý: Một số thông tin trong câu trả lời "
                        f"({claims_preview}) cần được xác minh thêm."
                    )
                    state["should_escalate"] = True

                # Add recommendation for verification
                if verification.reasoning:
                    logger.info(
                        f"Verification reasoning: {verification.reasoning}",
                        extra={"query_id": state["query_id"]}
                    )

            state["final_answer"] = final_answer
            state = update_state_step(state, "verified")

            logger.info(
                "Verification complete",
                extra={
                    "query_id": state["query_id"],
                    "is_grounded": state["is_grounded"],
                    "verification_tier": state["verification_tier"],
                    "has_unsupported_claims": len(state["unsupported_claims"]) > 0,
                    "should_escalate": state["should_escalate"]
                }
            )

        except Exception as e:
            logger.error(
                f"Verifier error: {e}",
                extra={"query_id": state["query_id"]},
                exc_info=True
            )
            # On error, still provide the draft answer
            state["final_answer"] = state.get("draft_answer", "Không có câu trả lời.")
            state["is_grounded"] = None  # Unknown
            state["verification_tier"] = None
            state = update_state_step(state, "verified")

        return state


# Singleton instance
verifier_agent = VerifierAgent(
    groundedness_threshold=0.75,
    enable_tier2_verification=True,
    freshness_threshold_days=180
)
