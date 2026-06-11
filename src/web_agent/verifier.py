"""Evidence verification rules for the web agent."""

from __future__ import annotations

from typing import Any

from .schemas import AgentEntities, AgentIntake, EvidenceCoverage, VerifierResult


BOUNDARY_TASKS = {"unsupported_image", "need_clarification", "unsupported_general", "invalid_premise"}


def _coerce_intake(intake: dict[str, Any] | AgentIntake) -> AgentIntake:
    if isinstance(intake, AgentIntake):
        return intake
    return AgentIntake.model_validate(intake)


def _combined_evidence_text(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in evidence:
        for key in ("title", "source_file", "preview", "corrective_query"):
            value = item.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _top_score(evidence: list[dict[str, Any]]) -> float | None:
    if not evidence:
        return None
    score = evidence[0].get("rerank_score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _required_terms(entities: AgentEntities) -> list[str]:
    hard_terms: list[str] = []
    for group in (entities.artists, entities.works, entities.techniques, entities.schools):
        hard_terms.extend(str(term).strip() for term in group if str(term).strip())
    if hard_terms:
        return list(dict.fromkeys(hard_terms))
    dynasty_terms = [str(term).strip() for term in entities.dynasties if str(term).strip()]
    return list(dict.fromkeys(dynasty_terms[:2]))


def evidence_coverage(intake: AgentIntake, evidence: list[dict[str, Any]]) -> EvidenceCoverage:
    terms = _required_terms(intake.entities)
    evidence_text = _combined_evidence_text(evidence)
    covered = [term for term in terms if term in evidence_text]
    missing = [term for term in terms if term not in evidence_text]
    return EvidenceCoverage(
        required_terms=terms,
        covered_terms=covered,
        missing_terms=missing,
        top_score=_top_score(evidence),
        evidence_count=len(evidence),
    )


def verify_agent_state(
    *,
    question: str,
    intake: dict[str, Any] | AgentIntake,
    evidence: list[dict[str, Any]],
    top_evidence_relevant: bool,
) -> dict[str, Any]:
    """Return a normalized verifier result for the current graph state."""

    parsed = _coerce_intake(intake)
    task_type = parsed.task_type
    coverage = evidence_coverage(parsed, evidence)

    if task_type == "unsupported_image":
        return VerifierResult(
            verdict="unsupported_image",
            can_continue=True,
            reason="图像请求不在中国山水画创作范围内。",
            coverage=coverage,
        ).as_dict()
    if task_type in {"need_clarification", "unsupported_general"}:
        return VerifierResult(
            verdict=task_type,
            can_continue=True,
            reason="问题需要澄清或不属于研究范围。",
            coverage=coverage,
        ).as_dict()
    if task_type == "invalid_premise":
        issue = parsed.premise_issue
        return VerifierResult(
            verdict="invalid_premise",
            can_continue=True,
            reason=issue.message if issue else "前提不成立。",
            coverage=coverage,
        ).as_dict()

    if not parsed.needs_retrieval:
        return VerifierResult(
            verdict="ok",
            can_continue=True,
            reason="该任务不需要文献检索。",
            coverage=coverage,
        ).as_dict()

    issue = parsed.premise_issue
    if issue and issue.severity == "needs_evidence" and not top_evidence_relevant:
        return VerifierResult(
            verdict="insufficient_evidence_for_direct_influence",
            can_continue=False,
            reason=issue.message,
            coverage=coverage,
        ).as_dict()

    if not top_evidence_relevant:
        return VerifierResult(
            verdict="low_relevance",
            can_continue=False,
            reason="证据相关性不足，无法支撑可靠回答。",
            coverage=coverage,
        ).as_dict()

    if coverage.required_terms and not coverage.covered_terms:
        return VerifierResult(
            verdict="entity_mismatch",
            can_continue=False,
            reason=(
                "检索结果未覆盖问题中的关键实体或技法，"
                f"缺失：{'、'.join(coverage.missing_terms[:4])}。"
            ),
            coverage=coverage,
        ).as_dict()

    if coverage.missing_terms:
        return VerifierResult(
            verdict="partial_coverage",
            can_continue=True,
            reason=(
                "证据可用于下一步，但部分关键词未直接覆盖："
                f"{'、'.join(coverage.missing_terms[:4])}。"
            ),
            coverage=coverage,
        ).as_dict()

    return VerifierResult(
        verdict="ok",
        can_continue=True,
        reason="证据可用于下一步。",
        coverage=coverage,
    ).as_dict()
