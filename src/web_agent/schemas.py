"""Typed schemas for the LangGraph web-agent contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TaskType = Literal[
    "direct",
    "general_art_qa",
    "research_qa",
    "research_then_image",
    "unsupported_image",
    "need_clarification",
    "unsupported_general",
    "invalid_premise",
]


class FlexibleModel(BaseModel):
    """Base model that preserves extra fields from runtime adapters."""

    model_config = ConfigDict(extra="allow")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class RouteDecision(FlexibleModel):
    label: str
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "unknown"


class AgentEntities(FlexibleModel):
    dynasties: list[str] = Field(default_factory=list)
    schools: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    artists: list[str] = Field(default_factory=list)
    works: list[str] = Field(default_factory=list)


class PremiseIssue(FlexibleModel):
    kind: str = "unknown"
    severity: str
    message: str
    recommended_action: str = ""


class AgentIntake(FlexibleModel):
    task_type: TaskType
    route: RouteDecision
    entities: AgentEntities = Field(default_factory=AgentEntities)
    needs_retrieval: bool = False
    needs_image: bool = False
    needs_clarification: bool = False
    premise_issue: PremiseIssue | None = None
    router: str = "unknown"


class AgentPlanStep(FlexibleModel):
    node: str
    title: str
    goal: str


class EvidenceItem(FlexibleModel):
    rank: int
    chunk_id: str | None = None
    legacy_milvus_id: str | int | None = None
    source_file: str | None = None
    title: str | None = None
    page_start: int | str | None = None
    page_end: int | str | None = None
    page_count: int | str | None = None
    rerank_score: float | str | None = None
    evidence_store_hit: bool | None = None
    corrective_query: str | None = None
    source_prior_sources: list[Any] = Field(default_factory=list)
    pdf_url: str | None = None
    page_image_url: str | None = None
    preview: str = ""


class EvidenceCoverage(FlexibleModel):
    required_terms: list[str] = Field(default_factory=list)
    covered_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    top_score: float | None = None
    evidence_count: int = 0


class VerifierResult(FlexibleModel):
    verdict: str
    can_continue: bool
    reason: str
    coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)


class ResearchBrief(FlexibleModel):
    topic: str
    key_points: list[str] = Field(default_factory=list)
    visual_constraints: list[str] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _normalize_citations(cls, value: Any) -> list[int]:
        if not value:
            return []
        normalized: list[int] = []
        for item in value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                continue
        return normalized


class ImageSpec(FlexibleModel):
    format: str = "square"
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1024, ge=256, le=4096)
    positive_prompt: str
    negative_prompt: str = ""
    style_notes: str = ""


ImageStatus = Literal["queued", "running", "success", "failed", "cancelled"]


class ImageResult(FlexibleModel):
    status: ImageStatus
    message: str | None = None
    error_type: str | None = None
    path: str | None = None
    url: str | None = None
    filename: str | None = None
    seed: int | None = None
    workflow: str | None = None
    prompt_id: str | None = None
    job_id: str | None = None


class ImageCriticResult(FlexibleModel):
    passed: bool
    score: float | None = None
    issues: list[str] = Field(default_factory=list)
    retry_recommended: bool = False
    rule: dict[str, Any] = Field(default_factory=dict)
    vlm: dict[str, Any] = Field(default_factory=dict)


class MemoryResult(FlexibleModel):
    saved: bool = False
    reason: str | None = None
    error: str | None = None
    insights: dict[str, Any] = Field(default_factory=dict)
