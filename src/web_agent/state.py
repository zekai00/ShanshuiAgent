"""State schema for the LangGraph-backed web agent."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class WebAgentState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    top_k: int
    final_k: int
    user_id: str
    thread_id: str

    intake: dict[str, Any]
    plan: list[dict[str, str]]
    task_type: str
    evidence: list[dict[str, Any]]
    verifier: dict[str, Any]
    brief: dict[str, Any]
    image_spec: dict[str, Any]
    image_result: dict[str, Any]
    critic: dict[str, Any]
    final_answer: str
    memory_result: dict[str, Any]
    mode: str
