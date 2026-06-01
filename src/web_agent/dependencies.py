"""Dependency contract for the LangGraph web agent.

The graph owns orchestration. Runtime capabilities such as retrieval, answer
generation, image generation, and memory writes are injected by the web server.
This keeps the graph package independent from the FastAPI script.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WebAgentDependencies:
    build_agent_intake: Callable[[str, list[dict[str, str]]], dict[str, Any]]
    build_agent_plan: Callable[[dict[str, Any]], list[dict[str, str]]]
    get_retriever: Callable[[int, int], Any]
    evidence_payload: Callable[[dict[str, Any], int], dict[str, Any]]
    evidence_is_relevant: Callable[[list[dict[str, Any]]], bool]
    non_research_answer: Callable[[str, dict[str, Any] | None], str]
    direct_art_answer: Callable[[str, list[dict[str, str]]], str]
    unsupported_image_answer: Callable[[str], str]
    premise_answer: Callable[[str, dict[str, Any]], str]
    low_relevance_answer: Callable[[str, list[dict[str, Any]]], str]
    stream_answer_deltas: Callable[[str, list[dict[str, Any]], list[dict[str, str]]], Iterable[str]]
    synthesize_research_brief: Callable[[str, list[dict[str, Any]], dict[str, Any]], dict[str, Any]]
    design_image_spec: Callable[[str, dict[str, Any]], dict[str, Any]]
    generate_image_with_comfyui: Callable[[dict[str, Any]], dict[str, Any]]
    critic_image_result: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    build_image_final_answer: Callable[
        [str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]],
        str,
    ]
    maybe_write_memory: Callable[[str, str], dict[str, Any]]
