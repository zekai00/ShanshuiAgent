"""Streaming event helpers shared by LangGraph web-agent nodes."""

from __future__ import annotations

import json
from typing import Any

from langgraph.config import get_stream_writer


AGENT_NODE_TITLES = {
    "intake": "任务理解",
    "planner": "计划制定",
    "researcher": "文献检索",
    "verifier": "证据核验",
    "research_synthesizer": "研究卷宗",
    "prompt_designer": "图像提示词",
    "image_generator": "图像生成",
    "image_critic": "图像检查",
    "final_writer": "最终回复",
    "memory_writer": "记忆写入",
}


def event_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def emit_event(payload: dict[str, Any]) -> None:
    """Emit a custom LangGraph stream event when a writer is available."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(payload)


def emit_phase(phase: str) -> None:
    emit_event({"type": "phase", "phase": phase})


def node_event(node: str, status: str, detail: str = "", data: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "node",
        "node": node,
        "title": AGENT_NODE_TITLES.get(node, node),
        "status": status,
        "detail": detail,
    }
    if data is not None:
        payload["data"] = data
    return payload


def emit_node(node: str, status: str, detail: str = "", data: Any | None = None) -> None:
    emit_event(node_event(node, status, detail, data))


def text_chunks(text: str, size: int = 28):
    for index in range(0, len(text), size):
        yield text[index:index + size]


def emit_text(answer: str) -> None:
    for chunk in text_chunks(answer):
        emit_event({"type": "delta", "delta": chunk})
