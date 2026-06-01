"""Streaming adapter from LangGraph custom events to NDJSON."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator

from .dependencies import WebAgentDependencies
from .events import event_line
from .graph import get_web_agent_graph
from .state import WebAgentState


def make_thread_id(message: str, history: list[dict[str, str]], user_id: str, provided: str | None = None) -> str:
    if provided:
        return provided
    basis = f"{user_id}|{len(history)}|{message}|{time.time_ns()}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return f"web-agent:{user_id or 'guest'}:{digest}"


def stream_web_agent_events(
    *,
    message: str,
    history: list[dict[str, str]],
    top_k: int,
    final_k: int,
    user_id: str,
    deps: WebAgentDependencies,
    thread_id: str | None = None,
) -> Iterator[str]:
    run_thread_id = make_thread_id(message, history, user_id, thread_id)
    initial_state: WebAgentState = {
        "question": message.strip(),
        "history": history,
        "top_k": top_k,
        "final_k": final_k,
        "user_id": user_id or "guest",
        "thread_id": run_thread_id,
    }
    config = {"configurable": {"thread_id": run_thread_id}}
    yielded_done = False
    yield event_line({"type": "agent_run", "engine": "langgraph", "thread_id": run_thread_id})
    try:
        graph = get_web_agent_graph(deps)
        for payload in graph.stream(initial_state, config=config, stream_mode="custom", durability="sync"):
            if isinstance(payload, dict):
                if payload.get("type") == "done":
                    yielded_done = True
                yield event_line(payload)
        if not yielded_done:
            yield event_line({"type": "done", "mode": "langgraph_agent"})
    except Exception as exc:
        yield event_line({"type": "error", "message": str(exc)})
        yield event_line({"type": "done", "mode": "error"})
