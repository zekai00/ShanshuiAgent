"""Graph builder for the LangGraph-backed web agent."""

from __future__ import annotations

import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config import WEB_AGENT_CHECKPOINT_DB, ensure_runtime_dirs

from .dependencies import WebAgentDependencies
from .nodes import (
    make_final_writer_node,
    make_image_critic_node,
    make_image_generator_node,
    make_intake_node,
    make_memory_writer_node,
    make_planner_node,
    make_prompt_designer_node,
    make_research_synthesizer_node,
    make_researcher_node,
    make_verifier_node,
    route_after_final_writer,
    route_after_planner,
    route_after_verifier,
)
from .state import WebAgentState


_compiled_graph = None
_checkpoint_connection: sqlite3.Connection | None = None
_compiled_deps_id: int | None = None
_compile_lock = threading.Lock()


def build_web_agent_graph(deps: WebAgentDependencies, checkpointer=None):
    builder = StateGraph(WebAgentState)

    builder.add_node("intake", make_intake_node(deps))
    builder.add_node("planner", make_planner_node(deps))
    builder.add_node("researcher", make_researcher_node(deps))
    builder.add_node("verifier", make_verifier_node(deps))
    builder.add_node("research_synthesizer", make_research_synthesizer_node(deps))
    builder.add_node("prompt_designer", make_prompt_designer_node(deps))
    builder.add_node("image_generator", make_image_generator_node(deps))
    builder.add_node("image_critic", make_image_critic_node(deps))
    builder.add_node("final_writer", make_final_writer_node(deps))
    builder.add_node("memory_writer", make_memory_writer_node(deps))

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "final_writer": "final_writer",
            "verifier": "verifier",
            "researcher": "researcher",
        },
    )
    builder.add_edge("researcher", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "final_writer": "final_writer",
            "research_synthesizer": "research_synthesizer",
        },
    )
    builder.add_edge("research_synthesizer", "prompt_designer")
    builder.add_edge("prompt_designer", "image_generator")
    builder.add_edge("image_generator", "image_critic")
    builder.add_edge("image_critic", "final_writer")
    builder.add_conditional_edges(
        "final_writer",
        route_after_final_writer,
        {
            "memory_writer": "memory_writer",
            END: END,
        },
    )
    builder.add_edge("memory_writer", END)

    return builder.compile(checkpointer=checkpointer, name="web_agent")


def get_web_agent_graph(deps: WebAgentDependencies):
    global _compiled_graph, _checkpoint_connection, _compiled_deps_id
    deps_id = id(deps)
    with _compile_lock:
        if _compiled_graph is None or _compiled_deps_id != deps_id:
            if _checkpoint_connection is not None:
                _checkpoint_connection.close()
            ensure_runtime_dirs()
            WEB_AGENT_CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
            _checkpoint_connection = sqlite3.connect(str(WEB_AGENT_CHECKPOINT_DB), check_same_thread=False)
            checkpointer = SqliteSaver(_checkpoint_connection)
            _compiled_graph = build_web_agent_graph(deps, checkpointer=checkpointer)
            _compiled_deps_id = deps_id
    return _compiled_graph
