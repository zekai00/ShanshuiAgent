from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.web_agent.dependencies import WebAgentDependencies
from src.web_agent.graph import build_web_agent_graph
from src.web_agent.verifier import verify_agent_state


class FakeRetriever:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def retrieve_and_rerank(self, question: str) -> list[dict[str, Any]]:
        return self.docs


def route_payload(label: str = "domain_research") -> dict[str, Any]:
    return {"label": label, "reason": "test", "confidence": 1.0, "source": "test"}


def intake_payload(
    task_type: str,
    *,
    entities: dict[str, list[str]] | None = None,
    premise_issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "route": route_payload("domain_research" if task_type.startswith("research") else task_type),
        "entities": entities or {"dynasties": [], "schools": [], "techniques": [], "artists": [], "works": []},
        "needs_retrieval": task_type in {"research_qa", "research_then_image"},
        "needs_image": task_type == "research_then_image",
        "needs_clarification": task_type == "need_clarification",
        "premise_issue": premise_issue,
        "router": "test",
    }


def evidence_payload(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": doc.get("chunk_id", f"chunk-{rank}"),
        "source_file": doc.get("source_file", "test.pdf"),
        "title": doc.get("title", "测试文献"),
        "page_start": doc.get("page_start", 1),
        "rerank_score": doc.get("rerank_score", 2.0),
        "preview": doc.get("preview", ""),
    }


def make_deps(
    *,
    intake: dict[str, Any],
    docs: list[dict[str, Any]] | None = None,
    relevant: bool = True,
) -> WebAgentDependencies:
    docs = docs or []

    def build_agent_plan(agent_intake: dict[str, Any]) -> list[dict[str, str]]:
        task_type = agent_intake["task_type"]
        if task_type == "direct":
            nodes = ["final_writer"]
        elif task_type == "research_then_image":
            nodes = [
                "researcher",
                "verifier",
                "research_synthesizer",
                "prompt_designer",
                "image_generator",
                "image_critic",
                "final_writer",
                "memory_writer",
            ]
        else:
            nodes = ["researcher", "verifier", "final_writer"]
        return [{"node": node, "title": node, "goal": f"run {node}"} for node in nodes]

    def stream_answer_deltas(question: str, evidence: list[dict[str, Any]], history: list[dict[str, str]]) -> Iterable[str]:
        yield "基于证据回答 [1]"

    return WebAgentDependencies(
        build_agent_intake=lambda question, history: intake,
        build_agent_plan=build_agent_plan,
        get_retriever=lambda top_k, final_k: FakeRetriever(docs),
        evidence_payload=evidence_payload,
        evidence_is_relevant=lambda evidence: relevant,
        non_research_answer=lambda question, route: "直接回答",
        direct_art_answer=lambda question, history: "一般美术史回答",
        unsupported_image_answer=lambda question: "不支持该图像任务",
        premise_answer=lambda question, issue: f"前提纠偏：{issue.get('message')}",
        low_relevance_answer=lambda question, evidence: "证据相关性不足",
        stream_answer_deltas=stream_answer_deltas,
        synthesize_research_brief=lambda question, evidence, intake: {
            "topic": question,
            "key_points": ["吴门画派重视江南经验 [1]"],
            "visual_constraints": ["江南水乡", "文人笔墨"],
            "citations": [1],
        },
        design_image_spec=lambda question, brief: {
            "format": "horizontal_scroll",
            "width": 1536,
            "height": 768,
            "positive_prompt": "Chinese shanshui landscape painting, ink wash, Jiangnan literati scene",
            "negative_prompt": "photorealistic, text, watermark",
            "style_notes": "江南水乡",
        },
        generate_image_with_comfyui=lambda spec: {
            "status": "queued",
            "job_id": "test-job",
            "seed": 123,
            "message": "图像任务已提交后台生成。",
        },
        critic_image_result=lambda image_result, image_spec, brief: {
            "passed": False,
            "issues": ["图像任务仍在后台生成，暂未执行视觉评审"],
            "retry_recommended": False,
            "rule": {"passed": False},
            "vlm": {"enabled": False},
        },
        build_image_final_answer=lambda question, brief, spec, image_result, critic, evidence: image_result["status"],
        maybe_write_memory=lambda user_id, question: {"saved": False, "reason": "guest_or_empty", "insights": {}},
    )


def invoke_graph(deps: WebAgentDependencies, question: str = "测试问题") -> dict[str, Any]:
    graph = build_web_agent_graph(deps)
    return graph.invoke({"question": question, "history": [], "top_k": 3, "final_k": 1, "user_id": "guest"})


def test_direct_task_short_circuits_to_final_writer() -> None:
    deps = make_deps(intake=intake_payload("direct"))
    result = invoke_graph(deps, "你好")
    assert result["mode"] == "direct_agent"
    assert result["final_answer"] == "直接回答"


def test_research_qa_uses_evidence_and_writes_answer() -> None:
    deps = make_deps(
        intake=intake_payload("research_qa", entities={"artists": ["董其昌"], "works": [], "techniques": [], "schools": [], "dynasties": []}),
        docs=[{"preview": "董其昌提出南北宗论，影响明清山水画论。", "rerank_score": 2.2}],
        relevant=True,
    )
    result = invoke_graph(deps, "董其昌南北宗论是什么？")
    assert result["mode"] == "research_qa"
    assert result["verifier"]["verdict"] == "ok"
    assert result["final_answer"] == "基于证据回答 [1]"


def test_low_relevance_research_qa_refuses_answer() -> None:
    deps = make_deps(
        intake=intake_payload("research_qa"),
        docs=[{"preview": "无关文本", "rerank_score": 0.1}],
        relevant=False,
    )
    result = invoke_graph(deps)
    assert result["mode"] == "low_relevance"
    assert result["final_answer"] == "证据相关性不足"


def test_research_then_image_can_return_queued_job_without_gpu() -> None:
    deps = make_deps(
        intake=intake_payload("research_then_image", entities={"schools": ["吴门"], "artists": [], "works": [], "techniques": [], "dynasties": []}),
        docs=[{"preview": "吴门画派常以江南园林溪桥入画。", "rerank_score": 2.0}],
        relevant=True,
    )
    result = invoke_graph(deps, "请根据吴门画派生成一幅江南山水长卷")
    assert result["mode"] == "research_then_image"
    assert result["image_result"]["status"] == "queued"
    assert result["memory_result"]["saved"] is False


def test_verifier_rejects_entity_mismatch_even_when_top_score_is_high() -> None:
    intake = intake_payload(
        "research_qa",
        entities={"artists": ["董其昌"], "works": [], "techniques": [], "schools": [], "dynasties": []},
    )
    verifier = verify_agent_state(
        question="董其昌南北宗论是什么？",
        intake=intake,
        evidence=[evidence_payload({"preview": "范宽以雄伟山体著称。", "rerank_score": 3.0}, 1)],
        top_evidence_relevant=True,
    )
    assert verifier["verdict"] == "entity_mismatch"
    assert verifier["can_continue"] is False
    assert verifier["coverage"]["missing_terms"] == ["董其昌"]
