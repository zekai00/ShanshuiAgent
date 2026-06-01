"""LangGraph node implementations for the product web agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from .dependencies import WebAgentDependencies
from .events import emit_event, emit_node, emit_phase, emit_text
from .state import WebAgentState


BOUNDARY_TASKS = {"unsupported_image", "need_clarification", "unsupported_general", "invalid_premise"}


def make_intake_node(deps: WebAgentDependencies):
    def intake_node(state: WebAgentState) -> dict[str, Any]:
        question = state["question"].strip()
        history = state.get("history") or []
        emit_phase("理解任务")
        emit_node("intake", "running", "分析任务类型、领域、图像意图和错误前提")
        intake = deps.build_agent_intake(question, history)
        detail = f"{intake['task_type']} / {intake['route']['reason']}"
        emit_node("intake", "done", detail, intake)
        return {"intake": intake, "task_type": intake["task_type"], "evidence": []}

    return intake_node


def make_planner_node(deps: WebAgentDependencies):
    def planner_node(state: WebAgentState) -> dict[str, Any]:
        intake = state["intake"]
        emit_node("planner", "running", "根据任务类型选择需要执行的节点")
        plan = deps.build_agent_plan(intake)
        emit_event({"type": "plan", "steps": plan})
        emit_node("planner", "done", f"生成 {len(plan)} 步计划", plan)
        return {"plan": plan}

    return planner_node


def make_researcher_node(deps: WebAgentDependencies):
    def researcher_node(state: WebAgentState) -> dict[str, Any]:
        question = state["question"]
        emit_phase("检索中")
        emit_node("researcher", "running", "调用 Milvus/evidence store 检索并重排")
        retriever = deps.get_retriever(int(state.get("top_k") or 15), int(state.get("final_k") or 5))
        results = retriever.retrieve_and_rerank(question)
        evidence = [deps.evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
        emit_event({"type": "evidence", "evidence": evidence})
        emit_node("researcher", "done", f"返回 {len(evidence)} 条证据")
        return {"evidence": evidence}

    return researcher_node


def make_verifier_node(deps: WebAgentDependencies):
    def verifier_node(state: WebAgentState) -> dict[str, Any]:
        intake = state.get("intake") or {}
        evidence = state.get("evidence") or []
        task_type = state.get("task_type") or intake.get("task_type")
        emit_phase("核验证据")
        emit_node("verifier", "running", "检查相关性、错误前提和是否需要拒答")
        if task_type == "unsupported_image":
            verifier = {
                "verdict": "unsupported_image",
                "can_continue": True,
                "reason": "图像请求不在中国山水画创作范围内。",
            }
        elif task_type in {"need_clarification", "unsupported_general"}:
            verifier = {
                "verdict": str(task_type),
                "can_continue": True,
                "reason": "问题需要澄清或不属于研究范围。",
            }
        elif task_type == "invalid_premise":
            issue = intake.get("premise_issue") or {}
            verifier = {
                "verdict": "invalid_premise",
                "can_continue": True,
                "reason": issue.get("message", "前提不成立。"),
            }
        elif (
            intake.get("premise_issue")
            and intake["premise_issue"].get("severity") == "needs_evidence"
            and not deps.evidence_is_relevant(evidence)
        ):
            verifier = {
                "verdict": "insufficient_evidence_for_direct_influence",
                "can_continue": False,
                "reason": intake["premise_issue"]["message"],
            }
        elif intake.get("needs_retrieval") and not deps.evidence_is_relevant(evidence):
            verifier = {
                "verdict": "low_relevance",
                "can_continue": False,
                "reason": "证据相关性不足，无法支撑可靠回答。",
            }
        else:
            verifier = {"verdict": "ok", "can_continue": True, "reason": "证据可用于下一步。"}
        emit_node("verifier", "done", verifier["reason"], verifier)
        return {"verifier": verifier}

    return verifier_node


def make_research_synthesizer_node(deps: WebAgentDependencies):
    def research_synthesizer_node(state: WebAgentState) -> dict[str, Any]:
        question = state["question"]
        evidence = state.get("evidence") or []
        intake = state.get("intake") or {}
        emit_phase("整理卷宗")
        emit_node("research_synthesizer", "running", "把证据整理成给画师使用的研究约束")
        brief = deps.synthesize_research_brief(question, evidence, intake)
        emit_event({"type": "brief", "brief": brief})
        emit_node("research_synthesizer", "done", f"提炼 {len(brief.get('visual_constraints', []))} 条视觉约束", brief)
        return {"brief": brief}

    return research_synthesizer_node


def make_prompt_designer_node(deps: WebAgentDependencies):
    def prompt_designer_node(state: WebAgentState) -> dict[str, Any]:
        question = state["question"]
        brief = state.get("brief") or {}
        emit_phase("设计 Prompt")
        emit_node("prompt_designer", "running", "生成英文 positive prompt、negative prompt 和尺寸")
        image_spec = deps.design_image_spec(question, brief)
        emit_event({"type": "image_spec", "spec": image_spec})
        emit_node(
            "prompt_designer",
            "done",
            f"{image_spec.get('width')}x{image_spec.get('height')} / {image_spec.get('format')}",
            image_spec,
        )
        return {"image_spec": image_spec}

    return prompt_designer_node


def make_image_generator_node(deps: WebAgentDependencies):
    def image_generator_node(state: WebAgentState) -> dict[str, Any]:
        image_spec = state.get("image_spec") or {}
        emit_phase("生成图像")
        emit_node("image_generator", "running", "调用 ComfyUI 工作流")
        image_result = deps.generate_image_with_comfyui(image_spec)
        emit_event({"type": "image", "image": image_result})
        detail = "图像生成成功" if image_result.get("status") == "success" else str(image_result.get("message") or "图像未生成")
        emit_node("image_generator", "done", detail, image_result)
        return {"image_result": image_result}

    return image_generator_node


def make_image_critic_node(deps: WebAgentDependencies):
    def image_critic_node(state: WebAgentState) -> dict[str, Any]:
        image_result = state.get("image_result") or {}
        image_spec = state.get("image_spec") or {}
        emit_phase("检查图像")
        emit_node("image_critic", "running", "检查图像文件、生成状态和 prompt 约束")
        critic = deps.critic_image_result(image_result, image_spec)
        detail = "通过" if critic.get("passed") else "；".join(critic.get("issues") or ["未通过"])
        emit_event({"type": "image_critic", "critic": critic})
        emit_node("image_critic", "done", detail, critic)
        return {"critic": critic}

    return image_critic_node


def make_final_writer_node(deps: WebAgentDependencies):
    def final_writer_node(state: WebAgentState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history") or []
        intake = state.get("intake") or {}
        evidence = state.get("evidence") or []
        verifier = state.get("verifier") or {"can_continue": True, "verdict": "ok"}
        task_type = state.get("task_type") or intake.get("task_type")

        if task_type == "direct":
            emit_node("final_writer", "running", "直接回复，不启动文献检索")
            answer = deps.non_research_answer(question, intake.get("route"))
            emit_text(answer)
            emit_node("final_writer", "done", "已完成直接回复")
            emit_event({"type": "done", "mode": "direct_agent"})
            return {"final_answer": answer, "mode": "direct_agent"}

        if task_type == "general_art_qa":
            emit_node("final_writer", "running", "回答一般中国绘画史问题，不启动文献检索")
            answer = deps.direct_art_answer(question, history)
            emit_text(answer)
            emit_node("final_writer", "done", "已完成一般美术史回答")
            emit_event({"type": "done", "mode": "general_art_qa"})
            return {"final_answer": answer, "mode": "general_art_qa"}

        if task_type == "unsupported_image":
            emit_node("final_writer", "running", "给出边界说明")
            answer = deps.unsupported_image_answer(question)
            emit_text(answer)
            emit_node("final_writer", "done", "已完成边界回复")
            emit_event({"type": "done", "mode": "unsupported_image"})
            return {"final_answer": answer, "mode": "unsupported_image"}

        if task_type in {"need_clarification", "unsupported_general"}:
            emit_node("final_writer", "running", "给出澄清建议")
            answer = deps.non_research_answer(question, intake.get("route"))
            emit_text(answer)
            emit_node("final_writer", "done", "已完成澄清回复")
            emit_event({"type": "done", "mode": task_type})
            return {"final_answer": answer, "mode": str(task_type)}

        if task_type == "invalid_premise":
            emit_node("final_writer", "running", "纠正错误前提")
            answer = deps.premise_answer(question, intake.get("premise_issue") or {})
            emit_text(answer)
            emit_node("final_writer", "done", "已完成前提纠偏")
            emit_event({"type": "done", "mode": "invalid_premise"})
            return {"final_answer": answer, "mode": "invalid_premise"}

        if not verifier.get("can_continue"):
            emit_node("final_writer", "running", "说明证据不足或前提风险")
            if verifier.get("verdict") == "low_relevance":
                answer = deps.low_relevance_answer(question, evidence)
            else:
                answer = (
                    f"当前资料库证据不足以支持这个直接影响关系：{verifier.get('reason')}\n\n"
                    "我不会默认该前提成立。建议改问具体的比较问题，例如比较构图、色彩或笔触，而不是直接师承关系。"
                )
            emit_text(answer)
            emit_node("final_writer", "done", "已完成证据不足回复")
            mode = str(verifier.get("verdict") or "verification_failed")
            emit_event({"type": "done", "mode": mode})
            return {"final_answer": answer, "mode": mode}

        if task_type == "research_qa":
            emit_phase("生成回答")
            emit_node("final_writer", "running", "基于核验后的证据生成研究回答")
            answer_parts: list[str] = []
            for delta in deps.stream_answer_deltas(question, evidence, history):
                if delta:
                    answer_parts.append(delta)
                    emit_event({"type": "delta", "delta": delta})
            answer = "".join(answer_parts)
            emit_node("final_writer", "done", "已完成研究回答")
            emit_event({"type": "done", "mode": "research_qa"})
            return {"final_answer": answer, "mode": "research_qa"}

        emit_phase("组织结果")
        emit_node("final_writer", "running", "交付图像、研究依据、Prompt 和来源")
        final_answer = deps.build_image_final_answer(
            question,
            state.get("brief") or {},
            state.get("image_spec") or {},
            state.get("image_result") or {},
            state.get("critic") or {},
            evidence,
        )
        emit_text(final_answer)
        emit_node("final_writer", "done", "已完成研究创作交付")
        return {"final_answer": final_answer, "mode": "research_then_image"}

    return final_writer_node


def make_memory_writer_node(deps: WebAgentDependencies):
    def memory_writer_node(state: WebAgentState) -> dict[str, Any]:
        question = state["question"]
        user_id = state.get("user_id") or "guest"
        emit_node("memory_writer", "running", "仅记录明确表达的用户偏好")
        memory_result = deps.maybe_write_memory(user_id, question)
        detail = "已写入偏好" if memory_result.get("saved") else "无可写入偏好或 guest 用户"
        emit_event({"type": "memory", "memory": memory_result})
        emit_node("memory_writer", "done", detail, memory_result)
        emit_event({"type": "done", "mode": "research_then_image"})
        return {"memory_result": memory_result}

    return memory_writer_node


def route_after_planner(state: WebAgentState) -> str:
    task_type = state.get("task_type") or (state.get("intake") or {}).get("task_type")
    if task_type in {"direct", "general_art_qa"}:
        return "final_writer"
    if task_type in BOUNDARY_TASKS:
        return "verifier"
    return "researcher"


def route_after_verifier(state: WebAgentState) -> str:
    verifier = state.get("verifier") or {}
    if not verifier.get("can_continue", True):
        return "final_writer"
    task_type = state.get("task_type") or (state.get("intake") or {}).get("task_type")
    if task_type == "research_then_image":
        return "research_synthesizer"
    return "final_writer"


def route_after_final_writer(state: WebAgentState) -> str:
    task_type = state.get("task_type") or (state.get("intake") or {}).get("task_type")
    verifier = state.get("verifier") or {"can_continue": True}
    if task_type == "research_then_image" and verifier.get("can_continue", True):
        return "memory_writer"
    return END
