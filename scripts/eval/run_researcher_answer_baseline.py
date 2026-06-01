#!/usr/bin/env python3
"""Run answer-level baseline for the Researcher agent.

The script assumes `scripts/run_retrieval_server.py` is already serving
POST /retrieve on localhost. It calls the current researcher node directly,
so the run includes the real prompt, tool binding, and ReAct loop used by
the application.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DATA_DIR, DOCS_DIR
from src.agent.main_nodes import researcher_node  # noqa: E402

DEFAULT_INPUT = DATA_DIR / "eval" / "test_researcher_v1.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "eval" / "baseline_researcher_answer_v1.jsonl"
DEFAULT_REPORT = DOCS_DIR / "研究员回答基线评测报告.md"
DEFAULT_MANUAL_OVERRIDES = DATA_DIR / "eval" / "researcher_answer_manual_overrides_v1.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {str(row.get("id")): row for row in rows if row.get("id")}


def load_manual_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manual overrides must be a JSON object: {path}")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def message_to_dict(message: Any) -> dict[str, Any]:
    record = {
        "type": message.__class__.__name__,
        "name": getattr(message, "name", None),
        "content": str(getattr(message, "content", "")),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        record["tool_calls"] = tool_calls
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = getattr(message, "tool_call_id", None)
    return record


SOURCE_PATTERN = re.compile(r"来源文献:\s*(.+?\.pdf)")
CITATION_PATTERN = re.compile(r"\[文献:\s*《(.+?\.pdf)》\]")


def normalize_source_name(name: str) -> str:
    return name.strip().replace("《", "").replace("》", "")


def extract_tool_text(messages: list[Any]) -> str:
    return "\n".join(str(msg.content) for msg in messages if isinstance(msg, ToolMessage))


def extract_sources_from_tool(tool_text: str) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for match in SOURCE_PATTERN.finditer(tool_text):
        source = normalize_source_name(match.group(1))
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def extract_citations(answer: str) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for match in CITATION_PATTERN.finditer(answer):
        source = normalize_source_name(match.group(1))
        if source not in seen:
            seen.add(source)
            citations.append(source)
    return citations


def count_tool_calls(messages: list[Any]) -> int:
    count = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            count += len(getattr(msg, "tool_calls", []) or [])
    return count


def heuristic_score(item: dict[str, Any], answer: str, tool_text: str) -> dict[str, Any]:
    retrieved_sources = extract_sources_from_tool(tool_text)
    citations = extract_citations(answer)
    gold_sources = [normalize_source_name(x) for x in item.get("gold_sources", [])]
    expected_entities = [str(x) for x in item.get("expected_entities", [])]

    retrieved_source_set = set(retrieved_sources)
    citation_set = set(citations)
    gold_source_set = set(gold_sources)

    expected_entity_hits = [entity for entity in expected_entities if entity and entity in answer]
    gold_source_hit = bool(gold_source_set & retrieved_source_set) if gold_source_set else False
    cited_gold_source_hit = bool(gold_source_set & citation_set) if gold_source_set else False
    cited_retrieved_source_hit = bool(citation_set & retrieved_source_set) if citation_set else False

    refusal_terms = [
        "不能说明",
        "无法说明",
        "资料不足",
        "没有证据",
        "未显示",
        "不成立",
        "错误前提",
        "并非",
        "不能直接",
        "不应",
    ]
    false_premise_handled = any(term in answer for term in refusal_terms)

    return {
        "answer_chars": len(answer),
        "tool_calls": count_tool_calls_from_text(tool_text),
        "retrieved_sources": retrieved_sources,
        "citations": citations,
        "has_citation": bool(citations),
        "gold_source_hit": gold_source_hit,
        "cited_gold_source_hit": cited_gold_source_hit,
        "cited_retrieved_source_hit": cited_retrieved_source_hit,
        "expected_entity_hits": expected_entity_hits,
        "expected_entity_hit_count": len(expected_entity_hits),
        "expected_entity_total": len(expected_entities),
        "false_premise_handled": false_premise_handled,
    }


def count_tool_calls_from_text(tool_text: str) -> int:
    return tool_text.count("【画院档案库检索结果】")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_answer(
    client: OpenAI,
    item: dict[str, Any],
    answer: str,
    tool_text: str,
    heuristic: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    context_excerpt = tool_text[:6000]
    prompt = f"""
你是中国山水画 RAG 系统的严格评测员。请只输出 JSON，不要 Markdown。

请根据【问题】、【期望实体】、【金标来源】、【检索证据摘录】和【Researcher 回答】判断回答质量。

评分维度均为 0、1、2：
- answer_relevance: 是否正面回答问题。
- evidence_grounding: 是否主要基于检索证据，是否避免脱离证据发挥。
- citation_quality: 是否按 [文献: 《xxx.pdf》] 引用，且引用与回答内容匹配。
- factual_consistency: 是否存在明显事实错误、时代错乱或张冠李戴。
- false_premise_handling: 如果 task_type 是 evidence_missing，是否能识别错误前提/证据不足；否则填 null。

overall_pass 只有在回答相关、证据支撑、引用基本合格、无明显事实错误时才为 true。
failure_type 只能取：
none, retrieval, answer_generation, citation, false_premise, insufficient_evidence, format。

【题目 ID】{item.get("id")}
【task_type】{item.get("task_type")}
【问题】{item.get("question")}
【期望实体】{item.get("expected_entities", [])}
【金标来源】{item.get("gold_sources", [])}
【启发式信息】{json.dumps(heuristic, ensure_ascii=False)}
【检索证据摘录】
{context_excerpt}

【Researcher 回答】
{answer}

请输出如下 JSON：
{{
  "answer_relevance": 0,
  "evidence_grounding": 0,
  "citation_quality": 0,
  "factual_consistency": 0,
  "false_premise_handling": null,
  "overall_pass": false,
  "failure_type": "none",
  "comments": "不超过80字"
}}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严格、保守、只输出 JSON 的自动评测员。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return parse_json_object(response.choices[0].message.content or "{}")


def run_one(item: dict[str, Any], judge_client: OpenAI | None, judge_model: str) -> dict[str, Any]:
    question = item["question"]
    state = {
        "messages": [HumanMessage(content=question)],
        "sender": "user",
        "summary": "",
        "long_term_memory": "",
        "user_id": "eval",
    }
    result = researcher_node(state)
    messages = result.get("messages", [])
    answer = str(result.get("research_dossier", "")).strip()
    tool_text = extract_tool_text(messages)
    heuristic = heuristic_score(item, answer, tool_text)

    row = {
        "id": item["id"],
        "task_type": item.get("task_type"),
        "question": question,
        "gold_sources": item.get("gold_sources", []),
        "expected_entities": item.get("expected_entities", []),
        "answer": answer,
        "heuristic": heuristic,
        "messages": [message_to_dict(msg) for msg in messages],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if judge_client is not None:
        try:
            row["judge"] = judge_answer(judge_client, item, answer, tool_text, heuristic, judge_model)
        except Exception as exc:
            row["judge_error"] = str(exc)

    return row


def effective_judge(row: dict[str, Any]) -> dict[str, Any]:
    manual = row.get("manual_review")
    if isinstance(manual, dict):
        return manual
    judge = row.get("judge")
    if isinstance(judge, dict):
        return judge
    return {}


def apply_manual_overrides(rows: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not overrides:
        return rows
    patched: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        override = overrides.get(str(row.get("id")))
        if override:
            row["manual_review"] = override
        patched.append(row)
    return patched


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if "error" not in row]
    judged = [row for row in attempted if effective_judge(row)]
    pass_rows = [row for row in judged if effective_judge(row).get("overall_pass") is True]

    by_task: dict[str, dict[str, Any]] = {}
    for task, task_rows_iter in group_by(attempted, "task_type").items():
        task_rows = list(task_rows_iter)
        task_judged = [row for row in task_rows if effective_judge(row)]
        by_task[task] = {
            "n": len(task_rows),
            "judged": len(task_judged),
            "pass": sum(1 for row in task_judged if effective_judge(row).get("overall_pass") is True),
            "citation_rate": rate(sum(1 for row in task_rows if row.get("heuristic", {}).get("has_citation")), len(task_rows)),
            "gold_source_hit_rate": rate(sum(1 for row in task_rows if row.get("heuristic", {}).get("gold_source_hit")), len(task_rows)),
            "avg_entity_hits": avg([row.get("heuristic", {}).get("expected_entity_hit_count", 0) for row in task_rows]),
        }

    failure_types = Counter(
        effective_judge(row).get("failure_type", "unjudged")
        for row in attempted
        if not effective_judge(row).get("overall_pass")
    )

    return {
        "attempted": len(attempted),
        "total": len(rows),
        "judged": len(judged),
        "pass": len(pass_rows),
        "pass_rate": rate(len(pass_rows), len(judged)),
        "citation_rate": rate(sum(1 for row in attempted if row.get("heuristic", {}).get("has_citation")), len(attempted)),
        "gold_source_hit_rate": rate(sum(1 for row in attempted if row.get("heuristic", {}).get("gold_source_hit")), len(attempted)),
        "avg_entity_hits": avg([row.get("heuristic", {}).get("expected_entity_hit_count", 0) for row in attempted]),
        "by_task": by_task,
        "failure_types": dict(failure_types),
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return grouped


def rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


def avg(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def score_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_DIR))
    except ValueError:
        return str(path)


def write_report(rows: list[dict[str, Any]], output_path: Path, baseline_path: Path, input_path: Path, title: str) -> None:
    summary = summarize_rows(rows)
    failed = [
        row
        for row in rows
        if effective_judge(row) and effective_judge(row).get("overall_pass") is not True
    ]
    failed = sorted(
        failed,
        key=lambda row: (
            str(effective_judge(row).get("failure_type", "")),
            str(row.get("id", "")),
        ),
    )
    manual_count = sum(1 for row in rows if isinstance(row.get("manual_review"), dict))

    lines = [
        f"# {title}",
        "",
        f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 测试集：`{display_path(input_path)}`",
        f"- 逐题结果：`{display_path(baseline_path)}`",
        "- 被测对象：当前 `researcher_node`，模型为 `deepseek-v4-flash`，工具为 `search_landscape_literature`。",
        f"- 说明：该评测使用自动启发式统计和 LLM-as-judge，并对明显误判做人工覆盖 {manual_count} 条；结论用于工程决策，关键论文结论仍建议人工复核。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 完成题数 | {summary['attempted']}/{summary['total']} |",
        f"| LLM 裁判覆盖题数 | {summary['judged']} |",
        f"| 自动通过率 | {summary['pass']}/{summary['judged']} ({summary['pass_rate']:.2f}%) |",
        f"| 回答中出现规范文献引用比例 | {summary['citation_rate']:.2f}% |",
        f"| 检索来源命中金标比例 | {summary['gold_source_hit_rate']:.2f}% |",
        f"| 平均期望实体命中数 | {summary['avg_entity_hits']:.2f} |",
        "",
        "## 分任务结果",
        "",
        "| 任务类型 | 题数 | 通过 | 引用率 | 金标来源命中率 | 平均实体命中 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for task, stats in summary["by_task"].items():
        lines.append(
            f"| `{task}` | {stats['n']} | {stats['pass']}/{stats['judged']} | "
            f"{stats['citation_rate']:.2f}% | {stats['gold_source_hit_rate']:.2f}% | "
            f"{stats['avg_entity_hits']:.2f} |"
        )

    lines.extend(["", "## 失败类型", "", "| 类型 | 数量 |", "| --- | ---: |"])
    for failure_type, count in sorted(summary["failure_types"].items()):
        lines.append(f"| `{failure_type}` | {count} |")

    lines.extend(["", "## 典型失败样例", ""])
    if not failed:
        lines.append("本次自动评测未发现失败样例。")
    else:
        for row in failed[:10]:
            judge = effective_judge(row)
            answer_preview = str(row.get("answer", "")).replace("\n", " ")[:220]
            lines.extend(
                [
                    f"### {row.get('id')} `{row.get('task_type')}`",
                    "",
                    f"- 问题：{row.get('question')}",
                    f"- 失败类型：`{judge.get('failure_type')}`",
                    f"- 裁判意见：{judge.get('comments')}",
                    f"- 回答摘录：{answer_preview}",
                    "",
                ]
            )

    lines.extend(["## 训练策略判断", ""])
    lines.extend(make_training_decision(summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def make_training_decision(summary: dict[str, Any]) -> list[str]:
    failures = Counter(summary.get("failure_types", {}))
    pass_rate = float(summary.get("pass_rate", 0.0))
    citation_rate = float(summary.get("citation_rate", 0.0))
    source_rate = float(summary.get("gold_source_hit_rate", 0.0))

    lines = []
    if source_rate < 80:
        lines.append(
            "1. 当前首要问题是检索覆盖不足，应先改检索数据、切片、召回和图谱，不建议立刻用 SFT/DPO 掩盖证据缺口。"
        )
    elif citation_rate < 90 or failures.get("citation", 0) > 0:
        lines.append(
            "1. 当前首要训练方向是 SFT：让本地小模型稳定学习“检索证据输入 -> 有边界的中文回答 -> 规范引用”的格式。"
        )
    elif failures.get("false_premise", 0) > 0 or failures.get("insufficient_evidence", 0) > 0:
        lines.append(
            "1. 当前需要补充 SFT 拒答/纠错样本，训练模型在证据不足、时代错乱、错误前提问题上先说明边界。"
        )
    elif pass_rate >= 85:
        lines.append(
            "1. 当前系统回答质量已经达到可用基线，训练应服务于本地化替代和风格稳定；DPO 只适合在 SFT 后做偏好收敛。"
        )
    else:
        lines.append(
            "1. 当前仍建议先做 SFT，而不是 DPO/PPO/GRPO；失败主要需要明确示范答案，而不是在线强化学习。"
        )

    lines.extend(
        [
            "2. 暂不建议 PPO：本系统没有在线奖励模型和 rollout 基础设施，成本高且目标不清晰。",
            "3. 暂不建议直接 GRPO：只有在后续把“引用是否存在、答案是否由证据支持、错误前提是否拒答”形式化成可验证 reward 后，才值得尝试。",
            "4. DPO 可作为第二阶段：当 SFT 模型已经会答，但仍在引用严谨性、拒答边界、啰嗦程度上有偏好问题时，再构造 chosen/rejected 对进行 DPO。",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--manual-overrides", default=str(DEFAULT_MANUAL_OVERRIDES))
    parser.add_argument("--title", default="研究员回答基线评测报告")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", default=os.environ.get("CL_EVAL_JUDGE_MODEL", "local-chat-model"))
    args = parser.parse_args()

    load_dotenv(WORKSPACE_DIR / ".env")

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    items = load_jsonl(input_path)
    if args.limit is not None:
        items = items[: args.limit]

    existing = load_existing(output_path) if args.resume else {}
    manual_overrides = load_manual_overrides(Path(args.manual_overrides))
    if not args.resume and output_path.exists():
        output_path.unlink()

    judge_client = None
    if not args.no_judge:
        api_key = os.environ.get("CL_LLM_API_KEY")
        if not api_key:
            print("[!] 未找到 CL_LLM_API_KEY，本次只做启发式评测。")
        else:
            judge_client = OpenAI(api_key=api_key, base_url=os.environ.get("CL_LLM_BASE_URL", "http://localhost:8000/v1"))

    rows: list[dict[str, Any]] = []
    rows.extend(existing.values())

    for index, item in enumerate(items, start=1):
        item_id = str(item.get("id"))
        if item_id in existing:
            print(f"[{index}/{len(items)}] skip {item_id}")
            continue

        print(f"[{index}/{len(items)}] run {item_id}: {item['question']}")
        try:
            row = run_one(item, judge_client, args.judge_model)
        except Exception as exc:
            row = {
                "id": item_id,
                "task_type": item.get("task_type"),
                "question": item.get("question"),
                "error": str(exc),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        append_jsonl(output_path, row)
        rows.append(row)

    report_rows = apply_manual_overrides(rows, manual_overrides)
    write_report(report_rows, report_path, output_path, input_path, args.title)
    summary = summarize_rows(report_rows)
    print(f"Wrote answers to {output_path}")
    print(f"Wrote report to {report_path}")
    print(
        f"Attempted {summary['attempted']}/{summary['total']}; "
        f"pass {summary['pass']}/{summary['judged']} ({summary['pass_rate']:.2f}%)."
    )


if __name__ == "__main__":
    main()
