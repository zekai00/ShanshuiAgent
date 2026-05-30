#!/usr/bin/env python3
"""Run retrieval-only baseline for researcher evaluation items.

The script assumes `scripts/run_retrieval_server.py` is already running and
serving POST /retrieve on localhost.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_INPUT = "/root/Workspace/ChineseLandscape/data/eval/test_researcher_v1.jsonl"
DEFAULT_OUTPUT = "/root/Workspace/ChineseLandscape/data/eval/baseline_retrieval_v1.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return items


def call_retrieval(endpoint: str, question: str, timeout: float) -> list[dict[str, Any]]:
    response = requests.post(endpoint, json={"query": question}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])


def infer_source_role(item: dict[str, Any]) -> str:
    explicit_role = item.get("source_role") or item.get("source_evaluation_mode")
    if explicit_role:
        return str(explicit_role)
    if item.get("task_type") == "evidence_missing":
        return "correction_evidence" if item.get("gold_sources") else "none"
    return "answer_evidence" if item.get("gold_sources") else "none"


def source_label(source_role: str) -> str:
    if source_role == "correction_evidence":
        return "纠错证据来源"
    if source_role == "answer_evidence":
        return "金标答案来源"
    return "无来源约束"


def score_item(item: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved_sources = [str(doc.get("source_file", "")) for doc in retrieved]
    combined_context = "\n".join(str(doc.get("raw_chunk_text") or doc.get("contextual_chunk", "")) for doc in retrieved)

    gold_sources = item.get("gold_sources", [])
    item_source_role = infer_source_role(item)
    expected_entities = item.get("expected_entities", [])
    retrieved_gold_sources = [source for source in gold_sources if source in retrieved_sources]
    source_hit = any(retrieved_gold_sources) if gold_sources else False
    all_gold_sources_hit = all(source in retrieved_sources for source in gold_sources) if gold_sources else False
    entity_hits = [entity for entity in expected_entities if entity and entity in combined_context]
    source_constraints = []
    for doc in retrieved:
        for source in doc.get("source_constraints", []) or []:
            if source not in source_constraints:
                source_constraints.append(source)

    return {
        "id": item["id"],
        "task_type": item["task_type"],
        "question": item["question"],
        "gold_sources": gold_sources,
        "evaluated_sources": gold_sources,
        "source_role": item_source_role,
        "source_label": source_label(item_source_role),
        "retrieved_sources": retrieved_sources,
        "retrieved_gold_sources": retrieved_gold_sources,
        "source_hit": source_hit,
        "all_gold_sources_hit": all_gold_sources_hit,
        "source_evaluated": bool(gold_sources),
        "expected_entities": expected_entities,
        "entity_hits": entity_hits,
        "entity_hit_count": len(entity_hits),
        "source_constraints": source_constraints,
        "source_constraint_match_count": sum(1 for doc in retrieved if doc.get("source_constraint_match")),
        "evidence_store_hit_count": sum(1 for doc in retrieved if doc.get("evidence_store_hit")),
        "retrieved_chunk_ids": [doc.get("chunk_id") for doc in retrieved],
        "top_results": [
            {
                "chunk_id": doc.get("chunk_id"),
                "legacy_milvus_id": doc.get("legacy_milvus_id") or doc.get("id"),
                "source_file": doc.get("source_file"),
                "title": doc.get("title"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "rerank_score": doc.get("rerank_score"),
                "evidence_store_hit": doc.get("evidence_store_hit"),
                "source_constraint_match": doc.get("source_constraint_match"),
                "context_preview": str(doc.get("raw_chunk_text") or doc.get("contextual_chunk", ""))[:300]
            }
            for doc in retrieved
        ],
        "review_status": item.get("review_status", "")
    }


def pct(numerator: int, denominator: int) -> str:
    return f"{(numerator / denominator * 100):.2f}%" if denominator else "N/A"


def write_report(rows: list[dict[str, Any]], report_path: Path, input_path: Path, output_path: Path) -> None:
    attempted = [row for row in rows if "error" not in row]
    errored = [row for row in rows if "error" in row]
    source_rows = [row for row in attempted if row.get("source_evaluated")]
    source_hits = sum(1 for row in source_rows if row.get("source_hit"))
    all_source_hits = sum(1 for row in source_rows if row.get("all_gold_sources_hit"))
    avg_entity_hits = (
        sum(int(row.get("entity_hit_count", 0)) for row in attempted) / len(attempted)
        if attempted else 0.0
    )
    evidence_docs = sum(len(row.get("top_results", [])) for row in attempted)
    evidence_store_hits = sum(int(row.get("evidence_store_hit_count", 0)) for row in attempted)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempted:
        by_task[str(row.get("task_type", ""))].append(row)

    lines = [
        "# 研究员检索评测报告",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## 输入输出",
        "",
        f"- 输入评测集：`{input_path}`",
        f"- 输出明细：`{output_path}`",
        "",
        "## 总览",
        "",
        f"- 总题数：{len(rows)}",
        f"- 成功请求：{len(attempted)}",
        f"- 请求失败：{len(errored)}",
        f"- 有来源约束题数：{len(source_rows)}",
        f"- 任一来源命中：{source_hits}/{len(source_rows)}（{pct(source_hits, len(source_rows))}）",
        f"- 全部来源命中：{all_source_hits}/{len(source_rows)}（{pct(all_source_hits, len(source_rows))}）",
        f"- 平均期望实体命中数：{avg_entity_hits:.2f}",
        f"- evidence store 命中证据块：{evidence_store_hits}/{evidence_docs}（{pct(evidence_store_hits, evidence_docs)}）",
        "",
        "## 口径说明",
        "",
        "- 普通事实/概念/比较题中的来源表示“金标答案来源”，用于判断检索是否取回可回答问题的文献。",
        "- 错误前提专项中的来源表示“纠错证据来源”，用于判断系统是否取回能支持纠错的文献，不表示这些来源支持错误前提。",
        "- 错误前提专项中 `gold_sources=[]` 的题目通常是时代错置、现代技术错置或常识性边界题，不纳入来源命中率，应主要评估回答是否开头纠错、拒绝顺着错误前提编造。",
        "",
        "## 分任务结果",
        "",
        "| 任务类型 | 题数 | 任一来源命中 | 全部来源命中 | 平均实体命中 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for task, task_rows in sorted(by_task.items()):
        task_source_rows = [row for row in task_rows if row.get("source_evaluated")]
        task_source_hits = sum(1 for row in task_source_rows if row.get("source_hit"))
        task_all_source_hits = sum(1 for row in task_source_rows if row.get("all_gold_sources_hit"))
        task_avg_entities = sum(int(row.get("entity_hit_count", 0)) for row in task_rows) / len(task_rows)
        lines.append(
            f"| {task} | {len(task_rows)} | "
            f"{task_source_hits}/{len(task_source_rows)}（{pct(task_source_hits, len(task_source_rows))}） | "
            f"{task_all_source_hits}/{len(task_source_rows)}（{pct(task_all_source_hits, len(task_source_rows))}） | "
            f"{task_avg_entities:.2f} |"
        )

    failures = [row for row in source_rows if not row.get("all_gold_sources_hit")]
    lines.extend([
        "",
        "## 来源未全命中的样本",
        "",
    ])
    if failures:
        for row in failures[:40]:
            lines.extend([
                f"- `{row.get('id')}` `{row.get('task_type')}`",
                f"  - 问题：{row.get('question')}",
                f"  - {row.get('source_label', '来源约束')}：{row.get('evaluated_sources', row.get('gold_sources'))}",
                f"  - 检索来源：{row.get('retrieved_sources')}",
            ])
    else:
        lines.append("所有有金标来源的样本均命中全部来源。")

    if errored:
        lines.extend(["", "## 请求失败样本", ""])
        for row in errored[:20]:
            lines.append(f"- `{row.get('id')}`：{row.get('error')}")

    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    items = load_jsonl(input_path)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['id']} {item['question']}")
        try:
            retrieved = call_retrieval(args.endpoint, item["question"], args.timeout)
            row = score_item(item, retrieved)
        except Exception as exc:
            row = {
                "id": item.get("id"),
                "task_type": item.get("task_type"),
                "question": item.get("question"),
                "error": str(exc),
                "source_hit": False,
                "entity_hit_count": 0
            }
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    attempted = [row for row in rows if "error" not in row]
    source_rows = [row for row in attempted if row.get("source_evaluated")]
    source_hits = sum(1 for row in source_rows if row.get("source_hit"))
    avg_entity_hits = (
        sum(int(row.get("entity_hit_count", 0)) for row in attempted) / len(attempted)
        if attempted else 0.0
    )
    print(f"Wrote baseline to {output_path}")
    print(f"Items attempted: {len(attempted)}/{len(rows)}")
    print(f"Source hit rate: {source_hits}/{len(source_rows)}")
    print(f"Average entity hits: {avg_entity_hits:.2f}")
    if args.report:
        report_path = Path(args.report)
        write_report(rows, report_path, input_path, output_path)
        print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
