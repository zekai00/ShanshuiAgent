#!/usr/bin/env python3
"""Calibrate query routing and RAG rerank thresholds without answer LLM calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DATA_DIR, DOCS_DIR  # noqa: E402

DEFAULT_INPUT = DATA_DIR / "eval" / "router_threshold_calibration_v1.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "eval" / "baseline_router_threshold_calibration_v1.jsonl"
DEFAULT_REPORT = DOCS_DIR / "路由与RAG阈值校准报告.md"
DEFAULT_RETRIEVE_URL = "http://127.0.0.1:7861/api/retrieve"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def accuracy(rows: list[dict[str, Any]], pred_key: str, gold_key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(pred_key) == row.get(gold_key)) / len(rows)


def confusion(rows: list[dict[str, Any]], pred_key: str, gold_key: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[str(row.get(gold_key))][str(row.get(pred_key))] += 1
    return {gold: dict(pred_counts) for gold, pred_counts in sorted(matrix.items())}


def import_rule_router(use_llm_router: bool):
    if not use_llm_router:
        os.environ["CL_ROUTER_LLM_ENABLED"] = "0"
    from scripts.run_web_app import route_question

    return route_question


def evaluate_rule_router(rows: list[dict[str, Any]], use_llm_router: bool) -> dict[str, Any]:
    route_question = import_rule_router(use_llm_router)
    evaluated = []
    for row in rows:
        route = route_question(row["query"])
        evaluated.append(
            {
                **row,
                "rule_route": route,
                "rule_pred_route": route["label"],
                "rule_should_retrieve": route["label"] == "domain_research",
            }
        )
    return {
        "rows": evaluated,
        "route_accuracy": accuracy(evaluated, "rule_pred_route", "expected_route"),
        "retrieve_accuracy": accuracy(evaluated, "rule_should_retrieve", "should_retrieve"),
        "confusion": confusion(evaluated, "rule_pred_route", "expected_route"),
        "false_rag": [
            row for row in evaluated
            if row["rule_should_retrieve"] and not row["should_retrieve"]
        ],
        "false_reject": [
            row for row in evaluated
            if not row["rule_should_retrieve"] and row["should_retrieve"]
        ],
    }


def evaluate_traditional_nlp_router(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    queries = [row["query"] for row in rows]
    labels = [row["expected_route"] for row in rows]
    counts = Counter(labels)
    n_splits = min(5, min(counts.values()))
    if n_splits < 2:
        return {"available": False, "error": "Not enough examples per class for cross validation."}

    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = cross_val_predict(pipeline, queries, labels, cv=splitter)
    label_order = sorted(counts)
    pred_rows = []
    for row, pred in zip(rows, predictions):
        pred_rows.append(
            {
                **row,
                "nlp_pred_route": str(pred),
                "nlp_should_retrieve": str(pred) == "domain_research",
            }
        )
    matrix = confusion_matrix(labels, predictions, labels=label_order)
    return {
        "available": True,
        "n_splits": n_splits,
        "route_accuracy": accuracy(pred_rows, "nlp_pred_route", "expected_route"),
        "retrieve_accuracy": accuracy(pred_rows, "nlp_should_retrieve", "should_retrieve"),
        "labels": label_order,
        "confusion": {
            label: {label_order[index]: int(value) for index, value in enumerate(matrix[row_index])}
            for row_index, label in enumerate(label_order)
        },
        "false_rag": [
            row for row in pred_rows
            if row["nlp_should_retrieve"] and not row["should_retrieve"]
        ],
        "false_reject": [
            row for row in pred_rows
            if not row["nlp_should_retrieve"] and row["should_retrieve"]
        ],
    }


def call_retrieve(url: str, query: str, top_k: int, final_k: int, timeout: float) -> list[dict[str, Any]]:
    response = requests.post(url, json={"query": query, "top_k": top_k, "final_k": final_k}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("evidence") or payload.get("data") or []


def evaluate_retrieval_scores(
    rows: list[dict[str, Any]],
    retrieve_url: str,
    top_k: int,
    final_k: int,
    timeout: float,
    limit: int | None,
    retrieve_all: bool,
    max_retrieval_per_category: int | None,
) -> list[dict[str, Any]]:
    output = []
    selected = rows[:limit] if limit else rows
    retrieval_counts: Counter[str] = Counter()
    for index, row in enumerate(selected, start=1):
        if not retrieve_all and not row.get("should_retrieve"):
            output.append(
                {
                    **row,
                    "retrieval_ok": False,
                    "retrieval_skipped": True,
                    "retrieval_skip_reason": "should_retrieve=false",
                    "top_score": None,
                    "top_results": [],
                }
            )
            continue
        retrieval_category = str(row.get("category") or row.get("expected_route") or "unknown")
        if max_retrieval_per_category and retrieval_counts[retrieval_category] >= max_retrieval_per_category:
            output.append(
                {
                    **row,
                    "retrieval_ok": False,
                    "retrieval_skipped": True,
                    "retrieval_skip_reason": f"max_retrieval_per_category={max_retrieval_per_category}",
                    "top_score": None,
                    "top_results": [],
                }
            )
            continue
        retrieval_counts[retrieval_category] += 1
        print(f"[{index}/{len(selected)}] retrieve: {row['id']} {row['query']}", flush=True)
        try:
            evidence = call_retrieve(retrieve_url, row["query"], top_k, final_k, timeout)
            top = evidence[0] if evidence else {}
            top_score = top.get("rerank_score")
            if top_score is not None:
                top_score = float(top_score)
            output.append(
                {
                    **row,
                    "retrieval_ok": True,
                    "top_score": top_score,
                    "top_source": top.get("source_file"),
                    "top_title": top.get("title"),
                    "top_page": top.get("page_start"),
                    "top_preview": str(top.get("preview") or top.get("raw_chunk_text") or top.get("contextual_chunk") or "")[:180],
                    "top_results": [
                        {
                            "source_file": item.get("source_file"),
                            "title": item.get("title"),
                            "page_start": item.get("page_start"),
                            "rerank_score": item.get("rerank_score"),
                        }
                        for item in evidence[:final_k]
                    ],
                }
            )
        except Exception as exc:
            output.append(
                {
                    **row,
                    "retrieval_ok": False,
                    "retrieval_error": str(exc),
                    "top_score": None,
                    "top_results": [],
                }
            )
    return output


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        if not row.get("retrieval_ok"):
            continue
        expected = bool(row.get("expected_sufficient_evidence"))
        score = row.get("top_score")
        predicted = score is not None and float(score) >= threshold
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        elif not predicted and expected:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    beta = 0.5
    f05 = ((1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f0_5": f05,
    }


def sweep_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = [round(-4.0 + index * 0.25, 2) for index in range(49)]
    return [threshold_metrics(rows, threshold) for threshold in thresholds]


def choose_threshold(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in metrics if row["precision"] >= 0.9 and row["recall"] >= 0.75]
    if feasible:
        best = sorted(feasible, key=lambda row: (row["f0_5"], row["precision"], row["recall"]), reverse=True)[0]
        return {**best, "meets_min_quality": True}
    best = sorted(metrics, key=lambda row: (row["f0_5"], row["precision"], row["recall"]), reverse=True)[0]
    return {**best, "meets_min_quality": False}


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_expected: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("retrieval_ok") and row.get("top_score") is not None:
            key = "sufficient" if row.get("expected_sufficient_evidence") else "insufficient"
            by_expected[key].append(float(row["top_score"]))

    summary = {}
    for key, values in by_expected.items():
        values = sorted(values)
        summary[key] = {
            "count": len(values),
            "min": values[0] if values else None,
            "median": values[len(values) // 2] if values else None,
            "max": values[-1] if values else None,
        }
    return summary


def example_rows(rows: list[dict[str, Any]], threshold: float, false_positive: bool) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if not row.get("retrieval_ok") or row.get("top_score") is None:
            continue
        predicted = float(row["top_score"]) >= threshold
        expected = bool(row.get("expected_sufficient_evidence"))
        if false_positive and predicted and not expected:
            examples.append(row)
        if not false_positive and (not predicted) and expected:
            examples.append(row)
    return sorted(examples, key=lambda item: float(item["top_score"]), reverse=not false_positive)[:8]


def timestamped_report_path(report_path: Path, generated_at: datetime) -> Path:
    timestamp = generated_at.strftime("%Y%m%d-%H%M")
    if report_path.name.startswith(f"{timestamp}-"):
        return report_path
    return report_path.with_name(f"{timestamp}-{report_path.name}")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    generated_at: datetime,
    input_path: Path,
    route_eval: dict[str, Any],
    nlp_eval: dict[str, Any],
    retrieval_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    current_threshold: float,
) -> Path:
    path = timestamped_report_path(path, generated_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = threshold_metrics(retrieval_rows, current_threshold)
    top_thresholds = sorted(threshold_rows, key=lambda row: row["f0_5"], reverse=True)[:8]
    summary = score_summary(retrieval_rows)
    attempted = sum(1 for row in retrieval_rows if row.get("retrieval_ok"))
    skipped = sum(1 for row in retrieval_rows if row.get("retrieval_skipped"))

    false_positive = example_rows(retrieval_rows, recommended["threshold"], false_positive=True)
    false_negative = example_rows(retrieval_rows, recommended["threshold"], false_positive=False)
    recommendation_text = (
        f"推荐阈值：{recommended['threshold']:.2f}"
        if recommended.get("meets_min_quality")
        else f"本轮不推荐调整线上阈值；最佳候选 {recommended['threshold']:.2f} 未达到 precision>=90% 且 recall>=75% 的最低要求。"
    )

    lines = [
        "# 路由与 RAG 阈值校准报告",
        "",
        f"生成时间：{generated_at.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## 数据",
        "",
        f"- 标注集：`{input_path}`",
        f"- 样本数：{len(route_eval['rows'])}",
        "- 本次不调用回答大模型；路由默认关闭 LLM，只评估规则与传统 NLP 小模型；检索只使用 `/api/retrieve` 的 reranker 分数。",
        f"- 检索评分样本：{attempted}；跳过非检索样本：{skipped}。",
        "",
        "## 路由评测",
        "",
        f"- 规则路由 route accuracy：{pct(route_eval['route_accuracy'])}",
        f"- 规则路由 should_retrieve accuracy：{pct(route_eval['retrieve_accuracy'])}",
        f"- 规则误进 RAG：{len(route_eval['false_rag'])}",
        f"- 规则误拒检索：{len(route_eval['false_reject'])}",
        "",
    ]
    if nlp_eval.get("available"):
        lines.extend(
            [
                "## 传统 NLP 小模型",
                "",
                "- 模型：TF-IDF 字符 n-gram(1-4) + Logistic Regression。",
                f"- 交叉验证折数：{nlp_eval['n_splits']}",
                f"- route accuracy：{pct(nlp_eval['route_accuracy'])}",
                f"- should_retrieve accuracy：{pct(nlp_eval['retrieve_accuracy'])}",
                f"- 误进 RAG：{len(nlp_eval['false_rag'])}",
                f"- 误拒检索：{len(nlp_eval['false_reject'])}",
                "",
            ]
        )
    else:
        lines.extend(["## 传统 NLP 小模型", "", f"- 未运行：{nlp_eval.get('error')}", ""])

    lines.extend(
        [
            "## 分数分布",
            "",
            f"- 期望有足够证据：{summary.get('sufficient', {})}",
            f"- 期望证据不足/不应引用：{summary.get('insufficient', {})}",
            "",
            "## 阈值校准",
            "",
            f"- 当前阈值：{current_threshold:.2f}",
            f"- 当前阈值 precision / recall / specificity：{pct(current['precision'])} / {pct(current['recall'])} / {pct(current['specificity'])}",
            f"- {recommendation_text}",
            f"- 推荐阈值 precision / recall / specificity：{pct(recommended['precision'])} / {pct(recommended['recall'])} / {pct(recommended['specificity'])}",
            "",
            "### F0.5 排名前 8 的阈值",
            "",
            markdown_table(
                top_thresholds,
                [
                    ("threshold", "threshold"),
                    ("precision", "precision"),
                    ("recall", "recall"),
                    ("specificity", "specificity"),
                    ("tp", "tp"),
                    ("fp", "fp"),
                    ("tn", "tn"),
                    ("fn", "fn"),
                    ("F0.5", "f0_5"),
                ],
            ),
            "",
            "## 推荐阈值下的误放样例",
            "",
        ]
    )
    if false_positive:
        lines.append(markdown_table(false_positive, [("id", "id"), ("query", "query"), ("score", "top_score"), ("top_title", "top_title")]))
    else:
        lines.append("- 无。")
    lines.extend(["", "## 推荐阈值下的误杀样例", ""])
    if false_negative:
        lines.append(markdown_table(false_negative, [("id", "id"), ("query", "query"), ("score", "top_score"), ("top_title", "top_title")]))
    else:
        lines.append("- 无。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 本轮校准重点是系统边界，不评估最终回答文采。",
            "- 如果没有阈值达到最低质量要求，不应仅靠调阈值上线；需要增加前提核验、时代错置识别或查询改写策略。",
            "- 传统 NLP 小模型可作为无 API 环境下的候选路由器，但是否接入需要看交叉验证误进 RAG 和误拒检索。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--retrieve-url", default=DEFAULT_RETRIEVE_URL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieve-all", action="store_true", help="Also retrieve rows whose should_retrieve label is false.")
    parser.add_argument("--max-retrieval-per-category", type=int)
    parser.add_argument("--reuse-output", action="store_true", help="Reuse an existing output JSONL instead of calling retrieval.")
    parser.add_argument("--current-threshold", type=float, default=float(os.getenv("CL_RAG_MIN_RERANK_SCORE", "1.0")))
    parser.add_argument("--use-llm-router", action="store_true", help="Evaluate the web router with LLM fallback enabled.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    route_eval = evaluate_rule_router(rows, args.use_llm_router)
    nlp_eval = evaluate_traditional_nlp_router(rows)
    if args.reuse_output and args.output.exists():
        retrieval_rows = load_jsonl(args.output)
    else:
        retrieval_rows = evaluate_retrieval_scores(
            rows,
            args.retrieve_url,
            args.top_k,
            args.final_k,
            args.timeout,
            args.limit,
            args.retrieve_all,
            args.max_retrieval_per_category,
        )
    threshold_rows = sweep_thresholds(retrieval_rows)
    recommended = choose_threshold(threshold_rows)
    write_jsonl(args.output, retrieval_rows)
    generated_at = datetime.now().astimezone()
    report_path = write_report(
        args.report,
        generated_at,
        args.input,
        route_eval,
        nlp_eval,
        retrieval_rows,
        threshold_rows,
        recommended,
        args.current_threshold,
    )
    print(f"Wrote results: {args.output}")
    print(f"Wrote report: {report_path}")
    if recommended.get("meets_min_quality"):
        print(f"Recommended threshold: {recommended['threshold']:.2f}")
    else:
        print(f"No safe threshold found; best candidate: {recommended['threshold']:.2f}")


if __name__ == "__main__":
    main()
