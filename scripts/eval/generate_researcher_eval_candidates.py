#!/usr/bin/env python3
"""Generate first-pass researcher evaluation candidates from PDF filenames.

This script deliberately generates candidates, not a final locked test set.
The output should be manually reviewed before changing `review_status` to
`reviewed`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WORKSPACE_DIR = Path("/root/Workspace/ChineseLandscape")
RAW_PDFS_DIR = WORKSPACE_DIR / "data" / "raw_pdfs"
DEFAULT_OUTPUT = WORKSPACE_DIR / "data" / "eval" / "researcher_eval_candidates.jsonl"


def infer_task_type(filename: str) -> str:
    title = filename.removesuffix(".pdf")
    if any(key in title for key in ["比较", "差异", "演变", "衍变", "发展", "变化", "流派"]):
        return "comparison"
    if any(key in title for key in ["布局", "空间", "位置", "桥梁", "建筑", "园林", "点景"]):
        return "composition_aesthetic"
    if any(key in title for key in ["皴法", "留白", "三远法", "画论", "笔墨", "禅境", "隐逸"]):
        return "concept_explain"
    return "factual_qa"


def infer_entities(filename: str) -> list[str]:
    title = filename.removesuffix(".pdf")
    candidates = [
        "先秦", "秦汉", "魏晋南北朝", "隋", "唐", "五代", "北宋", "南宋", "宋",
        "元", "明", "清", "近现代", "黄公望", "郭熙", "马远", "四王",
        "三远法", "高远", "深远", "平远", "皴法", "披麻皴", "留白",
        "桥梁", "建筑", "园林", "点景", "布局", "空间", "笔墨", "青绿",
        "禅境", "隐逸", "自然", "审美"
    ]
    entities = [term for term in candidates if term in title]
    if not entities:
        entities = [title[:12]]
    return entities


def build_question(filename: str, task_type: str) -> str:
    title = filename.removesuffix(".pdf")
    clean_title = title.replace("_", "")

    if task_type == "comparison":
        return f"根据《{clean_title}》，这个主题体现了哪些山水画风格或审美变化？"
    if task_type == "composition_aesthetic":
        return f"根据《{clean_title}》，相关画面元素在山水画构图和意境中有什么作用？"
    if task_type == "concept_explain":
        return f"请根据《{clean_title}》解释其中涉及的核心山水画概念。"
    return f"《{clean_title}》主要讨论了中国山水画中的什么问题？"


def build_item(index: int, filename: str) -> dict:
    task_type = infer_task_type(filename)
    entities = infer_entities(filename)
    return {
        "id": f"candidate_researcher_{index:04d}",
        "split": "dev",
        "task_type": task_type,
        "question": build_question(filename, task_type),
        "gold_sources": [filename],
        "expected_entities": entities,
        "must_cite": True,
        "answer_key": "应基于指定文献回答，提取核心观点，并在关键结论后标注文献出处。",
        "reject_if": [
            "没有引用检索文献",
            "引用不存在的 PDF",
            "脱离指定文献泛泛而谈"
        ],
        "review_status": "needs_human_review",
        "notes": "脚本自动生成候选题，需要人工核对题目价值、答案要点和 expected_entities。"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=str(RAW_PDFS_DIR), help="Directory containing source PDFs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of files to process. 0 means all.")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_path = Path(args.output)
    pdf_files = sorted(path.name for path in pdf_dir.glob("*.pdf"))
    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for index, filename in enumerate(pdf_files, start=1):
            f.write(json.dumps(build_item(index, filename), ensure_ascii=False) + "\n")

    print(f"Wrote {len(pdf_files)} candidate items to {output_path}")


if __name__ == "__main__":
    main()
