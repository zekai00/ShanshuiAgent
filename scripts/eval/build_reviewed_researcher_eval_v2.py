#!/usr/bin/env python3
"""Promote the v2 Researcher eval set to a reviewed benchmark.

The v2 generator intentionally marked every row as `needs_human_review`.
This script applies the first manual review pass:

- rewrite malformed, over-broad, or title-fragment questions;
- keep valid stress tests while strengthening false-premise rubrics;
- validate schema, ids, questions, and task distribution.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from src.config import DATA_DIR

EVAL_DIR = DATA_DIR / "eval"

DEFAULT_MAIN_INPUT = EVAL_DIR / "test_researcher_v2.jsonl"
DEFAULT_FALSE_INPUT = EVAL_DIR / "test_researcher_false_premise_v2.jsonl"
DEFAULT_MAIN_OUTPUT = EVAL_DIR / "test_researcher_v2_reviewed.jsonl"
DEFAULT_FALSE_OUTPUT = EVAL_DIR / "test_researcher_false_premise_v2_reviewed.jsonl"
DEFAULT_SUMMARY_OUTPUT = EVAL_DIR / "test_researcher_v2_reviewed_summary.json"
SCHEMA_PATH = EVAL_DIR / "researcher_eval_schema.json"

TASK_COUNTS = {
    "factual_qa": 40,
    "concept_explain": 40,
    "comparison": 35,
    "composition_aesthetic": 35,
    "artist_prompt": 25,
    "evidence_missing": 25,
}

FALSE_PREMISE_REJECT_IF = [
    "没有在开头明确指出前提错误、时代错置、概念混同或证据不足",
    "虽然先否定错误前提，但继续展开不必要的现代技术类比或编造过程",
    "把错误前提包装成合理比较命题继续论证",
]

FALSE_PREMISE_ANSWER_SUFFIX = (
    "回答应在开头直接纠错或说明证据不足；除非用户明确要求跨时代类比，"
    "不应继续展开现代技术、现代产业或现代工具的假设性说明。"
)

MAIN_REWRITES: dict[str, dict[str, Any]] = {
    "test_researcher_v2_002": {
        "question": "根据《明代山水画中桥梁意象研究》，文献如何解释明代山水画中的桥梁意象及其空间、隐喻功能？",
        "expected_entities": ["明代", "桥梁意象", "空间连接", "隐喻", "游观路径", "山水画"],
        "answer_key": "应基于《明代山水画中桥梁意象研究.pdf》说明桥梁意象在明代山水画中的空间组织、游观路径和象征/隐喻作用，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_015": {
        "question": "根据《从点、线、面看中国山水画基本构成》，北宋山水画的点、线、面关系如何体现其构成方法？",
        "expected_entities": ["北宋", "点", "线", "面", "构成", "山水画"],
        "answer_key": "应基于《从点、线、面看中国山水画基本构成_罗胜.pdf》说明北宋山水画中点、线、面的组织方式及其构成意义，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_017": {
        "question": "根据《中国古代山水画中桥梁的类型特征及位置经营》，文献如何概括山水画中桥梁类型及其位置经营？",
        "expected_entities": ["桥梁类型", "位置经营", "桥梁", "山水画", "空间", "构图"],
        "answer_key": "应基于《中国古代山水画中桥梁的类型特征及位置经营.pdf》说明桥梁类型、结构特征及其在画面位置经营中的作用，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_025": {
        "question": "根据《从点、线、面看中国山水画基本构成》，点、线、面如何构成中国山水画的基本视觉结构？",
        "expected_entities": ["点", "线", "面", "视觉结构", "构成方法", "山水画"],
        "answer_key": "应基于《从点、线、面看中国山水画基本构成_罗胜.pdf》说明点、线、面的视觉功能、组合关系及其对山水画构成的意义，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_028": {
        "question": "根据《隐逸与溪山——浅析隋唐至清末中国山水画的形式演变》，隐逸观念如何影响隋唐至清末山水画的形式演变？",
        "expected_entities": ["隐逸", "溪山", "隋唐", "清末", "形式演变", "山水画"],
        "answer_key": "应基于《_隐逸与溪山_——浅析隋唐至清末中国山水画的形式演变.pdf》说明隐逸观念与溪山图式如何参与山水画形式演变，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_032": {
        "question": "根据《读书与赏物：明代山水画中的“观看”事件》，题跋、印章和文献资料如何帮助重建明代山水画的观看事件？",
        "expected_entities": ["明代", "观看事件", "题跋", "印章", "文献资料", "山水画"],
        "answer_key": "应基于《读书与赏物：明代山水画中的“观看”事件.pdf》说明题跋、印章、文献资料在重建观看者、观看场景和作品流传中的作用，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_033": {
        "question": "根据《中国古代山水画中桥梁的类型特征及位置经营》，五代两宋山水画中的桥梁图像有哪些类型和位置经营特点？",
        "expected_entities": ["五代", "两宋", "桥梁图像", "桥梁类型", "位置经营", "山水画"],
        "answer_key": "应基于《中国古代山水画中桥梁的类型特征及位置经营.pdf》说明五代两宋山水画中桥梁的类型、结构及其位置经营特点，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_039": {
        "question": "根据《读书与赏物：明代山水画中的“观看”事件》，文献中的“观看事件”主要指什么，为什么对理解明代山水画重要？",
        "expected_entities": ["明代", "观看事件", "读书", "赏物", "题跋", "山水画"],
        "answer_key": "应基于《读书与赏物：明代山水画中的“观看”事件.pdf》解释观看事件的含义，以及它如何连接作品、观看者、题跋印章和明代文人生活，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_040": {
        "question": "根据《中国宋代山水画中的“黑白”与“虚实”》，宋代山水画如何通过黑白关系和虚实关系组织画面意境？",
        "expected_entities": ["宋代", "黑白", "虚实", "水墨", "意境", "山水画"],
        "answer_key": "应基于《中国宋代山水画中的“黑白”与“虚实”.pdf》说明黑白、水墨和虚实关系如何构成宋代山水画的空间与意境，并在关键结论后引用该 PDF。",
    },
    "test_researcher_v2_093": {
        "question": "根据《五代山水画的艺术美学》和《从点、线、面看中国山水画基本构成》，比较五代四大家的山水审美与北宋山水画点线面构成方法的差异。",
        "expected_entities": ["五代四大家", "荆浩", "关仝", "董源", "巨然", "北宋", "点线面构成"],
        "answer_key": "应分别引用两个来源，比较五代四大家的审美取向、笔墨图式与北宋山水点线面构成方法之间的差异，并避免只讲其中一边。",
    },
    "test_researcher_v2_094": {
        "question": "根据《从点、线、面看中国山水画基本构成》和《明代北宗山水画风格研究》，比较北宋山水的构成方法与明代北宗画家群的风格取向。",
        "expected_entities": ["北宋", "点线面构成", "明代北宗", "戴进", "吴伟", "唐寅", "仇英"],
        "answer_key": "应分别引用两个来源，比较北宋山水点线面构成与明代北宗画家群在笔墨、构图、题材和风格取向上的差异，并避免只讲其中一边。",
    },
    "test_researcher_v2_095": {
        "question": "根据《明代北宗山水画风格研究》和《中国古代山水画中桥梁的类型特征及位置经营》，比较明代北宗画家群的风格与山水画桥梁类型、位置经营研究的关注点。",
        "expected_entities": ["明代北宗", "戴进", "吴伟", "唐寅", "仇英", "桥梁类型", "位置经营"],
        "answer_key": "应分别引用两个来源，比较明代北宗画家群的风格谱系与桥梁类型、位置经营研究在对象、方法和画面功能上的差异，并避免只讲其中一边。",
    },
    "test_researcher_v2_096": {
        "question": "根据《中国古代山水画中桥梁的类型特征及位置经营》和《明代山水画中桥梁意象研究》，比较桥梁类型/位置经营与明代桥梁意象研究的侧重点。",
        "expected_entities": ["桥梁类型", "位置经营", "明代", "桥梁意象", "空间连接", "隐喻"],
        "answer_key": "应分别引用两个来源，比较桥梁类型和位置经营研究与明代桥梁意象研究在结构、空间、游观和象征意义上的侧重点，并避免只讲其中一边。",
    },
    "test_researcher_v2_097": {
        "question": "根据《明代山水画中桥梁意象研究》和《明代中期园林题材山水画研究》，比较桥梁与流水的布局关系和唐寅园林题材山水的表现重点。",
        "expected_entities": ["明代", "桥梁", "流水", "布局关系", "唐寅", "园林题材"],
        "answer_key": "应分别引用两个来源，比较桥梁与流水的空间组织、象征意义和唐寅园林题材山水在场景、题材、文人趣味上的表现重点，并避免只讲其中一边。",
    },
    "test_researcher_v2_099": {
        "question": "根据《从点、线、面看中国山水画基本构成》和《中国山水画的布局观察与透视》，比较点线面构成方法与布局观察、透视方法的差异。",
        "expected_entities": ["点", "线", "面", "构成方法", "布局", "观察", "透视"],
        "answer_key": "应分别引用两个来源，比较点线面构成方法与布局观察、透视方法在分析对象、视觉组织和创作指导上的差异，并避免只讲其中一边。",
    },
    "test_researcher_v2_101": {
        "question": "根据《浅论清代山水画中的“仿”“摹”“拟”“临”现象》和《隐逸与溪山——浅析隋唐至清末中国山水画的形式演变》，比较清代仿古中的《富春山居图》传统与隐逸山水形式演变的关系。",
        "expected_entities": ["清代", "仿", "摹", "拟", "临", "富春山居图", "隐逸", "形式演变"],
        "answer_key": "应分别引用两个来源，比较清代仿古实践中对《富春山居图》传统的接受，与隐逸山水在隋唐至清末形式演变中的关系，并避免只讲其中一边。",
    },
    "test_researcher_v2_102": {
        "question": "根据《隐逸与溪山——浅析隋唐至清末中国山水画的形式演变》和《中国古代山水画中桥梁的类型特征及位置经营》，比较隐逸山水的形式演变与桥梁栏杆结构、位置经营研究的不同关注点。",
        "expected_entities": ["隐逸", "溪山", "形式演变", "桥梁", "栏杆结构", "位置经营"],
        "answer_key": "应分别引用两个来源，比较隐逸山水形式演变与桥梁栏杆结构、位置经营研究在画史对象、图像细节和构图功能上的不同关注点，并避免只讲其中一边。",
    },
    "test_researcher_v2_103": {
        "question": "根据《中国古代山水画中桥梁的类型特征及位置经营》和《从点、线、面看中国山水画基本构成》，比较桥梁栏杆结构研究与点线面构成方法在画面分析中的作用。",
        "expected_entities": ["桥梁", "栏杆结构", "位置经营", "点", "线", "面", "画面分析"],
        "answer_key": "应分别引用两个来源，比较桥梁栏杆结构/位置经营与点线面构成方法在图像细节、空间组织和创作分析中的作用，并避免只讲其中一边。",
    },
    "test_researcher_v2_104": {
        "question": "根据《从点、线、面看中国山水画基本构成》和《清代山水画的笔墨美》，比较点线面构成方法与清代四王、四僧笔墨取向的差异。",
        "expected_entities": ["点", "线", "面", "构成方法", "清代", "四王", "四僧", "笔墨"],
        "answer_key": "应分别引用两个来源，比较点线面构成方法与清代四王、四僧笔墨审美在分析层级、风格取向和画学功能上的差异，并避免只讲其中一边。",
    },
    "test_researcher_v2_106": {
        "question": "根据《明代秋景山水画探析》和《读书与赏物：明代山水画中的“观看”事件》，比较吴门画家的秋景山水与题跋、印章、文献资料所揭示的观看事件。",
        "expected_entities": ["明代", "吴门画家", "秋景山水", "观看事件", "题跋", "印章", "文献资料"],
        "answer_key": "应分别引用两个来源，比较吴门画家秋景山水的题材、意境与题跋、印章、文献资料所揭示的观看事件和接受场景，并避免只讲其中一边。",
    },
    "test_researcher_v2_107": {
        "question": "根据《读书与赏物：明代山水画中的“观看”事件》和《北宋山水画点景建筑布局分析与应用研究》，比较观看事件研究与北宋点景建筑布局研究的不同重点。",
        "expected_entities": ["观看事件", "题跋", "印章", "北宋", "点景建筑", "布局"],
        "answer_key": "应分别引用两个来源，比较明代山水观看事件研究与北宋点景建筑布局研究在证据类型、图像对象和空间分析上的不同重点，并避免只讲其中一边。",
    },
    "test_researcher_v2_108": {
        "question": "根据《明代中期园林题材山水画研究》和《读书与赏物：明代山水画中的“观看”事件》，比较文徵明园林题材山水与明代山水画观看事件研究的关注点。",
        "expected_entities": ["明代", "文徵明", "园林题材", "观看事件", "读书", "赏物"],
        "answer_key": "应分别引用两个来源，比较文徵明园林题材山水的场景经营、文人趣味与明代山水画观看事件研究在观看者、物件和文本证据上的关注点，并避免只讲其中一边。",
    },
    "test_researcher_v2_109": {
        "question": "根据《读书与赏物：明代山水画中的“观看”事件》和《中国宋代山水画中的“黑白”与“虚实”》，比较明代山水画观看事件与宋代山水画黑白、虚实关系的研究重点。",
        "expected_entities": ["明代", "观看事件", "宋代", "黑白", "虚实", "山水画"],
        "answer_key": "应分别引用两个来源，比较明代山水画观看事件与宋代山水画黑白、虚实关系在证据、观看机制、空间和意境分析上的差异，并避免只讲其中一边。",
    },
    "test_researcher_v2_115": {
        "question": "根据《明代中期园林题材山水画研究》和《从点、线、面看中国山水画基本构成》，比较园林山水场景（如《东林图》）与点线面构成方法的表现重点。",
        "expected_entities": ["明代", "园林山水", "东林图", "点", "线", "面", "构成方法"],
        "answer_key": "应分别引用两个来源，比较园林山水场景的空间、题材和文人趣味，与点线面构成方法在视觉组织和创作分析上的表现重点，并避免只讲其中一边。",
    },
    "test_researcher_v2_175": {
        "question": "请基于《从点、线、面看中国山水画基本构成》中关于点、线、面构成方法的资料，整理一份画师可用的山水画创作要点。",
        "expected_entities": ["点", "线", "面", "构成方法", "山水画", "创作要点"],
        "answer_key": "应先给出《从点、线、面看中国山水画基本构成_罗胜.pdf》的考据依据，再转化为画师可执行的构图、笔墨、设色、题材、意境和负面约束；不得伪造已生成图片。",
    },
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_keep_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def append_sentence(text: str, sentence: str) -> str:
    text = str(text or "").strip()
    if sentence in text:
        return text
    if not text:
        return sentence
    if text.endswith(("。", "！", "？", ".", "!", "?")):
        return text + sentence
    return text + "。" + sentence


def strengthen_false_premise(item: dict[str, Any]) -> None:
    item["reject_if"] = dedupe_keep_order(list(item.get("reject_if", [])) + FALSE_PREMISE_REJECT_IF)
    item["answer_key"] = append_sentence(str(item.get("answer_key", "")), FALSE_PREMISE_ANSWER_SUFFIX)


def promote_main_item(item: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(item)
    row["review_status"] = "reviewed"

    rewrite = MAIN_REWRITES.get(str(row["id"]))
    if rewrite:
        row["original_question"] = row["question"]
        row.update(rewrite)
        row["review_decision"] = "rewrite"
        row["review_reason"] = "自动生成题干存在截断、过泛、题名片段或不自然比较，已改写为可评测的学术问题。"
        row["notes"] = (
            f"reviewed: 改写自动生成题干；原题：{row['original_question']} "
            "金标来源和 task_type 保持不变。"
        )
    else:
        row["review_decision"] = "keep"
        row["review_reason"] = "题干、金标来源和评分标准可用于当前阶段评测。"
        row["notes"] = append_sentence(str(row.get("notes", "")), "reviewed: 保留。")

    if row.get("task_type") == "evidence_missing":
        strengthen_false_premise(row)
        row["review_reason"] = append_sentence(
            str(row.get("review_reason", "")),
            "强化错误前提/证据不足题的拒答边界。",
        )
        row["notes"] = append_sentence(str(row.get("notes", "")), "强化错误前提评分边界。")
    return row


def promote_false_premise_item(item: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(item)
    row["review_status"] = "reviewed"
    row["review_decision"] = "keep"
    row["review_reason"] = "保留为错误前提专项压力测试，并强化拒答边界。"
    strengthen_false_premise(row)
    row["notes"] = append_sentence(str(row.get("notes", "")), "reviewed: 保留为错误前提专项压力测试，强化拒答边界。")
    return row


def validate_rows(rows: list[dict[str, Any]], schema: dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema)
    ids: list[str] = []
    questions: list[str] = []
    errors: list[str] = []

    for index, row in enumerate(rows, start=1):
        row_errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        for error in row_errors:
            path = ".".join(str(part) for part in error.path) or "<root>"
            errors.append(f"{label}:{index}:{path}: {error.message}")
        ids.append(str(row.get("id")))
        questions.append(str(row.get("question")))

    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    duplicate_questions = sorted([item for item, count in Counter(questions).items() if count > 1])
    if duplicate_ids:
        errors.append(f"{label}: duplicate ids: {duplicate_ids}")
    if duplicate_questions:
        errors.append(f"{label}: duplicate questions: {duplicate_questions[:5]}")

    if errors:
        raise ValueError("\n".join(errors[:50]))


def assert_distribution(rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row["task_type"]) for row in rows)
    if dict(counts) != TASK_COUNTS:
        raise ValueError(f"Unexpected main task distribution: {dict(counts)} != {TASK_COUNTS}")


def build_summary(main_rows: list[dict[str, Any]], false_rows: list[dict[str, Any]]) -> dict[str, Any]:
    main_decisions = Counter(str(row.get("review_decision", "")) for row in main_rows)
    false_decisions = Counter(str(row.get("review_decision", "")) for row in false_rows)
    rewritten = [
        {
            "id": row["id"],
            "task_type": row["task_type"],
            "original_question": row.get("original_question", ""),
            "reviewed_question": row["question"],
            "gold_sources": row["gold_sources"],
        }
        for row in main_rows
        if row.get("review_decision") == "rewrite"
    ]
    main_false_premise_ids = [row["id"] for row in main_rows if row.get("task_type") == "evidence_missing"]

    return {
        "input_files": {
            "main": str(DEFAULT_MAIN_INPUT),
            "false_premise": str(DEFAULT_FALSE_INPUT),
        },
        "output_files": {
            "main": str(DEFAULT_MAIN_OUTPUT),
            "false_premise": str(DEFAULT_FALSE_OUTPUT),
        },
        "main_total": len(main_rows),
        "false_premise_total": len(false_rows),
        "task_distribution": dict(Counter(str(row["task_type"]) for row in main_rows)),
        "main_review_decisions": dict(main_decisions),
        "false_premise_review_decisions": dict(false_decisions),
        "main_rewrite_count": len(rewritten),
        "deleted_count": 0,
        "rewritten_items": rewritten,
        "false_premise_rubric_strengthened": {
            "main_ids": main_false_premise_ids,
            "specialized_ids": [row["id"] for row in false_rows],
            "total": len(main_false_premise_ids) + len(false_rows),
        },
        "validation": {
            "schema": str(SCHEMA_PATH),
            "duplicate_ids": 0,
            "duplicate_questions": 0,
            "status": "passed",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-input", type=Path, default=DEFAULT_MAIN_INPUT)
    parser.add_argument("--false-input", type=Path, default=DEFAULT_FALSE_INPUT)
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN_OUTPUT)
    parser.add_argument("--false-output", type=Path, default=DEFAULT_FALSE_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    main_input = load_jsonl(args.main_input)
    false_input = load_jsonl(args.false_input)

    main_rows = [promote_main_item(item) for item in main_input]
    false_rows = [promote_false_premise_item(item) for item in false_input]

    validate_rows(main_rows, schema, "main")
    validate_rows(false_rows, schema, "false_premise")
    assert_distribution(main_rows)

    shared_ids = sorted({row["id"] for row in main_rows} & {row["id"] for row in false_rows})
    if shared_ids:
        raise ValueError(f"Main and false-premise sets share ids: {shared_ids}")

    write_jsonl(args.main_output, main_rows)
    write_jsonl(args.false_output, false_rows)

    summary = build_summary(main_rows, false_rows)
    summary["input_files"] = {
        "main": str(args.main_input),
        "false_premise": str(args.false_input),
    }
    summary["output_files"] = {
        "main": str(args.main_output),
        "false_premise": str(args.false_output),
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "main_total": len(main_rows),
        "false_premise_total": len(false_rows),
        "main_rewrites": summary["main_rewrite_count"],
        "false_premise_rubric_strengthened": summary["false_premise_rubric_strengthened"]["total"],
        "task_distribution": summary["task_distribution"],
        "validation": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
