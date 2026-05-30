#!/usr/bin/env python3
"""Build a context-grounded SFT dataset for a local Researcher model.

The dataset is generated from the existing Milvus RAG store, not from the
held-out researcher evaluation set. It teaches the model to answer from
provided evidence and cite the source PDF in the system's required format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymilvus import MilvusClient

from src.config import EXTRACTED_ARTWORKS_DIR, LLAMA_FACTORY_DIR, MILVUS_DB_PATH, RETRIEVAL_COLLECTION_NAME

DEFAULT_OUTPUT = LLAMA_FACTORY_DIR / "data" / "researcher_rag_sft_v2.json"
COLLECTION_NAME = RETRIEVAL_COLLECTION_NAME


def compact_space(text: str) -> str:
    image_root = re.escape(str(EXTRACTED_ARTWORKS_DIR))
    text = re.sub(rf"{image_root}[^\s：\]]+", "[图像]", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_chunk(chunk: str) -> tuple[str, str]:
    anchor = ""
    original = ""
    if "【全局上下文】" in chunk and "【原文资料】" in chunk:
        anchor = chunk.split("【全局上下文】", 1)[1].split("【原文资料】", 1)[0]
        original = chunk.split("【原文资料】", 1)[1]
    else:
        original = chunk
    return compact_space(anchor), compact_space(original)


def trim(text: str, limit: int) -> str:
    text = compact_space(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_unknown(value: Any) -> str:
    if isinstance(value, list):
        value = "、".join(str(x) for x in value if str(x).strip())
    value = str(value or "").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]").replace("'", "").replace('"', "")
    return "" if value in {"未知", "佚名", "未知画家", "[]", ""} else value


def build_input(row: dict[str, Any]) -> str:
    parts = [f"来源文献: {row['source_file']}"]
    dynasty = normalize_unknown(row.get("dynasty"))
    painter = normalize_unknown(row.get("painter"))
    subject = normalize_unknown(row.get("subject_matter"))
    if dynasty:
        parts.append(f"朝代: {dynasty}")
    if painter:
        parts.append(f"画家/流派: {painter}")
    if subject:
        parts.append(f"主题: {subject}")
    parts.append(compact_space(str(row["contextual_chunk"])))
    return "【检索资料】\n" + "\n".join(parts)


def make_answer(row: dict[str, Any]) -> str:
    anchor, original = parse_chunk(str(row["contextual_chunk"]))
    source = row["source_file"]
    if anchor:
        answer = f"根据检索资料，这段材料的核心意思是：{trim(anchor, 260)}"
    else:
        answer = "根据检索资料，这段材料主要围绕中国山水画的具体问题展开。"

    if original:
        answer += f" 原文依据可概括为：{trim(original, 320)}"
    answer += f" [文献: 《{source}》]"
    return answer


def make_examples(row: dict[str, Any]) -> list[dict[str, str]]:
    source = str(row.get("source_file", "")).strip()
    chunk = str(row.get("contextual_chunk", "")).strip()
    if not source or not chunk or source == "None":
        return []

    input_text = build_input(row)
    answer = make_answer(row)
    examples = [
        {
            "instruction": "请基于给定检索资料，概括这段资料的核心学术观点，并按要求引用来源。",
            "input": input_text,
            "output": answer,
        }
    ]

    painter = normalize_unknown(row.get("painter"))
    subject = normalize_unknown(row.get("subject_matter"))
    dynasty = normalize_unknown(row.get("dynasty"))

    painter_for_question = painter if painter and len(painter) <= 24 and painter.count("、") <= 2 else ""

    if painter_for_question:
        examples.append(
            {
                "instruction": f"请基于给定检索资料，说明“{painter_for_question}”相关内容的山水画史意义。",
                "input": input_text,
                "output": answer,
            }
        )
    elif subject:
        examples.append(
            {
                "instruction": f"请基于给定检索资料，解释“{subject}”在山水画研究中的含义。",
                "input": input_text,
                "output": answer,
            }
        )
    elif dynasty:
        examples.append(
            {
                "instruction": f"请基于给定检索资料，说明这段材料与“{dynasty}”山水画研究的关系。",
                "input": input_text,
                "output": answer,
            }
        )

    return examples


def make_refusal_examples(repeat: int) -> list[dict[str, str]]:
    no_evidence = "【检索资料】\n系统没有提供能够支持该说法的可靠文献片段。"
    base_cases = [
        {
            "instruction": "请说明清代四王如何使用 Stable Diffusion 生成山水画。",
            "input": no_evidence,
            "output": "不能这样说明。Stable Diffusion 是现代生成式图像模型，清代四王不可能使用这一技术；在没有相关文献证据时，应将该问题判定为时代错置，而不是编造答案。因此此处不提供伪造文献引用。",
        },
        {
            "instruction": "《富春山居图》是否为北宋郭熙所作？",
            "input": no_evidence,
            "output": "这个前提不成立。《富春山居图》通常归属元代黄公望，而不是北宋郭熙；如果检索资料没有提供相反证据，回答应先纠正作者与时代关系，不能顺着错误前提展开。因此此处不提供伪造文献引用。",
        },
        {
            "instruction": "清初四王是否属于元四家？",
            "input": no_evidence,
            "output": "不能把清初四王直接说成元四家。二者属于不同历史时期和不同画史概念；若检索资料不足，应说明证据边界，而不是混同流派。因此此处不提供伪造文献引用。",
        },
        {
            "instruction": "南宋马远是否主要以披麻皴著称？",
            "input": no_evidence,
            "output": "这个说法需要谨慎。马远通常与南宋院体、边角构图、较劲利的山石表现相关，不能在没有证据时直接说他主要以披麻皴著称。因此此处不提供伪造文献引用。",
        },
        {
            "instruction": "郭熙的三远法是否等同于西方焦点透视理论？",
            "input": no_evidence,
            "output": "不能等同。三远法是中国山水画观看与空间组织方法，西方焦点透视是另一套视觉几何体系；没有文献证据时，应明确二者不能简单合并。因此此处不提供伪造文献引用。",
        },
    ]
    examples: list[dict[str, str]] = []
    for _ in range(max(repeat, 1)):
        for case in base_cases:
            question = case["instruction"]
            output = case["output"]
            examples.extend(
                [
                    case,
                    {
                        "instruction": f"请基于给定检索资料回答：{question}",
                        "input": no_evidence,
                        "output": output,
                    },
                    {
                        "instruction": f"用户问题存在潜在错误前提。请先判断能否回答，再回答：{question}",
                        "input": no_evidence,
                        "output": output,
                    },
                ]
            )
    return examples


def fetch_rows(limit: int) -> list[dict[str, Any]]:
    client = MilvusClient(str(MILVUS_DB_PATH))
    client.load_collection(COLLECTION_NAME)
    try:
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter="id >= 0",
            limit=limit,
            output_fields=[
                "id",
                "contextual_chunk",
                "source_file",
                "dynasty",
                "painter",
                "subject_matter",
                "content_scope",
            ],
        )
    finally:
        client.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-rows", type=int, default=1200)
    parser.add_argument("--max-examples", type=int, default=1600)
    parser.add_argument("--refusal-repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    rows = fetch_rows(args.max_rows)
    examples: list[dict[str, str]] = []
    for row in rows:
        examples.extend(make_examples(row))

    random.seed(args.seed)
    random.shuffle(examples)
    examples = examples[: args.max_examples]
    examples.extend(make_refusal_examples(args.refusal_repeat))
    random.shuffle(examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(examples)} SFT examples to {output_path}")


if __name__ == "__main__":
    main()
