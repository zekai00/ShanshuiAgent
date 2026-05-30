#!/usr/bin/env python3
"""Generate a larger Researcher evaluation set from the local RAG corpus.

The output is a test candidate set, not a fully human-verified benchmark.
Items are intentionally marked `needs_human_review` so we can review or
promote selected rows later without pretending the script produced gold data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

WORKSPACE_DIR = Path("/root/Workspace/ChineseLandscape")
MILVUS_DB_PATH = WORKSPACE_DIR / "data" / "vector_store" / "milvus_landscape.db"
COLLECTION_NAME = "landscape_rag"

DEFAULT_MAIN_OUTPUT = WORKSPACE_DIR / "data" / "eval" / "test_researcher_v2.jsonl"
DEFAULT_FALSE_OUTPUT = WORKSPACE_DIR / "data" / "eval" / "test_researcher_false_premise_v2.jsonl"
DEFAULT_SUMMARY_OUTPUT = WORKSPACE_DIR / "data" / "eval" / "test_researcher_v2_summary.json"
V1_PATH = WORKSPACE_DIR / "data" / "eval" / "test_researcher_v1.jsonl"

TASK_COUNTS = {
    "factual_qa": 40,
    "concept_explain": 40,
    "comparison": 35,
    "composition_aesthetic": 35,
    "artist_prompt": 25,
    "evidence_missing": 25,
}

DOMAIN_TERMS = [
    "先秦", "秦汉", "魏晋南北朝", "隋", "唐", "五代", "北宋", "南宋", "宋",
    "辽金", "元", "明", "清", "近现代",
    "黄公望", "郭熙", "马远", "夏圭", "范宽", "李成", "董源", "巨然",
    "倪瓒", "王蒙", "吴镇", "赵孟頫", "董其昌", "石涛", "八大山人",
    "王时敏", "王鉴", "王翚", "王原祁", "四王", "四僧", "元四家",
    "披麻皴", "斧劈皴", "雨点皴", "卷云皴", "解索皴", "折带皴", "牛毛皴",
    "留白", "计白当黑", "虚实相生", "三远法", "高远", "深远", "平远",
    "青绿", "浅绛", "水墨", "笔墨", "皴法", "构图", "布局", "桥梁",
    "点景", "建筑", "园林", "禅境", "隐逸", "象征", "隐喻", "空间营造",
    "经营位置", "文人画", "山水画",
]

CONCEPT_TERMS = [
    "皴法", "披麻皴", "斧劈皴", "留白", "计白当黑", "虚实相生",
    "三远法", "高远", "深远", "平远", "青绿", "浅绛", "笔墨",
    "禅境", "隐逸", "象征", "隐喻", "文人画", "经营位置",
]

COMPOSITION_TERMS = [
    "构图", "布局", "空间", "空间营造", "桥梁", "建筑", "点景", "园林",
    "位置经营", "留白", "虚实", "溪流", "山路", "树", "屋", "亭",
]

GENERIC_TOPIC_TERMS = {
    "先秦", "秦汉", "隋", "唐", "宋", "元", "明", "清", "近现代",
    "山水", "山水画", "中国", "绘画", "研究",
}


def clean_text(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"/root/Workspace/ChineseLandscape/data/extracted_artworks/[^\s：\]]+", "[图像]", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_value(value: Any) -> str:
    if isinstance(value, list):
        value = "、".join(str(item) for item in value if str(item).strip())
    value = str(value or "").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]").replace("'", "").replace('"', "")
    return "" if value in {"", "未知", "[]", "None"} else value


def clean_title(source_file: str) -> str:
    title = source_file.removesuffix(".pdf")
    title = title.replace("_NormalPdf", "").replace("NormalPdf", "")
    parts = title.split("_")
    if len(parts) > 1 and 1 <= len(parts[-1]) <= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", parts[-1]):
        title = "_".join(parts[:-1])
    title = title.replace("_", "")
    return title.strip("《》 ")


def parse_chunk(chunk: str) -> tuple[str, str]:
    chunk = clean_text(chunk)
    if "【全局上下文】" in chunk and "【原文资料】" in chunk:
        anchor = chunk.split("【全局上下文】", 1)[1].split("【原文资料】", 1)[0]
        original = chunk.split("【原文资料】", 1)[1]
        return clean_text(anchor), clean_text(original)
    return "", chunk


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
    return [row for row in rows if is_usable_row(row)]


def is_usable_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source_file", ""))
    chunk = clean_text(row.get("contextual_chunk", ""))
    if not source.endswith(".pdf") or re.fullmatch(r"paper\d+\.pdf", source):
        return False
    if len(chunk) < 80:
        return False
    noisy_markers = ["【参考文献】", "参考文献]", "致谢", "目录", "摘要 Abstract"]
    if any(marker in chunk for marker in noisy_markers):
        return False
    return True


def terms_in_row(row: dict[str, Any], terms: list[str]) -> list[str]:
    text = " ".join(
        clean_text(row.get(field, ""))
        for field in ["contextual_chunk", "source_file", "dynasty", "painter", "subject_matter"]
    )
    found = [term for term in terms if term in text]
    return dedupe(found)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def topic_for_row(row: dict[str, Any], preferred_terms: list[str] | None = None) -> str:
    if preferred_terms:
        terms = [term for term in terms_in_row(row, preferred_terms) if is_meaningful_topic(term)]
        if terms:
            return terms[0]
    subject = normalize_value(row.get("subject_matter"))
    painter = normalize_value(row.get("painter"))
    terms = [term for term in terms_in_row(row, DOMAIN_TERMS) if is_meaningful_topic(term)]
    if is_meaningful_topic(subject) and len(subject) <= 18:
        return subject
    if painter and painter not in {"佚名"} and len(painter) <= 18:
        return painter
    if terms:
        return terms[0]
    return clean_title(str(row.get("source_file", "")))[:14]


def is_meaningful_topic(value: str) -> bool:
    value = normalize_value(value)
    if not value or len(value) <= 1:
        return False
    return value not in GENERIC_TOPIC_TERMS


def expected_entities(row: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    values: list[str] = []
    for field in ["dynasty", "painter", "subject_matter"]:
        value = normalize_value(row.get(field))
        if value and value not in {"佚名"}:
            values.extend(re.split(r"[、,，/ ]+", value))
    values.extend(terms_in_row(row, DOMAIN_TERMS))
    if extra:
        values.extend(extra)
    return dedupe([value for value in values if len(value) <= 18])[:8]


def common_item(
    item_id: str,
    task_type: str,
    question: str,
    gold_sources: list[str],
    entities: list[str],
    answer_key: str,
    reject_if: list[str],
    must_cite: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "split": "test",
        "task_type": task_type,
        "question": question,
        "gold_sources": gold_sources,
        "expected_entities": dedupe(entities)[:8],
        "must_cite": must_cite,
        "answer_key": answer_key,
        "reject_if": reject_if,
        "review_status": "needs_human_review",
        "notes": notes,
    }


def make_factual(item_id: str, row: dict[str, Any]) -> dict[str, Any]:
    source = str(row["source_file"])
    title = clean_title(source)
    topic = topic_for_row(row)
    question = f"根据《{title}》，文献中的“{topic}”主要说明了中国山水画研究中的什么问题？"
    return common_item(
        item_id,
        "factual_qa",
        question,
        [source],
        expected_entities(row, [topic]),
        f"应基于《{source}》说明“{topic}”相关段落的核心事实、画史背景或研究对象，并在关键结论后引用该 PDF。",
        ["没有引用检索文献", "引用不存在的 PDF", "脱离指定文献泛泛而谈", "把题目回答成无关朝代或无关画家"],
        notes="v2 自动生成事实题，需人工抽样复核。",
    )


def make_concept(item_id: str, row: dict[str, Any]) -> dict[str, Any]:
    source = str(row["source_file"])
    title = clean_title(source)
    term = topic_for_row(row, CONCEPT_TERMS)
    question = f"请根据《{title}》解释“{term}”在中国山水画中的含义或作用。"
    return common_item(
        item_id,
        "concept_explain",
        question,
        [source],
        expected_entities(row, [term]),
        f"应解释“{term}”的画学含义、技法/审美作用和文献语境，不能只给泛泛定义。",
        ["没有引用检索文献", "把概念解释成现代无关概念", "没有结合山水画语境", "引用不存在的 PDF"],
        notes="v2 自动生成概念解释题，需人工抽样复核。",
    )


def make_composition(item_id: str, row: dict[str, Any]) -> dict[str, Any]:
    source = str(row["source_file"])
    title = clean_title(source)
    term = topic_for_row(row, COMPOSITION_TERMS)
    question = f"根据《{title}》，“{term}”如何参与山水画的构图、空间或意境营造？"
    return common_item(
        item_id,
        "composition_aesthetic",
        question,
        [source],
        expected_entities(row, [term]),
        f"应说明“{term}”在构图、空间组织、视线引导、虚实关系或意境表达中的作用，并引用文献。",
        ["只讲现代设计或工程功能", "不联系山水画画面组织", "没有引用文献", "引用不存在的 PDF"],
        notes="v2 自动生成构图审美题，需人工抽样复核。",
    )


def make_artist(item_id: str, row: dict[str, Any]) -> dict[str, Any]:
    source = str(row["source_file"])
    title = clean_title(source)
    topic = topic_for_row(row)
    question = f"请基于《{title}》中关于“{topic}”的资料，整理一份画师可用的山水画创作要点。"
    return common_item(
        item_id,
        "artist_prompt",
        question,
        [source],
        expected_entities(row, [topic, "创作要点"]),
        "应先给出考据依据，再转化为画师可执行的构图、笔墨、设色、题材、意境和负面约束；不得伪造已生成图片。",
        ["没有考据依据直接出 prompt", "不引用文献", "编造已经生成图像", "创作要点与文献主题无关"],
        notes="v2 自动生成 researcher-to-artist 桥接题，需人工抽样复核。",
    )


def make_comparison(item_id: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_source = str(left["source_file"])
    right_source = str(right["source_file"])
    left_topic = topic_for_row(left)
    right_topic = topic_for_row(right)
    left_title = clean_title(left_source)
    right_title = clean_title(right_source)
    question = f"根据《{left_title}》和《{right_title}》，比较“{left_topic}”与“{right_topic}”在山水画表现或画学意义上的差异。"
    return common_item(
        item_id,
        "comparison",
        question,
        dedupe([left_source, right_source]),
        expected_entities(left, [left_topic])[:4] + expected_entities(right, [right_topic])[:4],
        "应分别引用两个来源，比较二者在朝代、题材、技法、构图、审美或画学功能上的差异，并避免只讲其中一边。",
        ["只引用一个来源", "没有进行比较", "把两个概念或时代混为一谈", "引用不存在的 PDF"],
        notes="v2 自动生成跨文献比较题，需人工抽样复核。",
    )


def false_premise_cases() -> list[dict[str, Any]]:
    """Return 50 non-v1 false-premise cases. First 25 go to main v2."""
    cases = [
        ("宋代范宽是否使用无人机航拍来确定《溪山行旅图》的构图？", [], ["宋代", "范宽", "无人机", "构图"], "应指出无人机为现代技术，宋代范宽不可能使用；可转而说明传统山水构图需基于文献考察。", False),
        ("元代倪瓒是否通过 Photoshop 图层来设计画面留白？", [], ["元代", "倪瓒", "Photoshop", "留白"], "应指出 Photoshop 为现代软件，元代画家不可能使用；可解释留白应从笔墨和构图传统理解。", False),
        ("董其昌是否训练 LoRA 模型来总结南北宗论？", [], ["董其昌", "LoRA", "南北宗论"], "应指出 LoRA 是现代机器学习概念，董其昌不可能使用；不得编造训练过程。", False),
        ("唐代青绿山水是否主要依靠数码相机色彩管理形成矿物色效果？", [], ["唐代", "青绿山水", "数码相机", "矿物色"], "应指出数码相机为现代技术，唐代青绿设色应从颜料、壁画和绘画传统理解。", False),
        ("宗炳《画山水序》是否讨论了 AI 绘画版权问题？", [], ["宗炳", "画山水序", "AI绘画", "版权"], "应指出 AI 绘画版权是现代议题，不能归入宗炳原始画论。", False),
        ("郭熙是否属于元四家之一？", ["谈郭熙“三远法”对山水画创作的影响_NormalPdf.pdf", "元代山水画衍变浅析.pdf"], ["郭熙", "元四家", "北宋", "元代"], "应纠正：郭熙是北宋画家，元四家是元代画家群体，二者不可混同。", True),
        ("黄公望是否属于清初四王？", ["元代山水画衍变浅析.pdf", "镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["黄公望", "清初四王", "元代", "清初"], "应纠正：黄公望是元代画家，不属于清初四王。", True),
        ("王原祁是否参与绘制《富春山居图》原作？", ["元代山水画衍变浅析.pdf", "镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["王原祁", "富春山居图", "黄公望", "清代"], "应纠正：王原祁为清代四王之一，不是《富春山居图》原作者。", True),
        ("董源是否以南宋马远式边角构图著称？", ["浅议宋代中国山水画的艺术特征.pdf", "中国山水画的皴法研究_王旭峰.pdf"], ["董源", "马远", "南宋", "边角构图"], "应区分董源与南宋马远，不应把马远式边角构图套到董源身上。", True),
        ("北宋范宽是否主要以倪瓒式三段式构图著称？", ["浅议宋代中国山水画的艺术特征.pdf", "元代山水画衍变浅析.pdf"], ["范宽", "倪瓒", "北宋", "三段式构图"], "应区分北宋范宽与元代倪瓒，不应将倪瓒式构图直接归给范宽。", True),
        ("四僧是否就是清初四王的另一个名称？", ["传统绘画的流派与创新——清代山水画概论.pdf", "镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["四僧", "四王", "清代"], "应说明四僧与四王是清代不同画家群体和画史概念，不能等同。", True),
        ("三远法是否由摄影术发明后传入中国山水画？", ["谈郭熙“三远法”对山水画创作的影响_NormalPdf.pdf"], ["三远法", "摄影术", "郭熙"], "应指出三远法是中国古代山水画空间理论，早于摄影术，不能说由摄影术传入。", True),
        ("披麻皴是否是一种现代图像压缩算法？", ["中国山水画的皴法研究_王旭峰.pdf", "中国山水画皴法研究_邱梅.pdf"], ["披麻皴", "皴法", "图像压缩算法"], "应纠正：披麻皴是山水画皴法，不是现代图像压缩算法。", True),
        ("留白是否表示画家忘记完成的未画区域？", ["山水画留白艺术探究.pdf", "简析山水画留白意境.pdf"], ["留白", "意境", "虚实"], "应说明留白是主动的空间和意境经营，不是忘记完成。", True),
        ("明代桥梁意象是否只代表现代交通效率？", ["明代山水画中桥梁意象研究.pdf"], ["明代", "桥梁意象", "交通效率"], "应说明桥梁在山水画中涉及空间连接、游观路径和意境，不应简化为现代交通效率。", True),
        ("禅境是否等同于寺庙建筑施工规范？", ["论中国宋代山水画中的“禅境”.pdf"], ["禅境", "寺庙", "山水画"], "应说明禅境是审美和精神境界，不等同于建筑施工规范。", True),
        ("清代四王是否属于法国印象派画家？", ["镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["清代四王", "印象派", "法国"], "应纠正：清代四王是中国清代山水画家群体，不属于法国印象派。", True),
        ("马远是否是元代隐逸画家？", ["浅议宋代中国山水画的艺术特征.pdf", "隐逸文化的视觉呈现——元代山水画的象征与隐喻.pdf"], ["马远", "元代", "隐逸"], "应纠正：马远属于南宋画史语境，不应说成元代隐逸画家。", True),
        ("敦煌唐代青绿山水是否属于清代四王正统派作品？", ["从敦煌壁画看唐代青绿山水.pdf", "传统绘画的流派与创新——清代山水画概论.pdf"], ["敦煌", "唐代", "青绿山水", "清代四王"], "应区分唐代敦煌材料与清代四王，不能跨时代归属。", True),
        ("中国山水画空间营造是否完全等同于单点透视？", ["中国传统山水画的空间营造_陈慧钧.pdf", "中国山水画的布局观察与透视_王新顺.pdf"], ["空间营造", "单点透视", "山水画"], "应说明不能完全等同，中国山水画有游观、多视点和意象空间等特征。", True),
        ("王蒙是否是南宋院体画家马远的学生？", ["元代山水画衍变浅析.pdf", "浅议宋代中国山水画的艺术特征.pdf"], ["王蒙", "马远", "南宋", "元代"], "应纠正时代和谱系混淆，不能把元代王蒙说成马远学生。", True),
        ("《林泉高致》是否是为了指导 ComfyUI 工作流节点连接而写？", [], ["林泉高致", "ComfyUI", "工作流"], "应指出 ComfyUI 是现代图像生成工具，《林泉高致》不可能为此而写。", False),
        ("宋代山水画的黑白虚实是否指 OLED 屏幕显示参数？", ["中国宋代山水画中的“黑白”与“虚实”.pdf"], ["宋代", "黑白", "虚实", "OLED"], "应纠正：这里的黑白虚实是画面笔墨和空间关系，不是现代屏幕参数。", True),
        ("元代隐逸山水是否主要描绘高铁沿线景观？", ["隐逸文化的视觉呈现——元代山水画的象征与隐喻.pdf"], ["元代", "隐逸山水", "高铁"], "应指出高铁是现代交通，不属于元代隐逸山水语境。", True),
        ("明代园林题材山水画是否记录现代房地产小区景观规划？", ["明代中期园林题材山水画研究.pdf"], ["明代", "园林题材", "房地产"], "应指出现代房地产小区规划不能套入明代园林题材山水画。", True),
        ("王维是否使用 Midjourney 生成诗中有画的山水效果？", [], ["王维", "Midjourney", "山水"], "应指出 Midjourney 是现代生成式工具，王维不可能使用。", False),
        ("荆浩《笔法记》是否提出了 Transformer 注意力机制？", [], ["荆浩", "笔法记", "Transformer"], "应指出 Transformer 是现代深度学习架构，不能归入古代画论。", False),
        ("倪瓒是否是清代宫廷正统派四王之一？", ["元代山水画衍变浅析.pdf", "镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["倪瓒", "清代", "四王", "元代"], "应纠正：倪瓒是元代画家，不是清代四王之一。", True),
        ("米氏云山是否是现代云计算架构图？", ["中国山水画的皴法研究_王旭峰.pdf"], ["米氏云山", "云计算", "山水画"], "应纠正：米氏云山是画史风格/技法相关概念，不是云计算架构图。", True),
        ("青绿山水是否因为使用 RGB 显示器才呈现青绿色？", ["从敦煌壁画看唐代青绿山水.pdf"], ["青绿山水", "RGB", "设色"], "应指出青绿山水与传统设色材料和审美有关，不是 RGB 显示器效果。", True),
        ("王时敏是否参与设计现代相机镜头透视规范？", ["镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["王时敏", "相机镜头", "透视"], "应指出王时敏为清代画家，不可能参与现代相机技术规范。", True),
        ("《富春山居图》是否是一幅 Stable Diffusion 生成图？", ["中国山水画皴法研究_邱梅.pdf", "元代山水画衍变浅析.pdf"], ["富春山居图", "Stable Diffusion", "黄公望"], "应纠正：《富春山居图》是传统绘画作品，不是现代生成式模型作品。", True),
        ("南宋边角构图是否由手机竖屏短视频比例决定？", ["浅议宋代中国山水画的艺术特征.pdf"], ["南宋", "边角构图", "手机短视频"], "应指出手机短视频比例为现代媒介条件，不能解释南宋边角构图起源。", True),
        ("清初四王的仿古是否等同于复制粘贴网页素材？", ["浅论清代山水画中的“仿”“摹”“拟”“临”现象.pdf", "镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["清初四王", "仿古", "复制粘贴"], "应说明仿古是画学学习和传统转化，不等同于现代复制粘贴。", True),
        ("三远法是否包括近景、中景、远景三段摄影景别？", ["谈郭熙“三远法”对山水画创作的影响_NormalPdf.pdf"], ["三远法", "摄影景别", "高远", "深远", "平远"], "应纠正：三远法为高远、深远、平远，不是摄影景别分类。", True),
        ("斧劈皴是否是木工斧头劈柴的施工安全规范？", ["中国山水画的皴法研究_王旭峰.pdf"], ["斧劈皴", "皴法", "施工规范"], "应解释斧劈皴是山水画皴法，不是施工规范。", True),
        ("明代秋景山水画是否主要记录空调制冷技术？", ["明代秋景山水画探析.pdf"], ["明代", "秋景山水", "空调"], "应指出空调技术为现代物质条件，不属于明代秋景山水画主题。", True),
        ("山水画中的点景建筑是否都是现代摩天楼？", ["浅析建筑点景在中国山水画中的表现.pdf"], ["点景建筑", "摩天楼", "山水画"], "应说明点景建筑需结合传统山水图像语境，不应替换为现代摩天楼。", True),
        ("王翚是否直接参与开发图像生成模型的采样器？", ["镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["王翚", "图像生成模型", "采样器"], "应指出王翚是清代画家，不可能参与现代生成模型开发。", True),
        ("唐代敦煌壁画中的青绿山水是否使用数位板绘制？", ["从敦煌壁画看唐代青绿山水.pdf"], ["唐代", "敦煌壁画", "数位板"], "应指出数位板是现代设备，不能用于解释唐代壁画创作。", True),
        ("元四家是否由王时敏、王鉴、王翚、王原祁组成？", ["元代山水画衍变浅析.pdf", "镣铐下的舞蹈——清代初期_四王_山水画现象透视.pdf"], ["元四家", "四王", "王时敏", "黄公望"], "应纠正：王时敏、王鉴、王翚、王原祁是清初四王，不是元四家。", True),
        ("山水画留白是否必须用白色油漆涂满画面？", ["山水画留白艺术探究.pdf"], ["留白", "白色油漆", "山水画"], "应说明留白是构图和意境方法，不是必须用白色油漆涂抹。", True),
        ("清代笔墨美是否主要由打印机分辨率决定？", ["清代山水画的笔墨美.pdf"], ["清代", "笔墨美", "打印机分辨率"], "应指出打印机分辨率是现代输出参数，不能决定清代山水画笔墨美。", True),
        ("北宋点景建筑是否是现代 BIM 模型的截图？", ["北宋山水画点景建筑布局分析与应用研究.pdf"], ["北宋", "点景建筑", "BIM"], "应指出 BIM 是现代建筑信息模型，不属于北宋山水画创作条件。", True),
        ("明代北宗山水画是否属于西方抽象表现主义？", ["明代北宗山水画风格研究.pdf"], ["明代", "北宗山水", "抽象表现主义"], "应区分中国明代画史风格与西方现代艺术流派。", True),
        ("山水画的经营位置是否等同于商业店铺选址？", ["中国山水画景物布局规律研究.pdf", "中国山水画的布局_朱恒.pdf"], ["经营位置", "商业选址", "布局"], "应说明经营位置是画面布局经营，不是商业选址。", True),
        ("宋代禅境山水是否由虚拟现实头显呈现？", ["论中国宋代山水画中的“禅境”.pdf"], ["宋代", "禅境", "虚拟现实"], "应指出虚拟现实头显是现代技术，不能解释宋代山水画禅境。", True),
        ("明代桥隐主题是否专门描写现代高速公路立交桥？", ["明代山水画中桥梁意象研究.pdf"], ["明代", "桥隐", "高速公路立交桥"], "应指出现代高速公路立交桥不能套入明代桥隐语境。", True),
        ("传统山水画中的水墨是否指电脑水冷散热和黑白墨盒？", ["中国山水画皴法研究_邱梅.pdf"], ["水墨", "电脑水冷", "墨盒"], "应纠正：水墨是传统绘画媒材和笔墨语言，不是电脑硬件或打印耗材。", True),
        ("赵孟頫是否通过区块链 NFT 发行元代山水画作品？", ["元代山水画衍变浅析.pdf"], ["赵孟頫", "区块链", "NFT", "元代"], "应指出区块链和 NFT 是现代数字技术，不能解释赵孟頫时代的山水画创作或流通。", True),
    ]
    return [
        {
            "question": question,
            "gold_sources": sources,
            "expected_entities": entities,
            "answer_key": answer_key,
            "must_cite": must_cite,
        }
        for question, sources, entities, answer_key, must_cite in cases
    ]


def build_false_item(item_id: str, case: dict[str, Any], notes: str) -> dict[str, Any]:
    reject_if = [
        "顺着错误前提编造答案",
        "没有指出时代错置、概念混同或实体错配",
        "引用不存在的 PDF",
    ]
    if not case["must_cite"]:
        reject_if.append("在无文献证据时伪造文献引用")
    return common_item(
        item_id,
        "evidence_missing",
        case["question"],
        case["gold_sources"],
        case["expected_entities"],
        case["answer_key"],
        reject_if,
        must_cite=case["must_cite"],
        notes=notes,
    )


def take_rows(rows: list[dict[str, Any]], count: int, preferred_terms: list[str] | None = None) -> list[dict[str, Any]]:
    pool = ordered_pool(rows, preferred_terms)
    selected: list[dict[str, Any]] = []
    used_sources: Counter[str] = Counter()
    for row in pool:
        source = str(row["source_file"])
        if used_sources[source] >= 5:
            continue
        selected.append(row)
        used_sources[source] += 1
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only selected {len(selected)} rows, need {count}.")
    return selected


def ordered_pool(rows: list[dict[str, Any]], preferred_terms: list[str] | None = None) -> list[dict[str, Any]]:
    if preferred_terms:
        preferred = [row for row in rows if terms_in_row(row, preferred_terms)]
        preferred_ids = {row["id"] for row in preferred}
        rest = [row for row in rows if row["id"] not in preferred_ids]
        return preferred + rest
    return list(rows)


def load_existing_questions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    questions: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.add(json.loads(line)["question"])
    return questions


def build_items(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)

    items: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    next_index = 1

    def next_id() -> str:
        nonlocal next_index
        item_id = f"test_researcher_v2_{next_index:03d}"
        next_index += 1
        return item_id

    def add_unique(item: dict[str, Any]) -> bool:
        if item["question"] in used_questions:
            return False
        item["id"] = next_id()
        items.append(item)
        used_questions.add(item["question"])
        return True

    def append_row_items(
        count: int,
        maker: Any,
        preferred_terms: list[str] | None = None,
        max_per_source: int = 5,
    ) -> None:
        added = 0
        used_sources: Counter[str] = Counter()
        for row in ordered_pool(rows, preferred_terms):
            source = str(row["source_file"])
            if used_sources[source] >= max_per_source:
                continue
            item = maker("__pending__", row)
            if not add_unique(item):
                continue
            used_sources[source] += 1
            added += 1
            if added == count:
                return
        raise RuntimeError(f"Only generated {added} unique items, need {count}.")

    def append_comparison_items(count: int) -> None:
        added = 0
        used_sources: Counter[str] = Counter()
        pool = ordered_pool(rows)
        for offset in range(1, min(len(pool), 300)):
            for left_index in range(0, len(pool) - offset):
                left = pool[left_index]
                right = pool[left_index + offset]
                left_source = str(left["source_file"])
                right_source = str(right["source_file"])
                if left_source == right_source:
                    continue
                if used_sources[left_source] >= 6 or used_sources[right_source] >= 6:
                    continue
                item = make_comparison("__pending__", left, right)
                if not add_unique(item):
                    continue
                used_sources[left_source] += 1
                used_sources[right_source] += 1
                added += 1
                if added == count:
                    return
        raise RuntimeError(f"Only generated {added} unique comparison items, need {count}.")

    append_row_items(TASK_COUNTS["factual_qa"], make_factual)
    append_row_items(TASK_COUNTS["concept_explain"], make_concept, CONCEPT_TERMS)
    append_comparison_items(TASK_COUNTS["comparison"])
    append_row_items(TASK_COUNTS["composition_aesthetic"], make_composition, COMPOSITION_TERMS)
    append_row_items(TASK_COUNTS["artist_prompt"], make_artist, DOMAIN_TERMS)

    cases = false_premise_cases()
    for case in cases[: TASK_COUNTS["evidence_missing"]]:
        add_unique(build_false_item("__pending__", case, "v2 主测试集错误前提/证据不足题。"))

    false_items = [
        build_false_item(f"test_researcher_false_premise_v2_{idx:03d}", case, "v2 独立错误前提专项集。")
        for idx, case in enumerate(cases[TASK_COUNTS["evidence_missing"] : TASK_COUNTS["evidence_missing"] + 25], start=1)
    ]

    return items, false_items


def validate_items(items: list[dict[str, Any]], expected_count: int, existing_questions: set[str]) -> dict[str, Any]:
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} items, got {len(items)}")
    ids = [item["id"] for item in items]
    questions = [item["question"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids found.")
    if len(questions) != len(set(questions)):
        raise ValueError("Duplicate questions found.")
    overlap = sorted(set(questions) & existing_questions)
    if overlap:
        raise ValueError(f"Questions overlap existing v1 set: {overlap[:3]}")
    for item in items:
        missing = [
            key
            for key in ["id", "split", "task_type", "question", "gold_sources", "expected_entities", "must_cite", "answer_key", "reject_if", "review_status"]
            if key not in item
        ]
        if missing:
            raise ValueError(f"{item.get('id')} missing fields: {missing}")
    return {
        "count": len(items),
        "task_counts": dict(Counter(item["task_type"] for item in items)),
        "must_cite_counts": dict(Counter(str(item["must_cite"]) for item in items)),
        "unique_gold_sources": len({source for item in items for source in item["gold_sources"]}),
        "needs_human_review": sum(1 for item in items if item["review_status"] == "needs_human_review"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-output", default=str(DEFAULT_MAIN_OUTPUT))
    parser.add_argument("--false-output", default=str(DEFAULT_FALSE_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--milvus-limit", type=int, default=5000)
    args = parser.parse_args()

    existing_questions = load_existing_questions(V1_PATH)
    rows = fetch_rows(args.milvus_limit)
    items, false_items = build_items(rows, args.seed)

    main_summary = validate_items(items, 200, existing_questions)
    false_summary = validate_items(false_items, 25, existing_questions | {item["question"] for item in items})

    main_output = Path(args.main_output)
    false_output = Path(args.false_output)
    summary_output = Path(args.summary_output)
    write_jsonl(main_output, items)
    write_jsonl(false_output, false_items)

    summary = {
        "seed": args.seed,
        "source_rows_loaded": len(rows),
        "main_output": str(main_output),
        "false_premise_output": str(false_output),
        "main": main_summary,
        "false_premise": false_summary,
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
