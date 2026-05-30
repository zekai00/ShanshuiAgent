#!/usr/bin/env python3
"""Build a faceted taxonomy index for the curated literature registry."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path("/root/Workspace/ChineseLandscape")
META_ROOT = PROJECT_ROOT / "data" / "metadata"
REGISTRY_PATH = META_ROOT / "文献级标注清单.jsonl"
IMPORT_QUEUE_PATH = META_ROOT / "高权重导入队列.jsonl"
FACETED_REGISTRY_PATH = META_ROOT / "文献多维标注清单.jsonl"
INDEX_JSON_PATH = META_ROOT / "文献多维分类索引.json"
INDEX_MD_PATH = META_ROOT / "文献多维分类索引.md"
STATS_PATH = META_ROOT / "文献级标注统计.json"

TZ = ZoneInfo("Asia/Shanghai")


PERSON_NAMES = [
    "张彦远",
    "宗炳",
    "王微",
    "谢赫",
    "謝赫",
    "荊浩",
    "荆浩",
    "李思训",
    "李思訓",
    "王维",
    "王維",
    "关仝",
    "關仝",
    "郭熙",
    "郭思",
    "郭若虚",
    "郭若虛",
    "米芾",
    "米友仁",
    "石涛",
    "石濤",
    "展子虔",
    "范宽",
    "范寬",
    "李唐",
    "董源",
    "巨然",
    "赵孟頫",
    "趙孟頫",
    "黄公望",
    "黃公望",
    "倪瓒",
    "倪瓚",
    "吴镇",
    "吳鎮",
    "王蒙",
    "沈周",
    "文徵明",
    "唐寅",
    "董其昌",
    "龚贤",
    "龔賢",
    "萧云从",
    "蕭雲從",
    "安岐",
    "王时敏",
    "王時敏",
    "王鉴",
    "王鑑",
    "王翚",
    "王翬",
    "王原祁",
    "弘仁",
    "渐江",
    "漸江",
    "髡残",
    "髡殘",
    "八大山人",
    "朱耷",
    "马麟",
    "馬麟",
    "恽寿平",
    "惲壽平",
    "吴历",
    "吳歷",
    "黄慎",
    "黃慎",
    "华喦",
    "華嵒",
    "华岩",
    "黄宾虹",
    "黃賓虹",
    "傅抱石",
    "李可染",
    "张大千",
    "張大千",
    "谢稚柳",
    "謝稚柳",
]

PERSON_NORMALIZATION = {
    "荊浩": "荆浩",
    "謝赫": "谢赫",
    "李思訓": "李思训",
    "王維": "王维",
    "關仝": "关仝",
    "郭若虛": "郭若虚",
    "石濤": "石涛",
    "范寬": "范宽",
    "趙孟頫": "赵孟頫",
    "黃公望": "黄公望",
    "倪瓚": "倪瓒",
    "吳鎮": "吴镇",
    "龔賢": "龚贤",
    "蕭雲從": "萧云从",
    "漸江": "渐江",
    "髡殘": "髡残",
    "馬麟": "马麟",
    "王時敏": "王时敏",
    "王鑑": "王鉴",
    "王翬": "王翚",
    "惲壽平": "恽寿平",
    "吳歷": "吴历",
    "黃慎": "黄慎",
    "華嵒": "华喦",
    "华岩": "华喦",
    "黃賓虹": "黄宾虹",
    "張大千": "张大千",
    "謝稚柳": "谢稚柳",
}

WORK_RULES = {
    "歷代名畫記": ["歷代名畫記", "历代名画记"],
    "林泉高致": ["林泉高致"],
    "敘畫": ["敘畫", "叙画"],
    "古畫品錄": ["古畫品錄", "古画品录"],
    "圖畫見聞誌": ["圖畫見聞誌", "图画见闻志"],
    "畫史": ["畫史", "画史"],
    "畫禪室隨筆": ["畫禪室隨筆", "画禅室随笔"],
    "宣和畫譜": ["宣和畫譜", "宣和画谱"],
    "筆法記": ["筆法記", "笔法记"],
    "畫山水賦": ["畫山水賦", "画山水赋", "山水诀"],
    "苦瓜和尚畫語錄": ["苦瓜和尚畫語錄", "苦瓜和尚画语录"],
    "游春图": ["游春图", "遊春圖"],
    "青绿山水图": ["青绿山水图", "青綠山水圖"],
    "山水图册": ["山水图册", "山水圖冊", "《山水图》册", "山水图》册", "《山水圖》冊", "山水圖》冊"],
    "重江叠嶂图": ["重江叠嶂图", "重江疊嶂圖"],
    "董范合参图": ["董范合参图", "董范合參圖"],
    "小中现大图": ["小中现大图", "小中現大圖"],
    "溪山行旅图": ["溪山行旅图", "溪山行旅圖"],
    "早春图": ["早春图", "早春圖"],
    "万壑松风图": ["万壑松风图", "萬壑松風圖"],
    "夏山图": ["夏山图", "夏山圖", "Summer Mountains"],
    "溪岸图": ["溪岸图", "溪岸圖", "Riverbank"],
    "虞山林壑图": ["虞山林壑图", "虞山林壑圖", "Woods and Valleys of Mount Yu"],
    "渔父图": ["渔父图", "漁父圖", "Fisherman"],
    "古木遥岑图": ["Old Trees, Level Distance", "古木遥岑", "古木遙岑"],
    "泰山图": ["Mount Taihang", "泰山", "太行"],
    "南巡图": ["南巡", "Southern Inspection Tour"],
    "虎丘十二景": ["Tiger Hill", "虎丘"],
    "江峡图": ["Yangzi River Gorge", "长江", "江峡"],
    "青卞山图": ["Mt. Qingbian", "青卞"],
    "桃花源": ["桃花源", "Peach Blossom Spring"],
    "松壑清泉图": ["松壑清泉图", "松壑清泉圖"],
    "秋日山居图": ["秋日山居图", "秋日山居圖"],
    "罗浮图": ["罗浮图", "羅浮圖"],
    "道统圣贤图": ["道统圣贤图", "道統聖賢圖"],
}

LINEAGE_RULES = {
    "青绿山水传统": ["青绿山水", "青绿", "敦煌", "游春图", "展子虔"],
    "北宋山水与三远传统": ["北宋山水", "巨碑式", "三远", "平远", "范宽", "郭熙", "李唐", "早春图", "溪山行旅图"],
    "董巨江南山水传统": ["董源", "巨然", "董巨", "江南", "董范合参图", "Dong Yuan", "Juran", "Riverbank", "披麻"],
    "元四家与元代文人山水": ["黄公望", "吴镇", "倪瓒", "王蒙", "元四家", "富春山居", "渔父", "虞山林壑"],
    "文人画传统": ["赵孟頫", "王蒙", "黄公望", "吴镇", "倪瓒", "元四家", "文人画", "元代隐逸"],
    "吴门画派": ["吴门", "吳門", "沈周", "文徵明", "唐寅"],
    "松江派与董其昌系统": ["松江派", "董其昌", "画禅", "董其昌系统"],
    "南北宗论述系统": ["南北宗", "南宗", "北宗"],
    "清初四王与正统派": ["四王", "王时敏", "王鉴", "王翚", "王原祁", "正统派", "南宗正脉"],
    "四僧与石涛系统": ["四僧", "弘仁", "渐江", "髡残", "八大山人", "朱耷", "石涛", "苦瓜和尚", "一画"],
    "恽寿平-王翚关系": ["恽寿平", "王翚", "逸品"],
    "金陵画派": ["龚贤", "金陵"],
    "新安画派": ["新安", "萧云从", "弘仁", "渐江"],
    "扬州画派": ["扬州", "黄慎", "华喦", "华岩", "Hua Yan", "八怪"],
    "近现代中西转型": ["近现代", "中西", "Twentieth", "Nineteenth", "modern", "黄宾虹", "傅抱石", "李可染", "张大千", "谢稚柳"],
}

STYLE_RULES = {
    "青绿设色": ["青绿", "设色", "礦物", "矿物", "石青", "石绿"],
    "水墨笔墨": ["水墨", "笔墨", "筆墨", "墨笔", "一画"],
    "气韵六法": ["气韵", "氣韻", "六法", "六要"],
    "三远与平远": ["三远", "平远", "高远", "深远"],
    "巨碑式山水": ["巨碑式", "范宽", "溪山行旅图"],
    "皴法与山石树法": ["皴", "披麻皴", "山石", "树石", "古松"],
    "空间构图与位置经营": ["构图", "图式", "位置经营", "布局", "空间", "点线面", "形式构成", "构成"],
    "留白与虚实": ["留白", "虚实", "黑白", "禅境"],
    "建筑桥梁点景": ["桥梁", "建筑点景", "点景建筑", "建筑", "点景"],
    "审美与自然观": ["审美", "自然观", "自然", "真我"],
    "园林庭园与隐逸": ["园林", "庭园", "桃花源", "隐逸", "清居"],
    "诗书画题跋": ["诗书画", "題畫", "题画", "题跋", "书画关系", "书法", "行书", "calligraphy", "poetry"],
    "仿古摹古临拟": ["仿古", "摹古", "临仿", "临古", "拟", "南宗正脉"],
    "鉴藏辨伪": ["鉴藏", "辨伪", "安岐", "品评", "收藏"],
    "作品图录与馆藏说明": ["作品图录", "馆藏", "藏品", "Collection", "catalog", "導覽", "导览", "C.C. Wang"],
    "中西交汇": ["中西", "近现代", "Two Cultures"],
}

THEME_RULES = {
    "画论原典": ["画论", "畫論", "林泉高致", "筆法記", "历代名画记", "苦瓜和尚"],
    "发展脉络": ["继承", "创新", "发展", "转变", "传统", "源流", "宋元", "唐宋元"],
    "风格流派": ["风格", "画派", "南北宗", "松江派", "吴门", "正统派", "四王"],
    "风格技法": ["留白", "虚实", "黑白", "空间", "布局", "皴法", "笔墨", "桥梁", "建筑", "点景", "构成", "禅境", "审美"],
    "作品个案": ["图卷", "图轴", "图册", "作品", "小中现大图", "游春图", "重江叠嶂图"],
    "鉴定与真伪": ["辨伪", "鉴", "品评", "鉴藏"],
    "教学与展览": ["导览", "導覽", "education", "resource", "展览"],
    "图录与收藏": ["图录", "catalog", "Collection", "收藏"],
}

SOURCE_ROLE_MAP = {
    "ancient_theory_scan_pdf": "原典影印",
    "ancient_theory_text_pdf": "原典文本整理",
    "museum_catalog": "博物馆图录",
    "symposium_proceedings": "学术论文集",
    "museum_exhibition_guide": "展览导览",
    "museum_education_resource": "教学资料",
    "palace_museum_article": "专题论文",
    "palace_museum_article_merged_pdf": "专题论文",
    "museum_collection_entry_text_pdf": "作品级馆藏条目",
    "legacy_project_pdf": "既有项目文献（待溯源）",
}

PERIOD_BY_DYNASTY = {
    "魏晋南北朝": "魏晋南北朝：早期画论与山水观念",
    "南朝": "魏晋南北朝：早期画论与山水观念",
    "刘宋": "魏晋南北朝：早期画论与山水观念",
    "南齐": "魏晋南北朝：早期画论与山水观念",
    "隋": "隋唐五代：早期山水与青绿传统",
    "唐": "隋唐五代：早期山水与青绿传统",
    "五代": "隋唐五代：早期山水与青绿传统",
    "北宋": "宋代：山水成熟与理论化",
    "南宋": "宋代：山水成熟与理论化",
    "宋": "宋代：山水成熟与理论化",
    "元": "元代：文人画与复古传统",
    "明": "明代：吴门、松江与南北宗",
    "清": "清代：正统派、四王四僧与笔墨再阐释",
    "近现代": "近现代：中国画转型",
}


def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M CST")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def row_text(row: dict[str, Any]) -> str:
    parts: list[str] = [
        str(row.get("doc_id", "")),
        str(row.get("title", "")),
        str(row.get("author", "")),
        str(row.get("source_type", "")),
        str(row.get("category", "")),
        str(row.get("curation_axis", "")),
    ]
    parts.extend(str(item) for item in row.get("dynasties", []))
    parts.extend(str(item) for item in row.get("topics", []))
    return " ".join(parts)


def match_rules(text: str, rules: dict[str, list[str]]) -> list[str]:
    matched: list[str] = []
    for label, needles in rules.items():
        if any(needle and needle.lower() in text.lower() for needle in needles):
            matched.append(label)
    return matched


def normalize_person(name: str) -> str:
    return PERSON_NORMALIZATION.get(name, name)


def classify(row: dict[str, Any]) -> dict[str, list[str]]:
    text = row_text(row)
    dynasties = list(dict.fromkeys(row.get("dynasties", [])))
    periods = [PERIOD_BY_DYNASTY[d] for d in dynasties if d in PERIOD_BY_DYNASTY]
    if len(set(periods)) >= 3:
        periods.append("跨朝代/通史")

    persons = []
    for name in PERSON_NAMES:
        if name in text:
            persons.append(normalize_person(name))

    works = match_rules(text, WORK_RULES)
    lineages = match_rules(text, LINEAGE_RULES)
    styles = match_rules(text, STYLE_RULES)
    themes = match_rules(text, THEME_RULES)
    source_roles = [SOURCE_ROLE_MAP.get(str(row.get("source_type")), str(row.get("source_type")))]

    if row.get("curation_axis") == "ancient_theory" and "画论原典" not in themes:
        themes.append("画论原典")
    if row.get("curation_axis") == "development_lineage" and "发展脉络" not in themes:
        themes.append("发展脉络")
    if row.get("curation_axis") == "style_school" and "风格流派" not in themes:
        themes.append("风格流派")
    if row.get("curation_axis") == "ancient_theory":
        styles = [style for style in styles if style != "作品图录与馆藏说明"]
    if styles and "风格技法" not in themes and row.get("curation_axis") == "style_technique":
        themes.append("风格技法")

    if not lineages and any(d in dynasties for d in ["宋", "元"]):
        lineages.append("宋元山水传统")

    return {
        "periods": sorted(set(periods)),
        "dynasties": dynasties,
        "lineages_schools": sorted(set(lineages)),
        "styles_techniques": sorted(set(styles)),
        "persons": sorted(set(persons)),
        "works": sorted(set(works)),
        "themes": sorted(set(themes)),
        "source_roles": source_roles,
    }


def doc_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "source_file": row["source_file"],
        "authority_level": row.get("authority_level"),
        "rag_import_weight": row.get("rag_import_weight"),
    }


def build_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "periods": "发展阶段，不等同于单一朝代；一篇跨朝代资料可进入多个阶段。",
        "dynasties": "朝代标签，保留原始文献覆盖范围。",
        "lineages_schools": "流派、谱系、理论系统，如吴门、松江、南北宗、四王。",
        "styles_techniques": "风格、技法、题材和观看问题，如青绿、笔墨、三远、园林。",
        "persons": "主要画家、理论家、鉴藏家。",
        "works": "主要作品或原典名称。",
        "themes": "问题域，用于评测/RAG 查询构造。",
        "source_roles": "文献在证据链中的角色，如原典、图录、专题论文、馆藏条目。",
    }
    index: dict[str, dict[str, list[dict[str, Any]]]] = {key: defaultdict(list) for key in dimensions}
    for row in rows:
        facets = row["facets"]
        summary = doc_summary(row)
        for dimension in dimensions:
            for label in facets.get(dimension, []):
                index[dimension][label].append(summary)

    return {
        "generated_at": now_str(),
        "taxonomy_version": "2026-05-30.faceted-v1",
        "principle": "文献不做单一归类，而做多维标签；物理目录只表达来源/用途，检索和评测使用 facets。",
        "dimensions": dimensions,
        "index": {dim: dict(sorted(labels.items())) for dim, labels in index.items()},
    }


def write_markdown(index: dict[str, Any]) -> None:
    lines = [
        "# 文献多维分类索引",
        "",
        f"生成时间：{index['generated_at']}",
        "",
        "## 分类原则",
        "",
        "不把一篇文献强行放进唯一目录。目录只解决文件管理；真正用于 RAG、训练和评测的是多维标签。",
        "",
        "一篇文献可以同时属于多个朝代、流派、风格技法、人物和主题。例如董其昌青绿山水相关文献，既属于明代，也属于松江派/董其昌系统、南北宗论述、青绿设色、文人画和作品个案。",
        "",
        "## 维度说明",
        "",
    ]
    for dimension, description in index["dimensions"].items():
        lines.append(f"- `{dimension}`：{description}")
    lines.append("")

    for dimension, labels in index["index"].items():
        lines.append(f"## {dimension}")
        lines.append("")
        for label, docs in sorted(labels.items(), key=lambda item: (-len(item[1]), item[0])):
            lines.append(f"### {label}（{len(docs)}）")
            for doc in docs:
                lines.append(f"- `{doc['doc_id']}`｜{doc['title']}")
            lines.append("")

    INDEX_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = read_jsonl(REGISTRY_PATH)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        facets = classify(row)
        row = dict(row)
        row["facets"] = facets
        for key, value in facets.items():
            row[key] = value
        row["taxonomy_version"] = "2026-05-30.faceted-v1"
        row["taxonomy_updated_at"] = now_str()
        enriched.append(row)

    enriched.sort(key=lambda r: (r.get("category", ""), r.get("doc_id", "")))
    write_jsonl(REGISTRY_PATH, enriched)
    write_jsonl(FACETED_REGISTRY_PATH, enriched)

    import_rows = sorted(enriched, key=lambda r: (-float(r["rag_import_weight"]), r["category"], r["source_file"]))
    write_jsonl(IMPORT_QUEUE_PATH, import_rows)

    index = build_index(enriched)
    INDEX_JSON_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(index)

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}
    stats["taxonomy_generated_at"] = index["generated_at"]
    stats["faceted_registry_path"] = str(FACETED_REGISTRY_PATH)
    stats["taxonomy_index_json_path"] = str(INDEX_JSON_PATH)
    stats["taxonomy_index_md_path"] = str(INDEX_MD_PATH)
    stats["by_period"] = {label: len(docs) for label, docs in index["index"]["periods"].items()}
    stats["by_lineage_school"] = {label: len(docs) for label, docs in index["index"]["lineages_schools"].items()}
    stats["by_style_technique"] = {label: len(docs) for label, docs in index["index"]["styles_techniques"].items()}
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "documents": len(enriched),
        "faceted_registry": str(FACETED_REGISTRY_PATH),
        "index_json": str(INDEX_JSON_PATH),
        "index_md": str(INDEX_MD_PATH),
        "periods": stats["by_period"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
