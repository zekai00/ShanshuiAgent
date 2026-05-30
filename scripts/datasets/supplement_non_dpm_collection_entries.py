#!/usr/bin/env python3
"""Add non-DPM authoritative museum object entries to the corpus.

This batch is intentionally object-level. It fills gaps where the RAG system
needs stable evidence for individual works, artists, accession numbers,
collection pages, and museum descriptions rather than another institution-level
essay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from curate_authority_corpus import (
    build_registry,
    ensure_dirs,
    finalize_entry,
    is_pdf_complete,
    load_sources,
    out_path_for,
    write_manifest_and_source_list,
    write_sources,
    write_text_pdf,
)


CATEGORY = "09_非故宫作品级馆藏条目"
SOURCE_TYPE = "museum_collection_entry_text_pdf"
TIMEOUT = (15, 60)


MET_OBJECTS: list[dict[str, Any]] = [
    {
        "id": "met_object_guoxi_old_trees_level_distance_39668",
        "object_id": 39668,
        "title": "郭熙《古木遥岑图》（Old Trees, Level Distance）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["北宋", "宋"],
        "topics": ["郭熙", "古木遥岑图", "Old Trees, Level Distance", "三远", "平远", "北宋山水"],
        "filename": "I01_Met_北宋郭熙_古木遥岑图_馆藏条目.pdf",
    },
    {
        "id": "met_object_mi_youren_cloudy_mountains_40007",
        "object_id": 40007,
        "title": "米友仁《云山图》（Cloudy Mountains）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["南宋", "宋"],
        "topics": ["米友仁", "米氏云山", "Cloudy Mountains", "水墨", "文人画", "南宋山水"],
        "filename": "I02_Met_南宋米友仁_云山图_馆藏条目.pdf",
    },
    {
        "id": "met_object_dongyuan_riverbank_39542",
        "object_id": 39542,
        "title": "董源《溪岸图》（Riverbank）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["五代", "宋"],
        "topics": ["董源", "溪岸图", "Riverbank", "董巨江南山水", "披麻皴", "江南山水"],
        "filename": "I03_Met_董源_溪岸图_馆藏条目.pdf",
    },
    {
        "id": "met_object_ni_zan_woods_valleys_mount_yu_45636",
        "object_id": 45636,
        "title": "倪瓒《虞山林壑图》（Woods and Valleys of Mount Yu）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["元"],
        "topics": ["倪瓒", "虞山林壑图", "Woods and Valleys of Mount Yu", "元四家", "文人画", "疏淡"],
        "filename": "I04_Met_元倪瓒_虞山林壑图_馆藏条目.pdf",
    },
    {
        "id": "met_object_ni_zan_wind_riverbank_41154",
        "object_id": 41154,
        "title": "倪瓒《江岸风林图》（Wind among the Trees on the Riverbank）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["元"],
        "topics": ["倪瓒", "Wind among the Trees on the Riverbank", "元四家", "文人画", "题跋"],
        "filename": "I05_Met_元倪瓒_江岸风林图_馆藏条目.pdf",
    },
    {
        "id": "met_object_wu_zhen_fisherman_41468",
        "object_id": 41468,
        "title": "吴镇《渔父图》（Fisherman）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["元"],
        "topics": ["吴镇", "渔父图", "Fisherman", "元四家", "文人画", "隐逸"],
        "filename": "I06_Met_元吴镇_渔父图_馆藏条目.pdf",
    },
    {
        "id": "met_object_wu_zhen_crooked_pine_41462",
        "object_id": 41462,
        "title": "吴镇《古松图》（Crooked Pine）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["元"],
        "topics": ["吴镇", "Crooked Pine", "元四家", "文人画", "水墨", "题跋"],
        "filename": "I07_Met_元吴镇_古松图_馆藏条目.pdf",
    },
    {
        "id": "met_object_wang_yuanqi_after_wu_zhen_49185",
        "object_id": 49185,
        "title": "王原祁《仿吴镇山水》（Landscape after Wu Zhen）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["清", "元"],
        "topics": ["王原祁", "吴镇", "仿古", "清初四王", "元四家", "正统派"],
        "filename": "I08_Met_清王原祁_仿吴镇山水_馆藏条目.pdf",
    },
    {
        "id": "met_object_wang_hui_colors_taihang_49151",
        "object_id": 49151,
        "title": "王翚《太行山色图》（The Colors of Mount Taihang）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["清"],
        "topics": ["王翚", "太行山色图", "The Colors of Mount Taihang", "清初四王", "仿古", "正统派"],
        "filename": "I09_Met_清王翚_太行山色图_馆藏条目.pdf",
    },
    {
        "id": "met_object_wang_hui_snow_clearing_40021",
        "object_id": 40021,
        "title": "王翚《仿李成雪霁山水》（Snow Clearing）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["清", "北宋"],
        "topics": ["王翚", "李成", "Snow Clearing", "清初四王", "仿古", "北宋传统"],
        "filename": "I10_Met_清王翚_仿李成雪霁山水_馆藏条目.pdf",
    },
    {
        "id": "met_object_shen_zhou_silent_fisherman_45682",
        "object_id": 45682,
        "title": "沈周《秋林渔隐》（Silent Fisherman in an Autumn Wood）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["明"],
        "topics": ["沈周", "吴门画派", "Silent Fisherman in an Autumn Wood", "隐逸", "文人画"],
        "filename": "I11_Met_明沈周_秋林渔隐_馆藏条目.pdf",
    },
    {
        "id": "met_object_shen_zhou_autumn_colors_45683",
        "object_id": 45683,
        "title": "沈周《秋山图》（Autumn Colors among Streams and Mountains）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["明"],
        "topics": ["沈周", "吴门画派", "Autumn Colors among Streams and Mountains", "文人画", "山水"],
        "filename": "I12_Met_明沈周_秋山图_馆藏条目.pdf",
    },
    {
        "id": "met_object_gong_xian_landscapes_poems_36131",
        "object_id": 36131,
        "title": "龚贤《山水诗册》（Landscapes with poems）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["清"],
        "topics": ["龚贤", "金陵画派", "Landscapes with poems", "诗书画题跋", "水墨笔墨"],
        "filename": "I13_Met_清龚贤_山水诗册_馆藏条目.pdf",
    },
    {
        "id": "met_object_huang_binhong_landscape_36438",
        "object_id": 36438,
        "title": "黄宾虹《山水》（Landscape）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["黄宾虹", "近现代", "山水", "水墨笔墨", "中西转型"],
        "filename": "I14_Met_近现代黄宾虹_山水_馆藏条目.pdf",
    },
    {
        "id": "met_object_huang_binhong_ten_thousand_valleys_41924",
        "object_id": 41924,
        "title": "黄宾虹《万壑浓荫》（Ten Thousand Valleys in Deep Shade）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["黄宾虹", "近现代", "Ten Thousand Valleys in Deep Shade", "积墨", "水墨笔墨"],
        "filename": "I15_Met_近现代黄宾虹_万壑浓荫_馆藏条目.pdf",
    },
    {
        "id": "met_object_fu_baoshi_yangzi_gorge_772742",
        "object_id": 772742,
        "title": "傅抱石《长江峡谷》（Yangzi River Gorge）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["傅抱石", "长江峡谷", "Yangzi River Gorge", "近现代", "水墨笔墨", "中西转型"],
        "filename": "I16_Met_近现代傅抱石_长江峡谷_馆藏条目.pdf",
    },
    {
        "id": "met_object_fu_baoshi_solitary_traveler_772743",
        "object_id": 772743,
        "title": "傅抱石《山中独行者》（Solitary traveler in the mountains）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["傅抱石", "Solitary traveler in the mountains", "近现代", "山水", "水墨笔墨"],
        "filename": "I17_Met_近现代傅抱石_山中独行者_馆藏条目.pdf",
    },
    {
        "id": "met_object_li_keran_cottage_by_river_44628",
        "object_id": 44628,
        "title": "李可染《河边茅舍》（Cottage by the River）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["李可染", "Cottage by the River", "近现代", "山水", "写生", "水墨笔墨"],
        "filename": "I18_Met_近现代李可染_河边茅舍_馆藏条目.pdf",
    },
    {
        "id": "met_object_zhang_daqian_recluse_waterfall_772745",
        "object_id": 772745,
        "title": "张大千《高士观瀑》（Recluse Gazing at a Waterfall）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["张大千", "Recluse Gazing at a Waterfall", "近现代", "山水", "设色"],
        "filename": "I19_Met_近现代张大千_高士观瀑_馆藏条目.pdf",
    },
    {
        "id": "met_object_xie_zhiliu_landscape_75680",
        "object_id": 75680,
        "title": "谢稚柳《山水》（Landscape）Met 馆藏条目",
        "author": "The Metropolitan Museum of Art",
        "dynasties": ["近现代"],
        "topics": ["谢稚柳", "近现代", "山水", "鉴藏", "书画传统"],
        "filename": "I20_Met_近现代谢稚柳_山水_馆藏条目.pdf",
    },
]


CLEVELAND_OBJECTS: list[dict[str, Any]] = [
    {
        "id": "cma_object_juran_buddhist_retreat_135836",
        "object_id": 135836,
        "title": "巨然《溪山兰若图》（Buddhist Retreat by Stream and Mountains）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["五代", "北宋", "宋"],
        "topics": ["巨然", "Buddhist Retreat by Stream and Mountains", "董巨江南山水", "江南山水", "披麻皴"],
        "filename": "I21_Cleveland_巨然_溪山兰若图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_huang_gongwang_summer_mountains_156183",
        "object_id": 156183,
        "title": "黄公望《仿董源夏山图》（Summer Mountains after Dong Yuan）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["元", "五代"],
        "topics": ["黄公望", "董源", "Summer Mountains", "元四家", "董巨江南山水", "仿古"],
        "filename": "I22_Cleveland_元黄公望_仿董源夏山图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_zhao_mengfu_river_village_149413",
        "object_id": 149413,
        "title": "赵孟頫《水村图》（River Village）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["元"],
        "topics": ["赵孟頫", "水村图", "River Village", "元代文人画", "书画题跋"],
        "filename": "I23_Cleveland_元赵孟頫_水村图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_ni_zan_mountains_immortals_160137",
        "object_id": 160137,
        "title": "倪瓒等《仙山图》（Mountains of the Immortals）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["元"],
        "topics": ["倪瓒", "Mountains of the Immortals", "元四家", "文人画", "题跋"],
        "filename": "I24_Cleveland_元倪瓒_仙山图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_wang_yuanqi_after_ni_zan_132016",
        "object_id": 132016,
        "title": "王原祁《仿倪瓒山水》（Landscape after Ni Zan）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清", "元"],
        "topics": ["王原祁", "倪瓒", "仿古", "清初四王", "元四家", "正统派"],
        "filename": "I25_Cleveland_清王原祁_仿倪瓒山水_馆藏条目.pdf",
    },
    {
        "id": "cma_object_gong_xian_dong_juran_144286",
        "object_id": 144286,
        "title": "龚贤《仿董源巨然山水》（Landscape in the Style of Dong Yuan and Juran）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清", "五代"],
        "topics": ["龚贤", "金陵画派", "董源", "巨然", "董巨江南山水", "仿古"],
        "filename": "I26_Cleveland_清龚贤_仿董源巨然山水_馆藏条目.pdf",
    },
    {
        "id": "cma_object_dong_qichang_mt_qingbian_149791",
        "object_id": 149791,
        "title": "董其昌《青卞山图》（Mt. Qingbian）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["明"],
        "topics": ["董其昌", "青卞山图", "Mt. Qingbian", "松江派", "南北宗", "文人画"],
        "filename": "I27_Cleveland_明董其昌_青卞山图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_dong_qichang_clear_autumn_135964",
        "object_id": 135964,
        "title": "董其昌《秋山晴霭图》（River and Mountains on a Clear Autumn Day）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["明"],
        "topics": ["董其昌", "River and Mountains on a Clear Autumn Day", "松江派", "南北宗", "文人画"],
        "filename": "I28_Cleveland_明董其昌_秋山晴霭图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_wen_zhengming_qin_valley_144829",
        "object_id": 144829,
        "title": "文徵明《幽谷鸣琴》（Playing the Qin in a Secluded Valley）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["明"],
        "topics": ["文徵明", "吴门画派", "Playing the Qin in a Secluded Valley", "园林隐逸", "文人画"],
        "filename": "I29_Cleveland_明文徵明_幽谷鸣琴_馆藏条目.pdf",
    },
    {
        "id": "cma_object_wen_zhengming_old_trees_159727",
        "object_id": 159727,
        "title": "文徵明《寒溪古木》（Old Trees by a Wintry Brook）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["明"],
        "topics": ["文徵明", "吴门画派", "Old Trees by a Wintry Brook", "文人画", "水墨笔墨"],
        "filename": "I30_Cleveland_明文徵明_寒溪古木_馆藏条目.pdf",
    },
    {
        "id": "cma_object_shen_zhou_tiger_hill_140440",
        "object_id": 140440,
        "title": "沈周《虎丘十二景》（Twelve Views of Tiger Hill）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["明"],
        "topics": ["沈周", "虎丘十二景", "Tiger Hill", "吴门画派", "园林庭园", "文人画"],
        "filename": "I31_Cleveland_明沈周_虎丘十二景_馆藏条目.pdf",
    },
    {
        "id": "cma_object_shen_zhou_thousand_acres_clouds_140452",
        "object_id": 140452,
        "title": "沈周《云千顷》（The Thousand Acres of Clouds）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["明"],
        "topics": ["沈周", "虎丘十二景", "The Thousand Acres of Clouds", "吴门画派", "园林庭园"],
        "filename": "I32_Cleveland_明沈周_云千顷_馆藏条目.pdf",
    },
    {
        "id": "cma_object_hua_yan_conversation_autumn_131673",
        "object_id": 131673,
        "title": "华喦《秋谈图》（Conversation in Autumn）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清"],
        "topics": ["华喦", "扬州画派", "Conversation in Autumn", "文人画", "山水"],
        "filename": "I33_Cleveland_清华喦_秋谈图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_hua_yan_landscape_album_151024",
        "object_id": 151024,
        "title": "华喦《古诗山水册》（Album of Landscape Painting Illustrating Old Poems）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清"],
        "topics": ["华喦", "扬州画派", "古诗山水册", "诗书画题跋", "山水"],
        "filename": "I34_Cleveland_清华喦_古诗山水册_馆藏条目.pdf",
    },
    {
        "id": "cma_object_xiao_yuncong_pure_tones_131672",
        "object_id": 131672,
        "title": "萧云从《山水清音》（Pure Tones among Hills and Waters）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清"],
        "topics": ["萧云从", "新安画派", "Pure Tones among Hills and Waters", "山水", "水墨笔墨"],
        "filename": "I35_Cleveland_清萧云从_山水清音_馆藏条目.pdf",
    },
    {
        "id": "cma_object_xiao_yuncong_seasonal_landscape_132837",
        "object_id": 132837,
        "title": "萧云从《四时山水册叶》（Album of Seasonal Landscapes）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清"],
        "topics": ["萧云从", "新安画派", "Album of Seasonal Landscapes", "山水册", "水墨笔墨"],
        "filename": "I36_Cleveland_清萧云从_四时山水册叶_馆藏条目.pdf",
    },
    {
        "id": "cma_object_kuncan_spring_landscape_142642",
        "object_id": 142642,
        "title": "髡残《春山图》（Spring Landscape）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清"],
        "topics": ["髡残", "清初四僧", "Spring Landscape", "水墨笔墨", "山水"],
        "filename": "I37_Cleveland_清髡残_春山图_馆藏条目.pdf",
    },
    {
        "id": "cma_object_bada_landscape_guo_zhongshu_132908",
        "object_id": 132908,
        "title": "八大山人《仿郭忠恕山水》（Landscape after Guo Zhongshu）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["清"],
        "topics": ["八大山人", "朱耷", "清初四僧", "Landscape after Guo Zhongshu", "山水", "水墨笔墨"],
        "filename": "I38_Cleveland_清八大山人_仿郭忠恕山水_馆藏条目.pdf",
    },
    {
        "id": "cma_object_ma_lin_scholar_clouds_136949",
        "object_id": 136949,
        "title": "马麟《高士观云》（Scholar Reclining and Watching Rising Clouds）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["南宋", "宋"],
        "topics": ["马麟", "Scholar Reclining and Watching Rising Clouds", "南宋绘画", "诗书画题跋", "人物山水"],
        "filename": "I39_Cleveland_南宋马麟_高士观云_馆藏条目.pdf",
    },
    {
        "id": "cma_object_yao_tingmei_leisure_132308",
        "object_id": 132308,
        "title": "姚廷美《山居闲适图》（Leisure Enough to Spare）Cleveland 馆藏条目",
        "author": "Cleveland Museum of Art",
        "dynasties": ["元"],
        "topics": ["姚廷美", "Leisure Enough to Spare", "元代山水", "隐逸", "文人画"],
        "filename": "I40_Cleveland_元姚廷美_山居闲适图_馆藏条目.pdf",
    },
]


def request_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "ChineseLandscapeAuthorityCorpus/1.0"})
    response.raise_for_status()
    return response.json()


def lines_from_value(label: str, value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        text = "；".join(str(item) for item in value if item not in (None, ""))
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return [f"{label}：{text}"] if text else []


def build_met_text(entry: dict[str, Any], obj: dict[str, Any]) -> str:
    tags = [tag.get("term") for tag in obj.get("tags") or [] if tag.get("term")]
    fields = [
        ("馆藏机构", "The Metropolitan Museum of Art"),
        ("API Object ID", obj.get("objectID")),
        ("Object URL", obj.get("objectURL")),
        ("标题", obj.get("title")),
        ("作者", obj.get("artistDisplayName")),
        ("作者信息", obj.get("artistDisplayBio")),
        ("文化", obj.get("culture")),
        ("时期/朝代", obj.get("period") or obj.get("dynasty")),
        ("年代", obj.get("objectDate")),
        ("材质", obj.get("medium")),
        ("尺寸", obj.get("dimensions")),
        ("分类", obj.get("classification")),
        ("部门", obj.get("department")),
        ("登录号", obj.get("accessionNumber")),
        ("来源/捐赠", obj.get("creditLine")),
        ("公共领域", obj.get("isPublicDomain")),
        ("主题标签", tags),
    ]
    lines = [
        entry["title"],
        "",
        "本 PDF 由 Met Collection API 的公开馆藏结构化数据整理生成，用于项目 RAG 检索测试；引用时应回到 Object URL 核对。",
        "",
    ]
    for label, value in fields:
        lines.extend(lines_from_value(label, value))
    lines.extend(["", "项目人工标注："])
    lines.extend(lines_from_value("朝代", entry.get("dynasties")))
    lines.extend(lines_from_value("主题", entry.get("topics")))
    return "\n".join(lines)


def creator_text(creators: list[dict[str, Any]]) -> str:
    parts = []
    for creator in creators:
        description = creator.get("description") or creator.get("title") or ""
        role = creator.get("role")
        parts.append(f"{description}（{role}）" if role else description)
    return "；".join(part for part in parts if part)


def build_cleveland_text(entry: dict[str, Any], obj: dict[str, Any]) -> str:
    fields = [
        ("馆藏机构", "Cleveland Museum of Art"),
        ("API Object ID", obj.get("id")),
        ("Object URL", obj.get("url")),
        ("标题", obj.get("title")),
        ("作者", creator_text(obj.get("creators") or [])),
        ("文化", obj.get("culture")),
        ("年代", obj.get("creation_date")),
        ("技术/材质", obj.get("technique")),
        ("尺寸", obj.get("measurements")),
        ("分类", obj.get("type")),
        ("部门", obj.get("department")),
        ("藏品编号", obj.get("accession_number")),
        ("来源/捐赠", obj.get("creditline")),
        ("当前位置", obj.get("current_location")),
        ("公共领域", obj.get("share_license_status")),
    ]
    lines = [
        entry["title"],
        "",
        "本 PDF 由 Cleveland Museum of Art Open Access API 的公开馆藏结构化数据整理生成，用于项目 RAG 检索测试；引用时应回到 Object URL 核对。",
        "",
    ]
    for label, value in fields:
        lines.extend(lines_from_value(label, value))
    if obj.get("wall_description"):
        lines.extend(["", "馆方说明：", str(obj["wall_description"])])
    if obj.get("description"):
        lines.extend(["", "描述：", str(obj["description"])])
    if obj.get("inscriptions"):
        lines.extend(["", "题识/铭文：", str(obj["inscriptions"])])
    if obj.get("provenance"):
        lines.extend(["", "流传/来源：", str(obj["provenance"])])
    lines.extend(["", "项目人工标注："])
    lines.extend(lines_from_value("朝代", entry.get("dynasties")))
    lines.extend(lines_from_value("主题", entry.get("topics")))
    return "\n".join(lines)


def process_entry(rows: dict[str, dict[str, Any]], entry: dict[str, Any], text: str, *, source_url: str, landing_page: str, license_note: str) -> dict[str, Any]:
    row_input = {
        **entry,
        "category": CATEGORY,
        "source_type": SOURCE_TYPE,
        "authority_level": "A",
        "curation_axis": "work_case",
        "source_url": source_url,
        "landing_page": landing_page,
        "license_note": license_note,
    }
    out_path = out_path_for(row_input)
    status = "exists" if is_pdf_complete(out_path) else "generated"
    error = None
    if status == "generated":
        try:
            write_text_pdf(row_input, text, out_path)
            if not is_pdf_complete(out_path):
                status = "failed"
                error = "generated PDF integrity check failed"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
    row = finalize_entry(row_input, status, error, out_path)
    rows[row["id"]] = row
    print(f"{status:10s} {row['id']}")
    return row


def process_met(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for entry in MET_OBJECTS:
        api_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{entry['object_id']}"
        try:
            obj = request_json(api_url)
            text = build_met_text(entry, obj)
            landing_page = obj.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{entry['object_id']}"
            row = process_entry(
                rows,
                entry,
                text,
                source_url=api_url,
                landing_page=landing_page,
                license_note="The Met Collection API public object data; use with attribution and verify against the object page.",
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                **entry,
                "category": CATEGORY,
                "source_type": SOURCE_TYPE,
                "authority_level": "A",
                "curation_axis": "work_case",
                "source_url": api_url,
                "landing_page": f"https://www.metmuseum.org/art/collection/search/{entry['object_id']}",
                "license_note": "The Met Collection API public object data; use with attribution and verify against the object page.",
                "download_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "local_path": str(out_path_for({**entry, "category": CATEGORY})),
            }
            rows[row["id"]] = row
            print(f"failed     {row['id']} {row['error']}")
        results.append(row)
    return results


def process_cleveland(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for entry in CLEVELAND_OBJECTS:
        api_url = f"https://openaccess-api.clevelandart.org/api/artworks/{entry['object_id']}/"
        try:
            obj = request_json(api_url).get("data") or {}
            text = build_cleveland_text(entry, obj)
            landing_page = obj.get("url") or f"https://clevelandart.org/art/{entry['object_id']}"
            row = process_entry(
                rows,
                entry,
                text,
                source_url=api_url,
                landing_page=landing_page,
                license_note="Cleveland Museum of Art Open Access API public object data; use with attribution and verify against the object page.",
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                **entry,
                "category": CATEGORY,
                "source_type": SOURCE_TYPE,
                "authority_level": "A",
                "curation_axis": "work_case",
                "source_url": api_url,
                "landing_page": f"https://clevelandart.org/art/{entry['object_id']}",
                "license_note": "Cleveland Museum of Art Open Access API public object data; use with attribution and verify against the object page.",
                "download_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "local_path": str(out_path_for({**entry, "category": CATEGORY})),
            }
            rows[row["id"]] = row
            print(f"failed     {row['id']} {row['error']}")
        results.append(row)
    return results


def main() -> int:
    ensure_dirs()
    rows = load_sources()
    complete_statuses = {"downloaded", "exists", "generated", "merged", "imported"}
    before_complete = sum(1 for row in rows.values() if row.get("download_status") in complete_statuses)
    results = process_met(rows)
    results.extend(process_cleveland(rows))
    write_sources(rows)
    manifest = write_manifest_and_source_list(rows)
    stats = build_registry(rows)
    successful = [row for row in results if row.get("download_status") in complete_statuses]
    summary = {
        "attempted": len(results),
        "successful": len(successful),
        "failed": [row["id"] for row in results if row.get("download_status") == "failed"],
        "complete_before": before_complete,
        "complete_after": manifest["complete_documents"],
        "registry_stats": stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
