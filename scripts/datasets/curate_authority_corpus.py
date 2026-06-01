#!/usr/bin/env python3
"""Curate authoritative Chinese landscape painting sources.

The script keeps unstable scan/download sources out of the ingest queue:

1. Download only direct PDFs that pass PDF header/tail checks.
2. Merge DPM multi-part article PDFs into one document when needed.
3. Convert public-domain / public web text pages into traceable text PDFs.
4. Upsert source-level metadata and project-level document registry files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import fitz
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AUTHORITY_CORPUS_DIR, METADATA_DIR

DATASET_ROOT = AUTHORITY_CORPUS_DIR
RAW_ROOT = DATASET_ROOT / "raw_pdfs"
META_ROOT = DATASET_ROOT / "metadata"
PARTS_ROOT = META_ROOT / "pdf_parts"
TEXT_ROOT = META_ROOT / "text_sources"
PROJECT_META_ROOT = METADATA_DIR

SOURCES_PATH = META_ROOT / "sources.jsonl"
MANIFEST_PATH = META_ROOT / "manifest.json"
SOURCE_LIST_PATH = META_ROOT / "来源清单.md"
README_PATH = DATASET_ROOT / "README.md"

FONT_NAME = "china-s"
TZ = ZoneInfo("Asia/Shanghai")
USER_AGENT = "ShanshuiAgentAuthorityCorpus/1.0"

AUTHORITY_WEIGHTS = {
    "A": 1.25,
    "A-": 1.15,
    "B": 1.0,
    "C": 0.85,
}

COMPLETE_STATUSES = {"downloaded", "exists", "generated", "merged", "imported"}


DIRECT_PDF_SOURCES: list[dict[str, Any]] = [
    {
        "id": "commons_lidai_minghua_ji_mingke_scan",
        "title": "歷代名畫記十卷（明刻本影印）",
        "author": "唐 張彥遠撰；明 毛晉訂；南京圖書館藏本",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_scan_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["唐"],
        "topics": ["历代名画记", "画史", "六法", "山水树石", "张彦远"],
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/NJlib-0118844-0004%20%E6%AD%B7%E4%BB%A3%E5%90%8D%E7%95%AB%E8%A8%98%E5%8D%81%E5%8D%B7.pdf",
        "landing_page": "https://zh.wikisource.org/zh-hant/File:NJlib-0118844-0004_%E6%AD%B7%E4%BB%A3%E5%90%8D%E7%95%AB%E8%A8%98%E5%8D%81%E5%8D%B7.pdf",
        "license_note": "Wikimedia Commons public file page; original classical text is public domain. Keep file-page attribution when used.",
        "filename": "A01_古代画论_歷代名畫記十卷_明刻本影印.pdf",
    },
    {
        "id": "commons_linquan_gaozhi_scan",
        "title": "林泉高致（国家图书馆藏本影印）",
        "author": "宋 郭熙述；郭思编",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_scan_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["宋"],
        "topics": ["林泉高致", "山水训", "可行可望可游可居", "郭熙", "画论"],
        "source_url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/NLC892-411999030321-146913%20%E6%9E%97%E6%B3%89%E9%AB%98%E8%87%B4.pdf",
        "landing_page": "https://commons.wikimedia.org/wiki/File:NLC892-411999030321-146913_%E6%9E%97%E6%B3%89%E9%AB%98%E8%87%B4.pdf",
        "license_note": "Wikimedia Commons public file page; original classical text is public domain. Keep file-page attribution when used.",
        "filename": "A02_古代画论_林泉高致_影印.pdf",
    },
    {
        "id": "dpm_dong_qichang_qinglv_2018",
        "title": "董其昌对晚明青绿山水画发展大转变的作用",
        "author": "颜晓军",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明", "清"],
        "topics": ["董其昌", "青绿山水", "南北宗", "文人画", "晚明"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2018/10/11/u5bbf0a0a2544b.pdf",
        "landing_page": "https://www.dpm.org.cn/journal/247900.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E01_故宫_董其昌对晚明青绿山水画发展大转变的作用.pdf",
    },
    {
        "id": "dpm_dong_qichang_she_se_shanshui_2021",
        "title": "欲以真率 当彼钜丽——浅述董其昌设色山水画及其",
        "author": "故宫博物院/紫禁城",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["董其昌", "设色山水", "青绿山水", "松江派", "文人画"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2021/07/23/u60fa997f994ca.pdf",
        "landing_page": "https://www.dpm.org.cn/explode/others/256104.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E02_故宫_董其昌设色山水画.pdf",
    },
    {
        "id": "dpm_dong_qichang_shanshui_bianwei_2020",
        "title": "董其昌《山水》册与《董范合参图》轴辨伪",
        "author": "故宫博物院/故宫博物院院刊",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["董其昌", "董源", "范宽", "辨伪", "山水册"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2020/04/07/u5e8c354700077.pdf",
        "landing_page": "https://www.dpm.org.cn/journal/252246.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E03_故宫_董其昌山水册与董范合参图轴辨伪.pdf",
    },
    {
        "id": "dpm_zhaomengfu_chongjiang_diezhang_2018",
        "title": "由《重江叠嶂图》卷谈赵孟頫对北宋平远山水的继承与创新",
        "author": "故宫博物院/故宫博物院院刊",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "development_lineage",
        "dynasties": ["元", "北宋"],
        "topics": ["赵孟頫", "重江叠嶂图", "平远山水", "北宋传统", "元代文人画"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2018/12/11/u5c0f58527bac5.pdf",
        "landing_page": "https://www.dpm.org.cn/journal/248112.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E04_故宫_赵孟頫对北宋平远山水的继承与创新.pdf",
    },
    {
        "id": "dpm_anqi_dong_qichang_zhaomengfu_2018",
        "title": "从安岐说董其昌论赵孟頫",
        "author": "故宫博物院/故宫博物院院刊",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "development_lineage",
        "dynasties": ["元", "明", "清"],
        "topics": ["安岐", "董其昌", "赵孟頫", "鉴藏", "文人画评价"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2018/12/11/u5c0f56b6df4a2.pdf",
        "landing_page": "https://www.dpm.org.cn/journal/248112.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E05_故宫_从安岐说董其昌论赵孟頫.pdf",
    },
    {
        "id": "dpm_yun_shouping_wang_hui_yipin_2023",
        "title": "游戏涂抹——恽寿平与王翚“逸品”山水的“笔墨”之思",
        "author": "故宫博物院/紫禁城",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["清"],
        "topics": ["恽寿平", "王翚", "逸品", "笔墨", "清代山水"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2023/09/25/u65113a5a52e7a.pdf",
        "landing_page": "https://www.dpm.org.cn/explode/others/261707.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E06_故宫_恽寿平与王翚逸品山水的笔墨之思.pdf",
    },
    {
        "id": "dpm_huang_shen_sese_shanshui_2019",
        "title": "对画家、画史、典范塑造的思考——以黄慎《设色山水》卷为中心",
        "author": "故宫博物院/故宫博物院院刊",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["清"],
        "topics": ["黄慎", "设色山水", "画史", "典范塑造", "扬州画派"],
        "source_url": "https://img.dpm.org.cn/Uploads/File/2019/11/14/u5dcd3498ee878.pdf",
        "landing_page": "https://www.dpm.org.cn/explode/others/250457.html",
        "license_note": "故宫博物院网站公开 PDF；纳入研究语料时保留来源。",
        "filename": "E07_故宫_黄慎设色山水与典范塑造.pdf",
    },
]


PART_PDF_SOURCES: list[dict[str, Any]] = [
    {
        "id": "dpm_wumen_mingshi_bian_merged",
        "title": "“吴门画派”名实辨",
        "author": "故宫博物院/故宫博物院刊物",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article_merged_pdf",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["吴门画派", "沈周", "文徵明", "明代山水", "画派"],
        "source_parts": [
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00069_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00070_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00071_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00072_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00073_00.pdf",
        ],
        "source_url": "https://img.dpm.org.cn/Uploads/pdf/1578/T00069_00.pdf",
        "landing_page": "https://www.dpm.org.cn/paints/talk/203810.html",
        "license_note": "故宫博物院网站公开分页 PDF；本地合并为单篇研究 PDF，纳入语料时保留来源。",
        "filename": "E08_故宫_吴门画派名实辨_合并.pdf",
    },
    {
        "id": "dpm_tang_yin_shanshui_huihua_merged",
        "title": "试论唐寅的山水绘画",
        "author": "故宫博物院/故宫博物院刊物",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article_merged_pdf",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["唐寅", "吴门画派", "山水绘画", "明代山水"],
        "source_parts": [
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00082_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00083_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00084_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00085_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00086_00.pdf",
        ],
        "source_url": "https://img.dpm.org.cn/Uploads/pdf/1578/T00082_00.pdf",
        "landing_page": "https://www.dpm.org.cn/paints/talk/203810.html",
        "license_note": "故宫博物院网站公开分页 PDF；本地合并为单篇研究 PDF，纳入语料时保留来源。",
        "filename": "E09_故宫_试论唐寅的山水绘画_合并.pdf",
    },
    {
        "id": "dpm_wumen_garden_painting_merged",
        "title": "聊以画图写清居——谈以园林、庭园为题材的吴门绘画",
        "author": "故宫博物院/故宫博物院刊物",
        "category": "05_发展脉络与风格流派",
        "source_type": "palace_museum_article_merged_pdf",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["吴门画派", "园林山水", "庭园题材", "明代绘画", "文人生活"],
        "source_parts": [
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00087_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00088_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00089_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00090_00.pdf",
            "https://img.dpm.org.cn/Uploads/pdf/1578/T00091_00.pdf",
        ],
        "source_url": "https://img.dpm.org.cn/Uploads/pdf/1578/T00087_00.pdf",
        "landing_page": "https://www.dpm.org.cn/paints/talk/203810.html",
        "license_note": "故宫博物院网站公开分页 PDF；本地合并为单篇研究 PDF，纳入语料时保留来源。",
        "filename": "E10_故宫_园林庭园题材的吴门绘画_合并.pdf",
    },
]


TEXT_PDF_SOURCES: list[dict[str, Any]] = [
    {
        "id": "wikisource_bifaji_text_pdf",
        "title": "筆法記（维基文库文本整理版）",
        "author": "荊浩",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_text_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["五代", "宋"],
        "topics": ["笔法记", "六要", "气韵", "笔墨", "山水物象"],
        "source_url": "https://zh.wikisource.org/zh-hant/%E7%AD%86%E6%B3%95%E8%A8%98",
        "landing_page": "https://zh.wikisource.org/zh-hant/%E7%AD%86%E6%B3%95%E8%A8%98",
        "license_note": "维基文库公开文本整理为本地 PDF；原典公有领域，页面文本按维基文库条款使用。",
        "filename": "A03_古代画论_筆法記_文本整理版.pdf",
    },
    {
        "id": "wikisource_linquan_gaozhi_text_pdf",
        "title": "林泉高致集（维基文库文本整理版）",
        "author": "郭熙述；郭思编",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_text_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["宋"],
        "topics": ["林泉高致", "山水训", "三远", "郭熙", "宋代山水画论"],
        "source_url": "https://zh.wikisource.org/zh-hant/%E6%9E%97%E6%B3%89%E9%AB%98%E8%87%B4",
        "landing_page": "https://zh.wikisource.org/zh-hant/%E6%9E%97%E6%B3%89%E9%AB%98%E8%87%B4",
        "license_note": "维基文库公开文本整理为本地 PDF；原典公有领域，页面文本按维基文库条款使用。",
        "filename": "A04_古代画论_林泉高致集_文本整理版.pdf",
    },
    {
        "id": "wikisource_huashanshui_fu_text_pdf",
        "title": "畫山水賦（四庫全書本，维基文库文本整理版）",
        "author": "题荊浩撰",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_text_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["五代", "清"],
        "topics": ["画山水赋", "山水诀", "荆浩", "四库全书", "山水画法"],
        "source_url": "https://zh.wikisource.org/zh-hant/%E7%95%AB%E5%B1%B1%E6%B0%B4%E8%B3%A6_(%E5%9B%9B%E5%BA%AB%E5%85%A8%E6%9B%B8%E6%9C%AC)",
        "landing_page": "https://zh.wikisource.org/zh-hant/%E7%95%AB%E5%B1%B1%E6%B0%B4%E8%B3%A6_(%E5%9B%9B%E5%BA%AB%E5%85%A8%E6%9B%B8%E6%9C%AC)",
        "license_note": "维基文库公开文本整理为本地 PDF；原典公有领域，页面文本按维基文库条款使用。",
        "filename": "A05_古代画论_畫山水賦_文本整理版.pdf",
    },
    {
        "id": "wikisource_lidai_minghua_ji_vol1_text_pdf",
        "title": "歷代名畫記卷第一（维基文库文本整理版）",
        "author": "張彥遠",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_text_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["唐"],
        "topics": ["历代名画记", "画史", "六法", "论画山水树石", "张彦远"],
        "source_url": "https://zh.wikisource.org/zh-hant/%E6%AD%B7%E4%BB%A3%E5%90%8D%E7%95%AB%E8%A8%98/%E5%8D%B7%E7%AC%AC%E4%B8%80",
        "landing_page": "https://zh.wikisource.org/zh-hant/%E6%AD%B7%E4%BB%A3%E5%90%8D%E7%95%AB%E8%A8%98/%E5%8D%B7%E7%AC%AC%E4%B8%80",
        "license_note": "维基文库公开文本整理为本地 PDF；原典公有领域，页面文本按维基文库条款使用。",
        "filename": "A06_古代画论_歷代名畫記卷第一_文本整理版.pdf",
    },
    {
        "id": "ctext_kugua_heshang_huayulu_text_pdf",
        "title": "苦瓜和尚畫語錄（中國哲學書電子化計劃文本整理版）",
        "author": "石濤",
        "category": "01_古代画论原典",
        "source_type": "ancient_theory_text_pdf",
        "authority_level": "A-",
        "curation_axis": "ancient_theory",
        "dynasties": ["清"],
        "topics": ["苦瓜和尚画语录", "石涛", "一画", "笔墨", "清代画论"],
        "source_url": "https://ctext.org/wiki.pl?chapter=623580&if=gb",
        "landing_page": "https://ctext.org/wiki.pl?chapter=623580&if=gb",
        "license_note": "中國哲學書電子化計劃公开网页文本整理为本地 PDF；原典公有领域，页面文本按来源站条款使用。",
        "filename": "A07_古代画论_苦瓜和尚畫語錄_文本整理版.pdf",
    },
]


COLLECTION_TEXT_SOURCES: list[dict[str, Any]] = [
    {
        "id": "dpm_work_youchuntu_text_pdf",
        "title": "展子虔《游春图》卷（故宫馆藏条目整理版）",
        "author": "故宫博物院",
        "category": "06_作品级馆藏条目",
        "source_type": "museum_collection_entry_text_pdf",
        "authority_level": "A-",
        "curation_axis": "development_lineage",
        "dynasties": ["隋"],
        "topics": ["游春图", "展子虔", "青绿山水", "早期山水", "故宫馆藏"],
        "source_url": "https://www.dpm.org.cn/collection/paint/234623.html",
        "landing_page": "https://www.dpm.org.cn/collection/paint/234623.html",
        "license_note": "故宫博物院公开馆藏条目整理为文本 PDF；用于作品级证据时保留页面来源。",
        "filename": "F01_故宫馆藏_展子虔游春图卷_文本整理版.pdf",
    },
    {
        "id": "dpm_work_dong_qichang_qinglv_text_pdf",
        "title": "董其昌《青绿山水图》轴（故宫馆藏条目整理版）",
        "author": "故宫博物院",
        "category": "06_作品级馆藏条目",
        "source_type": "museum_collection_entry_text_pdf",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["董其昌", "青绿山水", "南北宗", "松江派", "故宫馆藏"],
        "source_url": "https://www.dpm.org.cn/collection/paint/230629.html",
        "landing_page": "https://www.dpm.org.cn/collection/paint/230629.html",
        "license_note": "故宫博物院公开馆藏条目整理为文本 PDF；用于作品级证据时保留页面来源。",
        "filename": "F02_故宫馆藏_董其昌青绿山水图轴_文本整理版.pdf",
    },
    {
        "id": "dpm_work_dong_qichang_shanshui_ce_text_pdf",
        "title": "董其昌《山水图》册（故宫馆藏条目整理版）",
        "author": "故宫博物院",
        "category": "06_作品级馆藏条目",
        "source_type": "museum_collection_entry_text_pdf",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["明"],
        "topics": ["董其昌", "山水图册", "留白", "笔墨", "故宫馆藏"],
        "source_url": "https://www.dpm.org.cn/collection/paint/230715.html",
        "landing_page": "https://www.dpm.org.cn/collection/paint/230715.html",
        "license_note": "故宫博物院公开馆藏条目整理为文本 PDF；用于作品级证据时保留页面来源。",
        "filename": "F03_故宫馆藏_董其昌山水图册_文本整理版.pdf",
    },
    {
        "id": "dpm_work_wu_li_shanshui_ce_text_pdf",
        "title": "吴历《山水图》册（故宫馆藏条目整理版）",
        "author": "故宫博物院",
        "category": "06_作品级馆藏条目",
        "source_type": "museum_collection_entry_text_pdf",
        "authority_level": "A-",
        "curation_axis": "style_school",
        "dynasties": ["清"],
        "topics": ["吴历", "山水图册", "清代山水", "四王吴恽", "故宫馆藏"],
        "source_url": "https://www.dpm.org.cn/collection/paint/232540.html",
        "landing_page": "https://www.dpm.org.cn/collection/paint/232540.html",
        "license_note": "故宫博物院公开馆藏条目整理为文本 PDF；用于作品级证据时保留页面来源。",
        "filename": "F04_故宫馆藏_吴历山水图册_文本整理版.pdf",
    },
]


def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M CST")


def ensure_dirs() -> None:
    for path in [RAW_ROOT, META_ROOT, PARTS_ROOT, TEXT_ROOT, PROJECT_META_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        head = f.read(5)
        f.seek(max(0, path.stat().st_size - 4096))
        tail = f.read()
    return head == b"%PDF-" and b"%%EOF" in tail


def load_sources() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if SOURCES_PATH.exists():
        for line in SOURCES_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def write_sources(rows: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(rows.values(), key=lambda r: (r.get("category", ""), r.get("id", "")))
    SOURCES_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in ordered) + "\n",
        encoding="utf-8",
    )


def request_get(url: str, *, stream: bool = False, timeout: tuple[int, int] = (15, 90)) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            session = requests.Session()
            response = session.get(url, stream=stream, timeout=timeout, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 2:
                time.sleep(2 + attempt)
    assert last_error is not None
    raise last_error


def download_pdf(url: str, out_path: Path) -> tuple[str, str | None]:
    if is_pdf_complete(out_path):
        return "exists", None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(out_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with request_get(url, stream=True) as response:
                content_type = response.headers.get("Content-Type", "")
                if "html" in content_type.lower():
                    return "failed", f"URL returned HTML content-type: {content_type}"
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
            if not is_pdf_complete(tmp_path):
                failed_path = out_path.with_suffix(out_path.suffix + ".partial")
                shutil.move(str(tmp_path), failed_path)
                return "failed", f"PDF integrity check failed; partial saved to {failed_path}"
            shutil.move(str(tmp_path), out_path)
            return "downloaded", None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def merge_pdf_parts(entry: dict[str, Any], out_path: Path) -> tuple[str, str | None, list[str]]:
    if is_pdf_complete(out_path):
        return "exists", None, []

    part_dir = PARTS_ROOT / entry["id"]
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    for index, url in enumerate(entry["source_parts"], start=1):
        part_path = part_dir / f"{index:02d}_{Path(url).name}"
        status, error = download_pdf(url, part_path)
        if status == "failed":
            return "failed", f"part {index} failed: {error}", [str(p) for p in part_paths]
        part_paths.append(part_path)

    merged = fitz.open()
    try:
        for part_path in part_paths:
            with fitz.open(part_path) as part_doc:
                merged.insert_pdf(part_doc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(out_path)
    finally:
        merged.close()

    if not is_pdf_complete(out_path):
        return "failed", "merged PDF integrity check failed", [str(p) for p in part_paths]
    return "merged", None, [str(p) for p in part_paths]


def clean_extracted_lines(lines: list[str], *, source_url: str) -> list[str]:
    drop_exact = {
        "姊妹計劃",
        "數據項",
        "维基",
        "維基",
        "中國哲學書電子化計劃",
        "查看正文",
        "修改",
        "查看歷史",
        "新增語言",
        "目次",
        "閱讀",
        "編輯",
        "檢視歷史",
        "上一頁",
        "下一頁",
        "上一幅",
        "下一幅",
    }
    cleaned: list[str] = []
    for raw in lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if "ctext.org" in source_url and line == "URN":
            break
        if "ctext.org" in source_url and line.startswith("->"):
            continue
        if line in drop_exact:
            continue
        if line in {"[", "]", ":", "->", "→", "〈", "〉"}:
            continue
        if re.fullmatch(r"\d{1,4}", line) and "ctext.org" in source_url:
            continue
        if re.fullmatch(r"\[\s*編輯\s*\]", line):
            continue
        if len(line) <= 2 and line in {"大", "中", "小"}:
            continue
        cleaned.append(line)

    deduped: list[str] = []
    previous = ""
    for line in cleaned:
        if line == previous:
            continue
        deduped.append(line)
        previous = line
    return deduped


def html_to_main_text(source_url: str, html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for selector in [
        "script",
        "style",
        "noscript",
        ".mw-editsection",
        ".noprint",
        ".metadata",
        ".navbox",
        "#toc",
        "header",
        "footer",
        "nav",
    ]:
        for node in soup.select(selector):
            node.decompose()

    selectors = [
        "#mw-content-text .mw-parser-output",
        "#mw-content-text",
        "#content",
        ".content",
        ".text",
        ".detail",
        ".main",
        "article",
        "body",
    ]
    main = None
    for selector in selectors:
        main = soup.select_one(selector)
        if main is not None:
            break
    if main is None:
        main = soup
    lines = clean_extracted_lines(main.get_text("\n").splitlines(), source_url=source_url)
    return "\n".join(lines)


def wrap_text_for_pdf(text: str, width: int = 40) -> list[str]:
    output: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            output.append("")
            continue
        while len(paragraph) > width:
            cut = width
            for punct in "。！？；，、：」』）〉》":
                pos = paragraph.rfind(punct, 0, width + 1)
                if pos >= max(18, width - 10):
                    cut = pos + 1
                    break
            output.append(paragraph[:cut])
            paragraph = paragraph[cut:].strip()
        if paragraph:
            output.append(paragraph)
    return output


def write_text_pdf(entry: dict[str, Any], text: str, out_path: Path) -> None:
    text_path = TEXT_ROOT / f"{entry['id']}.txt"
    text_path.write_text(text, encoding="utf-8")

    lines = [
        entry["title"],
        f"作者/机构：{entry.get('author', '')}",
        f"来源：{entry['source_url']}",
        f"整理时间：{now_str()}",
        "说明：本 PDF 为公开网页文本整理版，用于本项目 RAG 测试；引用时应回到原始来源页核对。",
        "",
    ]
    lines.extend(wrap_text_for_pdf(text, width=39))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = None
    y = 0.0
    line_height = 17
    margin_x = 54
    margin_top = 56
    page_bottom = 790

    def new_page() -> fitz.Page:
        nonlocal y
        p = doc.new_page(width=595, height=842)
        y = margin_top
        return p

    page = new_page()
    for index, line in enumerate(lines):
        if y > page_bottom:
            page = new_page()
        fontsize = 14 if index == 0 else 10
        if line == "":
            y += line_height / 2
            continue
        page.insert_text((margin_x, y), line, fontname=FONT_NAME, fontsize=fontsize)
        y += 22 if index == 0 else line_height

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    doc.save(tmp_path)
    doc.close()
    shutil.move(str(tmp_path), out_path)


def generate_text_pdf(entry: dict[str, Any], out_path: Path) -> tuple[str, str | None]:
    force_rebuild = os.getenv("FORCE_TEXT_PDF_REBUILD") == "1"
    if is_pdf_complete(out_path) and not force_rebuild:
        return "exists", None
    try:
        response = request_get(entry["source_url"], stream=False, timeout=(15, 60))
        text = html_to_main_text(entry["source_url"], response.text)
        if len(text) < 200:
            return "failed", f"extracted text too short: {len(text)} characters"
        write_text_pdf(entry, text, out_path)
        if not is_pdf_complete(out_path):
            return "failed", "generated PDF integrity check failed"
        return "generated", None
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{type(exc).__name__}: {exc}"


def out_path_for(entry: dict[str, Any]) -> Path:
    return RAW_ROOT / entry["category"] / entry["filename"]


def finalize_entry(entry: dict[str, Any], status: str, error: str | None, out_path: Path) -> dict[str, Any]:
    row = dict(entry)
    row["local_path"] = str(out_path)
    row["downloaded_at"] = now_str()
    row["download_status"] = status
    row["authority_weight"] = AUTHORITY_WEIGHTS.get(row.get("authority_level", "B"), 1.0)
    row["import_priority"] = "high" if row["authority_weight"] >= 1.15 else "normal"
    if error:
        row["error"] = error
    else:
        row.pop("error", None)
    if status in COMPLETE_STATUSES and out_path.exists():
        row["file_size_bytes"] = out_path.stat().st_size
        row["sha256"] = sha256_file(out_path)
    return row


def curate_sources() -> list[dict[str, Any]]:
    rows = load_sources()
    results: list[dict[str, Any]] = []

    for entry in DIRECT_PDF_SOURCES:
        out_path = out_path_for(entry)
        status, error = download_pdf(entry["source_url"], out_path)
        row = finalize_entry(entry, status, error, out_path)
        rows[row["id"]] = row
        results.append(row)
        print(f"{status:10s} {row['id']}")

    for entry in PART_PDF_SOURCES:
        out_path = out_path_for(entry)
        status, error, part_paths = merge_pdf_parts(entry, out_path)
        row = finalize_entry(entry, status, error, out_path)
        if part_paths:
            row["part_local_paths"] = part_paths
        rows[row["id"]] = row
        results.append(row)
        print(f"{status:10s} {row['id']}")

    for entry in TEXT_PDF_SOURCES + COLLECTION_TEXT_SOURCES:
        out_path = out_path_for(entry)
        status, error = generate_text_pdf(entry, out_path)
        row = finalize_entry(entry, status, error, out_path)
        rows[row["id"]] = row
        results.append(row)
        print(f"{status:10s} {row['id']}")

    write_sources(rows)
    return results


def complete_sources(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    complete = []
    for row in rows.values():
        local_path = Path(str(row.get("local_path", "")))
        if row.get("download_status") in COMPLETE_STATUSES and local_path.exists() and is_pdf_complete(local_path):
            complete.append(row)
    return sorted(complete, key=lambda r: (r.get("category", ""), r.get("id", "")))


def build_registry(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    complete = complete_sources(rows)
    registry_rows: list[dict[str, Any]] = []
    for row in complete:
        local_path = Path(row["local_path"])
        weight = float(row.get("authority_weight") or AUTHORITY_WEIGHTS.get(row.get("authority_level", "B"), 1.0))
        curation_axis = row.get("curation_axis") or "seed_corpus"
        ingest_weight = weight
        if curation_axis == "ancient_theory":
            ingest_weight += 0.08
        if row.get("source_type") in {"museum_catalog", "symposium_proceedings"}:
            ingest_weight += 0.05

        registry_rows.append(
            {
                "doc_id": row["id"],
                "title": row.get("title"),
                "author": row.get("author"),
                "source_file": row.get("filename") or local_path.name,
                "raw_path": str(local_path),
                "dataset_root": str(DATASET_ROOT),
                "category": row.get("category"),
                "source_type": row.get("source_type"),
                "curation_axis": curation_axis,
                "authority_level": row.get("authority_level"),
                "authority_weight": round(weight, 3),
                "rag_import_weight": round(ingest_weight, 3),
                "import_priority": "high" if ingest_weight >= 1.15 else "normal",
                "dynasties": row.get("dynasties", []),
                "topics": row.get("topics", []),
                "source_url": row.get("source_url"),
                "landing_page": row.get("landing_page"),
                "license_note": row.get("license_note"),
                "sha256": row.get("sha256"),
                "file_size_bytes": row.get("file_size_bytes"),
                "download_status": row.get("download_status"),
                "registry_created_at": now_str(),
            }
        )

    PROJECT_META_ROOT.mkdir(parents=True, exist_ok=True)
    registry_path = PROJECT_META_ROOT / "文献级标注清单.jsonl"
    import_queue_path = PROJECT_META_ROOT / "高权重导入队列.jsonl"
    stats_path = PROJECT_META_ROOT / "文献级标注统计.json"

    registry_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in registry_rows) + "\n",
        encoding="utf-8",
    )

    import_rows = sorted(
        registry_rows,
        key=lambda r: (-float(r["rag_import_weight"]), r["category"], r["source_file"]),
    )
    import_queue_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in import_rows) + "\n",
        encoding="utf-8",
    )

    stats = {
        "generated_at": now_str(),
        "registry_path": str(registry_path),
        "import_queue_path": str(import_queue_path),
        "complete_documents": len(registry_rows),
        "total_bytes": sum(int(row.get("file_size_bytes") or 0) for row in registry_rows),
        "by_category": Counter(row.get("category") for row in registry_rows),
        "by_authority_level": Counter(row.get("authority_level") for row in registry_rows),
        "by_curation_axis": Counter(row.get("curation_axis") for row in registry_rows),
        "high_priority_count": sum(1 for row in registry_rows if row["import_priority"] == "high"),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def write_manifest_and_source_list(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "complete": 0, "bytes": 0})
    failed: list[str] = []
    complete_count = 0
    total_bytes = 0
    for row in rows.values():
        cat = row.get("category", "未分类")
        categories[cat]["count"] += 1
        if row.get("download_status") in COMPLETE_STATUSES:
            categories[cat]["complete"] += 1
            complete_count += 1
            size = int(row.get("file_size_bytes") or 0)
            categories[cat]["bytes"] += size
            total_bytes += size
        elif row.get("download_status") == "failed":
            failed.append(row["id"])

    manifest = {
        "updated_at": now_str(),
        "dataset_root": str(DATASET_ROOT),
        "raw_pdf_root": str(RAW_ROOT),
        "source_count": len(rows),
        "complete_documents": complete_count,
        "failed": sorted(failed),
        "total_bytes": total_bytes,
        "categories": dict(sorted(categories.items())),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 中国山水画权威语料来源清单",
        "",
        f"更新时间：{now_str()}",
        "",
        f"- 来源记录：{len(rows)}",
        f"- 完整可用文档：{complete_count}",
        f"- 失败/暂缓：{len(failed)}",
        "",
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows.values(), key=lambda r: (r.get("category", ""), r.get("id", ""))):
        by_category[row.get("category", "未分类")].append(row)
    for category, items in by_category.items():
        lines.append(f"## {category}")
        lines.append("")
        for row in items:
            status = row.get("download_status")
            size_mb = (int(row.get("file_size_bytes") or 0) / 1024 / 1024) if row.get("file_size_bytes") else 0
            lines.append(f"- [{status}] {row.get('id')}｜{row.get('title')}｜{row.get('authority_level')}｜{size_mb:.1f} MB")
            lines.append(f"  - 来源：{row.get('landing_page') or row.get('source_url')}")
            lines.append(f"  - 主题：{'、'.join(row.get('topics', []))}")
        lines.append("")
    SOURCE_LIST_PATH.write_text("\n".join(lines), encoding="utf-8")

    readme = [
        "# 中国山水画权威种子语料库",
        "",
        f"更新时间：{now_str()}",
        "",
        "## 目录",
        "",
        "- `raw_pdfs/`：已下载、合并或由公开文本整理生成，并通过基本 PDF 完整性检查的文档。",
        "- `metadata/sources.jsonl`：来源级元数据，每行一个来源。",
        "- `metadata/manifest.json`：语料统计。",
        "- `metadata/来源清单.md`：人工可读的来源清单。",
        "- `failed_partial_downloads/`：网络中断产生的未完整 PDF，不应进入 RAG。",
        "",
        "## 当前规模",
        "",
        f"- 完整可用文档：{complete_count} 个。",
        f"- 完整文档总量：约 {total_bytes / 1024 / 1024:.1f} MB。",
        f"- 来源记录：{len(rows)} 个。",
        f"- 失败/暂缓：{len(failed)} 个。",
        "",
        "## 来源分层",
        "",
        "1. `01_古代画论原典`：古代画论影印件与文本整理版。",
        "2. `02_博物馆图录专题`：MetPublications / MMA Libraries 图录与专题出版物。",
        "3. `03_作品教学资料`：Princeton 与台北故宫公开教学/展览资料。",
        "4. `04_故宫专题论文`：故宫公开 PDF 专题论文。",
        "5. `05_发展脉络与风格流派`：按发展脉络、风格、画派补充的专题论文。",
        "6. `06_作品级馆藏条目`：故宫公开馆藏条目的文本整理版。",
        "7. `07_既有项目文献`：从项目原始 PDF 迁入的既有文献，保留待溯源状态。",
        "8. `08_优先补充文献`：针对当前缺口优先补充的原典、故宫专题与作品个案。",
        "9. `09_非故宫作品级馆藏条目`：Met、Cleveland 等非故宫权威馆藏条目的文本整理版。",
        "",
        "## 项目导入清单",
        "",
        f"- 项目文献级标注：`{PROJECT_META_ROOT / '文献级标注清单.jsonl'}`",
        f"- 高权重导入队列：`{PROJECT_META_ROOT / '高权重导入队列.jsonl'}`",
        "",
        "## 使用约束",
        "",
        "这些文件只作为研究语料种子集。后续进入项目 RAG 前，应保留来源 URL、来源页、权威等级、哈希与引用说明。不要把失败或暂缓文件纳入解析或索引。",
        "",
    ]
    README_PATH.write_text("\n".join(readme), encoding="utf-8")
    return manifest


def main() -> int:
    ensure_dirs()
    print(f"started {now_str()}")
    curate_sources()
    rows = load_sources()
    manifest = write_manifest_and_source_list(rows)
    stats = build_registry(rows)
    print(json.dumps({"manifest": manifest, "registry_stats": stats}, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
