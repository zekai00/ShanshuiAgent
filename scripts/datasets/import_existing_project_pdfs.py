#!/usr/bin/env python3
"""Import existing project PDFs into the authority corpus as legacy sources."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_PDFS_DIR
from scripts.datasets.curate_authority_corpus import (
    AUTHORITY_WEIGHTS,
    DATASET_ROOT,
    RAW_ROOT,
    SOURCES_PATH,
    build_registry,
    is_pdf_complete,
    load_sources,
    sha256_file,
    write_manifest_and_source_list,
    write_sources,
)


LEGACY_RAW_ROOT = RAW_PDFS_DIR
DEST_CATEGORY = "07_既有项目文献"
DEST_ROOT = RAW_ROOT / DEST_CATEGORY
TZ = ZoneInfo("Asia/Shanghai")


def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M CST")


def stable_id(filename: str, digest: str) -> str:
    stem = Path(filename).stem
    ascii_hint = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    if ascii_hint:
        ascii_hint = ascii_hint[:40]
        return f"legacy_project_{ascii_hint}_{digest[:10]}"
    return f"legacy_project_pdf_{digest[:12]}"


def clean_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_NormalPdf$", "", stem)
    return stem


def infer_dynasties(title: str) -> list[str]:
    rules: list[tuple[str, list[str]]] = [
        ("隋唐至清末", ["隋", "唐", "五代", "宋", "元", "明", "清"]),
        ("唐、五代、宋", ["唐", "五代", "宋"]),
        ("唐五代宋", ["唐", "五代", "宋"]),
        ("五代宋元", ["五代", "宋", "元"]),
        ("五代北宋", ["五代", "北宋"]),
        ("元代以前", ["唐", "五代", "宋", "元"]),
        ("近现代", ["近现代"]),
        ("现代", ["近现代"]),
        ("北宋", ["北宋"]),
        ("南宋", ["南宋"]),
        ("宋代", ["宋"]),
        ("宋", ["宋"]),
        ("元代", ["元"]),
        ("元", ["元"]),
        ("明代", ["明"]),
        ("明", ["明"]),
        ("清代", ["清"]),
        ("清", ["清"]),
        ("隋唐", ["隋", "唐"]),
        ("唐代", ["唐"]),
        ("唐", ["唐"]),
        ("五代", ["五代"]),
        ("古代", ["唐", "五代", "宋", "元", "明", "清"]),
    ]
    dynasties: list[str] = []
    for needle, values in rules:
        if needle in title:
            dynasties.extend(values)
    return list(dict.fromkeys(dynasties))


def infer_topics(title: str) -> list[str]:
    topic_rules: list[tuple[str, list[str]]] = [
        ("画论", ["画论"]),
        ("画诀", ["画诀", "画论"]),
        ("画山水序", ["宗炳", "画山水序", "画论"]),
        ("三远", ["郭熙", "三远法", "林泉高致"]),
        ("郭熙", ["郭熙", "三远法"]),
        ("皴法", ["皴法", "笔墨"]),
        ("笔墨", ["笔墨"]),
        ("留白", ["留白", "虚实"]),
        ("黑白", ["黑白", "虚实"]),
        ("虚实", ["虚实"]),
        ("空间", ["空间营造"]),
        ("布局", ["布局", "构图"]),
        ("位置经营", ["位置经营", "构图"]),
        ("点、线、面", ["形式构成", "点线面"]),
        ("构成", ["形式构成"]),
        ("桥梁", ["桥梁意象", "位置经营"]),
        ("建筑", ["建筑点景", "点景"]),
        ("点景", ["点景"]),
        ("园林", ["园林山水"]),
        ("隐逸", ["隐逸文化"]),
        ("自然", ["自然观"]),
        ("禅境", ["禅境"]),
        ("审美", ["审美"]),
        ("青绿", ["青绿山水", "设色"]),
        ("北宗", ["北宗山水", "画派"]),
        ("四王", ["清初四王", "正统派"]),
        ("流派", ["流派"]),
        ("创新", ["流派与创新"]),
        ("发展", ["发展史"]),
        ("历史", ["发展史"]),
        ("演变", ["演变"]),
        ("衍变", ["演变"]),
        ("比较", ["比较研究"]),
        ("图式", ["图式结构"]),
        ("真我", ["自然表现", "表现境界"]),
        ("观看", ["观看事件", "明代山水"]),
        ("分类", ["中国画分类"]),
    ]
    topics: list[str] = ["既有项目文献"]
    for needle, values in topic_rules:
        if needle in title:
            topics.extend(values)
    return list(dict.fromkeys(topics))


def infer_axis(title: str) -> str:
    if any(key in title for key in ["画论", "画诀", "画山水序", "三远法"]):
        return "ancient_theory_context"
    if any(key in title for key in ["发展", "历史", "演变", "衍变", "比较", "形式演变"]):
        return "development_lineage"
    if any(key in title for key in ["流派", "风格", "北宗", "四王", "元代", "明代", "清代", "五代"]):
        return "style_school"
    if any(key in title for key in ["布局", "空间", "桥梁", "建筑", "留白", "皴法", "笔墨", "构成", "黑白", "虚实"]):
        return "style_technique"
    return "legacy_project"


def authority_level_for(filename: str) -> str:
    stem = Path(filename).stem
    if re.fullmatch(r"paper\d+", stem):
        return "C"
    return "B"


def existing_hash_index(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["sha256"]: row for row in rows.values() if row.get("sha256")}


def import_legacy_pdfs() -> dict[str, Any]:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_sources()
    hashes = existing_hash_index(rows)
    imported = 0
    duplicates = 0
    failed = 0
    source_records = 0

    for src_path in sorted(LEGACY_RAW_ROOT.glob("*.pdf")):
        source_records += 1
        digest = sha256_file(src_path)
        title = clean_title(src_path.name)
        source_id = stable_id(src_path.name, digest)
        authority_level = authority_level_for(src_path.name)
        base_row: dict[str, Any] = {
            "id": source_id,
            "title": title,
            "author": "待补",
            "category": DEST_CATEGORY,
            "source_type": "legacy_project_pdf",
            "authority_level": authority_level,
            "authority_weight": AUTHORITY_WEIGHTS.get(authority_level, 1.0),
            "curation_axis": infer_axis(title),
            "dynasties": infer_dynasties(title),
            "topics": infer_topics(title),
            "source_url": None,
            "landing_page": None,
            "license_note": "项目既有本地 PDF，来源 URL 与授权信息待补；进入 RAG 时应低于原典、博物馆图录和故宫专题论文。",
            "filename": src_path.name,
            "original_project_path": str(src_path),
            "sha256": digest,
            "downloaded_at": now_str(),
        }

        existing = hashes.get(digest)
        if existing and not str(existing.get("id", "")).startswith("legacy_project_"):
            row = dict(base_row)
            row.update(
                {
                    "download_status": "duplicate_skipped",
                    "duplicate_of": existing.get("id"),
                    "duplicate_local_path": existing.get("local_path"),
                    "local_path": str(src_path),
                    "file_size_bytes": src_path.stat().st_size,
                }
            )
            rows[source_id] = row
            duplicates += 1
            print(f"duplicate {src_path.name} -> {existing.get('id')}")
            continue

        dest_path = DEST_ROOT / src_path.name
        try:
            if not is_pdf_complete(src_path):
                row = dict(base_row)
                row.update({"download_status": "failed", "error": "source PDF integrity check failed", "local_path": str(src_path)})
                rows[source_id] = row
                failed += 1
                print(f"failed    {src_path.name}")
                continue
            if not dest_path.exists() or sha256_file(dest_path) != digest:
                shutil.copy2(src_path, dest_path)
            row = dict(base_row)
            row.update(
                {
                    "download_status": "imported",
                    "local_path": str(dest_path),
                    "file_size_bytes": dest_path.stat().st_size,
                }
            )
            rows[source_id] = row
            hashes[digest] = row
            imported += 1
            print(f"imported  {src_path.name}")
        except Exception as exc:  # noqa: BLE001
            row = dict(base_row)
            row.update({"download_status": "failed", "error": f"{type(exc).__name__}: {exc}", "local_path": str(src_path)})
            rows[source_id] = row
            failed += 1
            print(f"failed    {src_path.name}: {exc}")

    write_sources(rows)
    manifest = write_manifest_and_source_list(rows)
    registry_stats = build_registry(rows)
    return {
        "source_records": source_records,
        "imported": imported,
        "duplicates": duplicates,
        "failed": failed,
        "manifest": manifest,
        "registry_stats": registry_stats,
    }


def main() -> int:
    result = import_legacy_pdfs()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
