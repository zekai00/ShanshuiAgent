#!/usr/bin/env python3
"""Rename legacy paper1-paper5 PDFs with descriptive titles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from curate_authority_corpus import (
    build_registry,
    load_sources,
    write_manifest_and_source_list,
    write_sources,
)


PROJECT_RAW_ROOT = Path("/root/Workspace/ChineseLandscape/data/raw_pdfs")
DATASET_LEGACY_ROOT = Path("/root/datasets/chinese_landscape_authority_corpus/raw_pdfs/07_既有项目文献")


RENAMES: dict[str, dict[str, Any]] = {
    "legacy_project_paper1_7bc3e11e36": {
        "old_filename": "paper1.pdf",
        "new_filename": "山水密语_拨开中国山水画的知识迷雾_节选.pdf",
        "title": "山水密语：拨开中国山水画的知识迷雾（节选）",
        "topics": ["既有项目文献", "中国山水画", "意象", "空灵之美", "诗画互融"],
    },
    "legacy_project_paper2_3c4c06024a": {
        "old_filename": "paper2.pdf",
        "new_filename": "中国山水画技法与鉴赏研究_节选.pdf",
        "title": "中国山水画技法与鉴赏研究（节选）",
        "topics": ["既有项目文献", "中国山水画技法", "鉴赏", "董其昌", "文人画"],
    },
    "legacy_project_paper3_8381506b8f": {
        "old_filename": "paper3.pdf",
        "new_filename": "中国山水画概述_节选.pdf",
        "title": "中国山水画概述（节选）",
        "topics": ["既有项目文献", "中国山水画", "分类", "基本特征", "发展概述"],
    },
    "legacy_project_paper4_6b587742fe": {
        "old_filename": "paper4.pdf",
        "new_filename": "笔墨年轮_纵览中国山水画的发展图谱_节选.pdf",
        "title": "笔墨年轮：纵览中国山水画的发展图谱（节选）",
        "topics": ["既有项目文献", "中国山水画发展", "笔墨", "摹古", "朱耷", "清代山水"],
    },
    "legacy_project_paper5_36665b3a24": {
        "old_filename": "paper5.pdf",
        "new_filename": "中国水墨山水画技法图解一_图版节选.pdf",
        "title": "中国水墨山水画技法图解一（图版节选）",
        "topics": ["既有项目文献", "水墨山水", "技法图解", "刘海粟", "陈树人", "关山月"],
    },
}


def rename_file(root: Path, old_filename: str, new_filename: str) -> Path | None:
    old_path = root / old_filename
    new_path = root / new_filename
    if old_path.exists():
        if new_path.exists():
            raise FileExistsError(f"target already exists: {new_path}")
        old_path.rename(new_path)
        return new_path
    if new_path.exists():
        return new_path
    return None


def main() -> int:
    rows = load_sources()
    renamed = []
    for doc_id, info in RENAMES.items():
        row = rows.get(doc_id)
        if not row:
            print(f"missing metadata row: {doc_id}")
            continue

        old_filename = info["old_filename"]
        new_filename = info["new_filename"]
        dataset_path = rename_file(DATASET_LEGACY_ROOT, old_filename, new_filename)
        project_path = rename_file(PROJECT_RAW_ROOT, old_filename, new_filename)

        if dataset_path is None:
            raise FileNotFoundError(f"dataset PDF not found for {doc_id}: {old_filename} / {new_filename}")

        row["original_filename"] = row.get("original_filename") or old_filename
        row["filename"] = new_filename
        row["local_path"] = str(dataset_path)
        row["title"] = info["title"]
        row["topics"] = info["topics"]
        row["rename_note"] = "由 paper1-paper5 这类占位文件名改为根据页面标题/页眉识别的描述性名称。"
        if project_path is not None:
            row["project_raw_path"] = str(project_path)
        rows[doc_id] = row
        renamed.append((doc_id, old_filename, new_filename))

    write_sources(rows)
    manifest = write_manifest_and_source_list(rows)
    stats = build_registry(rows)
    for doc_id, old_filename, new_filename in renamed:
        print(f"renamed {doc_id}: {old_filename} -> {new_filename}")
    print({"renamed": len(renamed), "complete_documents": manifest["complete_documents"], "registry_documents": stats["complete_documents"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
