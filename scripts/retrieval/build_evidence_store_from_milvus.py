#!/usr/bin/env python3
"""Build a canonical evidence store from the legacy Milvus RAG collection.

This is a non-destructive migration layer. It does not rebuild embeddings or
reparse PDFs. Instead, it extracts the current Milvus chunk payloads into
versioned JSONL artifacts so retrieval can return structured evidence with
stable chunk ids and separated raw/generated context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

WORKSPACE_DIR = Path("/root/Workspace/ChineseLandscape")
MILVUS_DB_PATH = WORKSPACE_DIR / "data" / "vector_store" / "milvus_landscape.db"
COLLECTION_NAME = "landscape_rag"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "data" / "processed" / "documents"

IMAGE_REF_PATTERN = re.compile(r"/root/Workspace/ChineseLandscape/data/extracted_artworks/[^\]\s：:，,。；;）)]+")
IMAGE_PAGE_PATTERN = re.compile(r"_p(\d+)_")


def sha_id(prefix: str, *parts: Any, length: int = 16) -> str:
    payload = "\n".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def clean_title(source_file: str) -> str:
    title = str(source_file or "").removesuffix(".pdf")
    title = title.replace("_NormalPdf", "").replace("NormalPdf", "")
    parts = title.split("_")
    if len(parts) > 1 and 1 <= len(parts[-1]) <= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", parts[-1]):
        title = "_".join(parts[:-1])
    return title.replace("_", "").strip("《》 ")


def normalize_title(value: str) -> str:
    value = clean_title(value)
    value = value.removesuffix(".pdf")
    value = re.sub(r"[\s《》“”\"'：:，,。·_\-—()（）【】\[\]]+", "", value)
    return value.lower()


def parse_contextual_chunk(text: Any) -> tuple[str, str]:
    value = str(text or "").strip()
    if "【全局上下文】" in value and "【原文资料】" in value:
        prefix = value.split("【全局上下文】", 1)[1].split("【原文资料】", 1)[0].strip()
        raw = value.split("【原文资料】", 1)[1].strip()
        return prefix, raw
    return "", value


def extract_image_refs(text: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for match in IMAGE_REF_PATTERN.finditer(text):
        path = match.group(0)
        page_match = IMAGE_PAGE_PATTERN.search(Path(path).name)
        refs.append({
            "image_path": path,
            "page_number": int(page_match.group(1)) if page_match else None,
        })
    return refs


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in list(value) if str(item)]
    except TypeError:
        return [str(value)] if str(value) else []


def load_rows(db_path: Path, collection_name: str, limit: int) -> list[dict[str, Any]]:
    client = MilvusClient(str(db_path))
    try:
        client.load_collection(collection_name)
        return client.query(
            collection_name=collection_name,
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_store(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    pages: dict[tuple[str, int], dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    quality = Counter()

    for row in rows:
        source_file = str(row.get("source_file") or "").strip()
        if not source_file:
            quality["missing_source_file"] += 1
            continue

        title = clean_title(source_file)
        doc_id = sha_id("doc", source_file)
        legacy_id = str(row.get("id"))
        contextual_chunk = str(row.get("contextual_chunk") or "")
        contextual_prefix, raw_chunk_text = parse_contextual_chunk(contextual_chunk)
        chunk_id = sha_id("chunk", source_file, legacy_id, raw_chunk_text)
        image_refs = extract_image_refs(raw_chunk_text)
        page_numbers = sorted({ref["page_number"] for ref in image_refs if ref["page_number"]})

        documents.setdefault(doc_id, {
            "doc_id": doc_id,
            "source_file": source_file,
            "title": title,
            "normalized_title": normalize_title(title),
            "pdf_path": str(WORKSPACE_DIR / "data" / "raw_pdfs" / source_file),
            "provenance_status": "legacy_milvus_migration",
            "page_provenance_status": "partial_or_unknown",
            "chunk_count": 0,
            "image_ref_count": 0,
        })
        documents[doc_id]["chunk_count"] += 1
        documents[doc_id]["image_ref_count"] += len(image_refs)

        aliases[normalize_title(title)] = source_file
        aliases[normalize_title(source_file)] = source_file
        aliases[normalize_title(source_file.removesuffix(".pdf"))] = source_file

        chunk = {
            "chunk_id": chunk_id,
            "legacy_milvus_id": legacy_id,
            "doc_id": doc_id,
            "source_file": source_file,
            "title": title,
            "page_start": page_numbers[0] if page_numbers else None,
            "page_end": page_numbers[-1] if page_numbers else None,
            "section_title": None,
            "raw_chunk_text": raw_chunk_text,
            "contextual_prefix": contextual_prefix,
            "retrieval_text": contextual_chunk,
            "metadata": {
                "dynasty": string_list(row.get("dynasty", [])),
                "painter": str(row.get("painter", "")),
                "subject_matter": str(row.get("subject_matter", "")),
                "content_scope": str(row.get("content_scope", "")),
            },
            "quality": {
                "parse_status": "migrated_from_legacy_milvus",
                "raw_char_count": len(raw_chunk_text),
                "contextual_prefix_char_count": len(contextual_prefix),
                "has_contextual_prefix": bool(contextual_prefix),
                "image_ref_count": len(image_refs),
                "page_provenance": "image_inferred" if page_numbers else "unknown",
            },
        }
        chunks.append(chunk)

        if not contextual_prefix:
            quality["missing_contextual_prefix"] += 1
        if not raw_chunk_text:
            quality["missing_raw_chunk_text"] += 1
        if str(row.get("painter", "")) in {"", "未知"}:
            quality["unknown_painter"] += 1
        if str(row.get("subject_matter", "")) in {"", "未知"}:
            quality["unknown_subject_matter"] += 1
        if image_refs:
            quality["chunks_with_image_refs"] += 1

        for index, ref in enumerate(image_refs, start=1):
            image_id = sha_id("image", chunk_id, ref["image_path"], index)
            images.append({
                "image_id": image_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_file": source_file,
                "page_number": ref["page_number"],
                "image_path": ref["image_path"],
                "provenance_status": "legacy_path_from_raw_chunk",
            })
            if ref["page_number"]:
                key = (doc_id, int(ref["page_number"]))
                pages.setdefault(key, {
                    "page_id": sha_id("page", doc_id, ref["page_number"]),
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "page_number": int(ref["page_number"]),
                    "text_available": False,
                    "provenance_status": "image_reference_only",
                })

    documents_rows = sorted(documents.values(), key=lambda item: item["source_file"])
    chunks.sort(key=lambda item: (item["source_file"], item["legacy_milvus_id"]))
    images.sort(key=lambda item: (item["source_file"], item["page_number"] or 0, item["image_path"]))
    pages_rows = sorted(pages.values(), key=lambda item: (item["source_file"], item["page_number"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "documents.jsonl", documents_rows)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "images.jsonl", images)
    write_jsonl(output_dir / "pages.jsonl", pages_rows)
    (output_dir / "source_aliases.json").write_text(
        json.dumps(dict(sorted(aliases.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_counts = Counter(chunk["source_file"] for chunk in chunks)
    manifest = {
        "build_time": datetime.now().isoformat(timespec="seconds"),
        "builder": "scripts/retrieval/build_evidence_store_from_milvus.py",
        "source": {
            "milvus_db_path": str(MILVUS_DB_PATH),
            "collection_name": COLLECTION_NAME,
            "mode": "legacy_milvus_migration",
        },
        "outputs": {
            "documents": str(output_dir / "documents.jsonl"),
            "chunks": str(output_dir / "chunks.jsonl"),
            "pages": str(output_dir / "pages.jsonl"),
            "images": str(output_dir / "images.jsonl"),
            "source_aliases": str(output_dir / "source_aliases.json"),
        },
        "counts": {
            "documents": len(documents_rows),
            "chunks": len(chunks),
            "pages_with_image_refs": len(pages_rows),
            "images": len(images),
            "source_aliases": len(aliases),
        },
        "quality": dict(quality),
        "top_sources_by_chunk_count": source_counts.most_common(20),
        "limitations": [
            "Legacy Milvus chunks do not contain reliable PDF page numbers for text-only chunks.",
            "page_start/page_end are only inferred when an extracted image path encodes a page number.",
            "A full PDF reparse is still required for page-level and bbox-level citation.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=MILVUS_DB_PATH)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.db_path, args.collection, args.limit)
    manifest = build_store(rows, args.output_dir)
    print(json.dumps({
        "documents": manifest["counts"]["documents"],
        "chunks": manifest["counts"]["chunks"],
        "pages_with_image_refs": manifest["counts"]["pages_with_image_refs"],
        "images": manifest["counts"]["images"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
