#!/usr/bin/env python3
"""Build canonical page/chunk evidence from the curated literature registry.

This script is the deterministic evidence layer before embedding. It reads the
document registry and PDF text-extractability audit, extracts text-ready PDFs
page by page, and writes structured JSONL artifacts with stable ids and page
provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import AUTHORITY_EVIDENCE_DIR, METADATA_DIR


DEFAULT_REGISTRY = METADATA_DIR / "文献级标注清单.jsonl"
DEFAULT_AUDIT = METADATA_DIR / "PDF文本可抽取性审计.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def sha_id(prefix: str, *parts: Any, length: int = 16) -> str:
    payload = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def stable_int_id(*parts: Any) -> int:
    payload = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def compact_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_inline(text: str) -> str:
    return re.sub(r"\s+", " ", compact_text(text)).strip()


def list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = compact_text(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs or [text]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(end - overlap, start + 1)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) >= 40]


def metadata_prefix(row: dict[str, Any]) -> str:
    parts = [
        f"文献《{row.get('title', '')}》",
        f"作者/机构：{row.get('author', '未知')}",
        f"类别：{row.get('category', '')}",
        f"权威等级：{row.get('authority_level', '')}",
    ]
    facets = row.get("facets") if isinstance(row.get("facets"), dict) else {}
    for label, key in [
        ("时期", "periods"),
        ("朝代", "dynasties"),
        ("流派", "lineages_schools"),
        ("技法", "styles_techniques"),
        ("人物", "persons"),
        ("主题", "themes"),
    ]:
        values = list_value(facets.get(key) or row.get(key))
        if values:
            parts.append(f"{label}：{'、'.join(values[:8])}")
    return "；".join(part for part in parts if str(part).strip())


def extract_pages(pdf_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    pages: list[dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
        for index, page in enumerate(doc, start=1):
            text = compact_text(page.get_text("text"))
            pages.append(
                {
                    "page_number": index,
                    "text": text,
                    "char_count": len(text),
                }
            )
        doc.close()
        return pages, None
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def build_store(
    registry_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    output_dir: Path,
    max_chars: int,
    overlap: int,
    include_low_text: bool,
) -> dict[str, Any]:
    audit_by_doc = {str(row.get("doc_id")): row for row in audit_rows}
    documents: list[dict[str, Any]] = []
    pages_out: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    counts = Counter()

    for row in registry_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            counts["missing_doc_id"] += 1
            continue

        audit = audit_by_doc.get(doc_id, {})
        extractability = str(audit.get("extractability") or "unknown")
        if extractability != "text_ready" and not (include_low_text and extractability == "low_text"):
            skipped.append(
                {
                    "doc_id": doc_id,
                    "title": row.get("title"),
                    "raw_path": row.get("raw_path"),
                    "extractability": extractability,
                    "reason": "not_text_ready",
                }
            )
            counts[f"skipped_{extractability}"] += 1
            continue

        pdf_path = Path(str(row.get("raw_path") or ""))
        if not pdf_path.exists():
            skipped.append(
                {
                    "doc_id": doc_id,
                    "title": row.get("title"),
                    "raw_path": str(pdf_path),
                    "extractability": extractability,
                    "reason": "missing_pdf",
                }
            )
            counts["missing_pdf"] += 1
            continue

        pages, error = extract_pages(pdf_path)
        if error:
            skipped.append(
                {
                    "doc_id": doc_id,
                    "title": row.get("title"),
                    "raw_path": str(pdf_path),
                    "extractability": extractability,
                    "reason": "extract_error",
                    "error": error,
                }
            )
            counts["extract_error"] += 1
            continue

        prefix = metadata_prefix(row)
        doc_chunk_count = 0
        text_page_count = 0
        extracted_chars = 0

        for page in pages:
            text = page["text"]
            page_number = int(page["page_number"])
            if text:
                text_page_count += 1
                extracted_chars += len(text)

            page_id = sha_id("page", doc_id, page_number)
            pages_out.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "source_file": row.get("source_file"),
                    "title": row.get("title"),
                    "page_number": page_number,
                    "char_count": len(text),
                    "text": text,
                    "provenance_status": "pdf_text_extraction",
                }
            )

            for chunk_index, raw_chunk_text in enumerate(chunk_text(text, max_chars, overlap), start=1):
                chunk_id = sha_id("chunk", doc_id, page_number, chunk_index, raw_chunk_text)
                numeric_id = stable_int_id(doc_id, page_number, chunk_index, raw_chunk_text)
                retrieval_text = f"【全局上下文】{prefix}\n【原文资料】{raw_chunk_text}"
                facets = row.get("facets") if isinstance(row.get("facets"), dict) else {}
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "legacy_milvus_id": str(numeric_id),
                        "doc_id": doc_id,
                        "source_file": row.get("source_file"),
                        "title": row.get("title"),
                        "author": row.get("author"),
                        "page_start": page_number,
                        "page_end": page_number,
                        "section_title": None,
                        "raw_chunk_text": raw_chunk_text,
                        "contextual_prefix": prefix,
                        "retrieval_text": retrieval_text,
                        "metadata": {
                            "category": row.get("category"),
                            "source_type": row.get("source_type"),
                            "authority_level": row.get("authority_level"),
                            "rag_import_weight": row.get("rag_import_weight"),
                            "dynasty": list_value(row.get("dynasties") or facets.get("dynasties")),
                            "periods": list_value(row.get("periods") or facets.get("periods")),
                            "lineages_schools": list_value(row.get("lineages_schools") or facets.get("lineages_schools")),
                            "styles_techniques": list_value(row.get("styles_techniques") or facets.get("styles_techniques")),
                            "persons": list_value(row.get("persons") or facets.get("persons")),
                            "works": list_value(row.get("works") or facets.get("works")),
                            "themes": list_value(row.get("themes") or facets.get("themes")),
                            "topics": list_value(row.get("topics")),
                            "source_url": row.get("source_url"),
                            "landing_page": row.get("landing_page"),
                        },
                        "quality": {
                            "parse_status": "authority_pdf_text_extracted",
                            "extractability": extractability,
                            "raw_char_count": len(raw_chunk_text),
                            "contextual_prefix_char_count": len(prefix),
                            "has_contextual_prefix": bool(prefix),
                            "page_provenance": "page_text_extraction",
                            "chunking": {
                                "max_chars": max_chars,
                                "overlap": overlap,
                            },
                        },
                    }
                )
                doc_chunk_count += 1

        documents.append(
            {
                "doc_id": doc_id,
                "source_file": row.get("source_file"),
                "title": row.get("title"),
                "author": row.get("author"),
                "pdf_path": str(pdf_path),
                "dataset_root": row.get("dataset_root"),
                "category": row.get("category"),
                "source_type": row.get("source_type"),
                "authority_level": row.get("authority_level"),
                "authority_weight": row.get("authority_weight"),
                "rag_import_weight": row.get("rag_import_weight"),
                "import_priority": row.get("import_priority"),
                "source_url": row.get("source_url"),
                "landing_page": row.get("landing_page"),
                "license_note": row.get("license_note"),
                "sha256": row.get("sha256"),
                "page_count": len(pages),
                "text_page_count": text_page_count,
                "extracted_text_chars": extracted_chars,
                "chunk_count": doc_chunk_count,
                "extractability": extractability,
                "facets": row.get("facets", {}),
                "provenance_status": "authority_registry_pdf_text_extraction",
                "page_provenance_status": "page_text_extraction",
            }
        )

        aliases[compact_inline(str(row.get("title", "")))] = str(row.get("source_file", ""))
        aliases[compact_inline(str(row.get("source_file", "")))] = str(row.get("source_file", ""))
        counts["documents"] += 1
        counts["pages"] += len(pages)
        counts["chunks"] += doc_chunk_count

    documents.sort(key=lambda item: (str(item.get("category", "")), str(item.get("source_file", ""))))
    pages_out.sort(key=lambda item: (str(item.get("source_file", "")), int(item.get("page_number") or 0)))
    chunks.sort(key=lambda item: (str(item.get("source_file", "")), int(item.get("page_start") or 0), str(item.get("chunk_id"))))
    skipped.sort(key=lambda item: (str(item.get("reason", "")), str(item.get("doc_id", ""))))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "pages.jsonl", pages_out)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "skipped_documents.jsonl", skipped)
    (output_dir / "source_aliases.json").write_text(
        json.dumps(dict(sorted(aliases.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    category_counts = Counter(str(item.get("category", "")) for item in documents)
    extractability_counts = Counter(str(item.get("extractability", "")) for item in documents)
    manifest = {
        "build_time": datetime.now().astimezone().isoformat(timespec="minutes"),
        "builder": "scripts/retrieval/build_authority_evidence_store.py",
        "registry_path": str(DEFAULT_REGISTRY),
        "audit_path": str(DEFAULT_AUDIT),
        "output_dir": str(output_dir),
        "chunking": {
            "max_chars": max_chars,
            "overlap": overlap,
            "unit": "page_text",
        },
        "counts": dict(counts),
        "documents_by_category": dict(sorted(category_counts.items())),
        "documents_by_extractability": dict(sorted(extractability_counts.items())),
        "skipped_documents": len(skipped),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output-dir", default=str(AUTHORITY_EVIDENCE_DIR))
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--include-low-text", action="store_true")
    args = parser.parse_args()

    registry_rows = read_jsonl(Path(args.registry))
    audit_rows = read_jsonl(Path(args.audit))
    manifest = build_store(
        registry_rows=registry_rows,
        audit_rows=audit_rows,
        output_dir=Path(args.output_dir),
        max_chars=args.max_chars,
        overlap=args.overlap,
        include_low_text=args.include_low_text,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
