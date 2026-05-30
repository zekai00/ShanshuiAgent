#!/usr/bin/env python3
"""Audit whether corpus PDFs expose extractable text for RAG ingestion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import METADATA_DIR

META_ROOT = METADATA_DIR
REGISTRY_PATH = META_ROOT / "文献级标注清单.jsonl"
AUDIT_PATH = META_ROOT / "PDF文本可抽取性审计.jsonl"
OCR_QUEUE_PATH = META_ROOT / "需OCR文献清单.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def audit_pdf(path: Path) -> tuple[int, int, float, str | None]:
    try:
        doc = fitz.open(path)
        pages = doc.page_count
        chars = sum(len(page.get_text().strip()) for page in doc)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        return 0, 0, 0.0, f"{type(exc).__name__}: {exc}"

    chars_per_page = chars / pages if pages else 0.0
    return pages, chars, chars_per_page, None


def extractability_label(chars: int, chars_per_page: float) -> str:
    if chars == 0:
        return "needs_ocr"
    if chars < 500 or chars_per_page < 80:
        return "low_text"
    return "text_ready"


def main() -> int:
    rows = read_jsonl(REGISTRY_PATH)
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["raw_path"])
        pages, chars, chars_per_page, error = audit_pdf(path)
        label = "error" if error else extractability_label(chars, chars_per_page)
        audit_rows.append(
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "raw_path": row["raw_path"],
                "category": row.get("category"),
                "authority_level": row.get("authority_level"),
                "rag_import_weight": row.get("rag_import_weight"),
                "page_count": pages,
                "extractable_text_chars": chars,
                "extractable_text_chars_per_page": round(chars_per_page, 1),
                "extractability": label,
                "error": error,
            }
        )

    write_jsonl(AUDIT_PATH, audit_rows)
    ocr_rows = [row for row in audit_rows if row["extractability"] in {"needs_ocr", "low_text", "error"}]
    ocr_rows.sort(key=lambda row: (-float(row.get("rag_import_weight") or 0), row["category"], row["doc_id"]))
    write_jsonl(OCR_QUEUE_PATH, ocr_rows)
    summary = {
        "documents": len(audit_rows),
        "text_ready": sum(1 for row in audit_rows if row["extractability"] == "text_ready"),
        "needs_ocr": sum(1 for row in audit_rows if row["extractability"] == "needs_ocr"),
        "low_text": sum(1 for row in audit_rows if row["extractability"] == "low_text"),
        "error": sum(1 for row in audit_rows if row["extractability"] == "error"),
        "audit_path": str(AUDIT_PATH),
        "ocr_queue_path": str(OCR_QUEUE_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
