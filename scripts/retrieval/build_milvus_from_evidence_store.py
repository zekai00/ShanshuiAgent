#!/usr/bin/env python3
"""Build a Milvus Lite RAG index from canonical evidence chunks."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    AUTHORITY_COLBERT_TENSORS_PATH,
    AUTHORITY_EVIDENCE_DIR,
    AUTHORITY_MILVUS_DB_PATH,
    BGE_M3_PATH,
    MODEL_DEVICE,
    RETRIEVAL_COLLECTION_NAME,
)
from FlagEmbedding import BGEM3FlagModel


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


def ensure_clean_output(db_path: Path, colbert_path: Path, force: bool) -> None:
    if not force and (db_path.exists() or colbert_path.exists()):
        raise FileExistsError(
            "Output index already exists. Use --force to rebuild: "
            f"{db_path} / {colbert_path}"
        )
    for path in [db_path, colbert_path]:
        if path.exists():
            path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    colbert_path.parent.mkdir(parents=True, exist_ok=True)


def create_collection(client: MilvusClient, collection_name: str, dense_dim: int) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dynasty", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=16, max_length=64)
    schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="contextual_chunk", datatype=DataType.VARCHAR, max_length=8192)

    client.create_collection(collection_name=collection_name, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    client.create_index(collection_name=collection_name, index_params=index_params)


def as_text_list(value: Any, limit: int = 16) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item)[:64] for item in value if str(item).strip()][:limit]
    text = str(value).strip()
    return [text[:64]] if text else []


def joined(values: list[str], default: str = "未知") -> str:
    values = [value for value in values if value]
    return "、".join(values[:8]) if values else default


def build_row(chunk: dict[str, Any], dense_vec: Any, sparse_vec: Any) -> dict[str, Any]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    dynasties = as_text_list(metadata.get("dynasty"))
    persons = as_text_list(metadata.get("persons"))
    topics = as_text_list(metadata.get("topics") or metadata.get("themes"))
    row = {
        "id": int(chunk["legacy_milvus_id"]),
        "dense_vector": dense_vec.tolist() if hasattr(dense_vec, "tolist") else dense_vec,
        "sparse_vector": {int(k): float(v) for k, v in sparse_vec.items()} if isinstance(sparse_vec, dict) else sparse_vec,
        "contextual_chunk": str(chunk.get("retrieval_text") or ""),
        "source_file": str(chunk.get("source_file") or ""),
        "dynasty": dynasties or ["未知"],
        "painter": joined(persons),
        "subject_matter": joined(topics),
        "content_scope": str(metadata.get("category") or metadata.get("source_type") or "未知"),
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "doc_id": str(chunk.get("doc_id") or ""),
        "title": str(chunk.get("title") or ""),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "authority_level": str(metadata.get("authority_level") or ""),
        "rag_import_weight": float(metadata.get("rag_import_weight") or 0.0),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default=str(AUTHORITY_EVIDENCE_DIR / "chunks.jsonl"))
    parser.add_argument("--db-path", default=str(AUTHORITY_MILVUS_DB_PATH))
    parser.add_argument("--colbert-path", default=str(AUTHORITY_COLBERT_TENSORS_PATH))
    parser.add_argument("--collection", default=RETRIEVAL_COLLECTION_NAME)
    parser.add_argument("--model-path", default=str(BGE_M3_PATH))
    parser.add_argument("--device", default=MODEL_DEVICE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--insert-batch-size", type=int, default=128)
    parser.add_argument("--with-colbert", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    db_path = Path(args.db_path)
    colbert_path = Path(args.colbert_path)
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found: {chunks_path}")

    ensure_clean_output(db_path, colbert_path, args.force)
    use_fp16 = str(args.device).startswith("cuda")
    print(f"[*] Loading BGE-M3 from {args.model_path} on {args.device} (fp16={use_fp16})")
    encoder = BGEM3FlagModel(args.model_path, use_fp16=use_fp16, device=args.device)

    print(f"[*] Creating Milvus Lite collection: {db_path}::{args.collection}")
    client = MilvusClient(str(db_path))
    create_collection(client, args.collection, dense_dim=1024)

    colbert_db: dict[int, Any] = {}
    inserted = 0
    insert_buffer: list[dict[str, Any]] = []
    start_time = time.time()

    for start in tqdm(range(0, len(chunks), args.batch_size), desc="Encoding chunks"):
        batch = chunks[start : start + args.batch_size]
        texts = [str(chunk.get("retrieval_text") or "") for chunk in batch]
        features = encoder.encode(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=args.with_colbert,
        )

        rows = []
        for index, chunk in enumerate(batch):
            rows.append(
                build_row(
                    chunk,
                    dense_vec=features["dense_vecs"][index],
                    sparse_vec=features["lexical_weights"][index],
                )
            )
            if args.with_colbert:
                colbert_db[int(chunk["legacy_milvus_id"])] = features["colbert_vecs"][index]

        insert_buffer.extend(rows)
        if len(insert_buffer) >= args.insert_batch_size:
            client.insert(collection_name=args.collection, data=insert_buffer)
            inserted += len(insert_buffer)
            insert_buffer = []

    if insert_buffer:
        client.insert(collection_name=args.collection, data=insert_buffer)
        inserted += len(insert_buffer)

    client.load_collection(args.collection)
    client.close()

    if args.with_colbert:
        with colbert_path.open("wb") as f:
            pickle.dump(colbert_db, f)
    elif colbert_path.exists():
        colbert_path.unlink()

    manifest = {
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "builder": "scripts/retrieval/build_milvus_from_evidence_store.py",
        "chunks_path": str(chunks_path),
        "db_path": str(db_path),
        "collection": args.collection,
        "model_path": args.model_path,
        "device": args.device,
        "with_colbert": args.with_colbert,
        "inserted_chunks": inserted,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    manifest_path = db_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
