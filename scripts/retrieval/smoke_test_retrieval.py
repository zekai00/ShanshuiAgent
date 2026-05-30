#!/usr/bin/env python3
"""Run a small end-to-end retrieval smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.online_retrieval import OnlineHybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="郭熙的三远法是什么？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--final-k", type=int, default=3)
    args = parser.parse_args()

    retriever = OnlineHybridRetriever(top_k=args.top_k, final_k=args.final_k)
    try:
        results = retriever.retrieve_and_rerank(args.query)
    finally:
        retriever.close()

    summary = [
        {
            "rank": index,
            "source_file": item.get("source_file"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "chunk_id": item.get("chunk_id"),
            "rerank_score": item.get("rerank_score"),
            "preview": str(item.get("raw_chunk_text") or item.get("contextual_chunk") or "")[:180],
        }
        for index, item in enumerate(results, start=1)
    ]
    print(json.dumps({"query": args.query, "result_count": len(results), "results": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
