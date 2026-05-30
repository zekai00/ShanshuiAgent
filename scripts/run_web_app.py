#!/usr/bin/env python3
"""Run the modern ChineseLandscape web chat interface."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    FAST_LLM_MODEL,
    PROJECT_ROOT,
    RETRIEVAL_EVIDENCE_DIR,
)

UI_DIR = PROJECT_ROOT / "ui" / "modern"

app = FastAPI(title="ChineseLandscape Web", version="0.1.0")
app.mount("/assets", StaticFiles(directory=str(UI_DIR)), name="assets")

_retriever = None
_retriever_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)
    top_k: int = Field(default=15, ge=3, le=30)
    final_k: int = Field(default=5, ge=1, le=8)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=15, ge=3, le=30)
    final_k: int = Field(default=5, ge=1, le=8)


def get_retriever(top_k: int = 15, final_k: int = 5):
    global _retriever
    with _retriever_lock:
        if _retriever is None or _retriever.top_k != top_k or _retriever.final_k != final_k:
            if _retriever is not None:
                _retriever.close()
            from src.retrieval.online_retrieval import OnlineHybridRetriever

            _retriever = OnlineHybridRetriever(top_k=top_k, final_k=final_k)
    return _retriever


def evidence_payload(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    preview = str(doc.get("raw_chunk_text") or doc.get("contextual_chunk") or "")
    return {
        "rank": rank,
        "chunk_id": doc.get("chunk_id"),
        "legacy_milvus_id": doc.get("legacy_milvus_id") or doc.get("id"),
        "source_file": doc.get("source_file"),
        "title": doc.get("title"),
        "page_start": doc.get("page_start"),
        "page_end": doc.get("page_end"),
        "rerank_score": doc.get("rerank_score"),
        "evidence_store_hit": doc.get("evidence_store_hit"),
        "corrective_query": doc.get("corrective_query"),
        "source_prior_sources": doc.get("source_prior_sources") or [],
        "preview": preview[:700],
    }


def build_context(evidence: list[dict[str, Any]]) -> str:
    parts = []
    for doc in evidence:
        page = doc.get("page_start")
        page_label = f"页码：{page}" if page else "页码：未知"
        parts.append(
            "\n".join(
                [
                    f"[{doc.get('rank')}] {doc.get('source_file')} | {page_label} | chunk_id: {doc.get('chunk_id')}",
                    str(doc.get("preview", "")),
                ]
            )
        )
    return "\n\n".join(parts)


def load_manifest() -> dict[str, Any]:
    manifest_path = RETRIEVAL_EVIDENCE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    counts = manifest.get("counts", {})
    manifest["document_count"] = counts.get("documents", manifest.get("document_count"))
    manifest["chunk_count"] = counts.get("chunks", manifest.get("chunk_count"))
    manifest["page_count"] = counts.get("pages", manifest.get("page_count"))
    return manifest


def fallback_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "当前资料库没有检索到足够证据。"
    lines = [
        "前提判断：已完成检索；当前环境未配置在线回答模型，因此只返回证据摘要。",
        "",
        "依据与解释：",
    ]
    for doc in evidence[:5]:
        page = doc.get("page_start") or "未知"
        lines.append(
            f"- {doc.get('source_file')}，页码 {page}，证据块 {doc.get('chunk_id')}：{doc.get('preview', '')[:160]}"
        )
    lines.extend(["", f"原问题：{question}"])
    return "\n".join(lines)


def generate_answer(question: str, evidence: list[dict[str, Any]], history: list[dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        return fallback_answer(question, evidence)

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=80.0),
    )
    recent_history = history[-6:]
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是严谨的中国山水画研究员。只能依据给定证据回答。"
                "遇到是否类、时代错置、现代技术错置、人物流派混淆问题，先给出“前提判断”。"
                "回答结构使用：前提判断、依据与解释、来源。"
                "每条事实后尽量标注文献名、页码或 chunk_id。证据不足时明确说明。"
            ),
        }
    ]
    for item in recent_history:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:1200]})
    messages.append(
        {
            "role": "user",
            "content": f"问题：{question}\n\n证据：\n{build_context(evidence)}",
        }
    )
    response = client.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "ok": True,
        "llm_configured": bool(DEEPSEEK_API_KEY),
        "evidence_dir": str(RETRIEVAL_EVIDENCE_DIR),
        "manifest": manifest,
    }


@app.get("/api/corpus")
def corpus() -> dict[str, Any]:
    documents_path = RETRIEVAL_EVIDENCE_DIR / "documents.jsonl"
    manifest = load_manifest()
    documents = []
    if documents_path.exists():
        with documents_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(documents) >= 120:
                    break
                doc = json.loads(line)
                documents.append(
                    {
                        "source_file": doc.get("source_file"),
                        "title": doc.get("title"),
                        "authority_level": doc.get("authority_level"),
                        "category": doc.get("category"),
                        "chunk_count": doc.get("chunk_count"),
                        "page_count": doc.get("page_count"),
                    }
                )
    return {"manifest": manifest, "documents": documents}


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    try:
        retriever = get_retriever(req.top_k, req.final_k)
        results = retriever.retrieve_and_rerank(req.query)
        evidence = [evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
        return {"query": req.query, "evidence": evidence}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    try:
        retriever = get_retriever(req.top_k, req.final_k)
        results = retriever.retrieve_and_rerank(req.message)
        evidence = [evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
        answer = generate_answer(req.message, evidence, req.history)
        return {"answer": answer, "evidence": evidence}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
