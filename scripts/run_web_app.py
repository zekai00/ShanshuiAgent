#!/usr/bin/env python3
"""Run the modern ChineseLandscape web chat interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import fitz
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    BGE_M3_PATH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    FAST_LLM_MODEL,
    PROJECT_ROOT,
    RERANKER_PATH,
    RESEARCHER_LORA_PATH,
    RETRIEVAL_EVIDENCE_DIR,
)

UI_DIR = PROJECT_ROOT / "ui" / "modern"
PDF_PAGE_CACHE_DIR = PROJECT_ROOT / "data" / "runtime" / "pdf_pages"

app = FastAPI(title="ChineseLandscape Web", version="0.1.0")
app.mount("/assets", StaticFiles(directory=str(UI_DIR)), name="assets")

_retriever = None
_retriever_lock = threading.Lock()
_document_map: dict[str, dict[str, Any]] | None = None
_document_map_lock = threading.Lock()


@app.middleware("http")
async def no_cache_for_local_ui(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


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


def load_document_map() -> dict[str, dict[str, Any]]:
    global _document_map
    with _document_map_lock:
        if _document_map is not None:
            return _document_map

        documents_path = RETRIEVAL_EVIDENCE_DIR / "documents.jsonl"
        document_map: dict[str, dict[str, Any]] = {}
        if documents_path.exists():
            with documents_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    source_file = doc.get("source_file")
                    if source_file:
                        document_map[source_file] = doc
        _document_map = document_map
        return _document_map


def document_for_source(source_file: str) -> dict[str, Any]:
    doc = load_document_map().get(source_file)
    if not doc:
        raise HTTPException(status_code=404, detail="未找到登记文献")
    pdf_path = Path(str(doc.get("pdf_path") or ""))
    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF 文件不存在")
    doc["_resolved_pdf_path"] = pdf_path
    return doc


def pdf_query_url(route: str, source_file: str | None, page: Any | None = None) -> str | None:
    if not source_file:
        return None
    params: dict[str, Any] = {"source_file": source_file}
    if page:
        params["page"] = page
    return f"{route}?{urlencode(params)}"


def evidence_payload(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    preview = str(doc.get("raw_chunk_text") or doc.get("contextual_chunk") or "")
    source_file = doc.get("source_file")
    document_meta = load_document_map().get(source_file or "", {})
    page_start = doc.get("page_start")
    return {
        "rank": rank,
        "chunk_id": doc.get("chunk_id"),
        "legacy_milvus_id": doc.get("legacy_milvus_id") or doc.get("id"),
        "source_file": source_file,
        "title": doc.get("title") or document_meta.get("title"),
        "page_start": page_start,
        "page_end": doc.get("page_end"),
        "rerank_score": doc.get("rerank_score"),
        "evidence_store_hit": doc.get("evidence_store_hit"),
        "corrective_query": doc.get("corrective_query"),
        "source_prior_sources": doc.get("source_prior_sources") or [],
        "pdf_url": pdf_query_url("/api/pdf", source_file),
        "page_image_url": pdf_query_url("/api/pdf-page", source_file, page_start),
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
                    f"[{doc.get('rank')}] {doc.get('source_file')} | {page_label}",
                    str(doc.get("preview", "")),
                ]
            )
        )
    return "\n\n".join(parts)


def build_chat_messages(question: str, evidence: list[dict[str, Any]], history: list[dict[str, str]]) -> list[dict[str, str]]:
    recent_history = history[-6:]
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是严谨的中国山水画研究员。只能依据给定证据回答。"
                "前提正常的问题直接作答，不要输出“前提判断”段落。"
                "遇到是否类、时代错置、现代技术错置、人物流派混淆问题，先用一句话说明前提问题。"
                "不要输出“依据与解释：”这类标题。"
                "正文引用只能使用 [1]、[2] 这种证据编号，不要在正文括号里展开文件名、页码或 chunk_id。"
                "最后的“来源”部分按编号列出文献名和页码，不要列出 chunk_id。证据不足时明确说明。"
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
    return messages


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
    lines = ["当前环境未配置在线回答模型，因此只返回证据摘要。"]
    for doc in evidence[:5]:
        page = doc.get("page_start") or "未知"
        lines.append(
            f"- {doc.get('source_file')}，页码 {page}：{doc.get('preview', '')[:160]}"
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
    response = client.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=build_chat_messages(question, evidence, history),
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def stream_chat_answer(question: str, evidence: list[dict[str, Any]], history: list[dict[str, str]]):
    yield json.dumps({"type": "evidence", "evidence": evidence}, ensure_ascii=False) + "\n"

    if not DEEPSEEK_API_KEY:
        answer = fallback_answer(question, evidence)
        for index in range(0, len(answer), 24):
            yield json.dumps({"type": "delta", "delta": answer[index:index + 24]}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=80.0),
    )
    stream = client.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=build_chat_messages(question, evidence, history),
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield json.dumps({"type": "delta", "delta": delta}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "ok": True,
        "llm_configured": bool(DEEPSEEK_API_KEY),
        "answer_model": FAST_LLM_MODEL if DEEPSEEK_API_KEY else "evidence-summary-fallback",
        "answer_provider": "DeepSeek-compatible API" if DEEPSEEK_API_KEY else "local fallback",
        "trained_researcher_lora": str(RESEARCHER_LORA_PATH),
        "trained_researcher_lora_exists": RESEARCHER_LORA_PATH.exists(),
        "retriever_models": {
            "encoder": str(BGE_M3_PATH),
            "reranker": str(RERANKER_PATH),
        },
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


@app.get("/api/pdf")
def pdf_file(source_file: str = Query(min_length=1)) -> FileResponse:
    doc = document_for_source(source_file)
    pdf_path = doc["_resolved_pdf_path"]
    filename = doc.get("source_file") or pdf_path.name
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)


@app.get("/api/pdf-page")
def pdf_page_image(source_file: str = Query(min_length=1), page: int = Query(ge=1)) -> FileResponse:
    doc = document_for_source(source_file)
    pdf_path = doc["_resolved_pdf_path"]
    try:
        with fitz.open(pdf_path) as pdf:
            if page > pdf.page_count:
                raise HTTPException(status_code=404, detail="页码超出 PDF 范围")
            stat = pdf_path.stat()
            cache_key = hashlib.sha256(
                f"{pdf_path}:{stat.st_mtime_ns}:{stat.st_size}:{page}".encode("utf-8")
            ).hexdigest()[:24]
            cache_path = PDF_PAGE_CACHE_DIR / f"{cache_key}.png"
            if not cache_path.exists():
                PDF_PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                pdf_page = pdf.load_page(page - 1)
                pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
                pixmap.save(cache_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 页渲染失败：{exc}") from exc
    return FileResponse(cache_path, media_type="image/png")


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


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    try:
        retriever = get_retriever(req.top_k, req.final_k)
        results = retriever.retrieve_and_rerank(req.message)
        evidence = [evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
        return StreamingResponse(
            stream_chat_answer(req.message, evidence, req.history),
            media_type="application/x-ndjson",
        )
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
