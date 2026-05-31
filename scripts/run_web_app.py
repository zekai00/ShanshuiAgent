#!/usr/bin/env python3
"""Run the modern ChineseLandscape web chat interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
ROUTER_LLM_MODEL = os.getenv("CL_ROUTER_LLM_MODEL", FAST_LLM_MODEL)
ROUTER_LLM_ENABLED = os.getenv("CL_ROUTER_LLM_ENABLED", "1").lower() not in {"0", "false", "no"}
RAG_MIN_RERANK_SCORE = float(os.getenv("CL_RAG_MIN_RERANK_SCORE", "1.0"))

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


STRONG_DOMAIN_KEYWORDS = {
    "山水", "中国画", "國畫", "国画", "水墨", "青绿", "金碧", "浅绛", "没骨",
    "皴", "皴法", "笔墨", "设色", "画论", "画史", "画派", "流派", "画家",
    "构图", "意境", "气韵", "三远", "平远", "高远", "深远", "诗画",
    "荆浩", "关仝", "董源", "巨然", "范宽", "郭熙", "米芾", "马远", "夏圭",
    "王维", "赵孟頫", "黄公望", "王蒙", "倪瓒", "吴镇", "沈周", "文徵明", "唐寅",
    "仇英", "董其昌", "四王", "四僧", "石涛", "八大山人", "王时敏", "王鉴",
    "王翚", "王原祁", "吴门", "浙派", "南宗", "北宗",
    "富春山居图", "千里江山图", "溪山行旅图", "早春图", "游春图", "林泉高致",
    "笔法记", "苦瓜和尚画语录", "宣和画谱", "画禅室随笔",
}

AMBIGUOUS_DOMAIN_KEYWORDS = {
    "绘画", "美术", "宋代", "元代", "明代", "清代", "唐代", "五代", "隋代",
    "北宋", "南宋", "晚明", "宋元", "元明", "明清", "唐宋", "隋唐",
}

CASUAL_PATTERNS = (
    re.compile(r"^\s*(你好|您好|hello|hi|嗨)[啊呀!！。.\s]*$", re.IGNORECASE),
    re.compile(r"天气|下雨|晴天|阴天|气温|冷不冷|热不热"),
    re.compile(r"什么是爱|爱是什么"),
)

ROUTE_LABELS = {"domain_research", "casual", "unsupported_general", "need_clarification"}


def route_payload(label: str, reason: str, confidence: float, source: str) -> dict[str, Any]:
    return {
        "label": label if label in ROUTE_LABELS else "need_clarification",
        "reason": reason,
        "confidence": max(0.0, min(1.0, confidence)),
        "source": source,
    }


def classify_question_with_llm(question: str) -> dict[str, Any] | None:
    if not (ROUTER_LLM_ENABLED and DEEPSEEK_API_KEY):
        return None
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=20.0),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是中国山水画研究系统的路由分类器。只输出 JSON。"
                "label 只能是 domain_research、casual、unsupported_general、need_clarification。"
                "domain_research 表示问题需要基于中国山水画史、画论、画家、流派、技法、作品或文献证据回答。"
                "casual 表示寒暄、天气、闲聊。unsupported_general 表示一般知识或哲学生活问题。"
                "need_clarification 表示可能相关但问题过泛，需要用户补充山水画角度。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：{question}\n请输出 JSON：{{\"label\":\"...\",\"confidence\":0到1,\"reason\":\"不超过20字\"}}",
        },
    ]
    try:
        response = client.chat.completions.create(
            model=ROUTER_LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=120,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return None
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return route_payload(
        str(payload.get("label") or "need_clarification"),
        str(payload.get("reason") or "LLM 路由"),
        float(payload.get("confidence") or 0.5),
        "llm",
    )


def route_question(question: str) -> dict[str, Any]:
    normalized = question.strip()
    if not normalized:
        return route_payload("casual", "空问题", 1.0, "rule")
    if any(pattern.search(normalized) for pattern in CASUAL_PATTERNS):
        return route_payload("casual", "明显闲聊", 1.0, "rule")
    if any(keyword in normalized for keyword in STRONG_DOMAIN_KEYWORDS):
        return route_payload("domain_research", "命中山水画领域词", 1.0, "rule")
    if any(keyword in normalized for keyword in AMBIGUOUS_DOMAIN_KEYWORDS):
        llm_route = classify_question_with_llm(normalized)
        if llm_route:
            return llm_route
        return route_payload("need_clarification", "泛时代或泛艺术问题", 0.6, "rule")
    llm_route = classify_question_with_llm(normalized)
    if llm_route and llm_route["confidence"] >= 0.72:
        return llm_route
    return route_payload("unsupported_general", "未命中研究范围", 0.8, "rule")


def is_research_question(question: str) -> bool:
    return route_question(question)["label"] == "domain_research"


def non_research_answer(question: str, route: dict[str, Any] | None = None) -> str:
    normalized = question.strip()
    if any(token in normalized for token in {"你好", "您好", "hello", "hi", "嗨"}):
        return "你好。我主要用于回答中国山水画史、画论、技法、流派和文献证据相关问题。"
    if "天气" in normalized:
        return "听起来是个适合看画、读画论或者出门走走的天气。这类闲聊不需要检索山水画文献，所以我不会强行给出来源。"
    if "什么是爱" in normalized or normalized == "爱是什么":
        return "这是一般哲学或生活问题，不属于山水画史证据库的范围，我不会牵强引用山水文献。简单说，爱通常包含关切、责任、理解和持续投入。若你想问“山水画如何表达情感”，我可以基于文献证据回答。"
    if route and route.get("label") == "need_clarification":
        return "这个问题可能与中国山水画研究有关，但当前表述还不够明确。请补充你想讨论的山水画角度，例如朝代、画家、流派、作品或技法。"
    return "这个问题看起来不属于中国山水画史、画论、技法或文献证据范围，所以我不会启动 RAG 检索。你可以直接问山水画相关问题，例如“青绿山水是什么”或“董其昌南北宗论的影响是什么”。"


def stream_text_answer(answer: str):
    yield json.dumps({"type": "evidence", "evidence": []}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "phase", "phase": "直接回答"}, ensure_ascii=False) + "\n"
    for index in range(0, len(answer), 24):
        yield json.dumps({"type": "delta", "delta": answer[index:index + 24]}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"


def evidence_is_relevant(evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return False
    score = evidence[0].get("rerank_score")
    try:
        return float(score) >= RAG_MIN_RERANK_SCORE
    except (TypeError, ValueError):
        return True


def low_relevance_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    score = evidence[0].get("rerank_score") if evidence else None
    score_text = ""
    try:
        score_text = f"（当前最高相关性分数约为 {float(score):.2f}，低于阈值 {RAG_MIN_RERANK_SCORE:.2f}）"
    except (TypeError, ValueError):
        pass
    return (
        "我没有在当前山水画证据库中检索到足够相关的文献证据，因此不生成带来源的研究回答"
        f"{score_text}。你可以把问题改得更具体，例如补充画家、朝代、作品、流派或技法。\n\n"
        f"原问题：{question}"
    )


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
        "page_count": doc.get("page_count") or document_meta.get("page_count"),
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
                    f"[{doc.get('rank')}] {doc.get('title') or doc.get('source_file')} | {page_label}",
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
                "最后的“来源”部分按编号列出文献短标题和页码，不要列出 chunk_id 或冗长文件名。证据不足时明确说明。"
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
    yield json.dumps({"type": "phase", "phase": "生成中"}, ensure_ascii=False) + "\n"

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
        "router": {
            "llm_enabled": ROUTER_LLM_ENABLED and bool(DEEPSEEK_API_KEY),
            "router_model": ROUTER_LLM_MODEL if ROUTER_LLM_ENABLED and DEEPSEEK_API_KEY else "rule-only",
            "min_rerank_score": RAG_MIN_RERANK_SCORE,
        },
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
                doc = json.loads(line)
                facets = doc.get("facets") or {}
                documents.append(
                    {
                        "source_file": doc.get("source_file"),
                        "title": doc.get("title"),
                        "authority_level": doc.get("authority_level"),
                        "category": doc.get("category"),
                        "page_count": doc.get("page_count"),
                        "facets": {
                            "dynasties": facets.get("dynasties") or [],
                            "lineages_schools": facets.get("lineages_schools") or [],
                            "styles_techniques": facets.get("styles_techniques") or [],
                        },
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
        route = route_question(req.message)
        if route["label"] != "domain_research":
            return {"answer": non_research_answer(req.message, route), "evidence": [], "mode": "direct", "route": route}
        retriever = get_retriever(req.top_k, req.final_k)
        results = retriever.retrieve_and_rerank(req.message)
        evidence = [evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
        if not evidence_is_relevant(evidence):
            return {"answer": low_relevance_answer(req.message, evidence), "evidence": [], "mode": "low_relevance", "route": route}
        answer = generate_answer(req.message, evidence, req.history)
        return {"answer": answer, "evidence": evidence, "mode": "rag", "route": route}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    try:
        route = route_question(req.message)
        if route["label"] != "domain_research":
            return StreamingResponse(
                stream_text_answer(non_research_answer(req.message, route)),
                media_type="application/x-ndjson",
            )
        retriever = get_retriever(req.top_k, req.final_k)
        results = retriever.retrieve_and_rerank(req.message)
        evidence = [evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
        if not evidence_is_relevant(evidence):
            return StreamingResponse(
                stream_text_answer(low_relevance_answer(req.message, evidence)),
                media_type="application/x-ndjson",
            )
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
