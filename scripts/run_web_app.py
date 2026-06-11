#!/usr/bin/env python3
"""Run the modern ShanshuiAgent web interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

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
    COMFYUI_SERVER_URL,
    COMFYUI_WORKFLOW_PATH,
    FAST_LLM_MODEL,
    GENERATED_IMAGES_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    PROJECT_ROOT,
    RERANKER_PATH,
    RETRIEVAL_COLBERT_TENSORS_PATH,
    RESEARCHER_LORA_PATH,
    RETRIEVAL_EVIDENCE_DIR,
    RETRIEVAL_MILVUS_DB_PATH,
    VLM_CRITIC_MODEL_PATH,
)

UI_DIR = PROJECT_ROOT / "ui" / "modern"
PDF_PAGE_CACHE_DIR = PROJECT_ROOT / "data" / "runtime" / "pdf_pages"
ROUTER_LLM_MODEL = os.getenv("CL_ROUTER_LLM_MODEL", FAST_LLM_MODEL)
ROUTER_LLM_ENABLED = os.getenv("CL_ROUTER_LLM_ENABLED", "1").lower() not in {"0", "false", "no"}
AGENT_LLM_ROUTER_ENABLED = os.getenv("CL_AGENT_LLM_ROUTER_ENABLED", "1").lower() not in {"0", "false", "no"}
RAG_MIN_RERANK_SCORE = float(os.getenv("CL_RAG_MIN_RERANK_SCORE", "1.0"))
VLM_CRITIC_ENABLED = os.getenv("CL_VLM_CRITIC_ENABLED", "1").lower() not in {"0", "false", "no"}
VLM_CRITIC_PASS_SCORE = float(os.getenv("CL_VLM_CRITIC_PASS_SCORE", "0.65"))
VLM_CRITIC_MAX_NEW_TOKENS = int(os.getenv("CL_VLM_CRITIC_MAX_NEW_TOKENS", "700"))
COMFYUI_ASYNC_ENABLED = os.getenv("CL_COMFYUI_ASYNC_ENABLED", "1").lower() not in {"0", "false", "no"}
COMFYUI_JOB_WORKERS = int(os.getenv("CL_COMFYUI_JOB_WORKERS", "1"))

app = FastAPI(title="ShanshuiAgent Web", version="0.1.0")
app.mount("/assets", StaticFiles(directory=str(UI_DIR)), name="assets")
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated-images", StaticFiles(directory=str(GENERATED_IMAGES_DIR)), name="generated-images")

_retriever = None
_retriever_lock = threading.Lock()
_document_map: dict[str, dict[str, Any]] | None = None
_document_map_lock = threading.Lock()
_vlm_critic_bundle: dict[str, Any] | None = None
_vlm_critic_lock = threading.Lock()
_image_jobs: dict[str, dict[str, Any]] = {}
_image_jobs_lock = threading.Lock()
_image_job_executor = ThreadPoolExecutor(max_workers=max(1, COMFYUI_JOB_WORKERS), thread_name_prefix="comfyui-job")


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
    user_id: str = Field(default="guest", max_length=80)
    thread_id: str | None = Field(default=None, max_length=160)


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
    "王翚", "王原祁", "元四家", "吴门", "浙派", "南宗", "北宗",
    "富春山居图", "千里江山图", "溪山行旅图", "早春图", "游春图", "林泉高致",
    "笔法记", "苦瓜和尚画语录", "宣和画谱", "画禅室随笔",
}

AMBIGUOUS_DOMAIN_KEYWORDS = {
    "绘画", "美术", "宋代", "元代", "明代", "清代", "唐代", "五代", "隋代",
    "北宋", "南宋", "晚明", "宋元", "元明", "明清", "唐宋", "隋唐",
}

CASUAL_PATTERNS = (
    re.compile(r"^\s*(你好|您好|hello|hi|嗨)[啊呀!！。.\s]*$", re.IGNORECASE),
    re.compile(r"你是谁|随便聊聊|适合出门|出门吗"),
    re.compile(r"天气|下雨|晴天|阴天|气温|冷不冷|热不热"),
    re.compile(r"什么是爱|爱是什么"),
)

ROUTE_LABELS = {
    "direct",
    "domain_research",
    "casual",
    "general_art_qa",
    "research_qa",
    "research_then_image",
    "unsupported_image",
    "unsupported_general",
    "need_clarification",
    "invalid_premise",
}


def route_payload(label: str, reason: str, confidence: float, source: str) -> dict[str, Any]:
    return {
        "label": label if label in ROUTE_LABELS else "need_clarification",
        "reason": reason,
        "confidence": max(0.0, min(1.0, confidence)),
        "source": source,
    }


def classify_question_with_llm(question: str) -> dict[str, Any] | None:
    if not (ROUTER_LLM_ENABLED and LLM_API_KEY):
        return None
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
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
    return (
        "我没有在当前山水画证据库中检索到足够相关的文献证据，因此不生成带来源的研究回答"
        "。你可以把问题改得更具体，例如补充画家、朝代、作品、流派或技法。\n\n"
        f"原问题：{question}"
    )


AGENT_NODE_TITLES = {
    "intake": "任务理解",
    "planner": "计划制定",
    "researcher": "文献检索",
    "verifier": "证据核验",
    "research_synthesizer": "研究卷宗",
    "prompt_designer": "图像提示词",
    "image_generator": "图像生成",
    "image_critic": "图像检查",
    "final_writer": "最终回复",
    "memory_writer": "记忆写入",
}

IMAGE_INTENT_PATTERNS = (
    re.compile(r"(生成|绘制|创作|做|出|搞|弄|画(?!家|派|史|论|法|科|面))\s*(一幅|一张|一个)?[^，。！？\n]{0,30}(山水画|山水图|中国画|国画|水墨画|图像|图片|画面|图|长卷|立轴|斗方)"),
    re.compile(r"(山水画|山水图|中国画|国画|水墨画).*(生成|绘制|创作|出图|搞|弄|画)"),
    re.compile(r"(能|可以|可不可以|能不能).{0,12}(画|生成|绘制|创作).{0,12}(类似|这种|这个|那样|同样)"),
    re.compile(r"(image|picture|prompt|comfyui|stable diffusion)", re.IGNORECASE),
)

DYNASTY_TERMS = ["隋", "唐", "五代", "北宋", "南宋", "宋", "元", "明", "清", "晚明", "明清", "宋元"]
SCHOOL_TERMS = ["吴门", "浙派", "南宗", "北宗", "院体", "文人画", "四王", "四僧", "元四家", "青绿", "金碧", "浅绛", "水墨"]
TECHNIQUE_TERMS = ["皴", "皴法", "披麻皴", "斧劈皴", "点苔", "设色", "笔墨", "三远", "平远", "高远", "深远", "留白"]
ARTIST_TERMS = [
    "荆浩", "关仝", "董源", "巨然", "范宽", "郭熙", "米芾", "马远", "夏圭", "王维",
    "赵孟頫", "黄公望", "王蒙", "倪瓒", "吴镇", "沈周", "文徵明", "唐寅", "仇英",
    "董其昌", "石涛", "八大山人", "王时敏", "王鉴", "王翚", "王原祁",
]
GENERAL_ART_TERMS = {
    "清明上河图", "张择端", "韩熙载夜宴图", "顾闳中", "洛神赋图", "顾恺之",
    "步辇图", "阎立本", "簪花仕女图", "张萱", "周昉",
}


def event_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def node_event(node: str, status: str, detail: str = "", data: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "node",
        "node": node,
        "title": AGENT_NODE_TITLES.get(node, node),
        "status": status,
        "detail": detail,
    }
    if data is not None:
        payload["data"] = data
    return payload


def text_chunks(text: str, size: int = 28):
    for index in range(0, len(text), size):
        yield text[index:index + size]


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def has_image_intent(question: str) -> bool:
    normalized = question.strip()
    return any(pattern.search(normalized) for pattern in IMAGE_INTENT_PATTERNS)


def find_terms(question: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in question]


def extract_entities(question: str) -> dict[str, list[str]]:
    return {
        "dynasties": find_terms(question, DYNASTY_TERMS),
        "schools": find_terms(question, SCHOOL_TERMS),
        "techniques": find_terms(question, TECHNIQUE_TERMS),
        "artists": find_terms(question, ARTIST_TERMS),
    }


def detect_premise_issue(question: str) -> dict[str, Any] | None:
    modern_terms = r"(Stable\s*Diffusion|ComfyUI|Midjourney|Flux|AI|人工智能|生成式模型)"
    historical_terms = r"(清代|明代|宋代|元代|唐代|四王|元四家|董其昌|石涛|黄公望|范宽|郭熙|吴门|浙派)"
    if re.search(modern_terms, question, flags=re.I) and re.search(historical_terms, question):
        invalid_usage = re.search(
            rf"{historical_terms}.*?(如何|怎么|是否|是不是|能否|直接).*?(使用|用|接触|学习).*?{modern_terms}",
            question,
            flags=re.I,
        ) or re.search(
            rf"{modern_terms}.*?(是否|是不是).*?(清代|明代|宋代|元代|传统|古代|富春山居图|千里江山图)",
            question,
            flags=re.I,
        )
        modern_reference = re.search(r"(根据|参考|借鉴|以).*(风格|画法|笔墨).*(生成|绘制|创作)", question)
        if invalid_usage and not modern_reference:
            return {
                "kind": "modern_technology_anachronism",
                "severity": "invalid_premise",
                "message": "问题把现代图像生成技术放进古代画史语境，前提不成立。",
                "recommended_action": "应改为说明现代系统如何参考相关画派或画家风格进行创作。",
            }
    if "梵高" in question and ("元四家" in question or "山水画" in question) and re.search(r"(直接|是否|有没有).*?(学习|师法|临摹)", question):
        return {
            "kind": "unsupported_direct_influence",
            "severity": "needs_evidence",
            "message": "问题涉及跨文化直接影响，需要文献证据支持，不能默认成立。",
            "recommended_action": "先检索证据；证据不足时应说明不能证明直接学习关系。",
        }
    return None


AGENT_TASK_TYPES = {
    "direct",
    "general_art_qa",
    "research_qa",
    "research_then_image",
    "unsupported_image",
    "need_clarification",
    "unsupported_general",
    "invalid_premise",
}


def compact_history(history: list[dict[str, str]], limit: int = 6) -> str:
    lines = []
    for item in history[-limit:]:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


def history_has_landscape_context(history: list[dict[str, str]]) -> bool:
    context = compact_history(history, limit=8)
    if not context:
        return False
    if any(keyword in context for keyword in STRONG_DOMAIN_KEYWORDS):
        return True
    return any(keyword in context for keyword in {"山水", "中国画", "国画", "水墨", "画派", "画论"})


def classify_agent_intake_with_llm(question: str, history: list[dict[str, str]]) -> dict[str, Any] | None:
    if not (AGENT_LLM_ROUTER_ENABLED and LLM_API_KEY):
        return None
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=20.0),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是山水画研究创作智能体的任务理解器。只输出 JSON，不要解释。"
                "你必须结合最近对话理解省略和指代，例如“能画一幅类似的吗”“这个也生成一下”。"
                "task_type 只能是 direct、general_art_qa、research_qa、research_then_image、"
                "unsupported_image、need_clarification、unsupported_general、invalid_premise。"
                "research_then_image：用户要画、生成、绘制、搞个、弄个图像/山水图/长卷/类似画，即使说法口语化。"
                "research_qa：中国山水画史、画论、画家、流派、技法、作品的研究问答，需要文献证据。"
                "general_art_qa：中国绘画/美术史的简单事实问答但不一定属于山水画证据库，例如《清明上河图》作者。"
                "direct：寒暄、天气等闲聊。unsupported_general：完全无关的一般生活或百科。"
                "invalid_premise：把古代人物/画派当成使用 Stable Diffusion、ComfyUI 等现代技术的历史事实；"
                "但如果用户是让现代系统参考古代风格生成图像，应判为 research_then_image。"
                "entities 包含 dynasties、schools、techniques、artists、works 五个数组。"
                "needs_retrieval 和 needs_image 必须是布尔值。confidence 为 0 到 1。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"最近对话：\n{compact_history(history)}\n\n"
                f"当前用户问题：{question}\n\n"
                "输出 JSON："
                "{\"task_type\":\"...\",\"needs_retrieval\":true,\"needs_image\":false,"
                "\"entities\":{\"dynasties\":[],\"schools\":[],\"techniques\":[],\"artists\":[],\"works\":[]},"
                "\"reason\":\"不超过30字\",\"confidence\":0.0}"
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=ROUTER_LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=260,
        )
        payload = extract_json_object(response.choices[0].message.content or "")
    except Exception:
        return None
    if not payload:
        return None
    task_type = str(payload.get("task_type") or "")
    if task_type not in AGENT_TASK_TYPES:
        return None
    entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    normalized_entities = {
        "dynasties": [str(item) for item in entities.get("dynasties", []) if item],
        "schools": [str(item) for item in entities.get("schools", []) if item],
        "techniques": [str(item) for item in entities.get("techniques", []) if item],
        "artists": [str(item) for item in entities.get("artists", []) if item],
        "works": [str(item) for item in entities.get("works", []) if item],
    }
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "task_type": task_type,
        "route": route_payload(
            "domain_research" if task_type in {"research_qa", "research_then_image"} else task_type,
            str(payload.get("reason") or "LLM 语义路由"),
            confidence,
            "agent_llm",
        ),
        "entities": normalized_entities,
        "needs_retrieval": task_type in {"research_qa", "research_then_image"} or (
            bool(payload.get("needs_retrieval")) and task_type not in {"general_art_qa", "direct", "unsupported_general"}
        ),
        "needs_image": bool(payload.get("needs_image")) or task_type == "research_then_image",
        "needs_clarification": task_type == "need_clarification",
        "premise_issue": None,
        "router": "llm",
    }


def build_rule_agent_intake(question: str) -> dict[str, Any]:
    image_intent = has_image_intent(question)
    entities = extract_entities(question)
    premise_issue = detect_premise_issue(question)
    general_art = any(term in question for term in GENERAL_ART_TERMS)
    if general_art and not image_intent:
        return {
            "task_type": "general_art_qa",
            "route": route_payload("general_art_qa", "一般中国绘画史问题", 0.9, "rule"),
            "entities": entities,
            "needs_retrieval": False,
            "needs_image": False,
            "needs_clarification": False,
            "premise_issue": premise_issue,
            "router": "rule",
        }
    route = route_question(question)
    domain_research = route["label"] == "domain_research"
    if image_intent and not domain_research and any(term in question for term in {"山水", "国画", "水墨", "中国画"}):
        domain_research = True
    if route["label"] in {"casual", "unsupported_general"} and not image_intent:
        task_type = "general_art_qa" if general_art else "direct"
    elif premise_issue and premise_issue.get("severity") == "invalid_premise":
        task_type = "invalid_premise"
    elif image_intent and domain_research:
        task_type = "research_then_image"
    elif image_intent:
        task_type = "unsupported_image"
    else:
        task_type = "research_qa" if domain_research else route["label"]
    return {
        "task_type": task_type,
        "route": route,
        "entities": entities,
        "needs_retrieval": task_type in {"research_qa", "research_then_image"},
        "needs_image": task_type in {"research_then_image"},
        "needs_clarification": task_type == "need_clarification",
        "premise_issue": premise_issue,
        "router": "rule",
    }


def build_agent_intake(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    history = history or []
    rule_intake = build_rule_agent_intake(question)
    hard_issue = detect_premise_issue(question)
    if hard_issue and hard_issue.get("severity") == "invalid_premise":
        rule_intake["task_type"] = "invalid_premise"
        rule_intake["needs_retrieval"] = False
        rule_intake["needs_image"] = False
        rule_intake["premise_issue"] = hard_issue
        return rule_intake
    llm_intake = classify_agent_intake_with_llm(question, history)
    if has_image_intent(question) and history_has_landscape_context(history):
        contextual_intake = llm_intake or rule_intake
        contextual_intake["task_type"] = "research_then_image"
        contextual_intake["needs_retrieval"] = True
        contextual_intake["needs_image"] = True
        contextual_intake["needs_clarification"] = False
        contextual_intake["premise_issue"] = hard_issue
        contextual_intake["route"] = route_payload("domain_research", "上下文续问图像创作", 0.92, "contextual_agent")
        contextual_intake["router"] = "llm_context_guard" if llm_intake else "context_guard"
        return contextual_intake
    if rule_intake["task_type"] == "general_art_qa" and (
        not llm_intake or llm_intake["task_type"] != "research_then_image"
    ):
        if llm_intake and llm_intake["task_type"] == "general_art_qa" and llm_intake["route"]["confidence"] >= 0.5:
            return llm_intake
        return rule_intake
    if llm_intake and llm_intake["route"]["confidence"] >= 0.5:
        llm_intake["premise_issue"] = hard_issue
        if hard_issue and hard_issue.get("severity") == "needs_evidence":
            llm_intake["needs_retrieval"] = True
            if llm_intake["task_type"] in {"direct", "general_art_qa"}:
                llm_intake["task_type"] = "research_qa"
        return llm_intake
    return rule_intake


def build_agent_plan(intake: dict[str, Any]) -> list[dict[str, str]]:
    task_type = intake["task_type"]
    if task_type in {"direct", "general_art_qa"}:
        nodes = [("final_writer", "直接回答闲聊或非研究问题")]
    elif task_type in {"unsupported_image", "need_clarification", "unsupported_general", "invalid_premise"}:
        nodes = [("verifier", "判断边界或错误前提"), ("final_writer", "给出纠偏或澄清建议")]
    elif task_type == "research_qa":
        nodes = [
            ("researcher", "从文献库检索证据"),
            ("verifier", "核验证据相关性和问题前提"),
            ("final_writer", "基于证据生成研究回答"),
        ]
    else:
        nodes = [
            ("researcher", "从文献库检索画史和技法证据"),
            ("verifier", "核验证据是否足够支撑创作约束"),
            ("research_synthesizer", "整理给画师使用的研究卷宗"),
            ("prompt_designer", "把研究卷宗转成 ComfyUI 图像提示词"),
            ("image_generator", "调用 ComfyUI 生成图像"),
            ("image_critic", "规则与 VLM 视觉评审"),
            ("final_writer", "交付图像、提示词、证据来源"),
            ("memory_writer", "记录明确的用户偏好"),
        ]
    return [{"node": node, "title": AGENT_NODE_TITLES[node], "goal": goal} for node, goal in nodes]


def fallback_research_brief(question: str, evidence: list[dict[str, Any]], intake: dict[str, Any]) -> dict[str, Any]:
    key_points = []
    for item in evidence[:4]:
        title = item.get("title") or item.get("source_file") or "未命名文献"
        page = item.get("page_start") or "未知"
        preview = str(item.get("preview") or "").strip().replace("\n", " ")[:150]
        key_points.append(f"[{item.get('rank')}] {title}，第 {page} 页：{preview}")
    entities = intake.get("entities") or {}
    visual_constraints = []
    if "吴门" in entities.get("schools", []):
        visual_constraints.extend(["江南实景气息", "文人清雅", "平远空间", "园林溪桥题材"])
    if "青绿" in entities.get("schools", []):
        visual_constraints.extend(["石青石绿设色", "矿物色层染", "装饰性山体轮廓"])
    if "四王" in entities.get("schools", []):
        visual_constraints.extend(["仿古笔墨", "程式化山石", "稳健层叠山体"])
    if "宋" in entities.get("dynasties", []):
        visual_constraints.extend(["严整构图", "山体体量感", "烟岚层次"])
    if not visual_constraints:
        visual_constraints = ["中国山水画笔墨", "山石树木层次", "留白与云水空间", "纸本水墨质感"]
    return {
        "topic": question[:80],
        "key_points": key_points or ["当前证据不足，只能使用通用山水画创作约束。"],
        "visual_constraints": list(dict.fromkeys(visual_constraints))[:8],
        "citations": [item.get("rank") for item in evidence[:5] if item.get("rank")],
    }


def synthesize_research_brief(question: str, evidence: list[dict[str, Any]], intake: dict[str, Any]) -> dict[str, Any]:
    fallback = fallback_research_brief(question, evidence, intake)
    if not (LLM_API_KEY and evidence):
        return fallback
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=60.0),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是中国山水画研究卷宗整理器。只输出 JSON。"
                "必须严格依据给定证据，提炼给图像创作节点使用的画史、构图、笔墨和设色约束。"
                "JSON 字段：topic, key_points(list), visual_constraints(list), citations(list of evidence ranks)。"
            ),
        },
        {
            "role": "user",
            "content": f"用户任务：{question}\n\n证据：\n{build_context(evidence)}",
        },
    ]
    try:
        response = client.chat.completions.create(model=FAST_LLM_MODEL, messages=messages, temperature=0.1, max_tokens=900)
        payload = extract_json_object(response.choices[0].message.content or "")
    except Exception:
        payload = None
    if not payload:
        return fallback
    return {
        "topic": str(payload.get("topic") or fallback["topic"])[:120],
        "key_points": [str(item) for item in payload.get("key_points") or fallback["key_points"]][:6],
        "visual_constraints": [str(item) for item in payload.get("visual_constraints") or fallback["visual_constraints"]][:10],
        "citations": [int(item) for item in payload.get("citations") or fallback["citations"] if str(item).isdigit()][:5],
    }


def fallback_image_spec(question: str, brief: dict[str, Any]) -> dict[str, Any]:
    width, height, image_format = 1024, 1024, "square"
    if any(term in question for term in {"长卷", "横幅", "横图"}):
        width, height, image_format = 1536, 768, "horizontal_scroll"
    if any(term in question for term in {"立轴", "竖幅", "竖图"}):
        width, height, image_format = 768, 1536, "vertical_hanging_scroll"
    constraints = ", ".join(brief.get("visual_constraints") or [])
    positive_prompt = (
        "Chinese shanshui landscape painting, ink wash on xuan paper, "
        f"{constraints}, refined brushwork, layered mountains, mist, scholar atmosphere, "
        "authentic Chinese painting composition, museum quality"
    )
    negative_prompt = (
        "photorealistic, oil painting, western landscape, modern buildings, neon color, "
        "text, watermark, logo, low quality, distorted architecture"
    )
    return {
        "format": image_format,
        "width": width,
        "height": height,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "style_notes": "；".join(brief.get("visual_constraints") or []),
    }


def design_image_spec(question: str, brief: dict[str, Any]) -> dict[str, Any]:
    fallback = fallback_image_spec(question, brief)
    if not LLM_API_KEY:
        return fallback
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=60.0),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是 ComfyUI/Flux 图像提示词设计器。只输出 JSON。"
                "字段：format, width, height, positive_prompt, negative_prompt, style_notes。"
                "positive_prompt 必须是英文逗号分隔短语；必须忠于研究卷宗；不要加入现代建筑、摄影、油画风格。"
            ),
        },
        {"role": "user", "content": f"用户任务：{question}\n\n研究卷宗：{json.dumps(brief, ensure_ascii=False)}"},
    ]
    try:
        response = client.chat.completions.create(model=FAST_LLM_MODEL, messages=messages, temperature=0.15, max_tokens=900)
        payload = extract_json_object(response.choices[0].message.content or "")
    except Exception:
        payload = None
    if not payload:
        return fallback
    width = int(payload.get("width") or fallback["width"])
    height = int(payload.get("height") or fallback["height"])
    return {
        "format": str(payload.get("format") or fallback["format"]),
        "width": max(512, min(1920, width)),
        "height": max(512, min(1920, height)),
        "positive_prompt": str(payload.get("positive_prompt") or fallback["positive_prompt"]),
        "negative_prompt": str(payload.get("negative_prompt") or fallback["negative_prompt"]),
        "style_notes": str(payload.get("style_notes") or fallback["style_notes"]),
    }


def generated_image_url(path: Path) -> str:
    return f"/generated-images/{quote(path.name)}"


def _comfyui_preflight(server: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(proxy=None, trust_env=False, timeout=2.5) as client:
            client.get(f"{server}/system_stats")
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": "comfyui_offline",
            "message": f"ComfyUI 当前不可用：{exc}",
        }
    if not COMFYUI_WORKFLOW_PATH.exists():
        return {
            "status": "failed",
            "error_type": "workflow_missing",
            "message": f"ComfyUI 工作流不存在：{COMFYUI_WORKFLOW_PATH}",
        }
    return None


def _set_image_job(job_id: str, payload: dict[str, Any]) -> None:
    with _image_jobs_lock:
        current = _image_jobs.get(job_id, {})
        current.update(payload)
        current["updated_at"] = time.time()
        _image_jobs[job_id] = current


def _get_image_job(job_id: str) -> dict[str, Any] | None:
    with _image_jobs_lock:
        job = _image_jobs.get(job_id)
        return dict(job) if job else None


def _run_image_job(job_id: str, spec: dict[str, Any], seed: int) -> None:
    _set_image_job(job_id, {"status": "running", "message": "ComfyUI 正在生成图像"})
    result = _generate_image_with_comfyui_sync(spec, seed=seed, preflight_checked=True)
    result["job_id"] = job_id
    _set_image_job(job_id, {"status": result.get("status", "failed"), "image_result": result, "message": result.get("message")})


def _queue_image_generation(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:16]
    job = {
        "job_id": job_id,
        "status": "queued",
        "seed": seed,
        "created_at": time.time(),
        "updated_at": time.time(),
        "message": "图像任务已提交后台生成。",
        "image_result": None,
    }
    with _image_jobs_lock:
        _image_jobs[job_id] = job
    _image_job_executor.submit(_run_image_job, job_id, dict(spec), seed)
    return {
        "status": "queued",
        "job_id": job_id,
        "seed": seed,
        "message": "图像任务已提交后台生成，可通过 /api/image-jobs/{job_id} 查询。",
        "workflow": str(COMFYUI_WORKFLOW_PATH),
    }


def _generate_image_with_comfyui_sync(
    spec: dict[str, Any],
    *,
    seed: int | None = None,
    preflight_checked: bool = False,
) -> dict[str, Any]:
    seed = seed or random.randint(1, 2**31 - 1)
    server = COMFYUI_SERVER_URL.rstrip("/")
    if not preflight_checked:
        preflight_error = _comfyui_preflight(server)
        if preflight_error:
            preflight_error["seed"] = seed
            return preflight_error
    try:
        workflow_data = json.loads(COMFYUI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        workflow_data["68"]["inputs"]["text"] = spec["positive_prompt"]
        workflow_data["69"]["inputs"]["width"] = int(spec["width"])
        workflow_data["69"]["inputs"]["height"] = int(spec["height"])
        workflow_data["70"]["inputs"]["seed"] = seed
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": "workflow_error",
            "message": f"ComfyUI 工作流配置不可用：{exc}",
            "seed": seed,
        }
    try:
        with httpx.Client(proxy=None, trust_env=False, timeout=20.0) as client:
            response = client.post(f"{server}/prompt", json={"prompt": workflow_data})
            response.raise_for_status()
            prompt_id = response.json()["prompt_id"]
            deadline = time.time() + float(os.getenv("CL_COMFYUI_WAIT_SECONDS", "120"))
            while time.time() < deadline:
                history = client.get(f"{server}/history/{prompt_id}", timeout=20.0).json()
                if prompt_id not in history:
                    time.sleep(2)
                    continue
                outputs = history[prompt_id].get("outputs", {})
                for output in outputs.values():
                    images = output.get("images") or []
                    if not images:
                        continue
                    image_info = images[0]
                    image_response = client.get(
                        f"{server}/view",
                        params={
                            "filename": image_info.get("filename"),
                            "subfolder": image_info.get("subfolder", ""),
                            "type": image_info.get("type", "output"),
                        },
                        timeout=40.0,
                    )
                    image_response.raise_for_status()
                    filename = Path(str(image_info.get("filename") or f"agent_{seed}.png")).name
                    save_path = GENERATED_IMAGES_DIR / filename
                    save_path.write_bytes(image_response.content)
                    return {
                        "status": "success",
                        "path": str(save_path),
                        "url": generated_image_url(save_path),
                        "filename": save_path.name,
                        "seed": seed,
                        "workflow": str(COMFYUI_WORKFLOW_PATH),
                        "prompt_id": prompt_id,
                    }
                return {"status": "failed", "error_type": "empty_output", "message": "ComfyUI 未返回图像。", "seed": seed}
    except Exception as exc:
        return {"status": "failed", "error_type": "comfyui_error", "message": str(exc), "seed": seed}
    return {"status": "failed", "error_type": "timeout", "message": "ComfyUI 图像生成超时。", "seed": seed}


def generate_image_with_comfyui(spec: dict[str, Any]) -> dict[str, Any]:
    seed = random.randint(1, 2**31 - 1)
    server = COMFYUI_SERVER_URL.rstrip("/")
    preflight_error = _comfyui_preflight(server)
    if preflight_error:
        preflight_error["seed"] = seed
        return preflight_error
    if COMFYUI_ASYNC_ENABLED:
        return _queue_image_generation(spec, seed)
    return _generate_image_with_comfyui_sync(spec, seed=seed, preflight_checked=True)


def extract_json_payload(text: str) -> dict[str, Any] | None:
    payload = extract_json_object(text)
    if payload:
        return payload
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_vlm_critic_bundle() -> dict[str, Any]:
    global _vlm_critic_bundle
    with _vlm_critic_lock:
        if _vlm_critic_bundle is not None:
            return _vlm_critic_bundle
        if not VLM_CRITIC_MODEL_PATH.exists():
            raise FileNotFoundError(f"VLM 评审模型不存在：{VLM_CRITIC_MODEL_PATH}")

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(VLM_CRITIC_MODEL_PATH),
            torch_dtype="auto",
            device_map=os.getenv("CL_VLM_CRITIC_DEVICE_MAP", "auto"),
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(str(VLM_CRITIC_MODEL_PATH), local_files_only=True)
        _vlm_critic_bundle = {"model": model, "processor": processor}
        return _vlm_critic_bundle


def build_vlm_critic_prompt(spec: dict[str, Any], brief: dict[str, Any]) -> str:
    constraints = brief.get("visual_constraints") or []
    key_points = brief.get("key_points") or []
    return (
        "你是中国山水画生成结果的视觉评审员。请观察图片，并结合给定创作约束进行判断。"
        "只输出一个 JSON 对象，不要输出 Markdown，不要解释 JSON 之外的文字。\n\n"
        "评审目标：\n"
        "1. 判断图像是否像中国山水画，而不是摄影、油画、西方风景或数字插画。\n"
        "2. 判断是否基本符合研究卷宗中的构图、笔墨、设色、时代/流派约束。\n"
        "3. 检查是否有现代建筑、文字、水印、logo、明显畸形、低质量伪影。\n"
        "4. 给出是否建议重新生成，以及应如何修改 prompt。\n\n"
        f"研究约束：{json.dumps(constraints, ensure_ascii=False)}\n"
        f"研究依据要点：{json.dumps(key_points[:4], ensure_ascii=False)}\n"
        f"positive_prompt：{spec.get('positive_prompt')}\n"
        f"negative_prompt：{spec.get('negative_prompt')}\n\n"
        "JSON 格式："
        "{\"passed\":true,\"score\":0.0,\"style_alignment\":\"...\","
        "\"evidence_alignment\":\"...\",\"issues\":[\"...\"],"
        "\"retry_recommended\":false,\"prompt_revision_hints\":[\"...\"]}"
    )


def run_vlm_image_critic(image_path: Path, spec: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any]:
    bundle = get_vlm_critic_bundle()
    model = bundle["model"]
    processor = bundle["processor"]

    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_vlm_critic_prompt(spec, brief)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        inputs = inputs.to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=VLM_CRITIC_MAX_NEW_TOKENS,
        do_sample=False,
    )
    generated_ids_trimmed = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    payload = extract_json_payload(output)
    if not payload:
        return {
            "enabled": True,
            "model": str(VLM_CRITIC_MODEL_PATH),
            "passed": False,
            "score": 0.0,
            "issues": ["VLM 未返回可解析 JSON"],
            "raw_output": output[:1200],
            "retry_recommended": False,
        }
    try:
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "enabled": True,
        "model": str(VLM_CRITIC_MODEL_PATH),
        "passed": bool(payload.get("passed")) and score >= VLM_CRITIC_PASS_SCORE,
        "score": max(0.0, min(1.0, score)),
        "style_alignment": str(payload.get("style_alignment") or ""),
        "evidence_alignment": str(payload.get("evidence_alignment") or ""),
        "issues": [str(item) for item in payload.get("issues") or [] if item],
        "retry_recommended": bool(payload.get("retry_recommended")),
        "prompt_revision_hints": [str(item) for item in payload.get("prompt_revision_hints") or [] if item],
        "raw_output": output[:1200],
    }


def critic_image_result(image_result: dict[str, Any], spec: dict[str, Any], brief: dict[str, Any] | None = None) -> dict[str, Any]:
    if image_result.get("status") in {"queued", "running"}:
        return {
            "passed": False,
            "issues": [image_result.get("message") or "图像任务仍在后台生成，暂未执行视觉评审"],
            "retry_recommended": False,
            "rule": {"passed": False, "reason": "queued"},
            "vlm": {"enabled": False, "reason": "图像尚未生成，跳过 VLM 评审"},
        }
    if image_result.get("status") != "success":
        return {
            "passed": False,
            "issues": [image_result.get("message") or "图像未生成"],
            "retry_recommended": image_result.get("error_type") not in {"comfyui_offline", "workflow_error"},
            "rule": {"passed": False},
            "vlm": {"enabled": False, "reason": "图像未生成，跳过 VLM 评审"},
        }
    path = Path(str(image_result.get("path") or ""))
    if not path.exists() or path.stat().st_size == 0:
        return {
            "passed": False,
            "issues": ["图像文件不存在或为空"],
            "retry_recommended": True,
            "rule": {"passed": False},
            "vlm": {"enabled": False, "reason": "图像文件不可用，跳过 VLM 评审"},
        }
    missing = []
    prompt = str(spec.get("positive_prompt") or "").lower()
    for token in ["chinese", "landscape", "ink"]:
        if token not in prompt:
            missing.append(f"prompt 缺少 {token}")
    rule_result = {"passed": not missing, "issues": missing}
    vlm_result: dict[str, Any]
    if not VLM_CRITIC_ENABLED:
        vlm_result = {"enabled": False, "reason": "CL_VLM_CRITIC_ENABLED 已关闭"}
    else:
        try:
            vlm_result = run_vlm_image_critic(path, spec, brief or {})
        except Exception as exc:
            vlm_result = {
                "enabled": True,
                "model": str(VLM_CRITIC_MODEL_PATH),
                "passed": False,
                "score": 0.0,
                "issues": [f"VLM 评审失败：{exc}"],
                "retry_recommended": False,
            }
    issues = list(missing)
    if vlm_result.get("enabled"):
        issues.extend(str(item) for item in vlm_result.get("issues") or [] if item)
    passed = bool(rule_result["passed"])
    if vlm_result.get("enabled"):
        passed = passed and bool(vlm_result.get("passed"))
    retry_recommended = bool(vlm_result.get("retry_recommended")) if vlm_result.get("enabled") else False
    return {
        "passed": passed,
        "score": vlm_result.get("score") if vlm_result.get("enabled") else None,
        "issues": issues,
        "retry_recommended": retry_recommended,
        "rule": rule_result,
        "vlm": vlm_result,
    }


def source_lines(evidence: list[dict[str, Any]], ranks: list[int] | None = None) -> list[str]:
    selected = evidence
    if ranks:
        rank_set = {int(rank) for rank in ranks}
        selected = [item for item in evidence if int(item.get("rank") or 0) in rank_set]
    lines = []
    for item in selected[:5]:
        title = item.get("title") or item.get("source_file") or "未知来源"
        page = item.get("page_start") or "未知"
        lines.append(f"[{item.get('rank')}] 《{title}》，第 {page} 页")
    return lines


def build_image_final_answer(
    question: str,
    brief: dict[str, Any],
    spec: dict[str, Any],
    image_result: dict[str, Any],
    critic: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    citations = [int(item) for item in brief.get("citations") or [] if str(item).isdigit()]
    citation_text = " ".join(f"[{rank}]" for rank in citations[:4])
    if image_result.get("status") == "success":
        intro = f"已完成图像生成。创作方案基于检索到的山水画文献约束 {citation_text}。"
        image_line = f"图像文件：{image_result.get('filename')}，seed：{image_result.get('seed')}。"
    elif image_result.get("status") in {"queued", "running"}:
        intro = f"已完成研究、图像方案和 Prompt 设计，并已提交后台生图任务 {citation_text}。"
        image_line = (
            f"生图任务：{image_result.get('job_id')}，seed：{image_result.get('seed')}。"
            "图像完成后可通过系统轮询结果查看。"
        )
    else:
        intro = "已完成研究、图像方案和 Prompt 设计，但当前 ComfyUI 生图引擎不可用，所以没有实际生成图片。"
        image_line = f"生图状态：{image_result.get('message') or '未生成'}。"
    key_points = "\n".join(f"- {item}" for item in brief.get("key_points", [])[:4])
    constraints = "\n".join(f"- {item}" for item in brief.get("visual_constraints", [])[:6])
    sources = "\n".join(source_lines(evidence, citations))
    critic_line = "图像检查：通过。" if critic.get("passed") else f"图像检查：{'; '.join(critic.get('issues') or [])}。"
    vlm = critic.get("vlm") or {}
    vlm_line = ""
    if vlm.get("enabled"):
        score = vlm.get("score")
        score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "未知"
        vlm_line = (
            f"\nVLM 视觉评审：{score_text} 分。"
            f"{vlm.get('style_alignment') or ''}"
            f"{(' ' + vlm.get('evidence_alignment')) if vlm.get('evidence_alignment') else ''}"
        )
    return (
        f"{intro}\n\n"
        f"研究依据要点：\n{key_points}\n\n"
        f"创作约束：\n{constraints}\n\n"
        f"{image_line}\n{critic_line}{vlm_line}\n\n"
        f"ComfyUI positive prompt：\n{spec.get('positive_prompt')}\n\n"
        f"negative prompt：\n{spec.get('negative_prompt')}\n\n"
        f"来源：\n{sources}\n\n"
        f"原任务：{question}"
    )


def extract_memory_insights(question: str) -> dict[str, Any]:
    preferences = []
    feedback = []
    for pattern in [r"我喜欢([^。！？\n]+)", r"偏好([^。！？\n]+)"]:
        preferences.extend(match.strip() for match in re.findall(pattern, question))
    for pattern in [r"以后不要([^。！？\n]+)", r"不喜欢([^。！？\n]+)"]:
        feedback.extend(match.strip() for match in re.findall(pattern, question))
    return {
        "preferences": [item for item in preferences if item],
        "feedback": [item for item in feedback if item],
        "context": "",
    }


def maybe_write_memory(user_id: str, question: str) -> dict[str, Any]:
    insights = extract_memory_insights(question)
    if user_id and user_id != "guest" and (insights["preferences"] or insights["feedback"] or insights["context"]):
        try:
            from src.agent.memory.memory_manager import MemoryManager

            MemoryManager.save_memory(user_id, insights)
            return {"saved": True, "insights": insights}
        except Exception as exc:
            return {"saved": False, "error": str(exc), "insights": insights}
    return {"saved": False, "reason": "guest_or_empty", "insights": insights}


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
    if not LLM_API_KEY:
        return fallback_answer(question, evidence)

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=80.0),
    )
    response = client.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=build_chat_messages(question, evidence, history),
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def stream_answer_deltas(question: str, evidence: list[dict[str, Any]], history: list[dict[str, str]]):
    if not LLM_API_KEY:
        answer = fallback_answer(question, evidence)
        yield from text_chunks(answer, 24)
        return

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
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
            yield delta


def stream_chat_answer(question: str, evidence: list[dict[str, Any]], history: list[dict[str, str]]):
    yield json.dumps({"type": "evidence", "evidence": evidence}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "phase", "phase": "生成中"}, ensure_ascii=False) + "\n"

    for delta in stream_answer_deltas(question, evidence, history):
        yield json.dumps({"type": "delta", "delta": delta}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"


def premise_answer(question: str, issue: dict[str, Any]) -> str:
    return (
        f"这个问题的前提需要先纠正：{issue.get('message')}\n\n"
        f"处理建议：{issue.get('recommended_action')}\n\n"
        "如果你的目标是现代创作，可以改问：“请参考某一画派或画家的风格，生成一幅山水画”。"
        f"\n\n原问题：{question}"
    )


def direct_art_answer(question: str, history: list[dict[str, str]]) -> str:
    normalized = question.strip()
    if "清明上河图" in normalized and re.search(r"(谁|作者|画的|创作)", normalized):
        return "《清明上河图》一般认为是北宋画家张择端创作的。"
    if not LLM_API_KEY:
        return "这是中国绘画史的一般问题，当前未配置在线模型，无法可靠展开回答。"
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        http_client=httpx.Client(proxy=None, trust_env=False, timeout=40.0),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是中国绘画史问答助手。回答一般美术史事实问题，不要伪造文献来源，"
                "不要使用 [1] 这类引用编号。若问题需要本项目文献证据但当前未检索，应说明只是一般知识回答。"
            ),
        },
        {"role": "user", "content": f"最近对话：\n{compact_history(history)}\n\n当前问题：{question}"},
    ]
    try:
        response = client.chat.completions.create(model=FAST_LLM_MODEL, messages=messages, temperature=0.1, max_tokens=500)
        return response.choices[0].message.content or ""
    except Exception:
        return "这个问题属于一般中国绘画史问题，但当前在线模型调用失败，暂时无法展开。"


def unsupported_image_answer(question: str) -> str:
    return (
        "当前图像 Agent 只处理中国山水画相关创作任务。"
        "请补充山水画的朝代、画家、流派、技法、作品或画幅要求，例如："
        "“请根据明代吴门画派风格生成一幅江南山水长卷”。\n\n"
        f"原问题：{question}"
    )


def stream_final_text(answer: str):
    for chunk in text_chunks(answer, 28):
        yield event_line({"type": "delta", "delta": chunk})


def stream_agent_answer(req: ChatRequest):
    try:
        question = req.message.strip()
        yield event_line({"type": "phase", "phase": "理解任务"})
        yield event_line(node_event("intake", "running", "分析任务类型、领域、图像意图和错误前提"))
        intake = build_agent_intake(question, req.history)
        detail = f"{intake['task_type']} / {intake['route']['reason']}"
        yield event_line(node_event("intake", "done", detail, intake))

        yield event_line(node_event("planner", "running", "根据任务类型选择需要执行的节点"))
        plan = build_agent_plan(intake)
        yield event_line({"type": "plan", "steps": plan})
        yield event_line(node_event("planner", "done", f"生成 {len(plan)} 步计划", plan))

        evidence: list[dict[str, Any]] = []
        task_type = intake["task_type"]
        if task_type == "direct":
            yield event_line(node_event("final_writer", "running", "直接回复，不启动文献检索"))
            answer = non_research_answer(question, intake["route"])
            yield from stream_final_text(answer)
            yield event_line(node_event("final_writer", "done", "已完成直接回复"))
            yield event_line({"type": "done", "mode": "direct_agent"})
            return

        if task_type == "general_art_qa":
            yield event_line(node_event("final_writer", "running", "回答一般中国绘画史问题，不启动文献检索"))
            answer = direct_art_answer(question, req.history)
            yield from stream_final_text(answer)
            yield event_line(node_event("final_writer", "done", "已完成一般美术史回答"))
            yield event_line({"type": "done", "mode": "general_art_qa"})
            return

        if task_type == "unsupported_image":
            yield event_line(node_event("verifier", "done", "图像请求不在中国山水画创作范围内"))
            yield event_line(node_event("final_writer", "running", "给出边界说明"))
            yield from stream_final_text(unsupported_image_answer(question))
            yield event_line(node_event("final_writer", "done", "已完成边界回复"))
            yield event_line({"type": "done", "mode": "unsupported_image"})
            return

        if task_type in {"need_clarification", "unsupported_general"}:
            yield event_line(node_event("verifier", "done", "问题需要澄清或不属于研究范围"))
            yield event_line(node_event("final_writer", "running", "给出澄清建议"))
            yield from stream_final_text(non_research_answer(question, intake["route"]))
            yield event_line(node_event("final_writer", "done", "已完成澄清回复"))
            yield event_line({"type": "done", "mode": task_type})
            return

        if task_type == "invalid_premise":
            issue = intake.get("premise_issue") or {}
            yield event_line(node_event("verifier", "done", issue.get("message", "前提不成立"), issue))
            yield event_line(node_event("final_writer", "running", "纠正错误前提"))
            yield from stream_final_text(premise_answer(question, issue))
            yield event_line(node_event("final_writer", "done", "已完成前提纠偏"))
            yield event_line({"type": "done", "mode": "invalid_premise"})
            return

        if intake.get("needs_retrieval"):
            yield event_line({"type": "phase", "phase": "检索中"})
            yield event_line(node_event("researcher", "running", "调用 Milvus/evidence store 检索并重排"))
            retriever = get_retriever(req.top_k, req.final_k)
            results = retriever.retrieve_and_rerank(question)
            evidence = [evidence_payload(doc, index + 1) for index, doc in enumerate(results)]
            yield event_line({"type": "evidence", "evidence": evidence})
            yield event_line(node_event("researcher", "done", f"返回 {len(evidence)} 条证据"))

        yield event_line({"type": "phase", "phase": "核验证据"})
        yield event_line(node_event("verifier", "running", "检查相关性、错误前提和是否需要拒答"))
        verifier = verify_agent_state(
            question=question,
            intake=intake,
            evidence=evidence,
            top_evidence_relevant=evidence_is_relevant(evidence),
        )
        yield event_line(node_event("verifier", "done", verifier["reason"], verifier))

        if not verifier["can_continue"]:
            yield event_line(node_event("final_writer", "running", "说明证据不足或前提风险"))
            if verifier["verdict"] == "low_relevance":
                answer = low_relevance_answer(question, evidence)
            elif verifier["verdict"] == "insufficient_evidence_for_direct_influence":
                answer = (
                    f"当前资料库证据不足以支持这个直接影响关系：{verifier['reason']}\n\n"
                    "我不会默认该前提成立。建议改问具体的比较问题，例如比较构图、色彩或笔触，而不是直接师承关系。"
                )
            else:
                answer = (
                    f"当前资料库证据不足以支撑可靠回答：{verifier['reason']}\n\n"
                    "建议补充更明确的画家、作品、时代、流派或技法，或先扩大语料后再检索。"
                )
            yield from stream_final_text(answer)
            yield event_line(node_event("final_writer", "done", "已完成证据不足回复"))
            yield event_line({"type": "done", "mode": verifier["verdict"]})
            return

        if task_type == "research_qa":
            yield event_line({"type": "phase", "phase": "生成回答"})
            yield event_line(node_event("final_writer", "running", "基于核验后的证据生成研究回答"))
            for delta in stream_answer_deltas(question, evidence, req.history):
                yield event_line({"type": "delta", "delta": delta})
            yield event_line(node_event("final_writer", "done", "已完成研究回答"))
            yield event_line({"type": "done", "mode": "research_qa"})
            return

        yield event_line({"type": "phase", "phase": "整理卷宗"})
        yield event_line(node_event("research_synthesizer", "running", "把证据整理成给画师使用的研究约束"))
        brief = synthesize_research_brief(question, evidence, intake)
        yield event_line({"type": "brief", "brief": brief})
        yield event_line(node_event("research_synthesizer", "done", f"提炼 {len(brief.get('visual_constraints', []))} 条视觉约束", brief))

        yield event_line({"type": "phase", "phase": "设计 Prompt"})
        yield event_line(node_event("prompt_designer", "running", "生成英文 positive prompt、negative prompt 和尺寸"))
        image_spec = design_image_spec(question, brief)
        yield event_line({"type": "image_spec", "spec": image_spec})
        yield event_line(node_event("prompt_designer", "done", f"{image_spec.get('width')}x{image_spec.get('height')} / {image_spec.get('format')}", image_spec))

        yield event_line({"type": "phase", "phase": "生成图像"})
        yield event_line(node_event("image_generator", "running", "调用 ComfyUI 工作流"))
        image_result = generate_image_with_comfyui(image_spec)
        yield event_line({"type": "image", "image": image_result})
        image_detail = "图像生成成功" if image_result.get("status") == "success" else str(image_result.get("message") or "图像未生成")
        yield event_line(node_event("image_generator", "done", image_detail, image_result))

        yield event_line({"type": "phase", "phase": "检查图像"})
        yield event_line(node_event("image_critic", "running", "检查图像文件、生成状态和 prompt 约束"))
        critic = critic_image_result(image_result, image_spec, brief)
        critic_detail = "通过" if critic.get("passed") else "；".join(critic.get("issues") or ["未通过"])
        yield event_line({"type": "image_critic", "critic": critic})
        yield event_line(node_event("image_critic", "done", critic_detail, critic))

        yield event_line({"type": "phase", "phase": "组织结果"})
        yield event_line(node_event("final_writer", "running", "交付图像、研究依据、Prompt 和来源"))
        final_answer = build_image_final_answer(question, brief, image_spec, image_result, critic, evidence)
        yield from stream_final_text(final_answer)
        yield event_line(node_event("final_writer", "done", "已完成研究创作交付"))

        yield event_line(node_event("memory_writer", "running", "仅记录明确表达的用户偏好"))
        memory_result = maybe_write_memory(req.user_id, question)
        detail = "已写入偏好" if memory_result.get("saved") else "无可写入偏好或 guest 用户"
        yield event_line({"type": "memory", "memory": memory_result})
        yield event_line(node_event("memory_writer", "done", detail, memory_result))
        yield event_line({"type": "done", "mode": "research_then_image"})
    except Exception as exc:
        yield event_line({"type": "error", "message": str(exc)})
        yield event_line({"type": "done", "mode": "error"})


from src.web_agent.dependencies import WebAgentDependencies  # noqa: E402
from src.web_agent.streaming import stream_web_agent_events  # noqa: E402
from src.web_agent.verifier import verify_agent_state  # noqa: E402


WEB_AGENT_ENGINE = os.getenv("CL_WEB_AGENT_ENGINE", "langgraph").lower()
WEB_AGENT_DEPS = WebAgentDependencies(
    build_agent_intake=build_agent_intake,
    build_agent_plan=build_agent_plan,
    get_retriever=get_retriever,
    evidence_payload=evidence_payload,
    evidence_is_relevant=evidence_is_relevant,
    non_research_answer=non_research_answer,
    direct_art_answer=direct_art_answer,
    unsupported_image_answer=unsupported_image_answer,
    premise_answer=premise_answer,
    low_relevance_answer=low_relevance_answer,
    stream_answer_deltas=stream_answer_deltas,
    synthesize_research_brief=synthesize_research_brief,
    design_image_spec=design_image_spec,
    generate_image_with_comfyui=generate_image_with_comfyui,
    critic_image_result=critic_image_result,
    build_image_final_answer=build_image_final_answer,
    maybe_write_memory=maybe_write_memory,
)


def path_component(path: Path, *, required: bool = True) -> dict[str, Any]:
    exists = path.exists()
    if path.is_dir():
        kind = "directory"
    elif path.is_file():
        kind = "file"
    else:
        kind = "missing"
    return {
        "status": "ok" if exists else ("missing" if required else "optional_missing"),
        "path": str(path),
        "exists": exists,
        "kind": kind,
    }


def http_component(name: str, url: str, *, live: bool, headers: dict[str, str] | None = None) -> dict[str, Any]:
    component = {"status": "configured", "name": name, "url": url, "live_checked": False}
    if not live:
        return component
    component["live_checked"] = True
    try:
        with httpx.Client(proxy=None, trust_env=False, timeout=2.5, headers=headers) as client:
            response = client.get(url)
        component["status"] = "ok" if response.status_code < 500 else "error"
        component["http_status"] = response.status_code
    except Exception as exc:
        component["status"] = "error"
        component["error"] = str(exc)
    return component


def build_component_health(live: bool = False) -> dict[str, Any]:
    evidence_files = {
        "manifest": path_component(RETRIEVAL_EVIDENCE_DIR / "manifest.json"),
        "documents": path_component(RETRIEVAL_EVIDENCE_DIR / "documents.jsonl"),
        "chunks": path_component(RETRIEVAL_EVIDENCE_DIR / "chunks.jsonl"),
    }
    components = {
        "llm": {
            "status": "configured" if LLM_API_KEY else "fallback",
            "model": FAST_LLM_MODEL if LLM_API_KEY else "evidence-summary-fallback",
            "base_url": LLM_BASE_URL,
            "live": http_component(
                "llm",
                f"{LLM_BASE_URL}/models",
                live=live and bool(LLM_API_KEY),
                headers={"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else None,
            ),
        },
        "evidence_store": {
            "status": "ok" if all(item["exists"] for item in evidence_files.values()) else "missing",
            "directory": str(RETRIEVAL_EVIDENCE_DIR),
            "files": evidence_files,
        },
        "retrieval_index": {
            "milvus_lite": path_component(RETRIEVAL_MILVUS_DB_PATH),
            "colbert_tensors": path_component(RETRIEVAL_COLBERT_TENSORS_PATH, required=False),
            "encoder": path_component(BGE_M3_PATH),
            "reranker": path_component(RERANKER_PATH),
        },
        "comfyui": {
            "async_enabled": COMFYUI_ASYNC_ENABLED,
            "workers": max(1, COMFYUI_JOB_WORKERS),
            "workflow": path_component(COMFYUI_WORKFLOW_PATH),
            "service": http_component("comfyui", f"{COMFYUI_SERVER_URL.rstrip('/')}/system_stats", live=live),
        },
        "vlm_critic": {
            "enabled": VLM_CRITIC_ENABLED,
            "model": path_component(VLM_CRITIC_MODEL_PATH, required=VLM_CRITIC_ENABLED),
            "pass_score": VLM_CRITIC_PASS_SCORE,
        },
        "neo4j": {
            "status": "configured" if NEO4J_PASSWORD else "disabled_or_no_password",
            "uri": NEO4J_URI,
        },
        "image_jobs": {
            "queued_or_recent": len(_image_jobs),
            "workers": max(1, COMFYUI_JOB_WORKERS),
        },
    }
    return components


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health(live: bool = Query(default=False)) -> dict[str, Any]:
    manifest = load_manifest()
    components = build_component_health(live=live)
    return {
        "ok": True,
        "llm_configured": bool(LLM_API_KEY),
        "answer_model": FAST_LLM_MODEL if LLM_API_KEY else "evidence-summary-fallback",
        "answer_provider": "chat-completions endpoint" if LLM_API_KEY else "local fallback",
        "trained_researcher_lora": str(RESEARCHER_LORA_PATH),
        "trained_researcher_lora_exists": RESEARCHER_LORA_PATH.exists(),
        "retriever_models": {
            "encoder": str(BGE_M3_PATH),
            "reranker": str(RERANKER_PATH),
        },
        "evidence_dir": str(RETRIEVAL_EVIDENCE_DIR),
        "router": {
            "llm_enabled": ROUTER_LLM_ENABLED and bool(LLM_API_KEY),
            "router_model": ROUTER_LLM_MODEL if ROUTER_LLM_ENABLED and LLM_API_KEY else "rule-only",
            "min_rerank_score": RAG_MIN_RERANK_SCORE,
        },
        "agent": {
            "mode": "langgraph_web_agent" if WEB_AGENT_ENGINE == "langgraph" else "controlled_research_creation_agent",
            "engine": WEB_AGENT_ENGINE,
            "checkpointing": WEB_AGENT_ENGINE == "langgraph",
            "legacy_controlled_agent_available": True,
            "nodes": list(AGENT_NODE_TITLES.keys()),
            "image_generation": {
                "provider": "ComfyUI",
                "server": COMFYUI_SERVER_URL,
                "workflow": str(COMFYUI_WORKFLOW_PATH),
                "async_enabled": COMFYUI_ASYNC_ENABLED,
                "job_status_url": "/api/image-jobs/{job_id}",
                "generated_images_url": "/generated-images",
            },
            "image_critic": {
                "rule_check": True,
                "vlm_enabled": VLM_CRITIC_ENABLED,
                "vlm_model": str(VLM_CRITIC_MODEL_PATH),
                "vlm_model_exists": VLM_CRITIC_MODEL_PATH.exists(),
                "pass_score": VLM_CRITIC_PASS_SCORE,
            },
        },
        "components": components,
        "manifest": manifest,
    }


@app.get("/api/image-jobs/{job_id}")
def image_job(job_id: str) -> dict[str, Any]:
    job = _get_image_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到图像任务")
    return job


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
                        "author": doc.get("author"),
                        "authority_level": doc.get("authority_level"),
                        "category": doc.get("category"),
                        "source_type": doc.get("source_type"),
                        "page_count": doc.get("page_count"),
                        "pdf_url": pdf_query_url("/api/pdf", doc.get("source_file")),
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


@app.post("/api/agent/stream")
def agent_stream(req: ChatRequest) -> StreamingResponse:
    if WEB_AGENT_ENGINE in {"legacy", "controlled", "state_machine"}:
        iterator = stream_agent_answer(req)
    else:
        iterator = stream_web_agent_events(
            message=req.message,
            history=req.history,
            top_k=req.top_k,
            final_k=req.final_k,
            user_id=req.user_id,
            thread_id=req.thread_id,
            deps=WEB_AGENT_DEPS,
        )
    return StreamingResponse(iterator, media_type="application/x-ndjson")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
