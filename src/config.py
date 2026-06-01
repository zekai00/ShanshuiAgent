"""Central runtime configuration for ChineseLandscape.

The project is often run from scripts, API servers, and notebooks. Keeping
paths and service endpoints here avoids embedding workstation-specific absolute
paths in business code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _project_root() -> Path:
    raw = os.getenv("CL_PROJECT_ROOT") or os.getenv("CHINESE_LANDSCAPE_ROOT")
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()


def _path_env(name: str, default: str | Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else Path(default)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


DATA_DIR = _path_env("CL_DATA_DIR", PROJECT_ROOT / "data")
RAW_PDFS_DIR = _path_env("CL_RAW_PDFS_DIR", DATA_DIR / "raw_pdfs")
EXTRACTED_ARTWORKS_DIR = _path_env("CL_EXTRACTED_ARTWORKS_DIR", DATA_DIR / "extracted_artworks")
VECTOR_STORE_DIR = _path_env("CL_VECTOR_STORE_DIR", DATA_DIR / "vector_store")
PROCESSED_DIR = _path_env("CL_PROCESSED_DIR", DATA_DIR / "processed")
PROCESSED_DOCUMENTS_DIR = _path_env("CL_PROCESSED_DOCUMENTS_DIR", PROCESSED_DIR / "documents")
AUTHORITY_EVIDENCE_DIR = _path_env("CL_AUTHORITY_EVIDENCE_DIR", PROCESSED_DIR / "authority_evidence")
METADATA_DIR = _path_env("CL_METADATA_DIR", DATA_DIR / "metadata")
RUNTIME_DIR = _path_env("CL_RUNTIME_DIR", DATA_DIR / "runtime")
DOCS_DIR = _path_env("CL_DOCS_DIR", PROJECT_ROOT / "docs")

GENERATED_IMAGES_DIR = _path_env("CL_GENERATED_IMAGES_DIR", PROJECT_ROOT / "generated_images")
WORKFLOWS_DIR = _path_env("CL_WORKFLOWS_DIR", PROJECT_ROOT / "workflows")
AGENT_PROMPTS_DIR = _path_env("CL_AGENT_PROMPTS_DIR", PROJECT_ROOT / "src" / "agent" / "prompts")
UI_STATIC_DIR = _path_env("CL_UI_STATIC_DIR", PROJECT_ROOT / "ui" / "static")

MILVUS_DB_PATH = _path_env("CL_MILVUS_DB_PATH", VECTOR_STORE_DIR / "milvus_landscape.db")
COLBERT_TENSORS_PATH = _path_env("CL_COLBERT_TENSORS_PATH", VECTOR_STORE_DIR / "colbert_tensors.pkl")
AUTHORITY_MILVUS_DB_PATH = _path_env("CL_AUTHORITY_MILVUS_DB_PATH", VECTOR_STORE_DIR / "milvus_authority.db")
AUTHORITY_COLBERT_TENSORS_PATH = _path_env("CL_AUTHORITY_COLBERT_TENSORS_PATH", VECTOR_STORE_DIR / "colbert_authority_tensors.pkl")
RETRIEVAL_MILVUS_DB_PATH = _path_env("CL_RETRIEVAL_MILVUS_DB_PATH", AUTHORITY_MILVUS_DB_PATH)
RETRIEVAL_COLBERT_TENSORS_PATH = _path_env("CL_RETRIEVAL_COLBERT_TENSORS_PATH", AUTHORITY_COLBERT_TENSORS_PATH)
RETRIEVAL_EVIDENCE_DIR = _path_env("CL_RETRIEVAL_EVIDENCE_DIR", AUTHORITY_EVIDENCE_DIR)
INGESTION_STATE_PATH = _path_env("CL_INGESTION_STATE_PATH", VECTOR_STORE_DIR / "ingested_pdfs_state.json")
AGENT_CHECKPOINT_DB = _path_env("CL_AGENT_CHECKPOINT_DB", RUNTIME_DIR / "checkpoints.sqlite")
WEB_AGENT_CHECKPOINT_DB = _path_env("CL_WEB_AGENT_CHECKPOINT_DB", RUNTIME_DIR / "web_agent_checkpoints.sqlite")
USER_MEMORY_DB = _path_env("CL_USER_MEMORY_DB", RUNTIME_DIR / "user_memories.db")

MODELS_DIR = _path_env("CL_MODELS_DIR", "/root/models")
BGE_M3_PATH = _path_env("CL_BGE_M3_PATH", MODELS_DIR / "bge-m3")
RERANKER_PATH = _path_env("CL_RERANKER_PATH", MODELS_DIR / "bge-reranker-v2-m3")
RESEARCHER_BASE_MODEL_PATH = _path_env("CL_RESEARCHER_BASE_MODEL_PATH", MODELS_DIR / "Qwen3.5-9B")
MODEL_DEVICE = os.getenv("CL_MODEL_DEVICE", os.getenv("CUDA_DEVICE", "cuda:0"))
SPACY_MODEL_NAME = os.getenv("CL_SPACY_MODEL_NAME", "zh_core_web_sm")

DATASETS_ROOT = _path_env("CL_DATASETS_ROOT", "/root/datasets")
AUTHORITY_CORPUS_DIR = _path_env("CL_AUTHORITY_CORPUS_DIR", DATASETS_ROOT / "chinese_landscape_authority_corpus")
LLAMA_FACTORY_DIR = _path_env("CL_LLAMA_FACTORY_DIR", "/root/Workspace/LLaMA-Factory")
RESEARCHER_LORA_PATH = _path_env(
    "CL_RESEARCHER_LORA_PATH",
    LLAMA_FACTORY_DIR / "saves" / "Qwen3.5-9B-Base" / "lora" / "train_landscape_sft_v2",
)

RETRIEVAL_COLLECTION_NAME = os.getenv("CL_RETRIEVAL_COLLECTION", "landscape_rag")
RETRIEVAL_TOP_K = _int_env("CL_RETRIEVAL_TOP_K", 15)
RETRIEVAL_FINAL_K = _int_env("CL_RETRIEVAL_FINAL_K", 3)
RETRIEVAL_HOST = os.getenv("CL_RETRIEVAL_HOST", "127.0.0.1")
RETRIEVAL_PORT = _int_env("CL_RETRIEVAL_PORT", 8000)
RETRIEVAL_SERVICE_URL = os.getenv(
    "CL_RETRIEVAL_SERVICE_URL",
    f"http://{RETRIEVAL_HOST}:{RETRIEVAL_PORT}",
).rstrip("/")

COMFYUI_SERVER_URL = os.getenv("CL_COMFYUI_SERVER_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_WORKFLOW_PATH = _path_env("CL_COMFYUI_WORKFLOW_PATH", WORKFLOWS_DIR / "flux1_krea_dev-api.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
FAST_LLM_MODEL = os.getenv("CL_FAST_LLM_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))

NEO4J_URI = os.getenv("NEO4J_URI", os.getenv("CL_NEO4J_URI", "bolt://localhost:7687"))
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", os.getenv("CL_NEO4J_USERNAME", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("CL_NEO4J_PASSWORD")


def ensure_runtime_dirs() -> None:
    """Create local runtime directories that are safe to generate."""
    for path in [
        DATA_DIR,
        VECTOR_STORE_DIR,
        PROCESSED_DIR,
        PROCESSED_DOCUMENTS_DIR,
        AUTHORITY_EVIDENCE_DIR,
        RUNTIME_DIR,
        GENERATED_IMAGES_DIR,
        UI_STATIC_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
