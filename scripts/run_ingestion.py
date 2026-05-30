# /root/Workspace/ChineseLandscape/scripts/run_ingestion.py

import os
import sys
import sys
import warnings

# 🌟 全局屏蔽 pkg_resources 弃用警告，保持终端纯净
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import glob
import uuid

# 🌟 1. 强制将项目根目录加入寻址路径，解决跨目录 import 报错
WORKSPACE_DIR = "/root/Workspace/ChineseLandscape"
sys.path.append(WORKSPACE_DIR)

# 🌟 2. LangSmith 全局观测配置 (必须在引入其他包之前)
# 不在代码中写入 API Key；如需启用追踪，请在环境变量中配置 LANGCHAIN_API_KEY。
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "ChineseLandscape_Offline_Ingestion")

# 🌟 3. 从内部工厂提取组装零件
from src.ingestion.ingestion_manager import IngestionStateManager, LandscapeDatabaseManager
from src.ingestion.document_processor import DocumentProcessor
from Workspace.ChineseLandscape.src.retrieval.bge_m3_engine import BGEM3Engine

# 🌟 4. 全局物理路径重映射 (适配新目录架构)
PAPERS_FOLDER = os.path.join(WORKSPACE_DIR, "data", "raw_pdfs")
IMAGES_FOLDER = os.path.join(WORKSPACE_DIR, "data", "extracted_artworks") 
DB_FOLDER = os.path.join(WORKSPACE_DIR, "data", "vector_store")
TRACKING_FILE = os.path.join(DB_FOLDER, "ingested_pdfs_state.json")

MILVUS_DB_PATH = os.path.join(DB_FOLDER, "milvus_landscape.db")
COLBERT_FILE = os.path.join(DB_FOLDER, "colbert_tensors.pkl")

def interactive_startup_routing():
    """交互式防呆与状态校验引擎"""
    print("\n" + "="*70)
    print("🚀 数字敦煌 V3.0 离线工厂 (模块解耦版)")
    print("="*70)

    state_manager = IngestionStateManager(TRACKING_FILE)
    db_manager = LandscapeDatabaseManager(MILVUS_DB_PATH, COLBERT_FILE)
    
    print("\n[*] 正在扫描文件库与进行 MD5 指纹校验...")
    pdf_files = sorted(glob.glob(os.path.join(PAPERS_FOLDER, "*.pdf")))
    if not pdf_files:
        print("[!] ❌ 错误：在 data/raw_pdfs 中未找到任何 PDF 文件。程序退出。")
        sys.exit(0)

    current_files_md5 = {}
    for pdf in pdf_files:
        filename = os.path.basename(pdf)
        md5 = state_manager.calculate_md5(pdf)
        if md5: current_files_md5[filename] = md5
        
    pending_files = []
    for filename, current_md5 in current_files_md5.items():
        if state_manager.state.get(filename) != current_md5:
            pending_files.append(filename)

    db_exists = db_manager.db_exists()
    
    if not db_exists:
        print("[*] 侦测到本地无数据库，将执行【全量构建】。")
        pending_files = list(current_files_md5.keys())
        db_manager.init_database(force_rebuild=False)
    else:
        print(f"[*] 侦测到已有 Milvus 数据库。")
        print(f"    -> 历史入库文件数: {len(state_manager.state)}")
        print(f"    -> 发现新增或被修改的文件数: {len(pending_files)}")
        
        if len(pending_files) == 0:
            print("\n✅ 所有文件均已成功入库且未被修改！")
            choice = input("\n请选择: [1] 退出程序  [2] 强制清空旧库，从头全量重构: ")
            if choice.strip() == '2':
                state_manager.clear_state()
                db_manager.init_database(force_rebuild=True)
                pending_files = list(current_files_md5.keys())
            else:
                sys.exit(0)
        else:
            print("\n发现未处理的增量文件！")
            choice = input("请选择: [1] 仅追加增量文件 (断点续传)  [2] 彻底销毁旧库，全量重构: ")
            if choice.strip() == '2':
                state_manager.clear_state()
                db_manager.init_database(force_rebuild=True)
                pending_files = list(current_files_md5.keys())
            else:
                db_manager.init_database(force_rebuild=False)
                print("    -> 开启【增量追加】模式。")

    return pending_files, current_files_md5, db_manager, state_manager

# ---------------------------------------------------------
# 🏁 主业务流水线调度
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        pending_files, current_files_md5, db_manager, state_manager = interactive_startup_routing()
        if not pending_files: sys.exit(0)
            
        print("\n[*] 正在启动大模型引擎与 BGE-M3...")
        doc_processor = DocumentProcessor(images_save_dir=IMAGES_FOLDER)
        bge_m3_engine = BGEM3Engine(model_path="/root/models/bge-m3", device="cuda:0")
        
        for idx, filename in enumerate(pending_files):
            pdf_path = os.path.join(PAPERS_FOLDER, filename)
            print(f"\n▶️ [{idx+1}/{len(pending_files)}] 开始处理: {filename}", flush=True)
            
            clean_full_text = doc_processor.extract_and_crop(pdf_path, filename)
            if not clean_full_text: continue
            
            global_context = doc_processor.metadata_extractor.generate_global_context(clean_full_text)
            coarse_chunks = doc_processor.coarse_splitter.split_text(clean_full_text)
            
            milvus_insert_data = []
            new_colbert_tensors = {}
            
            for i, coarse_chunk in enumerate(coarse_chunks):
                fine_chunks = doc_processor.agentic_split(coarse_chunk)
                
                for fine_chunk in fine_chunks:
                    super_chunk = doc_processor.metadata_extractor.extract_super_chunk(fine_chunk, global_context)
                    metadata = super_chunk.get('metadata', {})
                    
                    if str(metadata.get('is_domain_relevant', True)).lower() == 'false':
                        print("      [🗑️ 拦截] 侦测到跨域噪音文本，已物理抛弃！")
                        continue
                    
                    contextual_chunk = f"【全局上下文】{super_chunk['contextual_enrichment']['context_anchor']}\n【原文资料】{fine_chunk}"
                    search_payload = f"{contextual_chunk}\n【潜在问题】{', '.join(super_chunk['contextual_enrichment']['multi_queries'])}\n【核心解答】{super_chunk['contextual_enrichment']['hyde_answer']}"
                    
                    chunk_id = abs(hash(str(uuid.uuid4()))) % (10 ** 15)
                    features = bge_m3_engine.encode_corpus([search_payload])
                    
                    dynasty_raw = metadata.get('dynasty', ["未知"])
                    sparse_weight = features['lexical_weights'][0]
                    
                    row_data = {
                        "id": chunk_id,
                        "dense_vector": features['dense_vecs'][0].tolist(),    
                        "sparse_vector": {int(k): float(v) for k, v in sparse_weight.items()} if isinstance(sparse_weight, dict) else sparse_weight, 
                        "contextual_chunk": contextual_chunk, 
                        "source_file": filename,      
                        "dynasty": dynasty_raw if isinstance(dynasty_raw, list) else [str(dynasty_raw)], 
                        "painter": str(metadata.get('painter_or_school', '未知')),
                        "subject_matter": str(metadata.get('subject_matter', '未知')),
                        "content_scope": str(metadata.get('content_scope', '未知'))
                    }
                    milvus_insert_data.append(row_data)
                    new_colbert_tensors[chunk_id] = features['colbert_vecs'][0]

                print(f"    -> 块 {i+1}/{len(coarse_chunks)}: {len(fine_chunks)} 个切片已就绪！", flush=True)

            if milvus_insert_data:
                db_manager.insert_batch(milvus_insert_data, new_colbert_tensors)
                state_manager.mark_as_completed(filename, current_files_md5[filename])
                print(f"  ✅ {filename} 数据已注入数据库并记录状态！", flush=True)

        print("\n🎉 离线工厂竣工！所有待办文档已处理完毕。")
        
    except KeyboardInterrupt:
        print("\n[!] 🛑 侦测到手动中断。下次启动时将自动恢复重试。")
