# /root/Workspace/ShanshuiAgent/scripts/build_full_graph.py

import sys
from pathlib import Path
from tqdm import tqdm
from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入你的核心图谱构建器
from src.config import MILVUS_DB_PATH, RETRIEVAL_COLLECTION_NAME
from src.ingestion.linear_rag_graph import LinearRAGBuilder

COLLECTION_NAME = RETRIEVAL_COLLECTION_NAME

def build_full_knowledge_graph():
    print("="*60)
    print("🕸️ 启动数字敦煌 V3.0 - 全域知识图谱构建流水线")
    print("="*60)

    # 1. 初始化 Neo4j 构建器
    builder = LinearRAGBuilder()
    if not builder.driver:
        print("[!] 致命错误：Neo4j 连接失败，请检查数据库是否启动以及密码配置。")
        sys.exit(1)

    # 交互式清屏检查
    builder.check_and_clear_database()

    # 2. 连接 Milvus 拉取全量上下文语料
    print("\n[*] 正在连接 Milvus 数据库提取语料...")
    if not MILVUS_DB_PATH.exists():
        print("[!] 未找到 Milvus 数据库，请先运行 run_ingestion.py！")
        sys.exit(1)
        
    milvus_client = MilvusClient(str(MILVUS_DB_PATH))
    milvus_client.load_collection(COLLECTION_NAME)

    # 通过 filter "id > 0" 暴力拉取所有主键和纯净文本
    # 注意：如果数据量达到百万级，需要改用 iterator 分页拉取，目前十万级以内可直接 query
    print("    -> 正在向 Milvus 发起全表扫描...")
    all_chunks = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter="id > 0",
        output_fields=["id", "contextual_chunk"]
    )
    
    total_chunks = len(all_chunks)
    if total_chunks == 0:
        print("[!] Milvus 库中没有数据，图谱构建终止。")
        sys.exit(0)
        
    print(f"  ✅ 成功从 Milvus 拉取到 {total_chunks} 条超级切片。")

    # 3. 逐块进行 NLP 抽取与 Cypher 写入
    print("\n[*] 正在启动 NLP 实体雷达并构建 Tri-Graph 拓扑网络...")
    
    # 记录统计信息
    total_entities_extracted = 0
    
    for chunk in tqdm(all_chunks, desc="图谱构建进度"):
        chunk_id = chunk["id"]
        contextual_chunk = chunk["contextual_chunk"]
        
        # 调用你在 linear_rag_graph.py 中写好的核心逻辑
        graph_data = builder.extract_tri_graph_elements(chunk_id, contextual_chunk)
        builder.execute_cypher(graph_data)
        
        # 统计挖掘到的实体数量
        for sent in graph_data.get('sentences', []):
            total_entities_extracted += len(sent.get('entities', []))

    # 4. 收尾清理
    builder.close()
    print("\n" + "="*60)
    print("🎉 知识图谱全量构建竣工！")
    print(f"📊 战报：共处理 {total_chunks} 个文本块，挖掘并写入了 {total_entities_extracted} 个实体关系。")
    print("="*60)

if __name__ == "__main__":
    build_full_knowledge_graph()
