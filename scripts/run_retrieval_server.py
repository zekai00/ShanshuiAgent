import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RETRIEVAL_FINAL_K, RETRIEVAL_HOST, RETRIEVAL_PORT, RETRIEVAL_TOP_K
from src.retrieval.online_retrieval import OnlineHybridRetriever

# 声明全局变量，但先不实例化
retriever = None

# 🌟 使用官方推荐的 Lifespan 管理生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("\n" + "="*60)
    print("⚡ 启动数字敦煌 V3.0 极速检索引擎")
    print("="*60)
    
    try:
        # 1. 在这里才真正去连接 Milvus 和加载大模型，避免 Import 时的死锁
        retriever = OnlineHybridRetriever(top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K)
        
        # 2. 执行模型 CUDA 预热
        print("🔥 [Lifespan] 正在执行底层模型 CUDA 预热 (防止首次调用超时)...")
        retriever.retrieve_and_rerank("预热请求")
        print("✅ [Lifespan] 预热完毕！服务现已达到极速状态。")
    except Exception as e:
        print(f"\n[!] 🚨 服务初始化或预热失败: {e}")
        print("请检查是否被其他进程锁死，或显存溢出。")
    
    # 挂起，交出控制权，开始接收 HTTP 请求
    yield 
    
    # 接收到 Ctrl+C 退出信号后，执行清理
    print("\n🛑 [Lifespan] 收到退出信号，正在释放显存与数据库连接...")
    if retriever and hasattr(retriever, 'close'):
        retriever.close()
        print("✅ 资源已安全释放。")

# 将 lifespan 绑定到 FastAPI 实例
app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    if not retriever:
        return {"data": [], "error": "检索引擎未就绪"}
    # 注意：这里直接调用同步的检索方法，FastAPI 会在主线程运行它，避免 AnyIO 多进程冲突
    results = retriever.retrieve_and_rerank(req.query)
    return {"data": results}

if __name__ == "__main__":
    import uvicorn
    # 强烈建议在此处直接通过模块名启动
    uvicorn.run("scripts.run_retrieval_server:app", host=RETRIEVAL_HOST, port=RETRIEVAL_PORT, reload=False)
