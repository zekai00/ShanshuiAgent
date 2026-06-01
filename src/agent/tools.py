import random
import time
import json
import requests
import urllib.parse
from langchain_core.tools import tool

from src.config import (
    COMFYUI_SERVER_URL,
    COMFYUI_WORKFLOW_PATH,
    GENERATED_IMAGES_DIR,
    RETRIEVAL_SERVICE_URL,
    ensure_runtime_dirs,
)

# ==========================================
# 底层引擎懒加载单例 (防止多次挂载 BGE 模型导致显存爆炸)
# ==========================================
_global_retriever = None
IMAGE_SAVE_DIR = GENERATED_IMAGES_DIR
ensure_runtime_dirs()
IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def get_online_retriever():
    global _global_retriever
    if _global_retriever is None:
        # 注意：此处引用已重构后的 ingestion 模块
        from src.retrieval.online_retrieval import OnlineHybridRetriever
        _global_retriever = OnlineHybridRetriever(top_k=15, final_k=3)
    return _global_retriever

@tool("search_landscape_literature")
def search_landscape_literature(query: str) -> str:
    """当用户询问中国山水画的技法、历史、画家、构图等理论知识时调用。"""
    print(f"\n[⚙️ Researcher 工具执行] 检索词: {query}")
    
    # 🌟 直接向常驻后台的检索引擎发请求，毫秒级返回，不再需要本地预热！
    try:
        response = requests.post(f"{RETRIEVAL_SERVICE_URL}/retrieve", json={"query": query}, timeout=30)
        results = response.json().get("data", [])
    except Exception as e:
        return f"检索引擎服务未启动或连接异常: {e}"
        
    if not results:
        return "抱歉，资料库中未检索到相关内容。请尝试更换关键词。"
        
    formatted_report = "【画院档案库检索结果】\n"
    for i, doc in enumerate(results):
        formatted_report += f"\n--- 档案 {i+1} ---\n"
        formatted_report += f"来源文献: {doc.get('source_file')}\n"
        formatted_report += f"证据块ID: {doc.get('chunk_id')}\n"
        if doc.get("page_start"):
            formatted_report += f"页码: {doc.get('page_start')}"
            if doc.get("page_end") and doc.get("page_end") != doc.get("page_start"):
                formatted_report += f"-{doc.get('page_end')}"
            formatted_report += "\n"
        if doc.get("contextual_prefix"):
            formatted_report += f"系统上下文: {doc.get('contextual_prefix')}\n"
        formatted_report += f"内容详情: {doc.get('raw_chunk_text') or doc.get('contextual_chunk')}\n"
    return formatted_report

@tool("generate_landscape_image")
def generate_landscape_image(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """当用户要求生成图像时调用。传入详尽的【英文】绘画提示词及尺寸。"""
    print(f"\n[🎨 工具执行] Artist 下发参数 -> 尺寸: {width}x{height} prompt是：{prompt}")
    proxies = {"http": None, "https": None}
    server_address = COMFYUI_SERVER_URL
    workflow_path = COMFYUI_WORKFLOW_PATH
    
    try:
        with workflow_path.open('r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        workflow_data["68"]["inputs"]["text"] = prompt
        workflow_data["69"]["inputs"]["width"] = width
        workflow_data["69"]["inputs"]["height"] = height
        workflow_data["70"]["inputs"]["seed"] = random.randint(1, 2**31 - 1)
    except KeyError as e:
        return f"JSON 节点匹配失败，请检查工作流中是否缺少该节点: {str(e)}"
    
    try:
        response = requests.post(f"{server_address}/prompt", json={"prompt": workflow_data}, proxies=proxies, timeout=10)
        response.raise_for_status() 
        prompt_id = response.json()['prompt_id']
        print(f"[*] 渲染任务已成功注入，任务 ID: {prompt_id}")
    except Exception as e:
        return f"网络请求被拦截或 ComfyUI 拒绝连接。真实报错: {str(e)}"

    retry_count = 0
    while retry_count < 100:
        try:
            hist_res = requests.get(f"{server_address}/history/{prompt_id}", proxies=proxies, timeout=10)
            history_data = hist_res.json()
            if prompt_id in history_data:
                outputs = history_data[prompt_id].get('outputs', {})
                if not outputs: return "ComfyUI 渲染失败！未生成图像。"
                for node_id in outputs:
                    if 'images' in outputs[node_id]:
                        img_info = outputs[node_id]['images'][0]
                        filename = img_info['filename']
                        img_url = f"{server_address}/view?filename={urllib.parse.quote(filename)}&subfolder={urllib.parse.quote(img_info['subfolder'])}&type={img_info['type']}"
                        img_data = requests.get(img_url, proxies=proxies, timeout=30).content
                        final_save_path = IMAGE_SAVE_DIR / filename
                        with final_save_path.open('wb') as f: f.write(img_data)
                        return f"图像渲染成功！已保存至: {final_save_path}"
            retry_count += 1
            time.sleep(3)
        except Exception as e:
            return f"获取引擎状态中断。真实报错: {str(e)}"
    return "渲染严重超时，强制终止。"

# ==========================================
# 更新工具白名单
# ==========================================
# 🌟 更新工具白名单，只保留核心检索
researcher_tools = [search_landscape_literature]
artist_tools = [generate_landscape_image]
tools_by_name = {t.name: t for t in (researcher_tools + artist_tools)}
