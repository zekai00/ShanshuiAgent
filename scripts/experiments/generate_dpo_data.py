import json
import sys
from pathlib import Path
from openai import OpenAI

# 确保能导入你自己的检索器
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLAMA_FACTORY_DIR,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
)
from src.retrieval.online_retrieval import OnlineHybridRetriever

# 初始化 DeepSeek 客户端
api_key = DEEPSEEK_API_KEY
if not api_key:
    raise RuntimeError("请先在环境变量中设置 DEEPSEEK_API_KEY，不能在代码中写入 API Key。")

client = OpenAI(
    api_key=api_key,
    base_url=DEEPSEEK_BASE_URL,
)

def generate_responses(query, retrieved_docs):
    """利用大模型同时生成优秀的 Chosen 回答和劣质的 Rejected 回答"""
    
    context = "\n".join([doc['contextual_chunk'] for doc in retrieved_docs])
    
    # 构建生成 Chosen 的 Prompt (拥有真实文献加持)
    chosen_prompt = f"""基于以下中国山水画的真实史料文献，专业、详实地回答用户问题。
【文献资料】：{context}
【用户问题】：{query}"""

    # 构建生成 Rejected 的 Prompt (故意引导它产生幻觉或时空错乱)
    rejected_prompt = f"""请故意用非常通俗、敷衍的现代网络语言回答以下问题，或者故意弄错朝代和作者。
【用户问题】：{query}"""

    try:
        chosen_res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": chosen_prompt}],
            temperature=0.3
        ).choices[0].message.content

        rejected_res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": rejected_prompt}],
            temperature=0.8
        ).choices[0].message.content
        
        return chosen_res, rejected_res
    except Exception as e:
        print(f"[!] API 调用失败: {e}")
        return None, None

if __name__ == "__main__":
    print("[*] 正在启动底层检索引擎...")
    retriever = OnlineHybridRetriever(top_k=min(RETRIEVAL_TOP_K, 10), final_k=RETRIEVAL_FINAL_K)
    
    # 你的原始问题列表 (可以从你的 4500 条数据里提取)
    queries = [
        "黄公望的富春山居图用了什么皴法？",
        "石涛的'一画论'核心思想是什么？"
    ]
    
    dpo_dataset = []
    
    for idx, query in enumerate(queries):
        print(f"\n▶️ 正在处理 [{idx+1}/{len(queries)}]: {query}")
        
        # 1. 真实检索
        docs = retriever.retrieve_and_rerank(query)
        if not docs:
            print("  -> 未检索到相关资料，跳过。")
            continue
            
        # 2. 调用大模型生成对比数据
        chosen, rejected = generate_responses(query, docs)
        
        if chosen and rejected:
            # 3. 组装为 LLaMA-Factory 标准 DPO 格式
            dpo_dataset.append({
                "instruction": query,
                "input": "",
                "chosen": chosen,
                "rejected": rejected
            })
            print("  ✅ 成功构建一对 DPO 样本！")
            
    # 保存结果
    output_path = LLAMA_FACTORY_DIR / "data" / "landscape_dpo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 竣工！DPO 数据已保存至: {output_path}")
    # 记得去 LLaMA-Factory 的 dataset_info.json 里注册这个新文件！
