import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from config.prompts.paper_prompts import GLOBAL_SUMMARY_PROMPT, SUPER_CHUNK_EXTRACTION_PROMPT

# 🌟 引入 LangSmith 包装器
from langsmith import wrappers

# 加载 .env 文件中的环境变量
load_dotenv()

class LocalLLMExtractor:
    def __init__(self, 
                 local_base_url: str = "http://localhost:8000/v1", 
                 local_model: str = "qwen3-4b-instruct",
                 api_base_url: str | None = None,
                 api_model: str | None = None):
        api_base_url = api_base_url or os.environ.get("CL_LONG_CONTEXT_LLM_BASE_URL", "http://localhost:8000/v1")
        api_model = api_model or os.environ.get("CL_LONG_CONTEXT_LLM_MODEL", "local-long-context-model")
        
        print(f"[*] 初始化双擎抽取器: 长文本引擎 ({api_model}) + 本地切片引擎 ({local_model}) ...")
        
        # 本地 Qwen-4B 客户端 (负责输出结构化 JSON)
        self.local_client = wrappers.wrap_openai(OpenAI(
            api_key=os.environ.get("CL_LOCAL_LLM_API_KEY", "local-dev"),
            base_url=local_base_url, 
            timeout=120.0
        ))
        self.local_model = local_model
        
        api_key = os.environ.get("CL_LONG_CONTEXT_LLM_API_KEY") or os.environ.get("CL_LLM_API_KEY")
        if not api_key:
            print("[!] 未配置长文本 LLM 凭证；远端大纲提取将失败。")
        
        # 长文本客户端负责通读 PDF 并提炼全局大纲。
        self.api_client = wrappers.wrap_openai(OpenAI(
            api_key=api_key or "not-configured",
            base_url=api_base_url, 
            # 100页 PDF 通读极其耗时，将 Timeout 延长至 10 分钟 (600秒) 以防中断
            timeout=600.0 
        ))
        self.api_model = api_model

    def generate_global_context(self, full_text: str) -> str:
        """调用超大杯模型，通读全文，提炼全局大纲"""
        print(f"    [LLM] 正在提炼全文 ({len(full_text)} 字符) 的全局大纲...")
        try:
            response = self.api_client.chat.completions.create(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": GLOBAL_SUMMARY_PROMPT},
                    {"role": "user", "content": f"【完整文献】：\n{full_text}"}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [!] 长文本 LLM 调用失败，降级为空大纲。报错: {e}")
            return "暂无全局大纲。"

    def extract_super_chunk(self, text_chunk: str, global_context: str) -> dict:
        """调用本地小模型，结合上帝视角生成超级切片"""
        prompt = SUPER_CHUNK_EXTRACTION_PROMPT.replace("{global_context}", global_context)
        
        try:
            response = self.local_client.chat.completions.create(
                model=self.local_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"【待处理文本片段】：\n{text_chunk}"}
                ],
                temperature=0.1,  
                max_tokens=1000, # 稍微放宽 token 限制，防止复杂 JSON 截断
                # 🌟 开启本地 Qwen-4B 的 JSON 强约束模式 (需要 vLLM/Ollama 底层支持)
                response_format={"type": "json_object"} 
            )
            
            result_text = response.choices[0].message.content
            clean_json_str = result_text[result_text.find('{'):result_text.rfind('}')+1]
            return json.loads(clean_json_str)
            
        except Exception as e:
            print(f"[!] JSON 解析失败: {e}")
            # 兜底结构
            return {
                "metadata": {
                    "is_domain_relevant": True, # 兜底默认放行
                    "dynasty": ["未知"], 
                    "painter_or_school": "未知", 
                    "composition_layout": "未知", 
                    "brushwork_technique": "未知",
                    "subject_matter": "未知",
                    "aesthetic_concept": "未知",
                    "content_scope": "未知"
                },
                "contextual_enrichment": {
                    "context_anchor": "无上下文锚点",
                    "multi_queries": [],
                    "hyde_answer": "无生成答案"
                }
            }
