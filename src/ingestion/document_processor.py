# /root/Workspace/ShanshuiAgent/src/core/document_processor.py

import os
import io
import re
import base64
import json
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from langsmith import wrappers
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 必须通过绝对路径引用其他内部模块
from src.ingestion.llm_extractor import LocalLLMExtractor

class DocumentProcessor:
    """🌟 专注于文档多模态解析与两级漏斗切分的引擎"""
    def __init__(self, images_save_dir: str):
        self.images_folder = images_save_dir
        # 被 LangSmith 拦截的本地调用
        self.vlm_client = wrappers.wrap_openai(OpenAI(api_key="sk-local", base_url="http://localhost:8001/v1", timeout=300.0))
        self.llm_client = wrappers.wrap_openai(OpenAI(api_key="sk-local", base_url="http://localhost:8000/v1", timeout=300.0))
        
        self.metadata_extractor = LocalLLMExtractor()
        self.coarse_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "。", "！"])
        
    def extract_and_crop(self, pdf_path: str, filename: str) -> str:
        """多模态解析：防御性 VLM 提取文本并裁剪古画"""
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"\n[!] ⚠️ 无法打开 {filename}，文件可能损坏，跳过: {e}", flush=True)
            return ""
            
        full_text = []
        system_prompt = (
            "你是一个专业的古籍排版恢复专家。请按照阅读顺序提取正文文字，忽略页眉页码。\n"
            "【特别指令】：如果遇到古画、图表或插图，请必须使用以下格式替代：\n"
            "[插图：坐标[x0,y0,x1,y1]：描述你看到的图片内容]\n"
            "注意：坐标必须是 0 到 1000 之间的整数，代表图片在当前页面的相对位置占比。"
        )
        
        for page_num in range(len(doc)):
            print(f"    -> 正在用 VLM 侦测第 {page_num+1}/{len(doc)} 页...", flush=True)
            try:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                response = self.vlm_client.chat.completions.create(
                    model="qwen-vl",
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]
                    }],
                    max_tokens=1500, temperature=0.1
                )
                vlm_text = response.choices[0].message.content.strip()
                
                def crop_and_replace(match):
                    try:
                        x0, y0, x1, y1 = map(int, match.groups()[:4])
                        desc = match.group(5)
                        rect = fitz.Rect(x0 / 1000.0 * page.rect.width, y0 / 1000.0 * page.rect.height, x1 / 1000.0 * page.rect.width, y1 / 1000.0 * page.rect.height)
                        img_name = f"{filename.replace('.pdf', '')}_p{page_num+1}_{x0}_{y0}.jpg"
                        img_path = os.path.join(self.images_folder, img_name)
                        page.get_pixmap(clip=rect, dpi=200).save(img_path)
                        return f"[插图：{img_path}：{desc}]"
                    except Exception as inner_e:
                        print(f"      [!] 裁剪坐标异常: {inner_e}")
                        return f"[插图：裁剪失败：{match.group(5)}]"
                        
                processed_text = re.sub(r'\[插图：坐标\[(\d+),(\d+),(\d+),(\d+)\]：(.*?)\]', crop_and_replace, vlm_text)
                full_text.append(processed_text)
            except Exception as e:
                print(f"    [!] 第 {page_num+1} 页 VLM 解析失败，跳过。报错: {e}")
                continue
                
        doc.close()
        merged_text = "\n\n".join(full_text)
        return re.sub(r'\[(中图分类号|文献标识码|文章编号|收稿日期|作者简介)\].*?\n|本文文献著录格式.*?\n', '', merged_text)

    # ========================================================
    # 🌟 修复重点：就是这个被你漏掉的 agentic_split 函数！
    # 已经去掉了 JSON Mode 紧箍咒，并加入了物理括号提取防线
    # ========================================================
    def agentic_split(self, text_block: str) -> list:
        """智能体精细切分，防抖与防空校验"""
        prompt = f"""请根据语义边界切分以下文本，输出纯 JSON 数组格式：["片段1", "片段2"]。
        如果当前文本已经是不可分割的单一逻辑闭环，请直接输出包含1个元素的数组；如果有多个主题，请切分为2到4个片段。\n【待切分文本】：\n{text_block}"""
        try:
            response = self.llm_client.chat.completions.create(
                model="qwen3-4b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
                # 🌟 物理切除：删除了 response_format={"type": "json_object"}
            )
            res_text = response.choices[0].message.content.strip()
            
            # 🌟 物理防线：强制寻找数组的中括号
            start_idx = res_text.find('[')
            end_idx = res_text.rfind(']')
            
            if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                raise ValueError(f"未找到有效的中括号 JSON 数组。模型原始输出: {res_text[:50]}...")
                
            clean_json = res_text[start_idx:end_idx+1]
            chunks = json.loads(clean_json)
            
            # 防御：剔除空块和超小无意义块
            return [c for c in chunks if isinstance(c, str) and len(c.strip()) > 30]
            
        except Exception as e:
            print(f"    [!] Agentic Split 降级（原样返回）: {e}")
            return [text_block]