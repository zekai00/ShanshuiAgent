import os
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.config import (
    COLBERT_TENSORS_PATH,
    EXTRACTED_ARTWORKS_DIR,
    INGESTION_STATE_PATH,
    MILVUS_DB_PATH,
    NEO4J_PASSWORD as CFG_NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    PROJECT_ROOT,
    RAW_PDFS_DIR,
    UI_STATIC_DIR,
    VECTOR_STORE_DIR,
)

WORKSPACE_DIR = str(PROJECT_ROOT)

# 全新命名的资源路径
PAPERS_FOLDER = RAW_PDFS_DIR
IMAGES_FOLDER = EXTRACTED_ARTWORKS_DIR
DB_FOLDER = VECTOR_STORE_DIR
TRACKING_FILE = INGESTION_STATE_PATH
COLBERT_FILE = COLBERT_TENSORS_PATH

import io
# Web 端可观测性配置应由部署环境自行注入；公开代码不内置外部服务端点。
import re
import base64  # <--- 新增这行
import gradio as gr
from langchain_core.messages import HumanMessage, ToolMessage
from src.agent.graph import app

# 🌟 新增导入：Neo4j 驱动与 PyVis 可视化引擎
from neo4j import GraphDatabase
from pyvis.network import Network

GRAPH_HTML_PATH = UI_STATIC_DIR / "graph.html"
GRAPH_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)

# Neo4j 数据库凭证 (请确认密码是你刚才改过的)
NEO4J_USER = NEO4J_USERNAME
NEO4J_PASSWORD = CFG_NEO4J_PASSWORD or ""

# ==========================================
# 核心组件 1：双路劫持器 (三通阀门)
# ==========================================
class DualLogger:
    def __init__(self, terminal_out, string_out):
        self.terminal_out = terminal_out
        self.string_out = string_out

    def write(self, text):
        self.terminal_out.write(text)
        self.string_out.write(text)

    def flush(self):
        self.terminal_out.flush()
        self.string_out.flush()

# ==========================================
# 核心组件 2：Neo4j 到 PyVis 的渲染引擎
# ==========================================
def generate_graph_html():
    """连接 Neo4j 抓取三元组，并渲染为带物理引擎的交互式 HTML"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 300")
            
            net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333", directed=True)
            nodes_added = set()
            
            for record in result:
                s = record["s"]
                o = record["o"]
                r = record["r"]
                
                # 🌟 修复 1：利用 hasattr 彻底避开旧版 s.id 的触碰，消除警告
                s_id = s.element_id if hasattr(s, "element_id") else s.id
                o_id = o.element_id if hasattr(o, "element_id") else o.id
                
                s_name = s.get("name", "未知")
                o_name = o.get("name", "未知")
                r_type = r.type
                
                if s_id not in nodes_added:
                    net.add_node(s_id, label=s_name, color="#14b8a6", size=20)
                    nodes_added.add(s_id)
                if o_id not in nodes_added:
                    net.add_node(o_id, label=o_name, color="#f59e0b", size=15)
                    nodes_added.add(o_id)
                    
                net.add_edge(s_id, o_id, title=r_type, label=r_type, color="#cbd5e1")
                
        driver.close()
        
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08 },
            "minVelocity": 0.75, "solver": "forceAtlas2Based"
          }
        }
        """)
        
        net.save_graph(str(GRAPH_HTML_PATH))
        
        # 🌟 修复 2：黑客级 Base64 注入，无视 Gradio 路由限制！
        with GRAPH_HTML_PATH.open('r', encoding='utf-8') as f:
            html_content = f.read()
            
        b64_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;charset=utf-8;base64,{b64_html}"
        
        return f'<iframe src="{data_uri}" width="100%" height="650px" style="border:none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></iframe>'
        
    except Exception as e:
        return f"<div style='padding: 20px; color: #ef4444;'>图谱渲染失败，请检查 Neo4j: {str(e)}</div>"

# ==========================================
# 核心组件 3：聊天主逻辑
# ==========================================
def chat_logic(user_message, history, current_state):
    if current_state is None:
        current_state = {"messages": [], "sender": "user"}
    
    history.append({"role": "user", "content": user_message})
    
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = DualLogger(old_stdout, captured_output)

    bot_reply_text = ""
    pdf_html = "<div style='padding: 20px; color: gray;'>等待命中相关文献...</div>"

    try:
        current_state["messages"].append(HumanMessage(content=user_message))
        current_state["sender"] = "user"
        previous_msg_count = len(current_state["messages"]) - 1
        
        yield history, history, current_state, "系统正在初始化引擎...\n", pdf_html

        final_state = app.invoke(current_state)
        current_state = final_state
        
        new_messages = final_state["messages"][previous_msg_count + 1:]
        for msg in new_messages:
            if isinstance(msg, ToolMessage): continue
            
            content = msg.content
            if isinstance(content, list):
                content = "".join([c.get("text","") for c in content if c.get("type")=="text"])
            
            if not content or not str(content).strip():
                continue 
                
            role_name = getattr(msg, 'name', 'System').upper()
            bot_reply_text += f"### {role_name}\n{content}\n\n---\n"
        
        system_logs = captured_output.getvalue()
        
        pdf_matches = re.findall(r'来源:\s*(.*?\.pdf)', system_logs)
        if pdf_matches:
            potential_path = str(PAPERS_FOLDER / pdf_matches[0].strip())
            if os.path.exists(potential_path):
                pdf_html = f'<embed src="/file={potential_path}#view=FitH" width="100%" height="650px" type="application/pdf">'

        history.append({"role": "assistant", "content": bot_reply_text.strip()})
        yield history, history, current_state, system_logs, pdf_html

    except Exception as e:
        error_msg = f"❌ 系统异常: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        yield history, history, current_state, captured_output.getvalue(), pdf_html
    finally:
        sys.stdout = old_stdout

# ==========================================
# 现代 UI 布局
# ==========================================
custom_css = """
#app-title { text-align: center; color: #14b8a6; margin-bottom: 20px; font-weight: bold; }
.gradio-container { max-width: 1500px !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# ⛰️ 中国山水画多智能体协作中枢", elem_id="app-title")
    state_var = gr.State(value=None)
    
    with gr.Row():
        # 左侧：聊天与指令
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=700, label="智能协作网络", avatar_images=(None, "🤖"))
            with gr.Row():
                msg_input = gr.Textbox(placeholder="请输入指令...", show_label=False, scale=8)
                submit_btn = gr.Button("发送", variant="primary", scale=1)
                
        # 右侧：多模态资源监控看板
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.TabItem("📖 考据文献"):
                    pdf_viewer = gr.HTML(label="原始文献")
                
                # 🌟 新增的知识图谱选项卡
                with gr.TabItem("🕸️ 知识图谱星空"):
                    # 页面加载时自动渲染一次图谱
                    graph_viewer = gr.HTML(value=generate_graph_html())
                    refresh_graph_btn = gr.Button("🔄 重新拉取底层图谱数据", size="sm")
                    refresh_graph_btn.click(fn=generate_graph_html, outputs=graph_viewer)
                
                with gr.TabItem("⚙️ 底层日志"):
                    log_output = gr.Code(label="终端实时流", language="shell")

    msg_input.submit(chat_logic, [msg_input, chatbot, state_var], [chatbot, chatbot, state_var, log_output, pdf_viewer])
    submit_btn.click(chat_logic, [msg_input, chatbot, state_var], [chatbot, chatbot, state_var, log_output, pdf_viewer])
    msg_input.submit(lambda: "", None, msg_input)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        # 🌟 必须授权工作区根目录，因为我们生成的 graph.html 存在那里
        allowed_paths=[WORKSPACE_DIR], 
        theme=gr.themes.Soft(primary_hue="teal"),
        css=custom_css
    )
