# /root/Workspace/ChineseLandscape/scripts/run_chat_agent.py

import os
import sys
import uuid
import json
import warnings

warnings.filterwarnings("ignore", message=".*extra_body.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKSPACE_DIR = "/root/Workspace/ChineseLandscape"
sys.path.append(WORKSPACE_DIR)

# GRPC 与 Phoenix 配置保持原样
os.environ["GRPC_KEEPALIVE_TIME_MS"] = "120000"
os.environ["GRPC_KEEPALIVE_TIMEOUT_MS"] = "20000"
os.environ["GRPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS"] = "120000"
os.environ["GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
os.environ["PHOENIX_PORT"] = "6006"
os.environ["PHOENIX_HOST"] = "0.0.0.0"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://127.0.0.1:6006/v1/traces"
if "PHOENIX_COLLECTOR_ENDPOINT" in os.environ:
    del os.environ["PHOENIX_COLLECTOR_ENDPOINT"]

import readline 
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 导入业务模块
from src.agent.graph import app 
from src.agent.memory.memory_manager import MemoryManager

def run_interactive_terminal():
    print("\n" + "="*60)
    print("🖌️ 欢迎使用中国传统山水画多模态协作系统 (星型中心路由版)")
    print("   (输入 'quit' 或 'exit' 退出交互)")
    print("="*60 + "\n")
    
    user_id = input("👉 请输入您的专属 User ID (直接回车默认 'guest'): ").strip() or "guest"
    session_id = str(uuid.uuid4())[:8]
    print(f"\n[*] 正在为您初始化环境 | User: [{user_id}] | Session Thread: [{session_id}]")
    
    ltm_data = MemoryManager.get_memory(user_id)
    has_memory = any(ltm_data.get(k) for k in ["preferences", "feedback", "context"])
    
    if has_memory:
        ltm_str = json.dumps(ltm_data, ensure_ascii=False)
        print(f"  ✅ 记忆提取成功！系统已记起您的画像：\n     \"{ltm_str[:80]}...\"")
    else:
        print(f"  ℹ️ 无历史画像，将以白板状态启动。")

    run_config = {
        "configurable": {"thread_id": session_id}, 
        "recursion_limit": 50 
    }
    
    printed_msg_count = 0
    
    while True:
        try:
            user_input = input("\n[🧑 您] > ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用，再见！")
                break
            if not user_input.strip():
                continue
            
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "user_id": user_id,
                "long_term_memory": json.dumps(ltm_data, ensure_ascii=False)
            }
            
            final_state = app.invoke(inputs, config=run_config)
            
            all_messages = final_state.get("messages", [])
            new_messages = all_messages[printed_msg_count:]
            printed_msg_count = len(all_messages)
            
            print("\n" + "-"*20 + " 核心流转与输出 " + "-"*20)
            for msg in new_messages:
                # 过滤人类消息和中间态
                if isinstance(msg, HumanMessage) or isinstance(msg, ToolMessage):
                    continue
                if getattr(msg, 'tool_calls', None) and not msg.content:
                    continue
                    
                name = msg.name if hasattr(msg, 'name') and msg.name else "大模型推理"
                
                # 🌟 极简映射，对应星型架构里的三个 Worker 节点
                role_prefix = f"[🤖 {name}]"
                if name == "chatter":
                    role_prefix = "[💬 接待员]"              
                elif name == "researcher":
                    role_prefix = "[📚 学术专员]"
                elif name == "artist":
                    role_prefix = "[🖌️ 美术指导]"
                
                print(f"\n{role_prefix}:")
                
                content = msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block.get("text"))
                else:
                    print(content)
            print("-" * 56)
            
            ltm_data = MemoryManager.get_memory(user_id)
                
        except KeyboardInterrupt:
            print("\n\n[!] 检测到中断信号，正在安全退出系统...")
            break
        except Exception as e:
            print(f"\n[❌ 系统异常]: {str(e)}")

if __name__ == "__main__":
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    
    try:
        session = px.launch_app(port=6006, host="0.0.0.0")
        tracer_provider = register(project_name="Landscape-Online", endpoint="http://127.0.0.1:6006/v1/traces")
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"🔥 Phoenix 全链路可观测大屏已就绪: http://localhost:6006")
    except Exception as e:
        print(f"⚠️ 监控启动失败。错误: {e}")
    
    run_interactive_terminal()