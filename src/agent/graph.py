# /root/Workspace/ShanshuiAgent/src/agent/graph.py

import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import AGENT_CHECKPOINT_DB, ensure_runtime_dirs
from .state import AgentState
from .main_nodes import (
    gateway_node, summarizer_node, supervisor_node,
    researcher_node, artist_node, chatter_node
)

# 1. 初始化计算图，绑定全局状态契约
main_builder = StateGraph(AgentState)

# 2. 注册所有物理节点
main_builder.add_node("gateway", gateway_node)
main_builder.add_node("summarizer", summarizer_node)
main_builder.add_node("supervisor", supervisor_node)
main_builder.add_node("researcher", researcher_node)
main_builder.add_node("artist", artist_node)
main_builder.add_node("chatter", chatter_node)

# ==========================================
# 3. 编排边 (Edges) 与条件路由 (Conditional Edges)
# ==========================================

# 【入口防线】：用户输入后，状态必须先流入网关
main_builder.add_edge(START, "gateway")

def route_from_gateway(state: AgentState):
    """
    内存保护机制：拦截过长的对话。
    如果 `messages` 数组长度过大，先引流到 summarizer 节点进行压缩，否则直接去主管节点报到。
    """
    if len(state["messages"]) > 10:
        return "summarizer"
    return "supervisor"

# 从网关出发的条件分流
main_builder.add_conditional_edges(
    "gateway",
    route_from_gateway,
    ["summarizer", "supervisor"]
)

# 内存压缩完毕后，回归主干线去找主管
main_builder.add_edge("summarizer", "supervisor")

def route_from_supervisor(state: AgentState):
    """
    派单机制：根据主管大模型输出的 JSON `next_node` 字段进行下游分发。
    """
    next_node = state.get("next_node", "chatter") # 默认去闲聊兜底
    if next_node == "finish":
         return END
    return next_node

# 从主管出发，呈放射状派单给具体的执行者，或宣告任务结束退出图谱
main_builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "artist": "artist",
        "chatter": "chatter",
        END: END
    }
)

# 【回流机制】：所有的 Worker 在自己内部干完活后，必须无条件退回集结地 (gateway)
# 这样图谱就会再次进入网关 -> 主管的轮询。主管会看到 Worker 刚才的输出，然后判定：
# “哦，画师已经把图画好了发给用户了，那我这轮输出 finish 吧。”从而优雅结束对话。
# ======= 修改前 =======
# main_builder.add_edge("researcher", "gateway")
# main_builder.add_edge("artist", "gateway")
# main_builder.add_edge("chatter", "gateway")

# ======= 修改后 =======
main_builder.add_edge("researcher", "gateway") # 考据完还得回去问主管要不要接着画图
main_builder.add_edge("artist", END)           # 画完图，本回合强制结束
main_builder.add_edge("chatter", END)          # 闲聊完，本回合强制结束
# ==========================================
# 4. 编译与物理持久化
# ==========================================
ensure_runtime_dirs()
AGENT_CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(AGENT_CHECKPOINT_DB), check_same_thread=False)
memory = SqliteSaver(conn)

# 编译为可执行应用
app = main_builder.compile(checkpointer=memory)
