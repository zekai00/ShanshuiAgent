from typing import Annotated, Sequence, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """回滚后的精简全局状态总线"""
    # 基础对话流
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 路由指令：由 Supervisor 决定下一步去哪 (researcher, artist, chatter, finish)
    next_node: str 
    # 发送指令的当前的agent
    sender: str
    # 🌟 新增：渐进式披露的物理隔离区
    research_dossier: str
    
    # 记忆与用户标识
    user_id: str
    long_term_memory: str
    summary: str  # 替代原先复杂的工件流转