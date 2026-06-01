import json
import yaml
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from src.config import AGENT_PROMPTS_DIR, FAST_LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from src.agent.memory.memory_manager import MemoryManager

from .state import AgentState
from .tools import researcher_tools, artist_tools, tools_by_name

# ==========================================
# 1. 核心模型配置 (分级算力)
# ==========================================
proxy_free_client = httpx.Client(proxy=None, trust_env=False)

# 🚀 极速模型：用于路由、闲聊及简单逻辑判断，关闭思考链
llm_fast = ChatOpenAI(
    model=FAST_LLM_MODEL,
    api_key=LLM_API_KEY or "not-configured",
    base_url=LLM_BASE_URL,
    temperature=0.1,
    http_client=proxy_free_client,
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}}
)

# 绑定专用工具的实例
simple_researcher_llm = llm_fast.bind_tools(researcher_tools)
artist_llm = llm_fast.bind_tools(artist_tools)

def get_prompt(yaml_filename: str) -> str:
    """动态读取 YAML 提示词，实现业务逻辑与指令分离"""
    file_path = AGENT_PROMPTS_DIR / yaml_filename
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("system_message", "")
    except FileNotFoundError:
        return ""

# ==========================================
# 2. 内部 ReAct 闭环引擎 (解决 V4 节点横跳的痛点)
# ==========================================
# 🌟 修复的底层执行引擎
def execute_react_loop(llm_with_tools, initial_messages, max_steps=5): # 放宽到 5 次
    new_messages = []
    current_messages = list(initial_messages)
    
    for step in range(max_steps):
        response = llm_with_tools.invoke(current_messages)
        new_messages.append(response)
        current_messages.append(response)
        
        # 退出条件：如果没有工具调用，说明大模型写出了最终结论
        if not getattr(response, "tool_calls", None):
            break
            
        for tool_call in response.tool_calls:
            t_name = tool_call["name"]
            if t_name in tools_by_name:
                try:
                    res = tools_by_name[t_name].invoke(tool_call["args"])
                except Exception as e:
                    res = f"执行异常: {str(e)}"
            else:
                res = f"拦截：未知工具 {t_name}"
                
            tool_msg = ToolMessage(content=str(res), tool_call_id=tool_call["id"], name=t_name)
            new_messages.append(tool_msg)
            current_messages.append(tool_msg)
            
    # 🌟 终极防线：如果循环耗尽，最后一条还是 ToolMessage，强制 LLM 进行文本总结！
    if isinstance(current_messages[-1], ToolMessage):
        print("  [!] 检索次数耗尽，正在强制大模型归纳当前发现...")
        # 暂时解除工具绑定，强迫它只输出文字
        final_response = llm_fast.invoke(current_messages) 
        new_messages.append(final_response)
            
    return new_messages

# ==========================================
# 3. 核心节点定义
# ==========================================

def supervisor_node(state: AgentState):
    """
    【中央主管节点 - 防死循环重制版】
    原理：通过显式检查 sender 状态，强迫主管在专家回复后重新评估是否该“下班”。
    """
    print("\n[🧠 Supervisor] 正在审阅全局上下文...")
    
    sys_prompt = get_prompt("supervisor_prompt.yaml")
    summary = state.get("summary", "")
    ltm = state.get("long_term_memory", "")
    last_sender = state.get("sender", "user") # 🌟 获取上一轮是谁在干活
    
    # 动态注入防死循环逻辑（核心改进点）
    # 如果上一个人不是用户，说明是某个 Worker 刚刚回答完
    anti_loop_context = ""
    if last_sender != "user":
        anti_loop_context = f"\n\n【🚨 实时状态警报】: 节点 [{last_sender}] 刚刚已经完成了它的任务并给出了回复。请阅读最后一条消息，如果用户的需求已经得到阶段性满足，请务必输出 'finish'，绝对禁止陷入死循环。"

    full_context = f"【历史提要】: {summary}\n【用户画像】: {ltm}{anti_loop_context}"
    
    # 为了防止历史太长干扰判断，我们只给 Supervisor 看最近 4 条消息
    messages = [SystemMessage(content=sys_prompt + "\n" + full_context)] + state["messages"][-4:]
    
    response = llm_fast.invoke(messages, response_format={"type": "json_object"})
    
    try:
        decision = json.loads(response.content)
        next_node = decision.get("next_node", "finish")
    except:
        next_node = "finish"
        
    print(f"  -> 决策流向: [{next_node}] (基于 {last_sender} 的执行结果)")
    return {"next_node": next_node, "sender": "supervisor"}

# 🌟 滚动摘要节点
def summarizer_node(state: AgentState):
    print("\n[🧹 Summarizer] 正在执行滚动状态压缩与记忆沉淀...")
    sys_prompt = get_prompt("summarizer_prompt.yaml")
    
    messages = state["messages"]
    old_summary = state.get("summary", "无")
    user_id = state.get("user_id", "guest")
    
    # 提取最近 6 条消息
    recent_chat = []
    for m in messages[-6:]:
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content):
            prefix = "User" if isinstance(m, HumanMessage) else "AI"
            recent_chat.append(f"{prefix}: {m.content}")
            
    recent_context = "\n".join(recent_chat)
    prompt = f"【原有背景摘要】: {old_summary}\n【新增对话记录】:\n{recent_context}"
    
    # 强制输出 JSON
    response = llm_fast.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)],
        response_format={"type": "json_object"}
    )
    
    try:
        res_json = json.loads(response.content)
        new_summary = res_json.get("summary", old_summary)
        insights = res_json.get("insights", {})
        
        # 如果不是 guest，且确实提取到了干货，则写入数据库
        if user_id != "guest" and any(insights.get(k) for k in ["preferences", "feedback", "context"]):
            MemoryManager.save_memory(user_id, insights)
            
        return {"summary": new_summary, "sender": "summarizer"}
    except Exception as e:
        print(f"  -> 压缩与提取失败: {e}")
        return {"sender": "summarizer"}

# 🌟 学术专员：查完资料后，写入独立卷宗
def researcher_node(state: AgentState):
    print("\n[📚 Researcher] 开启学术考据...")
    sys_prompt = get_prompt("researcher_prompt.yaml")
    ltm = state.get("long_term_memory", "")
    
    messages = [SystemMessage(content=sys_prompt + f"\n【用户偏好】: {ltm}")] + state["messages"]
    new_msgs = execute_react_loop(simple_researcher_llm, messages, max_steps=5)
    
    if new_msgs:
        new_msgs[-1].name = "researcher"
        
    # 将最终的文本答案提取出来，存入卷宗字段，供画师读取
    final_text = new_msgs[-1].content if new_msgs else "考据失败"
    
    return {
        "messages": new_msgs, 
        "sender": "researcher",
        "research_dossier": final_text # 渐进式披露：物理隔离的数据总线
    }

# 🌟 美术指导：被“强制致盲”，只读卷宗，不看历史
def artist_node(state: AgentState):
    print("\n[🖌️ Artist] 图像创作中...")
    sys_prompt = get_prompt("artist_prompt.yaml")
    
    # 获取隔离的卷宗
    dossier = state.get("research_dossier", "无前置考据卷宗。")
    ltm = state.get("long_term_memory", "")
    
    # 组装超级干净的上下文：人设 + 卷宗 + 偏好
    clean_sys_msg = SystemMessage(content=sys_prompt + f"\n【必须遵循的考据卷宗】:\n{dossier}\n\n【用户视觉偏好】:{ltm}")
    
    # 关键：只给画师看用户的【最后一句要求】，绝对不给看整个 state["messages"]
    last_user_req = state["messages"][-1] 
    
    messages = [clean_sys_msg, last_user_req]
    new_msgs = execute_react_loop(artist_llm, messages, max_steps=3)
    
    if new_msgs:
        new_msgs[-1].name = "artist"
    return {"messages": new_msgs, "sender": "artist"}

def chatter_node(state: AgentState):
    print("\n[💬 Chatter] 亲切接待中...")
    sys_prompt = get_prompt("chatter_prompt.yaml")
    
    # 闲聊直接输出，不进工具循环
    response = llm_fast.invoke([SystemMessage(content=sys_prompt)] + state["messages"])
    response.name = "chatter"
    return {"messages": [response], "sender": "chatter"}

def gateway_node(state: AgentState):
    """空转锚点，用于在 graph.py 中挂载条件边"""
    return {}
