import json
from langchain_core.messages import SystemMessage, HumanMessage
# 引入底层 LLM 配置。注意：这里我们特别引入了带思维链的 llm_reason
from .main_nodes import llm_reason, llm_fast, get_prompt, simple_researcher_llm
from .state import AgentState

# ==========================================
# 考据子图节点定义 (依赖深度推理的重型节点)
# ==========================================

def planner_node(state: AgentState):
    """
    【任务规划师节点】
    将复杂问题拆解为多个正交的子任务，供后续并发沙箱执行。
    """
    print("\n[🧠 Planner] 接收到高复杂度任务，正在启动【深度思考模式】进行 DAG 拆解...")
    prompt_text = get_prompt("planner_prompt.yaml")
    
# 🌟 修复：永远只取用户的第一句提问，彻底无视 Reviewer 的历史打回意见
    # 这样大模型就不会被报错信息干扰而输出 0 个任务
    user_input = state["messages"][0].content
    ltm = state.get("long_term_memory", "")
    
    context = f"\n【用户请求】: {user_input}\n【用户长期偏好】: {ltm}"
    messages = [SystemMessage(content=prompt_text + context)]
    
    # 强制要求大模型输出 JSON 格式
    response = llm_reason.invoke(messages, response_format={"type": "json_object"})
    
    try:
        plan_data = json.loads(response.content)
        task_plan = plan_data.get("task_plan", [])
    except Exception:
        # 兜底防止解析失败
        task_plan = [f"检索与 {user_input} 相关的文献"]
        
    print(f"  -> 拆解完成，生成了 {len(task_plan)} 个并发子任务。")
    
    return {
        "task_plan": task_plan, 
        "sender": "planner",
        # 🌟 每次重新规划时，向状态机发送一个特殊的覆盖指令，清空上一次循环残留的脏数据
        # (需要 state.py 中去掉 operator.add，或者依赖 LangGraph 的原生覆盖机制)
        "partial_reports": [] 
    }

def parallel_researcher_node(state: AgentState):
    """
    【并发研究员节点】
    🌟 核心：引入内部执行引擎，确保在并发沙箱里把资料查完再返回。
    """
    from .main_nodes import execute_react_loop, simple_researcher_llm, get_prompt # 引入依赖
    
    task_description = state["messages"][-1].content 
    print(f"\n[⚡ Parallel Worker] 启动并发沙箱，领受任务: '{task_description[:20]}...'")
    
    prompt_text = get_prompt("parallel_researcher_prompt.yaml")
    messages = [SystemMessage(content=prompt_text)] + state["messages"]
    
    # 在沙箱内部死磕，最多查 3 次
    new_msgs = execute_react_loop(simple_researcher_llm, messages, max_steps=3)
    
    # 提取最后得出的文本结论
    final_response = new_msgs[-1]
    
    # 如果最后一次还是想调用工具（说明查了3次都失败了强制熔断了）
    if getattr(final_response, 'tool_calls', None):
        return {"partial_reports": [f"关于任务 '{task_description}'：未检索到有效信息"]}
        
    print(f"  ✅ [Worker 完工] 任务完成，正在向全局状态注入简报。")
    return {"partial_reports": [final_response.content]}

def synthesizer_node(state: AgentState):
    """
    【逻辑融合器节点】 (Reduce 阶段)
    🌟 核心：使用 llm_reason。
    融合器需要阅读并理解多份并发报告，化解时间线和理论上的冲突，
    这项任务需要极强的分析与归纳能力，必须开启思维链（Thinking）。
    """
    print("\n[🧬 Synthesizer] 所有并发检索已收口，启动【深度思考模式】化解冲突生成卷宗...")
    prompt_text = get_prompt("synthesizer_prompt.yaml")
    
    reports = state.get("partial_reports", [])
    if not reports:
        return {"research_dossier": "检索失败，无有效史料返回。", "sender": "synthesizer"}
    
    combined_reports = "【以下是各并发研究员提交的碎片化简报】：\n\n"
    for i, report in enumerate(reports):
        combined_reports += f"--- 简报 {i+1} ---\n{report}\n\n"
        
    user_input = state["messages"][-1].content if state["messages"] else "无"
    
    dynamic_context = f"\n\n【用户原始问题】: {user_input}\n\n{combined_reports}"
    
    # 使用带有思考模式的推理模型，提炼出一份完美的研究卷宗
    response = llm_reason.invoke([SystemMessage(content=prompt_text + dynamic_context)])
    
    print("  ✅ 卷宗生成完毕！已写入全局 Artifacts 状态总线。")
    
    return {
        "research_dossier": response.content, 
        "sender": "synthesizer"
    }