# LangGraph 与当前 Agent 系统完整复盘报告

生成时间：2026-06-01 13:49 CST

> 说明：本文中的代码行号基于 2026-06-01 13:49 CST 当前工作区文件。后续继续修改代码后，行号可能会移动。

## 0. 结论先行

当前项目里存在两套“Agent”实现：

1. **真正的 LangGraph 图**：位于 `src/agent/graph.py`，有 6 个节点：`gateway`、`summarizer`、`supervisor`、`researcher`、`artist`、`chatter`。它通过 `scripts/run_chat_agent.py` 这个命令行入口运行，不是现在 7861 前端的主链路。  
   代码位置：`src/agent/graph.py:14-23`、`scripts/run_chat_agent.py:35`、`scripts/run_chat_agent.py:79`。

2. **当前 7861 前端实际使用的受控 Agent 状态机**：位于 `scripts/run_web_app.py`，通过 `/api/agent/stream` 对外提供服务。它不是 LangGraph 编译图，而是一个手写的流式状态机，节点包括 `intake`、`planner`、`researcher`、`verifier`、`research_synthesizer`、`prompt_designer`、`image_generator`、`image_critic`、`final_writer`、`memory_writer`。  
   代码位置：`scripts/run_web_app.py:294-305`、`scripts/run_web_app.py:1117-1264`、`scripts/run_web_app.py:1422-1424`、`ui/modern/app.js:523-527`。

因此，如果你问“现在 LangGraph 有哪些节点”，严格答案是：**LangGraph 只有旧图里的 6 个节点**。如果你问“当前前端正在跑的 Agent 有哪些节点”，答案是：**前端跑的是受控 Agent 状态机，不是 LangGraph，但它有 10 个业务节点**。

## 1. 入口与运行链路

### 1.1 LangGraph 命令行入口

LangGraph 的入口是 `scripts/run_chat_agent.py`：

- 导入 LangGraph app：`scripts/run_chat_agent.py:35`
- 从 `MemoryManager` 读取长期记忆：`scripts/run_chat_agent.py:48`
- 给 LangGraph 传入 `messages`、`user_id`、`long_term_memory`：`scripts/run_chat_agent.py:73-77`
- 调用 `app.invoke()`：`scripts/run_chat_agent.py:79`
- 用 `thread_id` 做 LangGraph checkpoint 线程标识：`scripts/run_chat_agent.py:57-60`

这条链路是终端交互，不是当前浏览器页面。

### 1.2 当前前端入口

当前前端页面 `http://127.0.0.1:7861/` 走的是 `scripts/run_web_app.py`：

- FastAPI app 定义：`scripts/run_web_app.py:53`
- 静态前端挂载：`scripts/run_web_app.py:54`
- 生成图片目录挂载：`scripts/run_web_app.py:55-56`
- 前端 JS 调用 `/api/agent/stream`：`ui/modern/app.js:523-527`
- 后端 `/api/agent/stream` 接口：`scripts/run_web_app.py:1422-1424`
- 实际执行函数：`scripts/run_web_app.py:1117-1264`

这条链路是当前产品主链路。

## 2. 当前模型配置

### 2.1 通用大模型配置

模型配置集中在 `src/config.py`：

- `DEEPSEEK_API_KEY`：`src/config.py:101`
- `DEEPSEEK_BASE_URL`：`src/config.py:102`
- `FAST_LLM_MODEL`：`src/config.py:103`

当前运行时读取到：

```text
FAST_LLM_MODEL = deepseek-v4-flash
ROUTER_LLM_MODEL = deepseek-v4-flash
DEEPSEEK_BASE_URL = https://api.deepseek.com/v1
DEEPSEEK_API_KEY_CONFIGURED = True
ROUTER_LLM_ENABLED = True
AGENT_LLM_ROUTER_ENABLED = True
```

`scripts/run_web_app.py` 中：

- `ROUTER_LLM_MODEL` 默认等于 `FAST_LLM_MODEL`：`scripts/run_web_app.py:48`
- 普通路由 LLM 是否启用：`scripts/run_web_app.py:49`
- Agent 语义路由 LLM 是否启用：`scripts/run_web_app.py:50`

### 2.2 检索模型配置

检索相关模型和路径在 `src/config.py`：

- BGE-M3 向量模型：`src/config.py:73-75`
- BGE reranker：`src/config.py:75`
- 检索 Milvus/evidence store 路径：`src/config.py:64-68`

当前 Web 端通过 `get_retriever()` 懒加载 `OnlineHybridRetriever`：

- retriever 全局缓存和锁：`scripts/run_web_app.py:58-61`
- 懒加载逻辑：`scripts/run_web_app.py:87-96`

### 2.3 图像生成配置

ComfyUI 配置在：

- 生成图片目录：`src/config.py:57`
- ComfyUI 服务地址：`src/config.py:98`
- ComfyUI workflow：`src/config.py:99`
- Web 端挂载 `/generated-images`：`scripts/run_web_app.py:55-56`

## 3. LangGraph 旧图：节点、上下文、模型、流转

LangGraph 图定义在 `src/agent/graph.py`。

### 3.1 LangGraph State

LangGraph 的 `AgentState` 在 `src/agent/state.py`：

| 字段 | 作用 | 代码 |
|---|---|---|
| `messages` | 对话消息流，使用 `add_messages` 合并 | `src/agent/state.py:8-9` |
| `next_node` | supervisor 决定下一步去哪 | `src/agent/state.py:11-12` |
| `sender` | 上一个发送者，用于防循环 | `src/agent/state.py:13-14` |
| `research_dossier` | researcher 给 artist 的隔离卷宗 | `src/agent/state.py:15-16` |
| `user_id` | 用户标识 | `src/agent/state.py:18-19` |
| `long_term_memory` | 长期记忆文本 | `src/agent/state.py:20` |
| `summary` | 滚动摘要 | `src/agent/state.py:21` |

### 3.2 LangGraph 图注册

节点注册：

- `gateway`：`src/agent/graph.py:18`
- `summarizer`：`src/agent/graph.py:19`
- `supervisor`：`src/agent/graph.py:20`
- `researcher`：`src/agent/graph.py:21`
- `artist`：`src/agent/graph.py:22`
- `chatter`：`src/agent/graph.py:23`

图编译和 checkpoint：

- 创建 `StateGraph(AgentState)`：`src/agent/graph.py:14-15`
- SQLite checkpoint：`src/agent/graph.py:87-90`
- 编译图：`src/agent/graph.py:93`

Checkpoint 数据库路径来自：

- `AGENT_CHECKPOINT_DB`：`src/config.py:70`

### 3.3 `gateway` 节点

实现位置：

- 注册：`src/agent/graph.py:18`
- 函数：`src/agent/main_nodes.py:213-215`

作用：

- 它是一个空转锚点，不调用 LLM，不调用工具。
- 用于统一进入后再根据消息长度分流。

能看到的信息：

- 理论上能拿到完整 `AgentState`，但当前实现直接 `return {}`，不读取任何字段。

是否搭载大模型：

- 没有。

可流转到：

- `summarizer`
- `supervisor`

触发条件：

- `len(state["messages"]) > 10` → `summarizer`
- 否则 → `supervisor`

代码位置：

- 条件函数：`src/agent/graph.py:32-39`
- 条件边：`src/agent/graph.py:41-46`

### 3.4 `summarizer` 节点

实现位置：

- 注册：`src/agent/graph.py:19`
- 函数：`src/agent/main_nodes.py:123-159`

作用：

- 对过长对话做滚动摘要。
- 从最近对话中提取用户偏好、反馈、上下文记忆。
- 必要时写入长期记忆数据库。

能看到的信息：

- `state["messages"]` 的最近 6 条：`src/agent/main_nodes.py:131-139`
- 原有 `summary`：`src/agent/main_nodes.py:128-139`
- `user_id`：`src/agent/main_nodes.py:129`

是否搭载大模型：

- 有，使用 `llm_fast`。
- `llm_fast` 是 `ChatOpenAI(model=FAST_LLM_MODEL, base_url=DEEPSEEK_BASE_URL)`：`src/agent/main_nodes.py:20-28`
- 当前运行模型是 `deepseek-v4-flash`。

Prompt：

- `summarizer_prompt.yaml`，读取逻辑在 `src/agent/main_nodes.py:34-42`
- prompt 内容：`src/agent/prompts/summarizer_prompt.yaml:1-24`

输出：

- 成功时返回 `summary` 和 `sender="summarizer"`：`src/agent/main_nodes.py:147-156`
- 失败时只返回 `sender="summarizer"`：`src/agent/main_nodes.py:157-159`

可流转到：

- 只能到 `supervisor`

代码位置：

- `summarizer -> supervisor`：`src/agent/graph.py:48-49`

### 3.5 `supervisor` 节点

实现位置：

- 注册：`src/agent/graph.py:20`
- 函数：`src/agent/main_nodes.py:88-120`

作用：

- 中央主管。
- 根据最近消息、历史摘要、长期记忆、上一个节点 `sender` 决定下一步去哪个 worker。
- 输出 JSON，关键字段是 `next_node`。

能看到的信息：

- `summary`：`src/agent/main_nodes.py:96`
- `long_term_memory`：`src/agent/main_nodes.py:97`
- `sender`：`src/agent/main_nodes.py:98`
- 最近 4 条 `messages`：`src/agent/main_nodes.py:108-109`

防循环机制：

- 如果 `last_sender != "user"`，会注入“节点刚刚完成任务，请判断是否 finish”的警报上下文：`src/agent/main_nodes.py:100-106`

是否搭载大模型：

- 有，使用 `llm_fast`。
- 调用时要求 JSON 输出：`src/agent/main_nodes.py:111`

Prompt：

- `supervisor_prompt.yaml`
- prompt 明确三个下属：`researcher`、`artist`、`chatter`：`src/agent/prompts/supervisor_prompt.yaml:5-8`
- 复杂任务先 researcher，再 artist：`src/agent/prompts/supervisor_prompt.yaml:10-12`
- JSON 输出格式：`src/agent/prompts/supervisor_prompt.yaml:15-20`

输出：

- `next_node`
- `sender="supervisor"`

代码位置：

- JSON 解析：`src/agent/main_nodes.py:111-117`
- 返回：`src/agent/main_nodes.py:119-120`

可流转到：

- `researcher`
- `artist`
- `chatter`
- `END`

触发条件：

- 由 LLM 输出的 `next_node` 决定。
- `next_node == "finish"` → `END`
- 其他值返回给条件边。

代码位置：

- 路由函数：`src/agent/graph.py:51-58`
- 条件边映射：`src/agent/graph.py:60-70`

### 3.6 `researcher` 节点

实现位置：

- 注册：`src/agent/graph.py:21`
- 函数：`src/agent/main_nodes.py:162-180`

作用：

- 学术研究员。
- 调用检索工具查山水画文献。
- 生成研究回答。
- 把最终文本写入 `research_dossier`，供 `artist` 使用。

能看到的信息：

- `long_term_memory`：`src/agent/main_nodes.py:165`
- 完整 `state["messages"]`：`src/agent/main_nodes.py:167`

是否搭载大模型：

- 有，使用 `simple_researcher_llm = llm_fast.bind_tools(researcher_tools)`：`src/agent/main_nodes.py:30-32`
- ReAct 循环在 `execute_react_loop()` 中执行：`src/agent/main_nodes.py:48-82`

工具：

- `search_landscape_literature`：`src/agent/tools.py:34-62`
- researcher 工具白名单：`src/agent/tools.py:148`
- 工具通过 `RETRIEVAL_SERVICE_URL/retrieve` 请求检索服务：`src/agent/tools.py:40-42`
- `RETRIEVAL_SERVICE_URL` 配置：`src/config.py:88-96`

Prompt：

- `researcher_prompt.yaml`
- 要求使用工具检索画院资料库：`src/agent/prompts/researcher_prompt.yaml:1-5`
- 要求事实溯源、引用来源：`src/agent/prompts/researcher_prompt.yaml:7-11`
- 错误前提核验规则：`src/agent/prompts/researcher_prompt.yaml:13-21`

输出：

- 新增消息 `messages`
- `sender="researcher"`
- `research_dossier=final_text`

代码位置：

- 设置 name：`src/agent/main_nodes.py:170-171`
- 写入 `research_dossier`：`src/agent/main_nodes.py:173-180`

可流转到：

- 只能回到 `gateway`

代码位置：

- `researcher -> gateway`：`src/agent/graph.py:81`

### 3.7 `artist` 节点

实现位置：

- 注册：`src/agent/graph.py:22`
- 函数：`src/agent/main_nodes.py:183-202`

作用：

- 美术指导。
- 将用户需求或 researcher 的研究卷宗转成英文生图 prompt。
- 调用 ComfyUI 工具生成图像。

能看到的信息：

- `research_dossier`：`src/agent/main_nodes.py:187-188`
- `long_term_memory`：`src/agent/main_nodes.py:189`
- 用户最后一句请求：`src/agent/main_nodes.py:194-195`

重要设计：

- artist 不看完整对话历史，只看研究卷宗、视觉偏好、最后一句请求。  
  代码位置：`src/agent/main_nodes.py:191-198`

是否搭载大模型：

- 有，使用 `artist_llm = llm_fast.bind_tools(artist_tools)`：`src/agent/main_nodes.py:30-32`
- ReAct 循环 max_steps=3：`src/agent/main_nodes.py:198`

工具：

- `generate_landscape_image`：`src/agent/tools.py:64-111`
- artist 工具白名单：`src/agent/tools.py:149`
- 工具会读取 ComfyUI workflow 并写入 prompt、宽高、seed：`src/agent/tools.py:72-80`
- 提交 ComfyUI `/prompt`：`src/agent/tools.py:82-88`
- 轮询 `/history/{prompt_id}` 并下载图片：`src/agent/tools.py:90-106`

Prompt：

- `artist_prompt.yaml`
- 要求必须调用 `generate_landscape_image`：`src/agent/prompts/artist_prompt.yaml:5-7`
- 分辨率策略：`src/agent/prompts/artist_prompt.yaml:8-11`
- ComfyUI 离线时不能伪造结果：`src/agent/prompts/artist_prompt.yaml:14-17`

输出：

- 新增消息
- `sender="artist"`

代码位置：

- 返回：`src/agent/main_nodes.py:200-202`

可流转到：

- 只能到 `END`

代码位置：

- `artist -> END`：`src/agent/graph.py:82`

### 3.8 `chatter` 节点

实现位置：

- 注册：`src/agent/graph.py:23`
- 函数：`src/agent/main_nodes.py:204-211`

作用：

- 闲聊和通用接待。
- 不调用工具。

能看到的信息：

- 完整 `state["messages"]`：`src/agent/main_nodes.py:209`

是否搭载大模型：

- 有，使用 `llm_fast`：`src/agent/main_nodes.py:209`

Prompt：

- `chatter_prompt.yaml`
- 负责日常问候、无关百科、身份介绍：`src/agent/prompts/chatter_prompt.yaml:1-9`

输出：

- 新增消息
- `sender="chatter"`

代码位置：

- 设置 name 并返回：`src/agent/main_nodes.py:209-211`

可流转到：

- 只能到 `END`

代码位置：

- `chatter -> END`：`src/agent/graph.py:83`

## 4. LangGraph 流转总图

代码流转：

```text
START
  -> gateway
      -> summarizer  当 len(messages) > 10
      -> supervisor  当 len(messages) <= 10
  -> summarizer
      -> supervisor
  -> supervisor
      -> researcher  当 next_node == "researcher"
      -> artist      当 next_node == "artist"
      -> chatter     当 next_node == "chatter"
      -> END         当 next_node == "finish"
  -> researcher
      -> gateway
  -> artist
      -> END
  -> chatter
      -> END
```

对应代码：

- `START -> gateway`：`src/agent/graph.py:29-30`
- `gateway -> summarizer/supervisor`：`src/agent/graph.py:32-46`
- `summarizer -> supervisor`：`src/agent/graph.py:48-49`
- `supervisor -> researcher/artist/chatter/END`：`src/agent/graph.py:51-70`
- `researcher -> gateway`：`src/agent/graph.py:81`
- `artist -> END`：`src/agent/graph.py:82`
- `chatter -> END`：`src/agent/graph.py:83`

## 5. 当前 Web 受控 Agent：节点、上下文、模型、流转

当前 Web Agent 节点名定义在：

- `scripts/run_web_app.py:294-305`

这套节点不是 LangGraph 节点，而是 `stream_agent_answer()` 中手写的阶段式执行。

主执行函数：

- `scripts/run_web_app.py:1117-1264`

### 5.1 Web Agent 的请求状态

后端请求结构：

- `message`：用户当前输入
- `history`：前端传来的历史对话
- `top_k`
- `final_k`
- `user_id`

代码位置：`scripts/run_web_app.py:73-78`

前端状态：

- `history`
- `evidence`
- `citedRanks`
- `agentTrace`
- `imageArtifacts`
- `imageSpec`
- PDF 预览、语料库筛选等 UI 状态

代码位置：`ui/modern/app.js:1-21`

前端发送请求前会把当前用户消息加入 `state.history`：

- `ui/modern/app.js:509`

请求体包含：

- `message`
- `history`
- `final_k`

代码位置：`ui/modern/app.js:523-527`

### 5.2 Web Agent 的运行时整体 state

严格说，当前 Web Agent 没有一个像 LangGraph `AgentState` 那样统一声明的 TypedDict。它的 state 是 `stream_agent_answer()` 中逐步产生和消费的一组局部变量，再通过 NDJSON 事件发给前端。

后端运行时 state：

| 字段/变量 | 产生位置 | 消费位置 | 作用 |
|---|---|---|---|
| `question` | `scripts/run_web_app.py:1119` | 后续所有节点 | 当前用户输入 |
| `intake` | `scripts/run_web_app.py:1122` | `planner`、`verifier`、`research_synthesizer`、分支判断 | 任务类型、路由理由、实体、是否检索、是否生图、错误前提 |
| `plan` | `scripts/run_web_app.py:1127` | 前端 Agent trace | 本轮将执行的节点序列 |
| `task_type` | `scripts/run_web_app.py:1132` | `scripts/run_web_app.py:1133-1174`、`scripts/run_web_app.py:1215` | 决定走哪类流程 |
| `evidence` | 初始化 `scripts/run_web_app.py:1131`；检索后更新 `scripts/run_web_app.py:1178-1180` | `verifier`、`final_writer`、`research_synthesizer`、来源栏 | RAG 返回的证据列表 |
| `verifier` | `scripts/run_web_app.py:1185-1198` | `scripts/run_web_app.py:1201-1215` | 判断是否能继续回答/生图 |
| `brief` | `scripts/run_web_app.py:1226` | `prompt_designer`、图像最终回复 | 研究卷宗和视觉约束 |
| `image_spec` | `scripts/run_web_app.py:1232` | `image_generator`、`image_critic`、图像最终回复 | 生图 prompt、尺寸、负面词、风格说明 |
| `image_result` | `scripts/run_web_app.py:1238` | `image_critic`、图像最终回复、前端图片展示 | ComfyUI 生成结果 |
| `critic` | `scripts/run_web_app.py:1245` | 图像最终回复 | 图像文件和 prompt 规则检查结果 |
| `final_answer` | `scripts/run_web_app.py:1252` | `stream_final_text()` | 研究创作任务的最终交付文本 |
| `memory_result` | `scripts/run_web_app.py:1257` | 前端事件和节点状态 | 记忆写入结果 |

后端返回给前端的事件 state：

- `phase`：阶段名，如理解任务、检索中、核验证据、生成回答。代码位置：`scripts/run_web_app.py:1120`、`scripts/run_web_app.py:1175`、`scripts/run_web_app.py:1183`、`scripts/run_web_app.py:1216`、`scripts/run_web_app.py:1224`、`scripts/run_web_app.py:1230`、`scripts/run_web_app.py:1236`、`scripts/run_web_app.py:1243`、`scripts/run_web_app.py:1250`
- `node`：节点运行状态，由 `node_event()` 包装。典型发送位置：`scripts/run_web_app.py:1121-1124`、`scripts/run_web_app.py:1126-1129`、`scripts/run_web_app.py:1176-1181`、`scripts/run_web_app.py:1184-1199`
- `plan`：计划步骤。代码位置：`scripts/run_web_app.py:1128`
- `evidence`：证据列表。代码位置：`scripts/run_web_app.py:1180`
- `brief`：研究卷宗。代码位置：`scripts/run_web_app.py:1227`
- `image_spec`：图像 prompt 规格。代码位置：`scripts/run_web_app.py:1233`
- `image`：图像生成结果。代码位置：`scripts/run_web_app.py:1239`
- `image_critic`：图像检查结果。代码位置：`scripts/run_web_app.py:1247`
- `delta`：流式回答片段。代码位置：`scripts/run_web_app.py:1218-1219`
- `memory`：记忆写入结果。代码位置：`scripts/run_web_app.py:1259`
- `done`：本轮结束和模式。代码位置：`scripts/run_web_app.py:1138`、`scripts/run_web_app.py:1146`、`scripts/run_web_app.py:1154`、`scripts/run_web_app.py:1162`、`scripts/run_web_app.py:1171`、`scripts/run_web_app.py:1212`、`scripts/run_web_app.py:1221`、`scripts/run_web_app.py:1261`

前端持有的 UI state：

| 字段 | 作用 | 代码 |
|---|---|---|
| `history` | 当前浏览器会话内的短期对话历史 | `ui/modern/app.js:1-3` |
| `evidence` | 当前回答的证据列表 | `ui/modern/app.js:3` |
| `citedRanks` | 当前回答正文中出现过的引用编号 | `ui/modern/app.js:4` |
| `health` | 后端健康状态 | `ui/modern/app.js:5` |
| `evidenceOpen` | 证据抽屉是否展开 | `ui/modern/app.js:6` |
| `pdfPreview`、`pdfZoom`、`pdfFit` | PDF 页图预览状态 | `ui/modern/app.js:7-9` |
| `corpusDocs`、`selectedCorpusDoc`、`corpusFilters` | 语料库列表和筛选状态 | `ui/modern/app.js:10-20` |
| `agentTrace` | 前端展示的 Agent 节点运行轨迹 | `ui/modern/app.js:12` |
| `imageArtifacts`、`imageSpec` | 图像生成结果和 prompt 规格 | `ui/modern/app.js:13-14` |

前端会在发送消息时清空本轮证据、Agent trace、图像结果和 prompt 状态：`ui/modern/app.js:497-509`。它逐行读取 NDJSON 流：`ui/modern/app.js:480-495`，再按事件类型更新 state：`ui/modern/app.js:534-581`。回答完成后把 assistant 消息写回 `history`：`ui/modern/app.js:583`。

### 5.3 `intake`：任务理解

代码位置：

- 节点标题：`scripts/run_web_app.py:294-305`
- LLM 语义路由：`scripts/run_web_app.py:441-522`
- 规则兜底：`scripts/run_web_app.py:525-564`
- 总入口：`scripts/run_web_app.py:567-601`
- 执行点：`scripts/run_web_app.py:1119-1124`

作用：

- 判断任务类型。
- 判断是否需要检索。
- 判断是否需要图像生成。
- 抽取朝代、流派、技法、画家等实体。
- 识别明显错误前提。
- 结合最近对话理解省略和指代。

能看到的信息：

- 当前问题 `question`
- `req.history`
- 规则词表：`STRONG_DOMAIN_KEYWORDS`、`AMBIGUOUS_DOMAIN_KEYWORDS`、`CASUAL_PATTERNS`
- 最近对话摘要 `compact_history(history)`

代码位置：

- 领域词表：`scripts/run_web_app.py:140-155`
- 闲聊正则：`scripts/run_web_app.py:157-162`
- 图像意图正则：`scripts/run_web_app.py:307-312`
- 最近历史压缩：`scripts/run_web_app.py:422-429`
- 上下文续问保护：`scripts/run_web_app.py:432-438`、`scripts/run_web_app.py:578-587`

是否搭载大模型：

- 有，优先使用 `deepseek-v4-flash` 做语义路由。
- 调用位置：`scripts/run_web_app.py:441-486`
- 模型名来自 `ROUTER_LLM_MODEL`：`scripts/run_web_app.py:48`

没有 LLM 时：

- 使用规则兜底 `build_rule_agent_intake()`：`scripts/run_web_app.py:525-564`

硬规则：

- 现代生成技术和古代画史混用的错误前提，先硬拦截：`scripts/run_web_app.py:379-407`、`scripts/run_web_app.py:570-576`
- 一般中国绘画史作品兜底：`scripts/run_web_app.py:322-325`、`scripts/run_web_app.py:529-540`

可流转到：

- 总是先到 `planner`

代码位置：

- `intake` 后马上进入 planner：`scripts/run_web_app.py:1126-1129`

### 5.4 `planner`：计划制定

代码位置：

- 函数：`scripts/run_web_app.py:604-627`
- 执行点：`scripts/run_web_app.py:1126-1129`

作用：

- 根据 `task_type` 生成本轮节点计划。
- 计划会作为 `plan` 事件发给前端。

能看到的信息：

- `intake` 产出的结构化字段，包括 `task_type`、`needs_retrieval`、`needs_image`、`entities`。

是否搭载大模型：

- 没有，纯规则。

可流转到：

- `direct` / `general_art_qa` → `final_writer`
- `unsupported_image` / `need_clarification` / `unsupported_general` / `invalid_premise` → `verifier` → `final_writer`
- `research_qa` → `researcher` → `verifier` → `final_writer`
- `research_then_image` → `researcher` → `verifier` → `research_synthesizer` → `prompt_designer` → `image_generator` → `image_critic` → `final_writer` → `memory_writer`

代码位置：

- 直接类任务：`scripts/run_web_app.py:606-607`
- 边界/错误前提类任务：`scripts/run_web_app.py:608-609`
- 研究问答：`scripts/run_web_app.py:610-615`
- 研究创作：`scripts/run_web_app.py:616-626`

### 5.5 `researcher`：文献检索

代码位置：

- 执行点：`scripts/run_web_app.py:1174-1181`
- retriever 懒加载：`scripts/run_web_app.py:87-96`
- evidence payload：`scripts/run_web_app.py:925-946`

作用：

- 调用当前 RAG 检索系统。
- 从 Milvus/evidence store 检索并重排。
- 返回结构化证据给前端和后续节点。

能看到的信息：

- 当前问题 `question`
- `top_k`、`final_k`
- 检索系统返回的 chunk 字段，包括 source_file、page_start、rerank_score、raw_chunk_text/contextual_chunk 等。

是否搭载大模型：

- researcher 节点本身不调用回答 LLM。
- 它依赖本地检索模型和 reranker。模型路径见 `src/config.py:73-75`。

可流转到：

- `verifier`

触发条件：

- `intake["needs_retrieval"] == True`

代码位置：

- `scripts/run_web_app.py:1174`

### 5.6 `verifier`：证据核验

代码位置：

- 相关性判断：`scripts/run_web_app.py:276-283`
- 错误前提识别：`scripts/run_web_app.py:379-407`
- 执行点：`scripts/run_web_app.py:1183-1199`

作用：

- 检查错误前提。
- 检查证据是否足够相关。
- 决定是否继续生成回答/图像，还是转入证据不足回复。

能看到的信息：

- `intake`
- `evidence`
- 最高 rerank 证据分数，但前端不再显示这个分数。

是否搭载大模型：

- 没有，当前是规则判断。

可流转到：

- 如果 `can_continue=False` → `final_writer`
- 如果 `task_type == research_qa` 且通过 → `final_writer`
- 如果 `task_type == research_then_image` 且通过 → `research_synthesizer`

代码位置：

- 不可继续：`scripts/run_web_app.py:1201-1213`
- 研究问答：`scripts/run_web_app.py:1215-1222`
- 研究创作继续：`scripts/run_web_app.py:1224-1228`

### 5.7 `research_synthesizer`：研究卷宗

代码位置：

- fallback 规则版：`scripts/run_web_app.py:630-654`
- LLM 版：`scripts/run_web_app.py:657-692`
- 执行点：`scripts/run_web_app.py:1224-1228`

作用：

- 把检索证据整理成图像创作可用的研究卷宗。
- 输出 topic、key_points、visual_constraints、citations。

能看到的信息：

- 当前问题
- 检索证据
- intake 中抽取的实体

是否搭载大模型：

- 有。若 `DEEPSEEK_API_KEY` 存在且 evidence 不为空，会调用 `FAST_LLM_MODEL`。
- 当前模型是 `deepseek-v4-flash`。
- 调用位置：`scripts/run_web_app.py:661-682`

没有 LLM 时：

- 使用规则 fallback：`scripts/run_web_app.py:630-654`

可流转到：

- `prompt_designer`

代码位置：

- `scripts/run_web_app.py:1230-1234`

### 5.8 `prompt_designer`：图像提示词设计

代码位置：

- fallback 规则版：`scripts/run_web_app.py:695-718`
- LLM 版：`scripts/run_web_app.py:721-757`
- 执行点：`scripts/run_web_app.py:1230-1234`

作用：

- 将研究卷宗转成 ComfyUI/Flux 可用的图像参数。
- 输出 `format`、`width`、`height`、`positive_prompt`、`negative_prompt`、`style_notes`。

能看到的信息：

- 当前问题
- `research_synthesizer` 生成的研究卷宗

是否搭载大模型：

- 有。若有 API key，会调用 `FAST_LLM_MODEL`。
- 当前模型是 `deepseek-v4-flash`。
- 调用位置：`scripts/run_web_app.py:725-743`

没有 LLM 时：

- 使用规则 fallback：`scripts/run_web_app.py:695-718`

可流转到：

- `image_generator`

代码位置：

- `scripts/run_web_app.py:1236-1241`

### 5.9 `image_generator`：图像生成

代码位置：

- 函数：`scripts/run_web_app.py:764-832`
- 执行点：`scripts/run_web_app.py:1236-1241`

作用：

- 检查 ComfyUI 是否在线。
- 读取 workflow。
- 写入 prompt、宽高、seed。
- 调用 ComfyUI `/prompt`。
- 轮询 `/history/{prompt_id}`。
- 下载图像到 `generated_images/`。
- 返回前端可访问 URL。

能看到的信息：

- `image_spec`
- `COMFYUI_SERVER_URL`
- `COMFYUI_WORKFLOW_PATH`
- `GENERATED_IMAGES_DIR`

是否搭载大模型：

- 没有。
- 它是工具调用节点，调用 ComfyUI。

可流转到：

- `image_critic`

失败处理：

- ComfyUI 不在线 → `comfyui_offline`：`scripts/run_web_app.py:767-776`
- workflow 错误 → `workflow_error`：`scripts/run_web_app.py:777-789`
- ComfyUI 调用异常 → `comfyui_error`：`scripts/run_web_app.py:790-831`
- 超时 → `timeout`：`scripts/run_web_app.py:832`

### 5.10 `image_critic`：图像检查

代码位置：

- 函数：`scripts/run_web_app.py:835-850`
- 执行点：`scripts/run_web_app.py:1243-1248`

作用：

- 检查图像是否生成成功。
- 检查文件是否存在且非空。
- 检查 prompt 中是否有基本关键词 `chinese`、`landscape`、`ink`。

能看到的信息：

- `image_result`
- `image_spec`
- 图像文件路径

是否搭载大模型：

- 没有。
- 当前不是视觉模型检查，只是文件和 prompt 级规则检查。

可流转到：

- `final_writer`

代码位置：

- `scripts/run_web_app.py:1250-1254`

### 5.11 `final_writer`：最终回复

代码位置：

- 研究问答 LLM 构造 prompt：`scripts/run_web_app.py:965-991`
- 非流式生成：`scripts/run_web_app.py:1017-1031`
- 流式生成：`scripts/run_web_app.py:1034-1054`
- 错误前提回复：`scripts/run_web_app.py:1066-1072`
- 一般绘画史回答：`scripts/run_web_app.py:1075-1100`
- 图像任务最终模板：`scripts/run_web_app.py:866-895`
- 主流程执行点：`scripts/run_web_app.py:1133-1147`、`scripts/run_web_app.py:1149-1172`、`scripts/run_web_app.py:1201-1222`、`scripts/run_web_app.py:1250-1254`

作用：

- 根据任务类型输出最终回答。
- 对 research_qa 使用证据增强回答。
- 对 research_then_image 输出研究依据、创作约束、Prompt、图像结果或失败原因。
- 对边界问题、错误前提、证据不足做自然语言说明。

能看到的信息：

- 当前问题
- 历史对话
- 证据
- verifier 结果
- brief
- image_spec
- image_result
- critic

是否搭载大模型：

- `research_qa`：使用 `FAST_LLM_MODEL`，当前是 `deepseek-v4-flash`。代码：`scripts/run_web_app.py:1034-1054`
- `general_art_qa`：优先有规则特判《清明上河图》，否则调用 `FAST_LLM_MODEL`。代码：`scripts/run_web_app.py:1075-1100`
- `research_then_image` 的最终组装是规则模板，不调用 LLM。代码：`scripts/run_web_app.py:866-895`
- 错误前提、证据不足、unsupported 等回复是规则文本。

可流转到：

- 对大多数任务：`END`
- 对 `research_then_image`：继续到 `memory_writer`

代码位置：

- `research_then_image` 中 `final_writer -> memory_writer`：`scripts/run_web_app.py:1256-1261`

### 5.12 `memory_writer`：记忆写入

代码位置：

- 偏好提取：`scripts/run_web_app.py:898-909`
- 写入入口：`scripts/run_web_app.py:912-922`
- 执行点：`scripts/run_web_app.py:1256-1260`

作用：

- 只记录用户明确表达的偏好或反馈。
- 当前只识别：
  - `我喜欢...`
  - `偏好...`
  - `以后不要...`
  - `不喜欢...`

能看到的信息：

- 当前用户问题
- `user_id`

是否搭载大模型：

- 没有，正则规则。

重要限制：

- 前端当前请求体没有传 `user_id`，只传 `message`、`history`、`final_k`：`ui/modern/app.js:523-527`
- 后端 `user_id` 默认是 `guest`：`scripts/run_web_app.py:78`
- 因此当前前端默认不会持久写入长期记忆。代码中明确 `user_id != "guest"` 才写入：`scripts/run_web_app.py:914`

可流转到：

- `END`

代码位置：

- `scripts/run_web_app.py:1260-1261`

## 6. 当前 Web 任务类型与流程

任务类型定义：

- `scripts/run_web_app.py:410-419`

### 6.1 `direct`

含义：

- 寒暄、天气、日常闲聊。

流程：

```text
intake -> planner -> final_writer -> END
```

代码：

- 计划：`scripts/run_web_app.py:606-607`
- 执行：`scripts/run_web_app.py:1133-1139`

### 6.2 `general_art_qa`

含义：

- 一般中国绘画史事实问题，不强行启动山水画 RAG。
- 例如“清明上河图是谁画的”。

流程：

```text
intake -> planner -> final_writer -> END
```

代码：

- 常见作品兜底：`scripts/run_web_app.py:322-325`、`scripts/run_web_app.py:529-540`
- 计划：`scripts/run_web_app.py:606-607`
- 执行：`scripts/run_web_app.py:1141-1147`
- 回答函数：`scripts/run_web_app.py:1075-1100`

### 6.3 `research_qa`

含义：

- 中国山水画史、画论、画家、流派、技法、作品研究问答。

流程：

```text
intake -> planner -> researcher -> verifier -> final_writer -> END
```

代码：

- 计划：`scripts/run_web_app.py:610-615`
- 检索：`scripts/run_web_app.py:1174-1181`
- 核验：`scripts/run_web_app.py:1183-1199`
- 回答：`scripts/run_web_app.py:1215-1222`

### 6.4 `research_then_image`

含义：

- 需要先查资料，再生成图像。
- 例如“来搞个宋代山水图生成下看看”“能画一副类似的画吗？”

流程：

```text
intake
 -> planner
 -> researcher
 -> verifier
 -> research_synthesizer
 -> prompt_designer
 -> image_generator
 -> image_critic
 -> final_writer
 -> memory_writer
 -> END
```

代码：

- 计划：`scripts/run_web_app.py:616-626`
- 检索：`scripts/run_web_app.py:1174-1181`
- 核验：`scripts/run_web_app.py:1183-1199`
- 卷宗：`scripts/run_web_app.py:1224-1228`
- Prompt：`scripts/run_web_app.py:1230-1234`
- 生图：`scripts/run_web_app.py:1236-1241`
- 图像检查：`scripts/run_web_app.py:1243-1248`
- 最终回复：`scripts/run_web_app.py:1250-1254`
- 记忆写入：`scripts/run_web_app.py:1256-1261`

### 6.5 `invalid_premise`

含义：

- 用户问题包含明显错误前提。
- 例如“清代四王如何使用 Stable Diffusion 生成山水画”。

流程：

```text
intake -> planner -> verifier -> final_writer -> END
```

代码：

- 错误前提识别：`scripts/run_web_app.py:379-407`
- 硬拦截：`scripts/run_web_app.py:570-576`
- 执行：`scripts/run_web_app.py:1165-1172`

### 6.6 `unsupported_image`

含义：

- 用户要生成图像，但不是中国山水画相关创作任务。

流程：

```text
intake -> planner -> verifier -> final_writer -> END
```

代码：

- 计划：`scripts/run_web_app.py:608-609`
- 执行：`scripts/run_web_app.py:1149-1155`

### 6.7 `need_clarification`

含义：

- 可能相关，但问题过泛，需要用户补充山水画角度。

流程：

```text
intake -> planner -> verifier -> final_writer -> END
```

代码：

- 普通路由：`scripts/run_web_app.py:240-244`
- 执行：`scripts/run_web_app.py:1157-1163`

### 6.8 `unsupported_general`

含义：

- 完全不相关的一般问题。

流程：

```text
intake -> planner -> verifier -> final_writer -> END
```

代码：

- 普通路由兜底：`scripts/run_web_app.py:245-248`
- 执行：`scripts/run_web_app.py:1157-1163`

## 7. 记忆系统复盘

### 7.1 LangGraph 记忆

LangGraph 有两层记忆：

1. **Checkpoint 记忆**：由 LangGraph `SqliteSaver` 保存执行状态。  
   代码：`src/agent/graph.py:87-93`

2. **长期用户画像记忆**：由 `MemoryManager` 存储到 SQLite。  
   代码：`src/agent/memory/memory_manager.py:11-97`

命令行入口中：

- 读取用户 ID：`scripts/run_chat_agent.py:44`
- 读取长期记忆：`scripts/run_chat_agent.py:48`
- 作为 `long_term_memory` 传入 graph：`scripts/run_chat_agent.py:73-77`
- 每轮后重新读取记忆：`scripts/run_chat_agent.py:115`

### 7.2 MemoryManager 的 SQLite 表

表结构：

```text
user_memories(
  user_id TEXT PRIMARY KEY,
  content TEXT,
  updated_at TEXT
)
```

代码位置：

- 初始化表：`src/agent/memory/memory_manager.py:13-27`
- 读取记忆：`src/agent/memory/memory_manager.py:29-51`
- 保存记忆：`src/agent/memory/memory_manager.py:53-97`

记忆内容格式：

```json
{
  "preferences": [],
  "feedback": [],
  "context": []
}
```

代码位置：

- 默认返回结构：`src/agent/memory/memory_manager.py:35-51`

保存策略：

- `preferences`、`feedback` 合并去重：`src/agent/memory/memory_manager.py:64-69`
- `context` 增加绝对时间戳：`src/agent/memory/memory_manager.py:71-80`
- 只保留最近 10 条 context：`src/agent/memory/memory_manager.py:79-80`

### 7.3 当前 Web 记忆

当前 Web 主链路没有启用 LangGraph checkpoint。

它有：

- 前端短期历史：`ui/modern/app.js:1-3`
- 后端请求历史字段：`scripts/run_web_app.py:73-78`
- Agent 中的可选长期记忆写入：`scripts/run_web_app.py:912-922`

但当前前端请求没有传 `user_id`：

- `ui/modern/app.js:523-527`

所以当前默认 `user_id="guest"`：

- `scripts/run_web_app.py:78`

而 `memory_writer` 只有在非 guest 且提取到偏好/反馈时才写入：

- `scripts/run_web_app.py:914-918`

结论：

> 当前 Web 页面主要依赖浏览器内存中的短期 `history`。长期记忆功能有代码，但默认未真正激活，因为前端没有用户 ID。

## 8. 前端如何显示 Agent 状态

前端 HTML：

- 产品名 `ShanshuiAgent`：`ui/modern/index.html:6`
- 首屏标题：`ui/modern/index.html:18-20`
- 工作台标题：`ui/modern/index.html:53-55`
- Agent trace 容器：`ui/modern/index.html:65-67`

前端 JS：

- 前端状态对象：`ui/modern/app.js:1-21`
- 设置计划：`ui/modern/app.js:218-227`
- 更新节点状态：`ui/modern/app.js:229-244`
- 渲染 Agent 工作流：`ui/modern/app.js:246-260`
- 渲染 Prompt 和图片：`ui/modern/app.js:263-299`
- 读取 NDJSON 流：`ui/modern/app.js:480-495`
- 发送 `/api/agent/stream`：`ui/modern/app.js:523-527`
- 处理 `plan`、`node`、`evidence`、`brief`、`image_spec`、`image`、`delta`、`done` 事件：`ui/modern/app.js:534-581`

## 9. 没有显式提到但重要的细节

### 9.1 当前 Web Agent 不是 LangGraph

虽然 UI 上叫 Agent，当前 7861 使用的是 `scripts/run_web_app.py` 中的手写状态机：

- `/api/agent/stream`：`scripts/run_web_app.py:1422-1424`
- 主流程：`scripts/run_web_app.py:1117-1264`

它没有调用 `src/agent/graph.py`。

### 9.2 旧 LangGraph researcher 依赖外部检索服务

旧 LangGraph 的 `search_landscape_literature` 工具调用：

- `RETRIEVAL_SERVICE_URL/retrieve`：`src/agent/tools.py:40-42`

这意味着使用 `scripts/run_chat_agent.py` 时，理论上需要另一个检索服务运行。

当前 Web Agent 不走这个工具，而是进程内懒加载 retriever：

- `scripts/run_web_app.py:87-96`
- `scripts/run_web_app.py:1174-1180`

### 9.3 图像生成有两套实现

旧 LangGraph 工具：

- `src/agent/tools.py:64-111`
- 返回文本路径。

当前 Web Agent：

- `scripts/run_web_app.py:764-832`
- 返回结构化结果，包括 `status`、`url`、`seed`、`workflow`、`prompt_id`。

当前 Web 版本更适合前端展示。

### 9.4 LoRA 存在但未接入当前回答模型

配置中有 researcher LoRA 路径：

- `src/config.py:83-86`

健康接口会报告其是否存在：

- `scripts/run_web_app.py:1280-1281`

但当前回答和路由使用的是 DeepSeek-compatible API 的 `FAST_LLM_MODEL`，不是本地 LoRA。

### 9.5 Neo4j、Milvus、BGE 等属于检索层，不属于 LangGraph 节点

当前 Web Agent 的 `researcher` 节点会触发检索器，检索器内部再用 Milvus、BGE-M3、reranker、Neo4j 等能力。  
从 Agent 编排角度看，它们是 `researcher` 节点下游的工具/子系统，不是 Agent 节点。

### 9.6 受控 Agent 的优势和代价

优势：

- 每个节点输入输出更清楚。
- 前端可显示每个阶段。
- 错误前提、ComfyUI 离线、证据不足等更容易控制。

代价：

- 它不是 LangGraph graph，没有 LangGraph checkpoint、状态回放、图可视化等原生能力。
- 如果未来要统一架构，需要把 `scripts/run_web_app.py` 的 10 个节点迁入新的 LangGraph `StateGraph`。

## 10. 建议

### 10.1 如果目标是“真正 LangGraph Agent”

建议下一步把当前 Web Agent 的 10 个节点迁到 LangGraph：

```text
intake -> planner -> researcher -> verifier -> research_synthesizer
       -> prompt_designer -> image_generator -> image_critic
       -> final_writer -> memory_writer
```

这样可以同时保留：

- 当前 Web Agent 的可控性。
- LangGraph 的 checkpoint、状态管理、可观测性、图式编排。

### 10.2 如果短期目标是稳定产品

可以继续保留当前手写状态机，把它作为生产链路；旧 `src/agent/graph.py` 保留为实验原型，但 README 和报告中要明确：

> 当前前端主链路是 controlled research creation agent，不是 LangGraph graph。
