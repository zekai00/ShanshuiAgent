# LangGraph 版 Web Agent 迁移报告

生成时间：2026-06-01 14:14 CST

## 1. 结论

已经把当前 `scripts/run_web_app.py` 里的 10 个产品化节点迁移成一套新的 LangGraph 图，并且没有覆盖旧版 `src/agent/` Agent。

当前前端 `/api/agent/stream` 默认走新版 LangGraph Web Agent：

- 新图位置：`src/web_agent/graph.py:38-86`
- 新状态定义：`src/web_agent/state.py:10-29`
- 新节点实现：`src/web_agent/nodes.py:17-304`
- 流式适配器：`src/web_agent/streaming.py:23-56`
- Web 接口切换位置：`scripts/run_web_app.py:1453-1467`

旧版受控状态机仍然保留：

- 旧函数仍在：`scripts/run_web_app.py:1117-1266`
- 回退开关：设置 `CL_WEB_AGENT_ENGINE=controlled`、`legacy` 或 `state_machine` 后，`/api/agent/stream` 会走旧函数。代码位置：`scripts/run_web_app.py:1455-1457`

## 2. 能否继续做产品化控制

可以。新版不是把控制权完全交给大模型，而是用 LangGraph 管理节点、状态、边、checkpoint；每个节点内部仍然保留产品化控制。

具体做法：

- 前端需要的 `phase`、`node`、`plan`、`evidence`、`brief`、`image_spec`、`image`、`image_critic`、`delta`、`memory`、`done` 事件都继续通过 NDJSON 输出。
- 节点内部使用 `langgraph.config.get_stream_writer()` 发 custom stream event。代码位置：`src/web_agent/events.py:29-39`、`src/web_agent/events.py:55-66`
- 流式适配器使用 `graph.stream(..., stream_mode="custom", durability="sync")`，把 LangGraph custom event 转为前端已有 NDJSON 协议。代码位置：`src/web_agent/streaming.py:44-56`

也就是说，现在是：

```text
LangGraph 负责：图、节点、边、state、checkpoint、恢复基础设施
产品代码负责：路由策略、证据展示、PDF 引用、生图状态、流式 delta、错误前提处理
```

## 3. 新增文件

### 3.1 `src/web_agent/dependencies.py`

作用：定义依赖注入契约。

代码位置：`src/web_agent/dependencies.py:15-36`

这里没有直接 import FastAPI 脚本，也不直接绑定检索器、LLM、生图器，而是声明需要哪些能力：

- `build_agent_intake`
- `build_agent_plan`
- `get_retriever`
- `evidence_payload`
- `evidence_is_relevant`
- `stream_answer_deltas`
- `synthesize_research_brief`
- `design_image_spec`
- `generate_image_with_comfyui`
- `critic_image_result`
- `maybe_write_memory`

这样图模块不依赖 `scripts/run_web_app.py` 的实现细节，降低耦合。

### 3.2 `src/web_agent/state.py`

作用：定义新版 LangGraph 的整体 state。

代码位置：`src/web_agent/state.py:10-29`

当前 state 字段：

```text
question, history, top_k, final_k, user_id, thread_id,
intake, plan, task_type, evidence, verifier,
brief, image_spec, image_result, critic,
final_answer, memory_result, mode
```

这比旧手写状态机更清楚，因为每个节点读写的字段都落在同一个 `WebAgentState` 中。

### 3.3 `src/web_agent/events.py`

作用：统一前端事件格式。

代码位置：

- 节点标题：`src/web_agent/events.py:11-22`
- NDJSON 行序列化：`src/web_agent/events.py:25-26`
- LangGraph custom event 发射：`src/web_agent/events.py:29-39`
- 节点事件结构：`src/web_agent/events.py:42-56`
- 文本分块流式输出：`src/web_agent/events.py:59-66`

### 3.4 `src/web_agent/nodes.py`

作用：实现 10 个产品节点。

节点对应代码：

- `intake`：`src/web_agent/nodes.py:17-28`
- `planner`：`src/web_agent/nodes.py:31-40`
- `researcher`：`src/web_agent/nodes.py:43-55`
- `verifier`：`src/web_agent/nodes.py:58-105`
- `research_synthesizer`：`src/web_agent/nodes.py:108-120`
- `prompt_designer`：`src/web_agent/nodes.py:123-139`
- `image_generator`：`src/web_agent/nodes.py:142-153`
- `image_critic`：`src/web_agent/nodes.py:156-168`
- `final_writer`：`src/web_agent/nodes.py:171-262`
- `memory_writer`：`src/web_agent/nodes.py:265-277`

条件路由：

- `planner` 后分流：`src/web_agent/nodes.py:280-286`
- `verifier` 后分流：`src/web_agent/nodes.py:289-296`
- `final_writer` 后分流：`src/web_agent/nodes.py:299-304`

### 3.5 `src/web_agent/graph.py`

作用：声明 LangGraph 图。

核心位置：

- 创建 `StateGraph(WebAgentState)`：`src/web_agent/graph.py:38-39`
- 注册 10 个节点：`src/web_agent/graph.py:41-50`
- 声明边和条件边：`src/web_agent/graph.py:52-84`
- 编译图：`src/web_agent/graph.py:86`
- SQLite checkpoint：`src/web_agent/graph.py:89-102`

新图结构：

```text
START
  -> intake
  -> planner
      -> final_writer      direct / general_art_qa
      -> verifier          unsupported_image / need_clarification / unsupported_general / invalid_premise
      -> researcher        research_qa / research_then_image
  -> researcher
      -> verifier
  -> verifier
      -> final_writer
      -> research_synthesizer
  -> research_synthesizer
      -> prompt_designer
  -> prompt_designer
      -> image_generator
  -> image_generator
      -> image_critic
  -> image_critic
      -> final_writer
  -> final_writer
      -> memory_writer     research_then_image 且 verifier 通过
      -> END
  -> memory_writer
      -> END
```

### 3.6 `src/web_agent/streaming.py`

作用：把 LangGraph custom stream 转成前端可读的 NDJSON。

代码位置：

- 生成或接收 `thread_id`：`src/web_agent/streaming.py:15-20`
- 构造初始 state：`src/web_agent/streaming.py:33-42`
- 发出 `agent_run` 元事件：`src/web_agent/streaming.py:44`
- 执行 LangGraph stream：`src/web_agent/streaming.py:46-53`
- 异常转为前端错误事件：`src/web_agent/streaming.py:54-56`

## 4. 对 `scripts/run_web_app.py` 的改动

### 4.1 请求体支持 `thread_id`

新增字段：

```text
thread_id: str | None
```

代码位置：`scripts/run_web_app.py:73-79`

作用：

- 如果前端或外部客户端传入同一个 `thread_id`，LangGraph checkpoint 可以把这次运行归入同一个线程。
- 当前前端没有传，所以后端会自动生成一次性 `web-agent:<user>:<hash>` 线程 ID。

### 4.2 修正 `direct` 路由标签

`ROUTE_LABELS` 增加 `direct`，避免 LLM intake 返回 `direct` 时被 `route_payload()` 降级为 `need_clarification`。

代码位置：`scripts/run_web_app.py:165-176`

### 4.3 注入依赖

新增 `WEB_AGENT_DEPS`，把原本 Web 端已有能力注入给新图。

代码位置：`scripts/run_web_app.py:1269-1292`

这样新图复用原能力，但不直接 import `scripts/run_web_app.py`，避免循环依赖。

### 4.4 `/health` 暴露新版 Agent 模式

新增：

```text
mode = langgraph_web_agent
engine = langgraph
checkpointing = true
legacy_controlled_agent_available = true
```

代码位置：`scripts/run_web_app.py:1320-1325`

### 4.5 `/api/agent/stream` 切换到 LangGraph

默认：

```text
/api/agent/stream -> stream_web_agent_events(...) -> LangGraph web agent
```

代码位置：`scripts/run_web_app.py:1453-1467`

回退：

```bash
CL_WEB_AGENT_ENGINE=controlled python scripts/run_web_app.py --host 127.0.0.1 --port 7861
```

## 5. 可恢复与可追踪

### 5.1 Checkpoint

新增独立 checkpoint 数据库：

- 配置项：`WEB_AGENT_CHECKPOINT_DB`
- 默认路径：`data/runtime/web_agent_checkpoints.sqlite`
- 代码位置：`src/config.py:70-72`

新图使用 `SqliteSaver`：

- 创建 SQLite 连接：`src/web_agent/graph.py:96-99`
- 编译时绑定 checkpointer：`src/web_agent/graph.py:100`

旧版 `src/agent/graph.py` 仍使用原来的 `AGENT_CHECKPOINT_DB`，没有被覆盖。

### 5.2 Trace

前端 trace 不需要重写，因为事件格式保持兼容：

- `node` 事件：节点运行、完成、失败状态
- `plan` 事件：本轮计划
- `phase` 事件：阶段提示
- `delta` 事件：回答流式文本
- `agent_run` 事件：新增加的 LangGraph 运行元信息，前端当前会忽略未知事件，不影响 UI

## 6. 任务类型流程

### 6.1 `direct`

```text
intake -> planner -> final_writer -> END
```

触发：寒暄、天气、闲聊。

### 6.2 `general_art_qa`

```text
intake -> planner -> final_writer -> END
```

触发：一般中国绘画史事实问题，例如《清明上河图》作者。

### 6.3 `research_qa`

```text
intake -> planner -> researcher -> verifier -> final_writer -> END
```

触发：中国山水画史、画论、技法、流派、作品研究问答。

### 6.4 `research_then_image`

```text
intake -> planner -> researcher -> verifier
  -> research_synthesizer -> prompt_designer
  -> image_generator -> image_critic
  -> final_writer -> memory_writer -> END
```

触发：需要先检索文献，再生成山水画图像的任务。

### 6.5 边界类任务

```text
intake -> planner -> verifier -> final_writer -> END
```

包括：

- `unsupported_image`
- `need_clarification`
- `unsupported_general`
- `invalid_premise`

## 7. 验证结果

已执行：

```bash
python -m py_compile src/web_agent/*.py scripts/run_web_app.py src/config.py
node --check ui/modern/app.js
git diff --check
```

均通过。

已重启本地服务：

```text
http://127.0.0.1:7861/
```

HTTP 冒烟测试：

1. `/health` 返回：

```text
agent.mode = langgraph_web_agent
agent.engine = langgraph
agent.checkpointing = true
```

2. `/api/agent/stream` 测试“你好”：

```text
agent_run -> intake -> planner -> final_writer -> done
```

3. `/api/agent/stream` 测试“清明上河图是谁画的”：

```text
agent_run -> intake -> planner -> final_writer -> done
```

4. 本地 stub 测试 `research_then_image` 全链路：

```text
intake -> planner -> researcher -> verifier -> research_synthesizer
-> prompt_designer -> image_generator -> image_critic
-> final_writer -> memory_writer -> done
```

## 8. 当前仍需注意

1. 新版 LangGraph 已经接管 `/api/agent/stream`，但旧 `/api/chat`、`/api/chat/stream` 仍是旧的非 Agent RAG 问答接口。
2. 当前前端没有传 `thread_id`，所以每次请求都会自动生成一次性线程 ID；如果后续要做“断点恢复到同一会话”，前端需要保存并传回 `thread_id`。
3. 当前长期记忆仍然受 `user_id != "guest"` 限制；前端默认不传用户 ID，所以仍以短期浏览器 `history` 为主。
4. 目前 `final_writer` 中的研究问答仍调用原有 `stream_answer_deltas()`，即回答模型仍是 `FAST_LLM_MODEL`，当前运行配置为 `deepseek-v4-flash`。

## 9. 总结

这次迁移后，系统已经从“手写受控状态机作为主链路”升级为“LangGraph 编排 + 产品化事件控制”的主链路。

关键收益：

- 图结构明确；
- state 明确；
- checkpoint 独立；
- 前端事件兼容；
- 旧版 Agent 未删除；
- 可通过环境变量回退；
- 后续可以继续接 LangGraph 的中断、恢复、人工确认、节点重试和更细粒度 tracing。
