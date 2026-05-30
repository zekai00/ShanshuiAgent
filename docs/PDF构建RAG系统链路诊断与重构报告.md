# PDF 构建 RAG 系统链路诊断与重构报告

日期：2026-05-30

## 一句话结论

当前系统的核心检索方向没有错：PDF 入库、Milvus 向量检索、dense/sparse/ColBERT 多路召回、RRF 融合、reranker 精排、Researcher 引用来源，这条主线是合理的。

但从第一性原理看，现在的问题不在“要不要继续做核验”，也不在“先不先训练模型”，而在于：**PDF 到证据块的构建链路还不够可信，证据溯源粒度太粗，指定文献约束没有被检索层强制执行，入库质量也缺少硬门禁**。

所以建议：

1. 先做 RAG 数据链路重构。
2. 再用 reviewed 评测集重新跑检索和回答。
3. 最后根据失败类型决定是否做 SFT/DPO。

不建议现在直接继续做 DPO、PPO、GRPO 或扩大回答模型训练。否则模型会学习到当前 RAG 链路里的错误证据、错误引用和错误前提处理方式。

## 参考资料

本报告参考了最新论文和官方资料，重点看“现代 RAG 系统应该如何组织证据、检索、重排、引用和评测”。

- Lewis 等，Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks，2020：RAG 的基本原则是生成模型要基于外部检索证据，而不是只靠参数记忆。  
  https://arxiv.org/abs/2005.11401
- Karpukhin 等，Dense Passage Retrieval for Open-Domain Question Answering，2020：dense retrieval 适合语义召回，但仍需要高质量 passage 和训练/评测闭环。  
  https://arxiv.org/abs/2004.04906
- Khattab 等，ColBERT，2020：late interaction/multi-vector 检索适合细粒度匹配，但存储与服务复杂度高。  
  https://arxiv.org/abs/2004.12832
- BGE-M3 技术报告，2024：BGE-M3 同时支持 dense、sparse、multi-vector，多语言、多粒度检索是它的设计目标。  
  https://arxiv.org/abs/2402.03216
- Anthropic Contextual Retrieval，2024：chunk 在入库前加入文档级上下文，有助于减少孤立片段造成的检索失败。  
  https://www.anthropic.com/engineering/contextual-retrieval
- Microsoft GraphRAG，2024：图谱适合做全局关系和社区摘要，但不应替代底层原文证据链。  
  https://arxiv.org/abs/2404.16130
- Milvus Hybrid Search 官方文档：Milvus 支持多向量字段、稠密/稀疏混合检索和 ranker 融合。  
  https://milvus.io/docs/hybrid_search_with_milvus.md
- RAGAS 论文，2023：RAG 评测应拆成检索相关性、证据忠实度、答案相关性等维度。  
  https://arxiv.org/abs/2309.15217
- OCR Hinders RAG，2024：OCR/文档解析错误会级联影响检索和生成，PDF 解析质量是 RAG 的上游关键变量。  
  https://arxiv.org/abs/2412.02592
- Docling 技术报告，2024：现代文档解析应保留页面结构、版面、表格、图片等结构化信息，服务后续 RAG。  
  https://arxiv.org/abs/2408.09869

## 当前项目实际链路

项目当前从 PDF 到 Researcher 回答的链路如下。

1. PDF 解析

`scripts/run_ingestion.py` 调用 `DocumentProcessor.extract_and_crop()`，逐页把 PDF 渲染成图片，再调用本地 VLM 提取正文和插图描述。

相关代码：

- `src/ingestion/document_processor.py`
- `scripts/run_ingestion.py`

2. 两级切分

先用 `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` 做粗切，再用本地 `qwen3-4b-instruct` 做 agentic split。

3. 上下文增强

每篇文献先调用 Kimi 生成全文全局大纲，再让本地 Qwen 为每个 chunk 生成：

- metadata
- context_anchor
- multi_queries
- hyde_answer

入库时的 `contextual_chunk` 由“全局上下文 + 原文资料”拼接而成。

相关代码：`scripts/run_ingestion.py:133`

4. Milvus 入库

每个 chunk 写入：

- `id`
- `dense_vector`
- `sparse_vector`
- `contextual_chunk`
- `source_file`
- `dynasty`
- `painter`
- `subject_matter`
- `content_scope`

相关代码：`scripts/run_ingestion.py:142`

5. 检索

在线检索器使用四路召回：

- dense vector
- sparse vector
- ColBERT multi-vector
- Neo4j graph

然后 RRF 融合，取候选 chunk，再用 BGE reranker 精排。

相关代码：`src/retrieval/online_retrieval.py:194`

6. 返回给 Researcher

检索器从 Milvus 取回：

```python
output_fields=["contextual_chunk", "source_file", "dynasty", "painter"]
```

然后工具格式化为：

```text
来源文献: xxx.pdf
内容详情: ...
```

相关代码：

- `src/retrieval/online_retrieval.py:229`
- `src/agent/tools.py:44`

7. Researcher 回答

Researcher prompt 要求回答时引用：

```text
[文献: 《XXX.pdf》]
```

相关代码：`src/agent/prompts/researcher_prompt.yaml:7`

## 本地数据状态

当前本地数据检查结果：

- `data/raw_pdfs` 中有 55 个 PDF。
- Milvus 集合 `landscape_rag` 中有 4703 个 chunk。
- Milvus 中实际覆盖 53 个 `source_file`。
- `data/vector_store/milvus_landscape.db` 约 32 MB。
- `data/vector_store/colbert_tensors.pkl` 约 6.3 GB。
- `data/extracted_artworks` 中有 1131 个图片文件。
- Milvus 中约 604 个 chunk 包含图片路径。
- `painter` 为未知的 chunk 有 2548 个。
- `subject_matter` 为未知的 chunk 有 953 个。

这说明系统已经有较完整的数据规模，但也暴露出两个问题：

1. metadata 抽取质量不稳定。
2. 图片和文本已经混在 `contextual_chunk` 里，但还没有真正形成多模态检索链路。

## 第一性原理判断

一个可用的学术 RAG 系统，本质上要满足五个条件。

### 1. 证据必须可追溯

用户不是只要一个“像真的回答”，而是要知道回答依据哪篇文献、哪一页、哪一段、哪一个 chunk。

当前系统只引用 PDF 文件名，没有 page、section、chunk_id、bbox。  
这会导致一个问题：即使模型引用了正确 PDF，也很难判断它到底是不是基于正确段落。

结论：这是结构性问题，需要重构。

### 2. 检索必须服从用户约束

如果用户问“根据《明代山水画中桥梁意象研究》……”，系统应该强制从这个 PDF 中检索，或者至少对该 PDF 加很高权重。

当前系统只是把文献标题放进 query，让 RAG 自己召回。评测集里的 `gold_sources` 只用于评分，不会传给模型，也不会约束检索。

结论：这是结构性问题，需要加 source-aware retrieval。

### 3. chunk 必须来自可信解析结果

RAG 的证据质量上限由 PDF 解析质量决定。OCR、VLM、版面恢复、表格/脚注/页眉处理一旦出错，后面 dense retrieval、reranker、LLM 都是在错误文本上工作。

当前系统直接逐页截图给 VLM 提取正文，优点是能处理扫描件和插图；缺点是：

- 没有优先使用 PDF 内嵌文本。
- 没有保存 page-level canonical parse artifact。
- 没有页码和 bbox。
- 没有 OCR/解析质量门禁。
- 解析结果没有独立落盘，Milvus 成了事实上的唯一证据库。

结论：这是当前最大结构性问题。

### 4. LLM 生成内容和原文证据必须分层

上下文增强、HyDE、multi_queries 是好方向。Anthropic 的 Contextual Retrieval 也支持“给 chunk 增加文档上下文”的做法。

但关键是：**LLM 生成的 context_anchor、hyde_answer 只能用于检索增强，不能和原文证据混成同一层证据。**

当前 `contextual_chunk` 把“全局上下文”和“原文资料”拼在一起返回给 Researcher。模型有可能把 LLM 生成的 anchor 当成原文证据引用。

结论：方向对，但需要重构存储结构和返回格式。

### 5. 评测必须在稳定链路上做

如果 RAG 链路本身证据不可追溯、指定文献不受约束、PDF 解析不可复验，那么继续做大规模回答评测和训练会污染判断。

但也不能完全不核验。正确做法是：

- 重构前只做小规模诊断核验，用来定位结构问题。
- 重构后再跑 reviewed 评测集，形成可比较指标。
- 不要在当前链路上继续做 DPO/SFT。

结论：现在不该继续训练，应该先重构，再评测。

## 当前系统做得对的地方

### 1. 多路召回方向正确

dense、sparse、ColBERT、graph 多路召回，再做 RRF 融合和 reranker 精排，这个方向是合理的。BGE-M3 本身也适合同时提供 dense、sparse、multi-vector 表征。

不需要推倒这个检索设计。

### 2. chunk 上下文增强方向正确

给 chunk 增加文档级背景，有助于解决孤立片段语义不足的问题。你当前的 `context_anchor` 思路是对的。

但它应该作为 `context_for_retrieval` 或 `contextual_prefix` 存储，而不是和原文证据不加区分地混在最终引用文本里。

### 3. Reranker 是必要组件

当前用 BGE reranker 对融合候选精排，这是对的。RAG 里只靠初召很容易拿到语义相近但证据不对的 chunk。

### 4. 错误前提专项评测方向正确

山水画领域很容易遇到“现代 AI 工具套古代画家”的问题。保留错误前提题是必要的。

问题不是题本身，而是 Researcher 必须先纠错，并且检索层不能为了迎合错误前提去找无关现代技术片段。

## 当前系统的结构性问题

### P0 问题 1：没有稳定的 canonical document store

当前 PDF 解析后没有形成独立的标准化文档层。Milvus 里保存了 `contextual_chunk`，但没有单独保存：

- doc_id
- page_id
- page_number
- section_title
- paragraph_id
- bbox
- raw_text
- normalized_text
- image_caption
- extraction_method
- parser_version
- pdf_checksum

这会导致后续所有东西都绑定到一次性入库结果，难以回溯、难以复建、难以定位错误。

建议：建立 `data/processed/documents/*.jsonl` 或 Parquet 作为 canonical document store。Milvus 只做索引，不做唯一真相源。

### P0 问题 2：引用粒度太粗

当前 Researcher 只能引用 `[文献: 《xxx.pdf》]`。这对论文问答不够。

至少应该引用：

```text
[文献: 《xxx.pdf》, 页: 12, chunk: abc123]
```

如果 PDF 是扫描件，也应至少有 page_number。若能保留 bbox，则可以进一步定位到页面区域。

### P0 问题 3：指定文献没有被检索层强制

评测题里有很多“根据《某某文献》”。但实际回答阶段只把 question 传给 Researcher，检索层没有 source filter。

应改成：

1. Query parser 解析出 `source_title_constraint`。
2. 如果用户指定文献，先做 source_file/title 匹配。
3. Milvus 检索加 scalar filter，或至少对指定 source 加权。
4. 如果指定文献不存在，明确告诉用户资料库没有该文献。

否则“根据 A 文献”的题可能检索到 B 文献，模型再引用 B，评测和真实体验都会受影响。

### P0 问题 4：LLM 生成 metadata 的兜底策略会放大噪音

`LocalLLMExtractor.extract_super_chunk()` 在 JSON 解析失败时默认：

```python
"is_domain_relevant": True
```

也就是说，一旦本地 Qwen 输出坏 JSON，系统会默认放行该 chunk。这在生产 RAG 里风险很高。

建议改成：

- JSON 解析失败时标记 `parse_status=failed`。
- 不直接入主索引。
- 放入 quarantine 队列。
- 允许人工或离线批处理修复。

### P0 问题 5：原文证据和增强文本混在一起

当前：

```text
【全局上下文】...
【原文资料】...
```

一起写进 `contextual_chunk`，检索和生成都用这个字段。

建议拆成：

- `raw_chunk_text`
- `contextual_prefix`
- `retrieval_text`
- `answer_evidence_text`

检索可以使用 `retrieval_text = contextual_prefix + raw_chunk_text + multi_queries + hyde_answer`。  
回答时只把 `raw_chunk_text` 和必要的 `contextual_prefix` 结构化返回，并明确告诉模型哪些是原文，哪些是系统生成背景。

### P1 问题 6：图谱不是证据层

Neo4j graph 可以帮助召回实体相关 chunk，但当前图谱服务离线时系统会降级。这是可以接受的。

但图谱不应作为回答事实来源。最终回答仍应引用原文 chunk。

建议：GraphRAG 用于 query expansion、entity expansion、community summary，但最终 answer grounding 必须回到 PDF chunk。

### P1 问题 7：ColBERT pickle 存储风险高

当前 `colbert_tensors.pkl` 有 6.3GB，而 Milvus DB 只有 32MB。

问题：

- pickle 不是长期稳定的数据存储格式。
- 没有 manifest 就难以知道它对应哪个 embedding model、哪个 chunk set、哪个 parser version。
- chunk 删除、增量更新、重建后容易和 Milvus 不一致。

建议至少增加：

- `index_manifest.json`
- model name/path
- model checksum or revision
- parser version
- chunk count
- source checksum list
- colbert tensor count

更长期可以考虑把 multi-vector 也迁移到支持多向量的索引后端，或将 ColBERT 作为可重建缓存。

### P1 问题 8：没有独立全文/关键词索引

BGE-M3 sparse_vector 可以做 lexical matching，但它不等价于传统 BM25/倒排全文索引。

对于中国美术史系统，精确词很重要：

- 文献题名
- 画家名
- 作品名
- 术语，如“披麻皴”“斧劈皴”
- 朝代
- 引文里的专名

建议补一个 BM25/full-text index，或使用 Milvus 支持的 BM25/full-text 能力。至少要能保证题名、PDF 文件名、画家名精确命中。

### P1 问题 9：多模态资产没有形成真正的多模态 RAG

系统已经裁剪出 1131 张图片，chunk 里也有图片路径。但当前 Researcher 检索主要还是文本。

如果未来要支持“请根据图像构图生成提示词”“这幅图像是什么桥梁布局”等任务，需要：

- image_id
- image_path
- page_number
- bbox
- image_caption
- CLIP/SigLIP/视觉 embedding
- 图像与文本 chunk 的关系

否则图片只是附在文本里的路径，不是可检索证据。

### P1 问题 10：脚本和配置中有工程化风险

发现的问题包括：

- `scripts/run_ingestion.py` 中硬编码了 LangSmith API key。
- `scripts/run_ingestion.py` 使用了不标准 import：`from Workspace.ChineseLandscape.src.retrieval.bge_m3_engine import BGEM3Engine`。
- 多处绝对路径写死 `/root/Workspace/ChineseLandscape`。
- ingestion、retrieval、agent 之间用字符串拼接格式传递证据，而不是结构化 JSON。

这些不是算法问题，但会影响可复现性、部署和后续维护。

## 推荐重构后的目标架构

### 1. 文档层

新增标准化文档产物：

```text
data/processed/documents/
  documents.jsonl
  pages.jsonl
  elements.jsonl
  chunks.jsonl
  images.jsonl
  manifest.json
```

建议字段：

```json
{
  "doc_id": "sha256:...",
  "source_file": "明代山水画中桥梁意象研究.pdf",
  "title": "明代山水画中桥梁意象研究",
  "pdf_path": "data/raw_pdfs/...",
  "pdf_checksum": "...",
  "parser_version": "pdf-rag-v1",
  "page_count": 24
}
```

chunk 字段：

```json
{
  "chunk_id": "docid_p12_c03",
  "doc_id": "...",
  "source_file": "...",
  "page_start": 12,
  "page_end": 13,
  "section_title": "桥梁意象的空间功能",
  "raw_chunk_text": "...",
  "contextual_prefix": "...",
  "retrieval_text": "...",
  "metadata": {
    "dynasty": ["明"],
    "painter": ["唐寅"],
    "concepts": ["桥梁", "流水", "游观路径"]
  },
  "quality": {
    "parser": "pymupdf+vlm",
    "parse_status": "ok",
    "char_count": 842,
    "garbled_ratio": 0.01
  }
}
```

### 2. 解析层

建议改成三段式：

1. 先用确定性 PDF text extraction 获取可复制文本。
2. 对扫描页或低质量页再用 OCR/VLM。
3. 对图片、表格、图注单独抽取 element。

每页都要保存：

- page text
- page image path
- extracted images
- OCR/VLM method
- parse warnings

### 3. 切分层

不要只依赖字符切分 + LLM split。

优先顺序建议：

1. 按章节标题切分。
2. 按段落切分。
3. 按语义窗口合并。
4. 对超长段落再做 sliding window。

LLM split 可以保留，但它应该是辅助，不是唯一结构来源。

### 4. 索引层

Milvus 存：

- `chunk_id`
- `doc_id`
- `source_file`
- `title`
- `page_start`
- `page_end`
- `raw_chunk_text`
- `contextual_prefix`
- `retrieval_text`
- `dense_vector`
- `sparse_vector`
- metadata scalar fields

另建：

- BM25/full-text index
- 可选 multi-vector index
- image vector index
- graph index

所有索引统一由 `manifest.json` 管理版本。

### 5. 检索层

检索流程建议改成：

1. Query parser
   - 识别是否指定 PDF
   - 识别画家、朝代、术语、作品名
   - 识别是否错误前提/现代技术错置
2. Constraint resolver
   - source filter
   - title alias
   - dynasty/painter filters
3. Candidate retrieval
   - BM25
   - dense
   - sparse
   - ColBERT
   - graph expansion
4. Fusion
   - RRF 或 weighted fusion
5. Rerank
   - cross-encoder reranker
6. Evidence packer
   - 控制 token
   - 保证多来源多样性
   - 返回 page/chunk citation

### 6. 回答层

Researcher 不应该只拿字符串，而应拿结构化 evidence：

```json
{
  "query": "...",
  "constraints": {
    "source_file": "明代山水画中桥梁意象研究.pdf"
  },
  "evidence": [
    {
      "chunk_id": "xxx",
      "source_file": "xxx.pdf",
      "page_start": 12,
      "page_end": 12,
      "raw_chunk_text": "...",
      "contextual_prefix": "...",
      "rerank_score": 0.83
    }
  ]
}
```

Prompt 要求：

- 每个事实必须绑定证据 chunk。
- 没证据就说证据不足。
- 指定文献不存在或未检索到时，直接说明。
- 错误前提题必须先纠错。

### 7. 引用校验层

生成后增加 citation verifier：

1. 检查回答里的 PDF 是否来自本轮 evidence。
2. 检查是否引用了不存在的 PDF。
3. 检查指定 source 的题是否引用了指定 source。
4. 检查 evidence_missing 题是否先纠错。
5. 必要时让 Researcher 重新生成。

这比直接训练模型更有效。

## 重构优先级计划

### 第 0 阶段：冻结训练

暂停：

- DPO
- PPO
- GRPO
- 大规模 SFT
- 继续扩充评测集

保留：

- 小规模诊断评测
- reviewed 评测集作为后续对比基准

### 第 1 阶段：证据层重构

目标：让每个回答都能追溯到 PDF 页码和 chunk。

任务：

1. 新建 `data/processed/documents`。
2. 为每个 PDF 生成 doc manifest。
3. 保存 page-level parsed text。
4. 保存 chunk-level raw text。
5. chunk_id 改为稳定 ID，例如 `sha256(doc_id + page + offset)`。
6. 入库时把 page、chunk_id、doc_id、source_file、title 写入 Milvus。

验收：

- 任意检索结果都能回到 `source_file + page + chunk_id`。
- Milvus 不再是唯一证据源。

### 第 2 阶段：指定文献检索约束

目标：用户问“根据《xxx》”时，检索必须优先或强制命中该文献。

任务：

1. 建立 title/source_file alias 表。
2. Query parser 提取文献名。
3. Milvus 检索支持 source_file scalar filter。
4. 没有指定文献时正常全库检索。
5. 指定文献不存在时明确返回。

验收：

- reviewed 评测集中所有 `根据《xxx》` 题，检索来源必须包含目标 PDF。

### 第 3 阶段：拆分 raw evidence 和 generated context

目标：LLM 生成内容只用于增强检索，不作为原文证据混用。

任务：

1. 拆字段：
   - `raw_chunk_text`
   - `contextual_prefix`
   - `multi_queries`
   - `hyde_answer`
   - `retrieval_text`
2. 检索用 `retrieval_text`。
3. 回答 evidence pack 以 `raw_chunk_text` 为主。
4. Prompt 明确区分“原文资料”和“系统生成上下文”。

验收：

- 回答引用必须落到 raw chunk。
- 不允许只引用 HyDE 或 context_anchor。

### 第 4 阶段：入库质量门禁

目标：坏解析、坏 JSON、跨域噪音不直接进入主索引。

任务：

1. JSON 解析失败时不默认 `is_domain_relevant=True`。
2. 增加 quarantine 文件。
3. 增加 chunk 质量统计：
   - 字符数
   - 异常符号比例
   - 图片路径比例
   - unknown metadata 比例
4. 增加入库报告。

验收：

- 每次 ingestion 后输出中文数据质量报告。
- 可清楚知道哪些 PDF/页面/块质量差。

### 第 5 阶段：全文索引和 citation verifier

目标：提高专名、题名、术语命中率，并减少幻觉引用。

任务：

1. 增加 BM25/full-text index。
2. 检索融合时加入 exact title/person/term boost。
3. 增加 citation verifier。
4. citation verifier 失败时触发重答或返回证据不足。

验收：

- 术语题、指定文献题、错误前提题通过率提升。
- 引用不存在 PDF 的问题接近 0。

### 第 6 阶段：重新评测

目标：判断还需不需要 SFT/DPO。

使用：

- `data/eval/test_researcher_v2_reviewed.jsonl`
- `data/eval/test_researcher_false_premise_v2_reviewed.jsonl`

评测重点：

- source hit
- chunk/page hit
- citation validity
- answer grounding
- false premise handling

如果主要失败是检索 source/chunk 不准，继续修 RAG。  
如果检索准但回答格式和拒答边界不稳定，做 SFT。  
如果 SFT 后偏好仍不稳定，再考虑 DPO。  
不建议在这个系统上优先考虑 PPO/GRPO。

## 是否还需要核验

需要，但核验的角色要变。

当前不建议投入大量人工核验或训练核验，因为底层 evidence pipeline 不稳定。  
但建议保留三类核验：

1. 重构前诊断核验：证明问题在哪里。
2. 重构中冒烟核验：每改一层确认没有破坏检索。
3. 重构后正式评测：用 reviewed 集做可比较结果。

所以不是“不做核验”，而是“不在错误层次上做核验”。

## 最终建议

当前系统不需要推倒重来，但需要做一次中等规模的 RAG 架构重构。

保留：

- BGE-M3 dense/sparse
- ColBERT 思路
- reranker
- contextual retrieval 思路
- Milvus
- reviewed 评测集
- 错误前提专项评测

重构：

- PDF canonical document store
- page/chunk provenance
- source-aware retrieval
- raw evidence 与 generated context 分层
- ingestion quality gate
- citation verifier
- index manifest/versioning

暂停：

- DPO/PPO/GRPO
- 基于当前回答日志的大规模 SFT
- 继续盲目扩充评测集

下一步最应该做的是：**重构证据层和指定文献检索约束**。这两项做完，系统的评测结果才真正有解释价值。
