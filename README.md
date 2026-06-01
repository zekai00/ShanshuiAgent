# ShanshuiAgent | 山水智研

<p>
  <a href="#中文"><strong>中文</strong></a>
  ·
  <a href="#english"><strong>English</strong></a>
</p>

## 中文

ShanshuiAgent（山水智研）是一个面向中国山水画研究与创作的证据型 Agent 系统。它从 PDF 文献构建文献级标注、页级文本、证据块、向量检索索引和可追溯来源，再通过 LangGraph 工作流完成任务理解、文献检索、证据核验、研究回答、图像提示词设计、ComfyUI 生图和 VLM 图像评审。

仓库地址：https://github.com/zekai00/ShanshuiAgent

### 当前能力

- 从权威 PDF 文献构建 evidence store，并保留文献、页码、证据块与原始 PDF 的对应关系。
- 使用 BGE-M3 向量召回、重排模型和 Milvus 索引完成多路检索。
- 可选接入 Neo4j 图谱，用于人物、朝代、流派、作品和技法关系的结构化查询。
- 通过 LangGraph 编排可追踪、可恢复的 Web Agent，覆盖研究问答和“先检索再创作”的图像任务。
- 通过现代化 Web UI 进行流式对话，正文引用以角标展示，点击可查看对应 PDF 页图并下载原 PDF。
- 支持 ComfyUI/Flux 图像生成，并用本地 Qwen3-VL 视觉模型检查生成图像与研究约束的一致性。
- 提供研究员回答、检索、错误前提识别等评测脚本与历史报告。
- 已保留本地 SFT/LoRA 训练产物；当前 Web UI 默认使用 `.env` 中配置的 DeepSeek-compatible API 回答模型。

### 目录结构

```text
src/
  agent/         LangGraph 研究员/创作/监督编排
  ingestion/     PDF 文本抽取与入库链路
  retrieval/     在线混合检索、BGE-M3、重排
scripts/
  retrieval/     evidence store 与 Milvus 构建脚本
  eval/          检索和回答评测脚本
  training/      研究员 SFT 数据构建脚本
  datasets/      权威文献整理、迁入、命名和分类脚本
ui/modern/       Web 对话界面
docs/            优化、评测、训练和资料整理报告
data/processed/  当前项目索引与评测数据
```

### 快速启动

```bash
pip install -r requirements.txt
cp .env.example .env
python scripts/run_web_app.py --host 127.0.0.1 --port 7861
```

打开 `http://127.0.0.1:7861`。

`.env` 中至少需要按实际环境配置回答模型 API；Milvus、Neo4j 和本地模型路径按本机部署情况配置。

### 重建证据索引

```bash
python scripts/retrieval/build_authority_evidence_store.py
python scripts/retrieval/build_milvus_from_evidence_store.py
python scripts/retrieval/smoke_test_retrieval.py
```

原始 PDF 目录、证据库输出目录和模型路径由本地配置决定；公开 README 不固定任何本机绝对路径。

### 评测与训练

```bash
python scripts/eval/run_retrieval_baseline.py
python scripts/eval/run_researcher_answer_baseline.py
python scripts/training/build_researcher_sft_dataset.py
```

当前建议优先继续扩大和校准测试集，再决定是否进行新的 SFT 或偏好优化。DPO/GRPO/PPO 不应替代证据链路；只有当 SFT 后仍存在稳定的偏好排序问题时，才考虑偏好对齐。

## English

ShanshuiAgent is an evidence-grounded agent system for Chinese landscape painting research and creation. It turns PDF literature into document-level metadata, page-level text, retrievable evidence chunks, vector indexes, and traceable citations, then uses a LangGraph workflow for task intake, retrieval, evidence verification, research answers, image prompt design, ComfyUI generation, and VLM image critique.

Repository: https://github.com/zekai00/ShanshuiAgent

### Capabilities

- Builds an evidence store from curated PDF literature while preserving document, page, chunk, and source PDF provenance.
- Uses BGE-M3 retrieval, a reranker, and Milvus for hybrid evidence retrieval.
- Optionally uses Neo4j for structured relations among artists, dynasties, schools, works, and techniques.
- Uses LangGraph to orchestrate a traceable and recoverable Web Agent for research QA and retrieval-grounded image creation.
- Provides a streaming Web UI with superscript citations; citation clicks open the corresponding PDF page image and allow PDF download.
- Supports ComfyUI/Flux image generation and local Qwen3-VL visual critique against research constraints.
- Includes evaluation scripts and historical reports for retrieval, researcher answers, and false-premise handling.
- Keeps local SFT/LoRA artifacts; the current Web UI answers with the DeepSeek-compatible API model configured in `.env`.

### Layout

```text
src/
  agent/         LangGraph researcher/artist/supervisor orchestration
  ingestion/     PDF extraction and ingestion
  retrieval/     online hybrid retrieval, BGE-M3, reranking
scripts/
  retrieval/     evidence store and Milvus build scripts
  eval/          retrieval and answer evaluation scripts
  training/      researcher SFT dataset builder
  datasets/      corpus curation, import, renaming, taxonomy scripts
ui/modern/       Web chat interface
docs/            optimization, evaluation, training, and corpus reports
data/processed/  local indexes and evaluation data
```

### Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python scripts/run_web_app.py --host 127.0.0.1 --port 7861
```

Open `http://127.0.0.1:7861`.

Configure the answer model API in `.env`; Milvus, Neo4j, and local model paths should match your machine.

### Rebuild The Evidence Index

```bash
python scripts/retrieval/build_authority_evidence_store.py
python scripts/retrieval/build_milvus_from_evidence_store.py
python scripts/retrieval/smoke_test_retrieval.py
```

The raw PDF directory, evidence-store output directory, and model paths are controlled by local configuration; this public README intentionally avoids machine-specific absolute paths.

### Evaluation And Training

```bash
python scripts/eval/run_retrieval_baseline.py
python scripts/eval/run_researcher_answer_baseline.py
python scripts/training/build_researcher_sft_dataset.py
```

The recommended next step is to keep expanding and calibrating the test set before launching more training. DPO, GRPO, or PPO should not compensate for weak evidence retrieval; use preference optimization only after SFT if a stable preference-ranking problem remains.
