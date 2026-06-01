const state = {
  history: [],
  evidence: [],
  citedRanks: new Set(),
  health: null,
  evidenceOpen: true,
  pdfPreview: null,
  pdfZoom: 1,
  pdfFit: true,
  corpusDocs: [],
  selectedCorpusDoc: null,
  agentTrace: [],
  imageArtifacts: [],
  imageSpec: null,
  corpusFilters: {
    dynasty: "",
    school: "",
    technique: "",
    authority: "",
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pageLabel(item) {
  if (!item.page_start) return "页码未知";
  if (item.page_end && item.page_end !== item.page_start) {
    return `页 ${item.page_start}-${item.page_end}`;
  }
  return `页 ${item.page_start}`;
}

function scoreLabel(item) {
  const score = Number(item.rerank_score);
  return Number.isFinite(score) ? score.toFixed(2) : "n/a";
}

function maxEvidenceScore(evidence) {
  const scores = (evidence || [])
    .map((item) => Number(item.rerank_score))
    .filter((score) => Number.isFinite(score));
  return scores.length ? Math.max(...scores) : null;
}

function citationRanksFromText(text) {
  const ranks = new Set();
  const content = String(text || "");
  for (const match of content.matchAll(/\[(\d+)\]/g)) {
    ranks.add(Number(match[1]));
  }
  return ranks;
}

function shortSourceName(value) {
  let name = String(value || "未知来源").replace(/\.pdf$/i, "");
  name = name.replace(/^[A-Z]\d{2}_/, "");
  const parts = name.split("_").filter(Boolean);
  if (parts.length > 1 && ["故宫", "古代画论", "DPM"].includes(parts[0])) {
    parts.shift();
  }
  name = parts.length ? parts.join(" ") : name.replaceAll("_", " ");
  return name.length > 34 ? `${name.slice(0, 33)}...` : name;
}

function sourceTitle(item) {
  return item.title || shortSourceName(item.source_file);
}

function fullSourceTitle(item) {
  const title = item.title || "";
  const source = item.source_file || "";
  return title && source && title !== source ? `${title} / ${source}` : (title || source || "未知来源");
}

function pageImageUrl(sourceFile, page) {
  if (!sourceFile || !page) return "";
  return `/api/pdf-page?source_file=${encodeURIComponent(sourceFile)}&page=${encodeURIComponent(page)}`;
}

function facetTags(doc) {
  const facets = doc.facets || {};
  return [
    ...(facets.dynasties || []),
    ...(facets.lineages_schools || []),
    ...(facets.styles_techniques || []),
  ].filter(Boolean).slice(0, 8);
}

function authorityLabel(level) {
  const value = String(level || "未评级").trim();
  const descriptions = {
    A: "优先引用",
    "A-": "优先引用",
    "B+": "辅助核验",
    B: "辅助核验",
    C: "背景参考",
  };
  return `${value} · ${descriptions[value] || "待核验"}`;
}

function authorityTitle(level) {
  const value = String(level || "未评级").trim();
  const descriptions = {
    A: "原典、馆藏、核心权威资料，回答时优先采用。",
    "A-": "权威整理、博物馆或高可信研究资料，回答时优先采用。",
    "B+": "质量较高的专题研究或学位论文，适合辅助核验。",
    B: "普通研究资料，适合补充背景，需要交叉核对。",
    C: "背景参考资料，不宜单独作为关键结论来源。",
  };
  return descriptions[value] || "尚未完成权威等级说明。";
}

function renderMessage(role, content, loading = false) {
  const node = document.createElement("div");
  node.className = `message ${role}${loading ? " loading" : ""}`;
  if (role === "assistant") {
    node.innerHTML = renderAnswerHtml(content);
  } else {
    node.textContent = content;
  }
  $("#messages").appendChild(node);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return node;
}

function citationButton(rank) {
  return `<sup><button class="citation-link" type="button" data-evidence-ref="${rank}" title="查看证据 ${rank}">[${rank}]</button></sup>`;
}

function stripChunkIds(text) {
  return String(text || "")
    .replace(/[，,；;]?\s*chunk_id\s*[：:]\s*[\w-]+/gi, "")
    .replace(/\s+（\s*）/g, "");
}

function cleanAnswerContent(content) {
  let text = stripChunkIds(String(content || "").replace(/\r\n/g, "\n"));
  text = text.replace(/[A-Z]\d{2}_[^\s，。；;、）)]+?\.pdf/gu, (name) => shortSourceName(name));
  text = text.replace(/^\s*前提判断[：:][^\n]*(?:\n+|$)/u, "");
  text = text.replace(/\n\s*前提判断[：:][^\n]*(?=\n|$)/gu, "\n");
  text = text.replace(/^\s*依据与解释[：:]\s*/u, "");
  text = text.replace(/\n\s*依据与解释[：:]\s*/gu, "\n");
  return text.trimStart();
}

function splitSourceSection(text) {
  const match = text.match(/\n?\s*来源[：:]\s*/u);
  if (!match || match.index === undefined) {
    return { body: text, sources: "", sourceCount: 0 };
  }
  const body = text.slice(0, match.index).trimEnd();
  const sources = text.slice(match.index + match[0].length).trim();
  const explicitCount = (sources.match(/(?:^|\n)\s*(?:[-*]\s*)?\[\d+\]/gu) || []).length;
  const fallbackCount = sources.split("\n").filter((line) => line.trim()).length;
  return { body, sources, sourceCount: explicitCount || fallbackCount };
}

function renderRichText(text) {
  let html = escapeHtml(text || "");
  const citations = [];
  const stashCitation = (rank) => {
    const token = `@@CITATION_${citations.length}@@`;
    citations.push(citationButton(rank));
    return token;
  };
  html = html.replace(/[（(]证据\[(\d+)\][：:][^）)]*[）)]/gu, (_match, rank) => stashCitation(rank));
  html = html.replace(/证据\[(\d+)\][：:][^\n。；;]*/gu, (_match, rank) => stashCitation(rank));
  html = html.replace(/\[(\d+)\]/gu, (_match, rank) => stashCitation(rank));
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\n/g, "<br>");
  citations.forEach((value, index) => {
    html = html.replace(`@@CITATION_${index}@@`, value);
  });
  return html;
}

function renderAnswerHtml(content) {
  const cleaned = cleanAnswerContent(content);
  const { body, sources, sourceCount } = splitSourceSection(cleaned);
  const bodyHtml = renderRichText(body);
  if (!sources) return bodyHtml;

  const sourceHtml = renderRichText(sources);
  if (sourceCount >= 5) {
    return `${bodyHtml}<details class="source-fold"><summary>来源（${sourceCount} 条）</summary><div>${sourceHtml}</div></details>`;
  }
  return `${bodyHtml}<div class="source-section"><div class="source-title">来源</div>${sourceHtml}</div>`;
}

function updateAnswerStatus({ phase = "", direct = false } = {}) {
  const node = $("#answer-status");
  const evidenceCount = state.evidence.length;
  const citedCount = state.citedRanks.size;
  const maxScore = maxEvidenceScore(state.evidence);
  const parts = [];
  if (direct || (!evidenceCount && phase === "直接回答")) {
    parts.push("未检索文献");
  } else if (evidenceCount) {
    parts.push(`已检索 ${evidenceCount} 条证据`);
    parts.push(`引用 ${citedCount} 条`);
    if (maxScore !== null) parts.push(`最高相关性 ${maxScore.toFixed(2)}`);
  } else {
    parts.push("等待检索");
  }
  if (phase) parts.push(phase);
  node.innerHTML = parts.map((part) => `<span>${escapeHtml(part)}</span>`).join("");
  node.hidden = false;
}

function setAgentPlan(steps) {
  state.agentTrace = (steps || []).map((step) => ({
    node: step.node,
    title: step.title || step.node,
    goal: step.goal || "",
    status: "pending",
    detail: "",
  }));
  renderAgentTrace();
}

function updateAgentNode(event) {
  const index = state.agentTrace.findIndex((item) => item.node === event.node);
  const next = {
    node: event.node,
    title: event.title || event.node,
    goal: index >= 0 ? state.agentTrace[index].goal : "",
    status: event.status || "running",
    detail: event.detail || "",
  };
  if (index >= 0) {
    state.agentTrace[index] = { ...state.agentTrace[index], ...next };
  } else {
    state.agentTrace.push(next);
  }
  renderAgentTrace();
}

function renderAgentTrace() {
  const target = $("#agent-trace");
  if (!state.agentTrace.length) {
    target.hidden = true;
    target.innerHTML = "";
    return;
  }
  target.hidden = false;
  const items = state.agentTrace.map((step) => `
    <li class="${escapeHtml(step.status || "pending")}">
      <span>${escapeHtml(step.title || step.node)}</span>
      <small>${escapeHtml(step.detail || step.goal || "")}</small>
    </li>
  `).join("");
  target.innerHTML = `<div class="trace-title">Agent 工作流</div><ol>${items}</ol>`;
}

function renderAgentArtifactsHtml() {
  const spec = state.imageSpec;
  const specHtml = spec
    ? `<details class="image-spec">
        <summary>图像 Prompt</summary>
        <dl>
          <div><dt>尺寸</dt><dd>${escapeHtml(spec.width)} x ${escapeHtml(spec.height)}</dd></div>
          <div><dt>格式</dt><dd>${escapeHtml(spec.format || "未指定")}</dd></div>
        </dl>
        <pre>${escapeHtml(spec.positive_prompt || "")}</pre>
        <pre>${escapeHtml(spec.negative_prompt || "")}</pre>
      </details>`
    : "";
  const imageHtml = state.imageArtifacts.map((item) => {
    if (item.status === "success" && item.url) {
      return `
        <figure class="generated-image">
          <img src="${escapeHtml(item.url)}" alt="生成的山水画" />
          <figcaption>
            <span>seed ${escapeHtml(item.seed || "")}</span>
            <a class="mini-button" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer" download>下载图片</a>
          </figcaption>
        </figure>
      `;
    }
    return `
      <div class="image-error">
        <strong>图像未生成</strong>
        <span>${escapeHtml(item.message || "生图引擎未返回图片")}</span>
      </div>
    `;
  }).join("");
  return specHtml || imageHtml ? `<div class="agent-artifacts">${specHtml}${imageHtml}</div>` : "";
}

function renderAssistantOutput(node, answer) {
  node.innerHTML = `${renderAnswerHtml(answer || "")}${renderAgentArtifactsHtml()}`;
}

function renderEvidence(target, evidence) {
  target.innerHTML = "";
  if (!evidence.length) {
    target.innerHTML = '<div class="empty">暂无证据</div>';
    return;
  }
  for (const item of evidence) {
    const node = document.createElement("article");
    const cited = state.citedRanks.has(Number(item.rank));
    node.className = `evidence-item${cited ? " cited-evidence" : " unreferenced-evidence"}`;
    node.id = `evidence-${item.rank}`;
    node.dataset.rank = String(item.rank);
    node.evidenceItem = item;
    const pageAction = item.page_image_url
      ? `<button class="mini-button preview-page" type="button" data-rank="${escapeHtml(item.rank)}">查看页</button>`
      : "";
    const downloadAction = item.pdf_url
      ? `<a class="mini-button" href="${escapeHtml(item.pdf_url)}" target="_blank" rel="noreferrer">下载 PDF</a>`
      : "";
    node.innerHTML = `
      <strong title="${escapeHtml(fullSourceTitle(item))}">${escapeHtml(sourceTitle(item))}</strong>
      <div class="evidence-meta">
        <span>#${escapeHtml(item.rank)}</span>
        <span>${escapeHtml(pageLabel(item))}</span>
        <span>${escapeHtml(scoreLabel(item))}</span>
        ${cited ? "<span>已引用</span>" : ""}
      </div>
      <details class="evidence-preview">
        <summary>查看证据文本</summary>
        <p>${escapeHtml(item.preview || "")}</p>
      </details>
      <div class="evidence-actions">${pageAction}${downloadAction}</div>
    `;
    target.appendChild(node);
  }
}

function updateEvidence(evidence) {
  state.evidence = evidence || [];
  $("#evidence-count").textContent = String(state.evidence.length);
  renderEvidence($("#evidence-list"), state.evidence);
  updateAnswerStatus({ phase: state.evidence.length ? "检索完成" : "直接回答", direct: !state.evidence.length });
}

function setHealth(ok, text) {
  const dot = $("#health-dot");
  dot.classList.remove("pending", "ok", "error");
  dot.classList.add(ok ? "ok" : "error");
  $("#health-text").textContent = text;
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(payload.detail || `HTTP ${res.status}`);
  }
  return payload;
}

async function loadHealth() {
  try {
    const data = await fetchJson("/health");
    state.health = data;
    setHealth(true, data.llm_configured ? "LLM 已配置" : "证据模式");
    $("#runtime-line").textContent = data.agent?.mode ? "Agent：研究、核验、创作、交付" : "证据页可预览，可下载原始 PDF";
    renderSystem(data);
  } catch (error) {
    setHealth(false, "服务异常");
    $("#system-panel").innerHTML = `<div class="metric"><span>错误</span><strong>${escapeHtml(error.message)}</strong></div>`;
  }
}

function jumpToEvidence(rank) {
  switchView("chat");
  const node = document.querySelector(`#evidence-${CSS.escape(String(rank))}`);
  const item = state.evidence.find((evidence) => String(evidence.rank) === String(rank));
  if (node) {
    node.scrollIntoView({ behavior: "smooth", block: "center" });
    node.classList.add("active-evidence");
    window.setTimeout(() => node.classList.remove("active-evidence"), 1400);
  }
  if (item) openPdfPreview(item);
}

function renderPdfPreview() {
  const item = state.pdfPreview;
  if (!item) return;
  const page = Number(item.current_page || item.page_start || 1);
  const total = Number(item.page_count || 0);
  $("#pdf-title").textContent = sourceTitle(item);
  $("#pdf-title").title = fullSourceTitle(item);
  $("#pdf-subtitle").textContent = total ? `第 ${page} / ${total} 页` : `第 ${page} 页`;
  const image = $("#pdf-page-img");
  image.removeAttribute("src");
  image.src = pageImageUrl(item.source_file, page) || item.page_image_url;
  image.alt = `${sourceTitle(item)} 第 ${page} 页`;
  image.style.width = state.pdfFit ? "min(100%, 920px)" : `${Math.round(820 * state.pdfZoom)}px`;
  $("#pdf-page-input").value = String(page);
  $("#pdf-page-input").max = total ? String(total) : "";
  const download = $("#download-pdf");
  if (item.pdf_url) {
    download.href = item.pdf_url;
    download.hidden = false;
  } else {
    download.hidden = true;
  }
  $("#prev-pdf-page").disabled = page <= 1;
  $("#next-pdf-page").disabled = Boolean(total && page >= total);
  $("#fit-pdf").textContent = state.pdfFit ? "原始大小" : "适应宽度";
}

function openPdfPreview(item, page = null) {
  if (!item || !item.page_image_url) return;
  state.pdfPreview = {
    ...item,
    current_page: Number(page || item.page_start || 1),
  };
  state.pdfZoom = 1;
  state.pdfFit = true;
  renderPdfPreview();
  $("#pdf-modal").hidden = false;
  document.body.classList.add("modal-open");
}

function closePdfPreview() {
  $("#pdf-modal").hidden = true;
  $("#pdf-page-img").removeAttribute("src");
  state.pdfPreview = null;
  document.body.classList.remove("modal-open");
}

function movePdfPage(delta) {
  const item = state.pdfPreview;
  if (!item) return;
  const total = Number(item.page_count || 0);
  const current = Number(item.current_page || item.page_start || 1);
  const next = current + delta;
  if (next < 1 || (total && next > total)) return;
  item.current_page = next;
  renderPdfPreview();
}

function jumpPdfPage(value) {
  const item = state.pdfPreview;
  if (!item) return;
  const total = Number(item.page_count || 0);
  let page = Number(value);
  if (!Number.isFinite(page)) return;
  page = Math.max(1, Math.floor(page));
  if (total) page = Math.min(total, page);
  item.current_page = page;
  renderPdfPreview();
}

function zoomPdf(delta) {
  state.pdfFit = false;
  state.pdfZoom = Math.max(0.55, Math.min(2.4, state.pdfZoom + delta));
  renderPdfPreview();
}

function togglePdfFit() {
  state.pdfFit = !state.pdfFit;
  renderPdfPreview();
}

function handleEvidenceClick(event) {
  const previewButton = event.target.closest(".preview-page");
  if (!previewButton) return;
  const card = previewButton.closest(".evidence-item");
  if (card?.evidenceItem) {
    openPdfPreview(card.evidenceItem);
  }
}

async function readNdjsonStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

async function sendChat(message) {
  renderMessage("user", message);
  $("#example-prompts").classList.add("hidden");
  state.citedRanks = new Set();
  state.evidence = [];
  state.agentTrace = [];
  state.imageArtifacts = [];
  state.imageSpec = null;
  $("#evidence-count").textContent = "0";
  renderEvidence($("#evidence-list"), state.evidence);
  renderAgentTrace();
  setEvidenceRail(false);
  state.history.push({ role: "user", content: message });
  const loadingNode = renderMessage("assistant", "准备分析", true);
  updateAnswerStatus({ phase: "Agent 准备" });
  $("#send-button").disabled = true;
  let answer = "";
  const phaseTimers = [
    window.setTimeout(() => {
      if (loadingNode.classList.contains("loading")) loadingNode.textContent = "理解任务";
    }, 180),
    window.setTimeout(() => {
      if (loadingNode.classList.contains("loading")) loadingNode.textContent = "执行计划";
    }, 900),
  ];
  try {
    const response = await fetch("/api/agent/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: state.history, final_k: 5 }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    loadingNode.classList.remove("loading");
    loadingNode.innerHTML = "";
    await readNdjsonStream(response, (event) => {
      if (event.type === "plan") {
        setAgentPlan(event.steps || []);
        updateAnswerStatus({ phase: "计划已生成", direct: !state.evidence.length });
      }
      if (event.type === "node") {
        updateAgentNode(event);
      }
      if (event.type === "evidence") {
        updateEvidence(event.evidence || []);
        if (!(event.evidence || []).length) {
          setEvidenceRail(false);
        } else if (!window.matchMedia("(max-width: 980px)").matches) {
          setEvidenceRail(true);
        }
      }
      if (event.type === "brief") {
        updateAnswerStatus({ phase: "研究卷宗完成", direct: !state.evidence.length });
      }
      if (event.type === "image_spec") {
        state.imageSpec = event.spec || null;
        renderAssistantOutput(loadingNode, answer);
        updateAnswerStatus({ phase: "Prompt 已生成", direct: !state.evidence.length });
      }
      if (event.type === "image") {
        state.imageArtifacts.push(event.image || {});
        renderAssistantOutput(loadingNode, answer);
        updateAnswerStatus({ phase: "图像生成完成", direct: !state.evidence.length });
      }
      if (event.type === "phase" && !answer) {
        loadingNode.textContent = event.phase || "";
        updateAnswerStatus({ phase: event.phase || "", direct: !state.evidence.length });
      }
      if (event.type === "delta") {
        answer += event.delta || "";
        state.citedRanks = citationRanksFromText(answer);
        renderAssistantOutput(loadingNode, answer);
        renderEvidence($("#evidence-list"), state.evidence);
        updateAnswerStatus({ phase: "生成中", direct: !state.evidence.length });
        $("#messages").scrollTop = $("#messages").scrollHeight;
      }
      if (event.type === "error") {
        answer += `\n请求失败：${event.message || "未知错误"}`;
        renderAssistantOutput(loadingNode, answer);
      }
      if (event.type === "done") {
        updateAnswerStatus({ phase: "完成", direct: !state.evidence.length });
      }
    });
    state.history.push({ role: "assistant", content: answer || loadingNode.textContent });
  } catch (error) {
    loadingNode.classList.remove("loading");
    loadingNode.textContent = `请求失败：${error.message}`;
  } finally {
    phaseTimers.forEach((timer) => window.clearTimeout(timer));
    $("#send-button").disabled = false;
    $("#messages").scrollTop = $("#messages").scrollHeight;
  }
}

async function retrieve(query) {
  const target = $("#retrieve-results");
  target.innerHTML = '<div class="empty">检索中</div>';
  try {
    const data = await fetchJson("/api/retrieve", {
      method: "POST",
      body: JSON.stringify({ query, final_k: 5 }),
    });
    renderEvidence(target, data.evidence || []);
  } catch (error) {
    target.innerHTML = `<div class="empty">请求失败：${escapeHtml(error.message)}</div>`;
  }
}

function facetValues(doc, key) {
  return doc.facets?.[key] || [];
}

function uniqueSorted(values) {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
}

function populateSelect(selector, label, values) {
  const select = $(selector);
  const current = select.value;
  select.innerHTML = `<option value="">${escapeHtml(label)}</option>${values
    .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
    .join("")}`;
  select.value = values.includes(current) ? current : "";
}

function populateCorpusFilters(docs) {
  populateSelect("#filter-dynasty", "全部朝代", uniqueSorted(docs.flatMap((doc) => facetValues(doc, "dynasties"))));
  populateSelect("#filter-school", "全部流派", uniqueSorted(docs.flatMap((doc) => facetValues(doc, "lineages_schools"))));
  populateSelect("#filter-technique", "全部技法", uniqueSorted(docs.flatMap((doc) => facetValues(doc, "styles_techniques"))));
  populateSelect("#filter-authority", "全部等级", uniqueSorted(docs.map((doc) => doc.authority_level)));
  state.corpusFilters.dynasty = $("#filter-dynasty").value;
  state.corpusFilters.school = $("#filter-school").value;
  state.corpusFilters.technique = $("#filter-technique").value;
  state.corpusFilters.authority = $("#filter-authority").value;
}

function filteredCorpusDocs() {
  return state.corpusDocs.filter((doc) => {
    const filters = state.corpusFilters;
    if (filters.dynasty && !facetValues(doc, "dynasties").includes(filters.dynasty)) return false;
    if (filters.school && !facetValues(doc, "lineages_schools").includes(filters.school)) return false;
    if (filters.technique && !facetValues(doc, "styles_techniques").includes(filters.technique)) return false;
    if (filters.authority && doc.authority_level !== filters.authority) return false;
    return true;
  });
}

function renderCorpusTable() {
  const table = $("#corpus-table");
  const docs = filteredCorpusDocs();
  if (state.selectedCorpusDoc && !docs.some((doc) => doc.source_file === state.selectedCorpusDoc.source_file)) {
    state.selectedCorpusDoc = docs[0] || null;
    renderCorpusDetail(state.selectedCorpusDoc);
  }
  $("#corpus-summary").textContent = state.corpusDocs.length
    ? `已显示 ${docs.length} 篇文献`
    : "文献索引已加载";
  table.innerHTML = "";
  if (!docs.length) {
    table.innerHTML = '<div class="empty">没有符合筛选条件的文献</div>';
    return;
  }
  for (const doc of docs) {
    const row = document.createElement("div");
    row.className = `doc-row${state.selectedCorpusDoc?.source_file === doc.source_file ? " selected-doc" : ""}`;
    row.tabIndex = 0;
    row.docItem = doc;
    row.innerHTML = `
      <strong title="${escapeHtml(doc.source_file || doc.title || "")}">${escapeHtml(doc.title || shortSourceName(doc.source_file))}</strong>
      <span class="doc-category">${escapeHtml(doc.category || "未分类")}</span>
      <span class="pill" title="${escapeHtml(authorityTitle(doc.authority_level))}">${escapeHtml(authorityLabel(doc.authority_level))}</span>
      <span>${escapeHtml(doc.page_count || 0)} 页</span>
    `;
    table.appendChild(row);
  }
}

function renderCorpusDetail(doc) {
  const target = $("#corpus-detail");
  if (!doc) {
    target.innerHTML = '<div class="empty">选择一篇文献查看详情</div>';
    return;
  }
  const tags = facetTags(doc);
  const download = doc.pdf_url
    ? `<a class="mini-button" href="${escapeHtml(doc.pdf_url)}" target="_blank" rel="noreferrer">下载 PDF</a>`
    : "";
  target.innerHTML = `
    <article class="doc-detail">
      <span class="pill" title="${escapeHtml(authorityTitle(doc.authority_level))}">${escapeHtml(authorityLabel(doc.authority_level))}</span>
      <h2>${escapeHtml(doc.title || shortSourceName(doc.source_file))}</h2>
      <dl>
        <div><dt>分类</dt><dd>${escapeHtml(doc.category || "未分类")}</dd></div>
        <div><dt>作者</dt><dd>${escapeHtml(doc.author || "未标注")}</dd></div>
        <div><dt>类型</dt><dd>${escapeHtml(doc.source_type || "未标注")}</dd></div>
        <div><dt>页数</dt><dd>${escapeHtml(doc.page_count || 0)}</dd></div>
        <div><dt>权威等级说明</dt><dd>${escapeHtml(authorityTitle(doc.authority_level))}</dd></div>
      </dl>
      <div class="tag-list">${tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("") || "<span>暂无标签</span>"}</div>
      <p title="${escapeHtml(doc.source_file || "")}">${escapeHtml(doc.source_file || "")}</p>
      <div class="evidence-actions">${download}</div>
    </article>
  `;
}

async function loadCorpus() {
  const table = $("#corpus-table");
  table.innerHTML = '<div class="empty">加载中</div>';
  try {
    const data = await fetchJson("/api/corpus");
    state.corpusDocs = data.documents || [];
    populateCorpusFilters(state.corpusDocs);
    state.selectedCorpusDoc = state.corpusDocs[0] || null;
    renderCorpusTable();
    renderCorpusDetail(state.selectedCorpusDoc);
  } catch (error) {
    table.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderSystem(data) {
  const modelName = (value) => String(value || "未知").split("/").filter(Boolean).pop() || "未知";
  const routerText = data.router?.llm_enabled
    ? `规则 + ${data.router.router_model} + 阈值 ${data.router.min_rerank_score}`
    : `规则 + 阈值 ${data.router?.min_rerank_score ?? "n/a"}`;
  const basicMetrics = [
    ["回答模型", data.answer_model || (data.llm_configured ? "可用" : "未配置")],
    ["Agent 模式", data.agent?.mode || "未启用"],
    ["证据库", data.evidence_dir ? "已加载" : "未知"],
    ["路由策略", routerText],
  ];
  const diagnostics = [
    ["服务提供方", data.answer_provider || "未知"],
    ["向量模型", modelName(data.retriever_models?.encoder)],
    ["重排模型", modelName(data.retriever_models?.reranker)],
    ["生图引擎", data.agent?.image_generation?.provider || "未配置"],
    ["生图工作流", modelName(data.agent?.image_generation?.workflow)],
    ["训练模型", data.trained_researcher_lora_exists ? "已存在，当前未用于前端回答" : "未发现"],
  ];
  const metricHtml = (items) => items
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
  $("#system-panel").innerHTML = `
    <section class="system-section">
      <h2>运行状态</h2>
      <div class="metric-grid">${metricHtml(basicMetrics)}</div>
    </section>
    <details class="system-section diagnostic-section">
      <summary>开发诊断</summary>
      <div class="metric-grid">${metricHtml(diagnostics)}</div>
    </details>
  `;
}

function setEvidenceRail(open) {
  state.evidenceOpen = Boolean(open);
  const layout = document.querySelector(".chat-layout");
  layout?.classList.toggle("evidence-collapsed", !state.evidenceOpen);
  const toggle = $("#toggle-evidence");
  toggle.textContent = state.evidenceOpen ? "隐藏证据" : "显示证据";
  toggle.setAttribute("aria-expanded", String(state.evidenceOpen));
}

function switchView(view) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $$(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
  if (view === "corpus") loadCorpus();
  if (view === "system" && state.health) renderSystem(state.health);
}

function setupEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
  $("#enter-app").addEventListener("click", () => {
    $("#intro-screen").classList.add("intro-hidden");
    $("#chat-input").focus();
  });
  $("#toggle-evidence").addEventListener("click", () => setEvidenceRail(!state.evidenceOpen));
  $("#example-prompts").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-prompt]");
    if (!button) return;
    const prompt = button.dataset.prompt;
    $("#chat-input").value = prompt;
    $("#chat-form").requestSubmit();
  });
  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("#chat-input");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    sendChat(message);
  });
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#retrieve-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const query = $("#retrieve-input").value.trim();
    if (query) retrieve(query);
  });
  $("#clear-chat").addEventListener("click", () => {
    state.history = [];
    state.citedRanks = new Set();
    state.agentTrace = [];
    state.imageArtifacts = [];
    state.imageSpec = null;
    updateEvidence([]);
    renderAgentTrace();
    $("#messages").innerHTML = "";
    $("#answer-status").hidden = true;
    $("#example-prompts").classList.remove("hidden");
    renderMessage("assistant", "请输入一个山水画史问题。");
  });
  $("#refresh-corpus").addEventListener("click", loadCorpus);
  [
    ["#filter-dynasty", "dynasty"],
    ["#filter-school", "school"],
    ["#filter-technique", "technique"],
    ["#filter-authority", "authority"],
  ].forEach(([selector, key]) => {
    $(selector).addEventListener("change", (event) => {
      state.corpusFilters[key] = event.target.value;
      renderCorpusTable();
    });
  });
  $("#corpus-table").addEventListener("click", (event) => {
    const row = event.target.closest(".doc-row");
    if (!row?.docItem) return;
    state.selectedCorpusDoc = row.docItem;
    renderCorpusTable();
    renderCorpusDetail(row.docItem);
  });
  $("#corpus-table").addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    const row = event.target.closest(".doc-row");
    if (!row?.docItem) return;
    state.selectedCorpusDoc = row.docItem;
    renderCorpusTable();
    renderCorpusDetail(row.docItem);
  });
  $("#messages").addEventListener("click", (event) => {
    const target = event.target.closest(".citation-link");
    if (target) jumpToEvidence(target.dataset.evidenceRef);
  });
  $("#evidence-list").addEventListener("click", handleEvidenceClick);
  $("#retrieve-results").addEventListener("click", handleEvidenceClick);
  $("#close-pdf").addEventListener("click", closePdfPreview);
  $("#prev-pdf-page").addEventListener("click", () => movePdfPage(-1));
  $("#next-pdf-page").addEventListener("click", () => movePdfPage(1));
  $("#pdf-page-input").addEventListener("change", (event) => jumpPdfPage(event.target.value));
  $("#zoom-out-pdf").addEventListener("click", () => zoomPdf(-0.15));
  $("#zoom-in-pdf").addEventListener("click", () => zoomPdf(0.15));
  $("#fit-pdf").addEventListener("click", togglePdfFit);
  $("#pdf-modal").addEventListener("click", (event) => {
    if (event.target.id === "pdf-modal") closePdfPreview();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#pdf-modal").hidden) closePdfPreview();
  });
}

function setupInkField() {
  const canvas = $("#ink-field");
  const ctx = canvas.getContext("2d");
  const particles = [];
  const count = 72;

  function resize() {
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  }

  function seed() {
    particles.length = 0;
    for (let i = 0; i < count; i += 1) {
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        r: 1 + Math.random() * 2.6,
        vx: -0.16 + Math.random() * 0.32,
        vy: 0.08 + Math.random() * 0.28,
        hue: Math.random() > 0.55 ? "99, 197, 183" : "217, 168, 78",
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.y > window.innerHeight + 20) p.y = -20;
      if (p.x < -20) p.x = window.innerWidth + 20;
      if (p.x > window.innerWidth + 20) p.x = -20;

      ctx.strokeStyle = `rgba(${p.hue}, 0.14)`;
      ctx.lineWidth = Math.max(0.6, p.r * 0.5);
      ctx.beginPath();
      ctx.moveTo(p.x - p.r * 8, p.y);
      ctx.quadraticCurveTo(p.x, p.y + p.r * 2, p.x + p.r * 10, p.y - p.r * 1.4);
      ctx.stroke();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", () => {
    resize();
    seed();
  });
  resize();
  seed();
  draw();
}

setupEvents();
setupInkField();
setEvidenceRail(!window.matchMedia("(max-width: 980px)").matches);
loadHealth();
renderMessage("assistant", "请输入一个山水画史问题。");
