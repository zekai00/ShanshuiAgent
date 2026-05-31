const state = {
  history: [],
  evidence: [],
  health: null,
  evidenceOpen: true,
  pdfPreview: null,
  corpusDocs: [],
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
  text = text.replace(/^\s*前提判断[：:]\s*(可以回答|可以作答|问题可以回答|前提正常)[。.]?\s*/u, "");
  text = text.replace(/^\s*前提判断[：:]\s*/u, "");
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

function renderEvidence(target, evidence) {
  target.innerHTML = "";
  if (!evidence.length) {
    target.innerHTML = '<div class="empty">暂无证据</div>';
    return;
  }
  for (const item of evidence) {
    const node = document.createElement("article");
    node.className = "evidence-item";
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
      </div>
      <p>${escapeHtml(item.preview || "")}</p>
      <div class="evidence-actions">${pageAction}${downloadAction}</div>
    `;
    target.appendChild(node);
  }
}

function updateEvidence(evidence) {
  state.evidence = evidence || [];
  $("#evidence-count").textContent = String(state.evidence.length);
  renderEvidence($("#evidence-list"), state.evidence);
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
    $("#runtime-line").textContent = "证据页可预览，可下载原始 PDF";
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
  const download = $("#download-pdf");
  if (item.pdf_url) {
    download.href = item.pdf_url;
    download.hidden = false;
  } else {
    download.hidden = true;
  }
  $("#prev-pdf-page").disabled = page <= 1;
  $("#next-pdf-page").disabled = Boolean(total && page >= total);
}

function openPdfPreview(item, page = null) {
  if (!item || !item.page_image_url) return;
  state.pdfPreview = {
    ...item,
    current_page: Number(page || item.page_start || 1),
  };
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
  state.history.push({ role: "user", content: message });
  const loadingNode = renderMessage("assistant", "准备检索", true);
  $("#send-button").disabled = true;
  let answer = "";
  const phaseTimers = [
    window.setTimeout(() => {
      if (loadingNode.classList.contains("loading")) loadingNode.textContent = "检索中";
    }, 180),
    window.setTimeout(() => {
      if (loadingNode.classList.contains("loading")) loadingNode.textContent = "重排中";
    }, 900),
  ];
  try {
    const response = await fetch("/api/chat/stream", {
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
      if (event.type === "evidence") {
        updateEvidence(event.evidence || []);
      }
      if (event.type === "phase" && !answer) {
        loadingNode.textContent = event.phase || "";
      }
      if (event.type === "delta") {
        answer += event.delta || "";
        loadingNode.innerHTML = renderAnswerHtml(answer);
        $("#messages").scrollTop = $("#messages").scrollHeight;
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
    row.className = "doc-row";
    row.innerHTML = `
      <strong title="${escapeHtml(doc.source_file || doc.title || "")}">${escapeHtml(doc.title || shortSourceName(doc.source_file))}</strong>
      <span class="doc-category">${escapeHtml(doc.category || "未分类")}</span>
      <span class="pill" title="${escapeHtml(authorityTitle(doc.authority_level))}">${escapeHtml(authorityLabel(doc.authority_level))}</span>
      <span>${escapeHtml(doc.page_count || 0)} 页</span>
    `;
    table.appendChild(row);
  }
}

async function loadCorpus() {
  const table = $("#corpus-table");
  table.innerHTML = '<div class="empty">加载中</div>';
  try {
    const data = await fetchJson("/api/corpus");
    state.corpusDocs = data.documents || [];
    populateCorpusFilters(state.corpusDocs);
    renderCorpusTable();
  } catch (error) {
    table.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderSystem(data) {
  const modelName = (value) => String(value || "未知").split("/").filter(Boolean).pop() || "未知";
  const metrics = [
    ["回答模型", data.answer_model || (data.llm_configured ? "可用" : "未配置")],
    ["服务提供方", data.answer_provider || "未知"],
    ["向量模型", modelName(data.retriever_models?.encoder)],
    ["重排模型", modelName(data.retriever_models?.reranker)],
    ["证据库", data.evidence_dir ? "已加载" : "未知"],
    ["路由策略", data.router?.llm_enabled ? `规则 + ${data.router.router_model} + 阈值 ${data.router.min_rerank_score}` : `规则 + 阈值 ${data.router?.min_rerank_score ?? "n/a"}`],
    ["训练模型", data.trained_researcher_lora_exists ? "已存在，当前未用于前端回答" : "未发现"],
  ];
  $("#system-panel").innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
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
    updateEvidence([]);
    $("#messages").innerHTML = "";
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
  $("#messages").addEventListener("click", (event) => {
    const target = event.target.closest(".citation-link");
    if (target) jumpToEvidence(target.dataset.evidenceRef);
  });
  $("#evidence-list").addEventListener("click", handleEvidenceClick);
  $("#retrieve-results").addEventListener("click", handleEvidenceClick);
  $("#close-pdf").addEventListener("click", closePdfPreview);
  $("#prev-pdf-page").addEventListener("click", () => movePdfPage(-1));
  $("#next-pdf-page").addEventListener("click", () => movePdfPage(1));
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

      const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 10);
      gradient.addColorStop(0, `rgba(${p.hue}, 0.2)`);
      gradient.addColorStop(1, `rgba(${p.hue}, 0)`);
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 10, 0, Math.PI * 2);
      ctx.fill();
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
