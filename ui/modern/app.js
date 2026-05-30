const state = {
  history: [],
  evidence: [],
  health: null,
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

function renderMessage(role, content, loading = false) {
  const node = document.createElement("div");
  node.className = `message ${role}${loading ? " loading" : ""}`;
  node.textContent = content;
  $("#messages").appendChild(node);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return node;
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
    node.innerHTML = `
      <strong>${escapeHtml(item.source_file || item.title || "未知来源")}</strong>
      <div class="evidence-meta">
        <span>#${escapeHtml(item.rank)}</span>
        <span>${escapeHtml(pageLabel(item))}</span>
        <span>${escapeHtml(scoreLabel(item))}</span>
      </div>
      <p>${escapeHtml(item.preview || "")}</p>
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
    const manifest = data.manifest || {};
    const chunks = manifest.chunk_count || manifest.chunks || "ready";
    $("#runtime-line").textContent = `Authority evidence: ${chunks}`;
    renderSystem(data);
  } catch (error) {
    setHealth(false, "服务异常");
    $("#system-panel").innerHTML = `<div class="metric"><span>错误</span><strong>${escapeHtml(error.message)}</strong></div>`;
  }
}

async function sendChat(message) {
  renderMessage("user", message);
  state.history.push({ role: "user", content: message });
  const loadingNode = renderMessage("assistant", "检索中", true);
  $("#send-button").disabled = true;
  try {
    const data = await fetchJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, history: state.history, final_k: 5 }),
    });
    loadingNode.classList.remove("loading");
    loadingNode.textContent = data.answer || "未生成回答";
    state.history.push({ role: "assistant", content: loadingNode.textContent });
    updateEvidence(data.evidence || []);
  } catch (error) {
    loadingNode.classList.remove("loading");
    loadingNode.textContent = `请求失败：${error.message}`;
  } finally {
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

async function loadCorpus() {
  const table = $("#corpus-table");
  table.innerHTML = '<div class="empty">加载中</div>';
  try {
    const data = await fetchJson("/api/corpus");
    const manifest = data.manifest || {};
    const docs = data.documents || [];
    $("#corpus-summary").textContent = `${manifest.document_count || docs.length} 篇文献，${manifest.chunk_count || "?"} 个证据块`;
    table.innerHTML = "";
    if (!docs.length) {
      table.innerHTML = '<div class="empty">暂无文献清单</div>';
      return;
    }
    for (const doc of docs) {
      const row = document.createElement("div");
      row.className = "doc-row";
      row.innerHTML = `
        <strong>${escapeHtml(doc.title || doc.source_file)}</strong>
        <span class="pill">${escapeHtml(doc.authority_level || "n/a")}</span>
        <span>${escapeHtml(doc.chunk_count || 0)} 块</span>
        <span>${escapeHtml(doc.page_count || 0)} 页</span>
      `;
      table.appendChild(row);
    }
  } catch (error) {
    table.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderSystem(data) {
  const manifest = data.manifest || {};
  const metrics = [
    ["LLM", data.llm_configured ? "可用" : "未配置"],
    ["Evidence Dir", data.evidence_dir || "未知"],
    ["Documents", manifest.document_count ?? "未知"],
    ["Chunks", manifest.chunk_count ?? "未知"],
  ];
  $("#system-panel").innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function switchView(view) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $$(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
  if (view === "corpus") loadCorpus();
  if (view === "system" && state.health) renderSystem(state.health);
}

function setupEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("#chat-input");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    sendChat(message);
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
loadHealth();
renderMessage("assistant", "请输入一个山水画史问题。");
