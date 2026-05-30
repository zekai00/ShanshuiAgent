# /root/Workspace/ChineseLandscape/src/retrieval/online_retrieval.py

import os
import sys
import warnings
import json
import re

# 🌟 1. 全局环境变量与屏蔽配置 (必须在最前)
os.environ["GRPC_KEEPALIVE_TIME_MS"] = "120000"
os.environ["GRPC_KEEPALIVE_TIMEOUT_MS"] = "20000"
os.environ["GRPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS"] = "120000"
os.environ["GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 隔离大模型显存
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import time
import pickle
import spacy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel, FlagReranker

load_dotenv()

# 🌟 2. 统一收口全局绝对路径 (杜绝变量覆写)
WORKSPACE_DIR = "/root/Workspace/ChineseLandscape"
sys.path.append(WORKSPACE_DIR)

PAPERS_FOLDER = os.path.join(WORKSPACE_DIR, "data", "raw_pdfs")
IMAGES_FOLDER = os.path.join(WORKSPACE_DIR, "data", "extracted_artworks")
DB_FOLDER = os.path.join(WORKSPACE_DIR, "data", "vector_store")
TRACKING_FILE = os.path.join(DB_FOLDER, "ingestion_state.json") 

MILVUS_DB_PATH = os.path.join(DB_FOLDER, "milvus_landscape.db")
COLBERT_FILE = os.path.join(DB_FOLDER, "colbert_tensors.pkl")
EVIDENCE_STORE_DIR = os.path.join(WORKSPACE_DIR, "data", "processed", "documents")
EVIDENCE_CHUNKS_FILE = os.path.join(EVIDENCE_STORE_DIR, "chunks.jsonl")
SOURCE_ALIASES_FILE = os.path.join(EVIDENCE_STORE_DIR, "source_aliases.json")

BGE_M3_PATH = "/root/models/bge-m3"
RERANKER_PATH = "/root/models/bge-reranker-v2-m3"


def clean_title(source_file: str) -> str:
    title = str(source_file or "").removesuffix(".pdf")
    title = title.replace("_NormalPdf", "").replace("NormalPdf", "")
    parts = title.split("_")
    if len(parts) > 1 and 1 <= len(parts[-1]) <= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", parts[-1]):
        title = "_".join(parts[:-1])
    return title.replace("_", "").strip("《》 ")


def normalize_title(value: str) -> str:
    value = clean_title(value)
    value = value.removesuffix(".pdf")
    value = re.sub(r"[\s《》“”\"'：:，,。·_\-—()（）【】\[\]]+", "", value)
    return value.lower()


def parse_contextual_chunk(text: str) -> tuple[str, str]:
    value = str(text or "").strip()
    if "【全局上下文】" in value and "【原文资料】" in value:
        prefix = value.split("【全局上下文】", 1)[1].split("【原文资料】", 1)[0].strip()
        raw = value.split("【原文资料】", 1)[1].strip()
        return prefix, raw
    return "", value

# ==========================================
# 在线检索引擎核心类
# ==========================================
class OnlineHybridRetriever:
    def __init__(self, top_k=15, final_k=5):
        self.top_k = top_k
        self.final_k = final_k
        print("\n" + "="*60)
        print("⚡ 启动数字敦煌 V3.0 极速检索引擎 (多路并发 + 智能截断)")
        print("="*60)
        
        print("[*] 正在挂载 BGE-M3 (Query 编码器)...")
        self.encoder = BGEM3FlagModel(BGE_M3_PATH, use_fp16=True, device="cuda:0")
        
        print("[*] 正在挂载 BGE-Reranker-v2-m3 (交叉精排器)...")
        self.reranker = FlagReranker(RERANKER_PATH, use_fp16=True, device="cuda:0")
        
        print("[*] 正在连接 Milvus Lite 双擎数据库...")
        self.milvus_client = MilvusClient(MILVUS_DB_PATH)
        self.collection_name = "landscape_rag"
        self.milvus_client.load_collection(self.collection_name)

        self.evidence_by_legacy_id = {}
        self.source_to_legacy_ids = defaultdict(list)
        self.source_aliases = {}
        self._load_evidence_store()
        
        print("[*] 正在载入 ColBERT 细粒度张量阵列...")
        if os.path.exists(COLBERT_FILE):
            with open(COLBERT_FILE, "rb") as f:
                self.colbert_db = pickle.load(f)
        else:
            self.colbert_db = {}
            
        print("[*] 正在连接 Neo4j 知识图谱...")
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USERNAME", "neo4j")
        neo4j_pwd = os.environ.get("NEO4J_PASSWORD", "neo4j")
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
            self.neo4j_driver.verify_connectivity()
            print(f"  [*] Neo4j 连接成功: {neo4j_uri} ({neo4j_user})")
        except Exception as exc:
            self.neo4j_driver = None
            print(f"  [!] 图谱连接失败，将降级为无图谱检索模式: {exc}")
        
        self.nlp = spacy.load("zh_core_web_sm")
        # 🌟 统一更新 1：山水画核心领域词典 (Domain Lexicon)
        # 必须涵盖受控词表中的所有朝代及其常见变体
        self.domain_lexicon = {
            # --- 朝代 (全量覆盖) ---
            "先秦": "DYNASTY", "秦汉": "DYNASTY", "魏晋南北朝": "DYNASTY", 
            "隋": "DYNASTY", "隋代": "DYNASTY", "隋朝": "DYNASTY",
            "唐": "DYNASTY", "唐代": "DYNASTY", "唐朝": "DYNASTY", "大唐": "DYNASTY",
            "五代十国": "DYNASTY", "五代": "DYNASTY",
            "宋": "DYNASTY", "宋代": "DYNASTY", "宋朝": "DYNASTY", "大宋": "DYNASTY",
            "北宋": "DYNASTY", "南宋": "DYNASTY", 
            "辽金西夏": "DYNASTY",
            "元": "DYNASTY", "元代": "DYNASTY", "元朝": "DYNASTY", "大元": "DYNASTY",
            "明": "DYNASTY", "明代": "DYNASTY", "明朝": "DYNASTY", "大明": "DYNASTY",
            "清": "DYNASTY", "清代": "DYNASTY", "清朝": "DYNASTY", "大清": "DYNASTY",
            "近现代": "DYNASTY",
            
            # --- 核心名家/别号/流派 (保持不变) ---
            "董源": "PERSON", "巨然": "PERSON", "郭熙": "PERSON", "马远": "PERSON", "夏圭": "PERSON",
            "倪瓒": "PERSON", "倪云林": "PERSON", "云林子": "PERSON",
            "黄公望": "PERSON", "黄大痴": "PERSON", "一峰道人": "PERSON",
            "王蒙": "PERSON", "吴镇": "PERSON",
            "石涛": "PERSON", "原济": "PERSON", "大涤子": "PERSON", "苦瓜和尚": "PERSON",
            "弘仁": "渐江", "渐江": "PERSON", "髡残": "PERSON", "石溪": "PERSON",
            "八大山人": "PERSON", "朱耷": "PERSON",
            "王时敏": "PERSON", "王鉴": "PERSON", "王翚": "PERSON", "王原祁": "PERSON",
            "四王": "SCHOOL", "四僧": "SCHOOL", "元四家": "SCHOOL", "华亭派": "SCHOOL", "吴门画派": "SCHOOL",
            
            # --- 核心笔墨与构图理论 (保持不变) ---
            "披麻皴": "TECHNIQUE", "斧劈皴": "TECHNIQUE", "雨点皴": "TECHNIQUE", "卷云皴": "TECHNIQUE",
            "解索皴": "TECHNIQUE", "折带皴": "TECHNIQUE", "牛毛皴": "TECHNIQUE", "荷叶皴": "TECHNIQUE",
            "泼墨": "TECHNIQUE", "没骨": "TECHNIQUE", "浅绛": "TECHNIQUE", "金碧": "TECHNIQUE", "青绿": "TECHNIQUE",
            "留白": "CONCEPT", "一河两岸": "CONCEPT", "经营位置": "CONCEPT", "置阵布势": "CONCEPT",
            "三远法": "CONCEPT", "高远": "CONCEPT", "深远": "CONCEPT", "平远": "CONCEPT",
            "虚实相生": "CONCEPT", "计白当黑": "CONCEPT", "疏密相间": "CONCEPT", "纵横开合": "CONCEPT"
        }

        # 🌟 统一更新 2：实体对齐字典 (Entity Normalization)
        # 核心原则：将所有俗称，强行映射为 paper_prompts.py 中的标准受控词！
        # 绝对禁止将“北宋”抹杀为“宋”，保留知识层级！
        self.alias_dict = {
            # --- 朝代归一化 (向单个汉字的受控主键看齐) ---
            "隋代": "隋", "隋朝": "隋",
            "唐代": "唐", "唐朝": "唐", "大唐": "唐",
            "五代": "五代十国",
            "宋代": "宋", "宋朝": "宋", "大宋": "宋",
            # 注意：不映射北宋和南宋，让它们保持原样（因为北宋本身就是受控词表里的标准词）
            "元代": "元", "元朝": "元", "大元": "元",
            "明代": "明", "明朝": "明", "大明": "明",
            "清代": "清", "清朝": "清", "大清": "清",
            
            # --- 画家归一化 ---
            "倪云林": "倪瓒", "云林子": "倪瓒",
            "黄大痴": "黄公望", "一峰道人": "黄公望",
            "石涛": "原济", "大涤子": "原济", "苦瓜和尚": "原济",
            "弘仁": "渐江", "髡残": "石溪", "八大山人": "朱耷"
        }
        
        print("✅ 检索引擎启动完毕！")

    def _load_evidence_store(self):
        if not os.path.exists(EVIDENCE_CHUNKS_FILE):
            print("  [!] 未发现 canonical evidence store，将使用 Milvus 原始字段降级检索。")
            return

        loaded = 0
        with open(EVIDENCE_CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                legacy_id = str(chunk.get("legacy_milvus_id", ""))
                source_file = str(chunk.get("source_file", ""))
                if not legacy_id or not source_file:
                    continue
                self.evidence_by_legacy_id[legacy_id] = chunk
                self.source_to_legacy_ids[source_file].append(legacy_id)
                self.source_aliases[normalize_title(source_file)] = source_file
                self.source_aliases[normalize_title(chunk.get("title", ""))] = source_file
                loaded += 1

        if os.path.exists(SOURCE_ALIASES_FILE):
            with open(SOURCE_ALIASES_FILE, "r", encoding="utf-8") as f:
                aliases = json.load(f)
            self.source_aliases.update({str(k): str(v) for k, v in aliases.items()})

        print(f"[*] 已挂载 canonical evidence store: {loaded} 个 chunk, {len(self.source_to_legacy_ids)} 篇文献。")

    def _extract_query_entities(self, query: str) -> list:
        doc = self.nlp(query)
        entities = set()
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "WORK_OF_ART", "NORP"} and len(ent.text) > 1:
                entities.add(self.alias_dict.get(ent.text, ent.text))
        for kw in self.domain_lexicon.keys():
            if kw in query:
                entities.add(self.alias_dict.get(kw, kw))
        return list(entities)

    def _resolve_source_constraints(self, query: str) -> list:
        sources = []
        for title in re.findall(r"《([^》]+)》", query):
            norm = normalize_title(title)
            source = self.source_aliases.get(norm)
            if not source:
                for alias, candidate in self.source_aliases.items():
                    if norm and (norm in alias or alias in norm):
                        source = candidate
                        break
            if source and source not in sources:
                sources.append(source)
        return sources

    def _query_terms(self, query: str) -> list:
        text = re.sub(r"《[^》]+》", " ", query)
        stop_terms = {
            "根据", "文献", "资料", "主要", "说明", "什么", "问题", "如何", "是否",
            "比较", "差异", "中的", "中国", "山水画", "研究", "请基于", "整理",
        }
        terms = {kw for kw in self.domain_lexicon.keys() if kw in query}
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,16}", text):
            if token not in stop_terms:
                terms.add(token)
        return sorted(terms, key=len, reverse=True)

    def _search_source_lexical(self, query: str, source_constraints: list) -> list:
        if not source_constraints or not self.evidence_by_legacy_id:
            return []

        terms = self._query_terms(query)
        results = []
        for source in source_constraints:
            for legacy_id in self.source_to_legacy_ids.get(source, []):
                evidence = self.evidence_by_legacy_id.get(legacy_id, {})
                text = "\n".join([
                    str(evidence.get("title", "")),
                    str(evidence.get("raw_chunk_text", "")),
                    str(evidence.get("contextual_prefix", "")),
                ])
                score = 0
                for term in terms:
                    score += text.count(term) * max(len(term), 2)
                if score > 0:
                    results.append((legacy_id, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return [legacy_id for legacy_id, _ in results[: max(self.top_k * max(len(source_constraints), 1), self.top_k)]]
    
    def _search_dense(self, dense_vec) -> list:
        res = self.milvus_client.search(collection_name=self.collection_name, data=[dense_vec], anns_field="dense_vector", limit=self.top_k, output_fields=["id"])
        return [str(hit["id"]) for hit in res[0]] if res and res[0] else []

    def _search_sparse(self, sparse_dict) -> list:
        clean_sparse = {int(k): float(v) for k, v in sparse_dict.items()}
        res = self.milvus_client.search(collection_name=self.collection_name, data=[clean_sparse], anns_field="sparse_vector", limit=self.top_k, output_fields=["id"])
        return [str(hit["id"]) for hit in res[0]] if res and res[0] else []

    def _search_colbert(self, query_colbert_vec) -> list:
        if not self.colbert_db: return []
        scores = []
        for chunk_id, doc_tensor in self.colbert_db.items():
            score = self.encoder.colbert_score(query_colbert_vec, doc_tensor)
            scores.append((chunk_id, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [str(item[0]) for item in scores[:self.top_k]]

    # 修改 src/retrieval/online_retrieval.py
    def _search_graph(self, entities: list) -> list:
        if not entities or not self.neo4j_driver: 
            return []
        
        cypher_query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(s:Sentence)<-[:HAS_SENTENCE]-(c:Chunk)
        WHERE e.name IN $entities
        RETURN c.id AS chunk_id, count(s) AS hit_weight
        ORDER BY hit_weight DESC LIMIT $limit
        """
        
        try:
            # 🌟 增加 try-except，防止 Neo4j 离线导致全盘崩溃
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query, entities=entities, limit=self.top_k)
                return [str(record["chunk_id"]) for record in result]
        except Exception as e:
            print(f"  [⚠️ Graph RAG Warning] 图谱检索执行异常（可能服务已断开）: {e}")
            return []

    def retrieve_and_rerank(self, query: str):
        start_time = time.time()
        print(f"\n[1/5] 提纯 Query 特征: '{query}'")
        source_constraints = self._resolve_source_constraints(query)
        if source_constraints:
            print(f"    -> Source-aware 约束: {source_constraints}")

        q_features = self.encoder.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=True)
        dense_vec = q_features['dense_vecs'][0].tolist()
        sparse_vec = q_features['lexical_weights'][0]
        colbert_vec = q_features['colbert_vecs'][0]
        
        entities = self._extract_query_entities(query)
        
        print(f"[2/5] 四路并发穿透检索...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_dense = executor.submit(self._search_dense, dense_vec)
            future_sparse = executor.submit(self._search_sparse, sparse_vec)
            future_colbert = executor.submit(self._search_colbert, colbert_vec)
            future_graph = executor.submit(self._search_graph, entities)
            
            dense_hits = future_dense.result()
            sparse_hits = future_sparse.result()
            colbert_hits = future_colbert.result()
            graph_hits = future_graph.result()

        source_hits = self._search_source_lexical(query, source_constraints)
            
        print(f"[3/5] 执行 RRF 排位融合...")
        rrf_scores = defaultdict(float)
        k_constant = 60
        hit_lists = [dense_hits, sparse_hits, colbert_hits, graph_hits]
        if source_hits:
            # Source-constrained candidates are intentionally weighted higher:
            # if the user names a PDF, the evidence must come from that PDF first.
            hit_lists.extend([source_hits, source_hits])
        for hit_list in hit_lists:
            for rank, chunk_id in enumerate(hit_list):
                rrf_scores[str(chunk_id)] += 1.0 / (k_constant + rank + 1)
                
        fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        candidate_ids = [cid for cid, score in fused_candidates]
        if not candidate_ids: return []

        print(f"[4/5] 提取 {len(candidate_ids)} 个精英候选全文...")
        id_str = ",".join(map(str, candidate_ids))
        raw_docs = self.milvus_client.query(
            collection_name=self.collection_name, filter=f"id in [{id_str}]",
            output_fields=["contextual_chunk", "source_file", "dynasty", "painter", "subject_matter", "content_scope"]
        )
        doc_map = {str(doc["id"]): doc for doc in raw_docs}
        
        print(f"[5/5] Reranker 交叉精排及阈值熔断判定...")
        sentence_pairs = []
        valid_candidate_ids = []
        for cid in candidate_ids:
            if cid in doc_map:
                evidence = self.evidence_by_legacy_id.get(str(cid), {})
                rerank_text = evidence.get("retrieval_text") or doc_map[cid].get("contextual_chunk", "")
                sentence_pairs.append([query, rerank_text])
                valid_candidate_ids.append(cid)
                
        rerank_scores = self.reranker.compute_score(sentence_pairs)
        if isinstance(rerank_scores, float): rerank_scores = [rerank_scores]
        
        final_results = []
        for i, cid in enumerate(valid_candidate_ids):
            doc = doc_map[cid]
            legacy_id = str(doc.get("id", cid))
            evidence = self.evidence_by_legacy_id.get(legacy_id, {})
            legacy_contextual_chunk = str(doc.get("contextual_chunk", ""))
            fallback_prefix, fallback_raw = parse_contextual_chunk(legacy_contextual_chunk)
            source_file = str(evidence.get("source_file") or doc.get("source_file", ""))
            raw_chunk_text = str(evidence.get("raw_chunk_text") or fallback_raw)
            contextual_prefix = str(evidence.get("contextual_prefix") or fallback_prefix)
            retrieval_text = str(evidence.get("retrieval_text") or legacy_contextual_chunk)
            # 🌟 核心修复：强行构造纯净的 Python 标准字典，杜绝 Milvus/Numpy 脏数据类型
            clean_doc = {
                "id": legacy_id,
                "legacy_milvus_id": legacy_id,
                "chunk_id": str(evidence.get("chunk_id") or f"legacy_{legacy_id}"),
                "doc_id": str(evidence.get("doc_id") or ""),
                "title": str(evidence.get("title") or clean_title(source_file)),
                "contextual_chunk": retrieval_text,
                "raw_chunk_text": raw_chunk_text,
                "contextual_prefix": contextual_prefix,
                "retrieval_text": retrieval_text,
                "source_file": source_file,
                "page_start": evidence.get("page_start"),
                "page_end": evidence.get("page_end"),
                "section_title": evidence.get("section_title"),
                "dynasty": str(doc.get("dynasty", "")),
                "painter": str(doc.get("painter", "")),
                "subject_matter": str(doc.get("subject_matter", "")),
                "content_scope": str(doc.get("content_scope", "")),
                "source_constraints": source_constraints,
                "source_constraint_match": source_file in source_constraints if source_constraints else None,
                "evidence_store_hit": bool(evidence),
                "rerank_score": float(rerank_scores[i])
            }
            final_results.append(clean_doc)
            
        final_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        filtered_results = [doc for doc in final_results if doc["rerank_score"] >= -3]
        
        if not filtered_results:
            print("\n[!] 🚨 裁判长判定熔断：得分过低，返回兜底 Top-1 以免大模型交白卷。")
            # 兜底机制：就算全不及格，也把排第一的死马当活马医返回去
            return final_results[:1]
            
        dynamic_final_k = max(self.final_k, len(source_constraints) * 2) if source_constraints else self.final_k
        if source_constraints:
            constrained = [doc for doc in filtered_results if doc["source_file"] in set(source_constraints)]
            if constrained:
                selected = []
                selected_ids = set()
                for source in source_constraints:
                    for doc in constrained:
                        if doc["source_file"] == source and doc["chunk_id"] not in selected_ids:
                            selected.append(doc)
                            selected_ids.add(doc["chunk_id"])
                            break
                for doc in constrained:
                    if len(selected) >= dynamic_final_k:
                        break
                    if doc["chunk_id"] not in selected_ids:
                        selected.append(doc)
                        selected_ids.add(doc["chunk_id"])
                top_k_results = selected[:dynamic_final_k]
            else:
                top_k_results = filtered_results[:dynamic_final_k]
        else:
            top_k_results = filtered_results[:dynamic_final_k]
        print(f"\n✅ 检索闭环完成！总耗时: {time.time() - start_time:.3f} 秒")
        return top_k_results

    def close(self):
        if self.neo4j_driver: self.neo4j_driver.close()
        self.milvus_client.close()

if __name__ == "__main__":
    retriever = OnlineHybridRetriever(top_k=15, final_k=3)
    test_queries = ["黄公望的富春山居图用了什么皴法？", "外星人是否访问过画院？"]
    for i, query in enumerate(test_queries):
        print("\n" + "▼"*60)
        print(f"🎯 [测试任务 {i+1}] 用户提问: {query}")
        results = retriever.retrieve_and_rerank(query)
        if not results:
            print("[结果] 抱歉，资料库中未检索到相关内容。")
            continue
        print("\n" + "🏆"*15 + " 黄金榜单 " + "🏆"*15)
        for rank, doc in enumerate(results):
            print(f"\n【第 {rank+1} 名】| 得分: {doc['rerank_score']:.4f}")
            print(f" 📂 文献: {doc.get('source_file')} | 🗿: {doc.get('dynasty')} / {doc.get('painter')}")
    retriever.close()
