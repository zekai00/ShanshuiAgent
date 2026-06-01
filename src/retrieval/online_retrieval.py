# /root/Workspace/ShanshuiAgent/src/retrieval/online_retrieval.py

import os
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
from neo4j import GraphDatabase
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from src.config import (
    BGE_M3_PATH,
    MODEL_DEVICE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    RERANKER_PATH,
    RETRIEVAL_COLBERT_TENSORS_PATH,
    RETRIEVAL_COLLECTION_NAME,
    RETRIEVAL_EVIDENCE_DIR,
    RETRIEVAL_MILVUS_DB_PATH,
    SPACY_MODEL_NAME,
)
from src.domain_terms import (
    LANDSCAPE_ALIAS_DICT,
    LANDSCAPE_DOMAIN_LEXICON,
    MODERN_NOISE_TERMS,
    QUERY_STOP_TERMS,
)

EVIDENCE_CHUNKS_FILE = RETRIEVAL_EVIDENCE_DIR / "chunks.jsonl"
SOURCE_ALIASES_FILE = RETRIEVAL_EVIDENCE_DIR / "source_aliases.json"
SOURCE_PRIOR_STOP_TERMS = {
    "中国山水画",
    "山水画",
    "画史",
    "时代",
    "流派",
    "笔墨",
    "构图",
    "现代",
    "是否",
    "是不是",
}


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
        use_fp16 = str(MODEL_DEVICE).startswith("cuda")
        self.encoder = BGEM3FlagModel(str(BGE_M3_PATH), use_fp16=use_fp16, device=MODEL_DEVICE)
        
        print("[*] 正在挂载 BGE-Reranker-v2-m3 (交叉精排器)...")
        self.reranker = FlagReranker(str(RERANKER_PATH), use_fp16=use_fp16, device=MODEL_DEVICE)
        
        print("[*] 正在连接 Milvus Lite 双擎数据库...")
        self.milvus_client = MilvusClient(str(RETRIEVAL_MILVUS_DB_PATH))
        self.collection_name = RETRIEVAL_COLLECTION_NAME
        self.milvus_client.load_collection(self.collection_name)

        self.evidence_by_legacy_id = {}
        self.source_to_legacy_ids = defaultdict(list)
        self.source_aliases = {}
        self.source_profile_text = {}
        self._load_evidence_store()
        
        print("[*] 正在载入 ColBERT 细粒度张量阵列...")
        if RETRIEVAL_COLBERT_TENSORS_PATH.exists():
            with RETRIEVAL_COLBERT_TENSORS_PATH.open("rb") as f:
                self.colbert_db = pickle.load(f)
        else:
            self.colbert_db = {}
            
        print("[*] 正在连接 Neo4j 知识图谱...")
        if not NEO4J_PASSWORD:
            self.neo4j_driver = None
            print("  [!] 未配置 NEO4J_PASSWORD，将降级为无图谱检索模式。")
        else:
            try:
                self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                self.neo4j_driver.verify_connectivity()
                print(f"  [*] Neo4j 连接成功: {NEO4J_URI} ({NEO4J_USERNAME})")
            except Exception as exc:
                self.neo4j_driver = None
                print(f"  [!] 图谱连接失败，将降级为无图谱检索模式: {exc}")
        
        self.nlp = spacy.load(SPACY_MODEL_NAME)
        self.domain_lexicon = LANDSCAPE_DOMAIN_LEXICON
        self.alias_dict = LANDSCAPE_ALIAS_DICT
        
        print("✅ 检索引擎启动完毕！")

    def _load_evidence_store(self):
        if not EVIDENCE_CHUNKS_FILE.exists():
            print("  [!] 未发现 canonical evidence store，将使用 Milvus 原始字段降级检索。")
            return

        loaded = 0
        with EVIDENCE_CHUNKS_FILE.open("r", encoding="utf-8") as f:
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
                if source_file not in self.source_profile_text:
                    self.source_profile_text[source_file] = "\n".join([
                        str(chunk.get("title", "")),
                        clean_title(source_file),
                        str(chunk.get("contextual_prefix", "")),
                    ])
                loaded += 1

        if SOURCE_ALIASES_FILE.exists():
            with SOURCE_ALIASES_FILE.open("r", encoding="utf-8") as f:
                aliases = json.load(f)
            for alias, source in aliases.items():
                self.source_aliases[str(alias)] = str(source)
                self.source_aliases[normalize_title(str(alias))] = str(source)

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
        terms = {kw for kw in self.domain_lexicon.keys() if kw in query}
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,16}", text):
            if token not in QUERY_STOP_TERMS:
                terms.add(token)
        return sorted(terms, key=len, reverse=True)

    def _contains_modern_noise(self, query: str) -> bool:
        normalized = query.lower()
        return any(term.lower() in normalized for term in MODERN_NOISE_TERMS)

    def _strip_modern_noise(self, query: str) -> str:
        cleaned = query
        for term in sorted(MODERN_NOISE_TERMS, key=len, reverse=True):
            cleaned = re.sub(re.escape(term), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"是否|是不是|能否|可否|为什么|如何|主要|因为|才|直接|参与|使用|属于|等同于|等同|决定|记录|呈现|开发|设计|发行", " ", cleaned)
        cleaned = re.sub(r"[？?，,。；;：:、（）()【】\[\]\"“”'‘’]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _build_corrective_query(self, query: str, entities: list[str], source_constraints: list[str]) -> str:
        if source_constraints:
            return ""
        if not self._contains_modern_noise(query) and not re.search(r"是否|是不是|能否|可否", query):
            return ""

        cleaned = self._strip_modern_noise(query)
        quoted_titles = re.findall(r"《([^》]+)》", query)
        terms = []
        for value in [*entities, *quoted_titles, *self._query_terms(cleaned)]:
            value = str(value).strip()
            if value and value not in QUERY_STOP_TERMS and value not in terms:
                terms.append(value)

        for anchor in ["中国山水画", "画史", "时代", "流派", "笔墨", "构图"]:
            if anchor not in terms:
                terms.append(anchor)

        return " ".join(terms[:16])

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

    def _search_source_profile(self, query: str) -> tuple[list[str], list[str]]:
        if not query or not self.source_profile_text:
            return [], []

        terms = [
            term for term in self._query_terms(query)
            if len(term) >= 2 and term not in SOURCE_PRIOR_STOP_TERMS
        ]
        if not terms:
            return [], []

        scored_sources = []
        for source, profile_text in self.source_profile_text.items():
            score = 0
            compact_title = normalize_title(source)
            compact_profile = normalize_title(profile_text)
            for term in terms:
                compact_term = normalize_title(term)
                if not compact_term:
                    continue
                if compact_term in compact_title:
                    score += max(len(compact_term), 2) * 3
                elif compact_term in compact_profile:
                    score += max(len(compact_term), 2)
            if score >= 8:
                scored_sources.append((source, score))

        scored_sources.sort(key=lambda item: item[1], reverse=True)
        sources = [source for source, _score in scored_sources[:3]]
        return self._search_source_lexical(query, sources), sources
    
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

        entities = self._extract_query_entities(query)
        corrective_query = self._build_corrective_query(query, entities, source_constraints)
        query_variants = [query]
        if corrective_query and corrective_query != query:
            query_variants.append(corrective_query)
            print(f"    -> 错误前提去噪 Query: {corrective_query}")
            corrective_entities = self._extract_query_entities(corrective_query)
            entities = list(dict.fromkeys([*entities, *corrective_entities]))

        q_features = self.encoder.encode(query_variants, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        
        print(f"[2/5] 多路并发穿透检索...")
        weighted_hit_lists = []
        with ThreadPoolExecutor(max_workers=max(4, len(query_variants) * 3 + 1)) as executor:
            futures = []
            for idx, _variant in enumerate(query_variants):
                dense_vec = q_features['dense_vecs'][idx].tolist()
                sparse_vec = q_features['lexical_weights'][idx]
                colbert_vec = q_features['colbert_vecs'][idx]
                # Corrective queries are more reliable for false-premise questions
                # because they remove modern anachronistic distractors.
                weight = 2.0 if idx > 0 else 1.0
                futures.extend([
                    (weight, executor.submit(self._search_dense, dense_vec)),
                    (weight, executor.submit(self._search_sparse, sparse_vec)),
                    (weight, executor.submit(self._search_colbert, colbert_vec)),
                ])
            futures.append((1.0, executor.submit(self._search_graph, entities)))

            for weight, future in futures:
                weighted_hit_lists.append((weight, future.result()))

        source_hits = self._search_source_lexical(query, source_constraints)
        source_prior_query = self._strip_modern_noise(query)
        source_prior_hits, source_prior_sources = self._search_source_profile(source_prior_query)
        if source_prior_sources and not source_constraints:
            print(f"    -> 文献标题/主题软先验: {source_prior_sources}")
            
        print(f"[3/5] 执行 RRF 排位融合...")
        rrf_scores = defaultdict(float)
        k_constant = 60
        hit_lists = weighted_hit_lists
        if source_hits:
            # Source-constrained candidates are intentionally weighted higher:
            # if the user names a PDF, the evidence must come from that PDF first.
            hit_lists.extend([(2.0, source_hits), (2.0, source_hits)])
        if source_prior_hits:
            hit_lists.append((1.5, source_prior_hits))
        for weight, hit_list in hit_lists:
            for rank, chunk_id in enumerate(hit_list):
                rrf_scores[str(chunk_id)] += weight * (1.0 / (k_constant + rank + 1))
                
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
        rerank_query = corrective_query or query
        for cid in candidate_ids:
            if cid in doc_map:
                evidence = self.evidence_by_legacy_id.get(str(cid), {})
                rerank_text = evidence.get("retrieval_text") or doc_map[cid].get("contextual_chunk", "")
                sentence_pairs.append([rerank_query, rerank_text])
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
                "source_prior_sources": source_prior_sources,
                "source_prior_match": source_file in source_prior_sources if source_prior_sources else None,
                "query_variants": query_variants,
                "corrective_query": corrective_query,
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
            
        if source_constraints:
            dynamic_final_k = max(self.final_k, len(source_constraints) * 2)
        elif corrective_query:
            dynamic_final_k = max(self.final_k, 5)
        else:
            dynamic_final_k = self.final_k
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
