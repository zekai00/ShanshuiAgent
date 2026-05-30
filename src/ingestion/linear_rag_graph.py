# /root/Workspace/ChineseLandscape/src/core/linear_rag_graph.py
import os
import spacy
import re
from typing import List, Dict
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==========================================
# 1. 核心构建器 (Tri-Graph + 领域词库 + 实体对齐)
# ==========================================
class LinearRAGBuilder:
    def __init__(self, uri=None, user=None):
        print("[*] 正在加载本地 spaCy 中文 NER 模型 (兜底泛化用)...")
        self.nlp = spacy.load("zh_core_web_sm")
        self.target_labels = {"PERSON", "GPE", "LOC", "ORG", "WORK_OF_ART", "NORP"}
        
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

        # 安全读取连接信息
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            print("[!] 致命警告: 未在 .env 或环境变量中找到 NEO4J_PASSWORD！连接大概率失败。")

        print(f"[*] 正在连接 Neo4j 数据库 ({uri})...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("  ✅ Neo4j 连接成功！")
        except Exception as e:
            print(f"  ❌ Neo4j 连接失败: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def check_and_clear_database(self):
        """交互式防呆设计：检查数据库是否非空，询问用户是否清屏"""
        if not self.driver: return
        with self.driver.session() as session:
            # 统计当前节点总数
            result = session.run("MATCH (n) RETURN count(n) as cnt")
            node_count = result.single()["cnt"]
            
            if node_count > 0:
                print(f"\n[*] ⚠️ 侦测到 Neo4j 数据库中已有 {node_count} 个节点！")
                choice = input("    请选择操作 [1] 继承旧图谱追加数据  [2] 彻底销毁旧图谱，从0重构: ")
                if choice.strip() == '2':
                    print("    [!] 正在执行核弹清屏指令：MATCH (n) DETACH DELETE n ...")
                    session.run("MATCH (n) DETACH DELETE n")
                    print("    ✅ 旧图谱已彻底销毁，白纸准备就绪。")
                else:
                    print("    -> 选择追加模式，保留历史图谱节点。")
            else:
                print("\n[*] Neo4j 数据库为空，准备直接构建。")

    def _normalize_entity(self, entity_name: str) -> str:
        """根据 alias_dict 进行强力纠偏对齐"""
        return self.alias_dict.get(entity_name, entity_name)

    def extract_tri_graph_elements(self, chunk_id: int, contextual_chunk: str) -> Dict:
        """基于 NLP + 领域词库 + 正则的复合三层节点提取"""
        doc = self.nlp(contextual_chunk)
        graph_data = {"chunk_id": chunk_id, "chunk_text": contextual_chunk, "sentences": []}
        
        for sent_idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if len(sent_text) < 5: continue
            
            sentence_data = {"sent_id": f"{chunk_id}_{sent_idx}", "text": sent_text, "entities": []}
            extracted_entities = set()
            
            # [策略 A]：spaCy 原生兜底提取
            for ent in sent.ents:
                if ent.label_ in self.target_labels and len(ent.text) > 1:
                    norm_name = self._normalize_entity(ent.text)
                    extracted_entities.add((norm_name, ent.label_))
            
            # [策略 B]：正则强提作品名
            books = re.findall(r'《(.*?)》', sent_text)
            for b in books:
                extracted_entities.add((b, "WORK_OF_ART"))
                
            # 🌟 [策略 C]：领域词典雷达强扫
            for kw, kw_type in self.domain_lexicon.items():
                if kw in sent_text:
                    norm_name = self._normalize_entity(kw)
                    extracted_entities.add((norm_name, kw_type))
                    
            for name, e_type in extracted_entities:
                sentence_data["entities"].append({"name": name, "type": e_type})
                
            graph_data["sentences"].append(sentence_data)
            
        return graph_data

    def execute_cypher(self, graph_data: Dict):
        """将复合 Tri-Graph 数据写入数据库"""
        if not self.driver: return
        c_id = graph_data['chunk_id']
        c_text = graph_data['chunk_text'].replace("'", "\\'") 
        
        with self.driver.session() as session:
            # 创建 Chunk
            session.run(f"MERGE (c:Chunk {{id: {c_id}}}) ON CREATE SET c.text = '{c_text[:50]}...'")
            
            for sent in graph_data['sentences']:
                s_id = sent['sent_id']
                # 创建 Sentence 及关联
                session.run(f"MERGE (s:Sentence {{id: '{s_id}'}})")
                session.run(f"MATCH (c:Chunk {{id: {c_id}}}), (s:Sentence {{id: '{s_id}'}}) MERGE (c)-[:HAS_SENTENCE]->(s)")
                
                for ent in sent['entities']:
                    e_name = ent['name'].replace("'", "\\'")
                    e_type = ent['type']
                    # 创建 Entity 及关联
                    session.run(f"MERGE (e:Entity {{name: '{e_name}'}}) ON CREATE SET e.type = '{e_type}'")
                    session.run(f"MATCH (s:Sentence {{id: '{s_id}'}}), (e:Entity {{name: '{e_name}'}}) MERGE (e)-[:MENTIONED_IN]->(s)")

# ==========================================
# 2. 沙箱测试入口
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🕸️ LinearRAG Tri-Graph 构建器 (生产级配置版)")
    print("="*60)
    
    # 初始化构建器
    builder = LinearRAGBuilder() 
    
    # 交互式检查数据库
    builder.check_and_clear_database()
    
    sample_chunk_id = 10086
    real_text = """【全局上下文】本段综述了元代山水画的巅峰之作及四大家的艺术特色。
【原文资料】大元时期是文人画的鼎盛时期，其中“元四家”最具代表性。黄大痴晚年隐居富春江，历时数年绘制了传世名作《富春山居图》，全图用披麻皴，笔墨苍简，展现了道家的隐逸思想。而倪云林则常作一河两岸的构图，画面多枯木竹石，极度留白，其《容膝斋图》便透出一种清冷孤傲的禅意。清初的四王和四僧，对这些技法也多有继承。"""
    
    print("\n[*] 正在对真实语料进行抽取与实体归一化...")
    graph_data = builder.extract_tri_graph_elements(sample_chunk_id, real_text)
    
    for sent in graph_data['sentences']:
        print(f"\n  📝 句子: {sent['text']}")
        if sent['entities']:
            entities_str = [f"{e['name']}({e['type']})" for e in sent['entities']]
            print(f"     -> 🎯 提取实体: {entities_str}")
            
    print("\n[*] 正在将 Tri-Graph 结构写入 Neo4j ...")
    builder.execute_cypher(graph_data)
    builder.close()
    print("\n🎉 沙箱执行完毕！图谱状态已更新。")
