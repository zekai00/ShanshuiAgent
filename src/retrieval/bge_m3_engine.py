# /root/Workspace/ChineseLandscape/src/core/bge_m3_engine.py
import os
import torch
from pprint import pprint
# 这里必须使用智源官方的 FlagEmbedding，LangChain 的不支持三输出
from FlagEmbedding import BGEM3FlagModel 

class BGEM3Engine:
    def __init__(self, model_path: str = "/root/models/bge-m3", device: str = "cuda:0"):
        """
        原理：加载 BGE-M3 权重。这是一个多任务混合架构的模型。
        use_fp16=True 可以节省一半显存，且几乎不影响检索精度。
        """
        print(f"[*] 正在将 BGE-M3 三栖引擎挂载至 {device} (启用 FP16 加速)...")
        self.model = BGEM3FlagModel(model_path, use_fp16=True, device=device)
        print("[*] BGE-M3 挂载完毕！")

    def encode_corpus(self, texts: list):
        """
        离线建库专用：一次性计算出 Dense(稠密), Sparse(稀疏词频), ColBERT(多向量)
        """
        print(f"[*] 正在提取 {len(texts)} 条文本的三维特征...")
        # batch_size 不要太大，ColBERT 极其吃显存
        embeddings = self.model.encode(texts, 
                                       batch_size=2, 
                                       max_length=512, 
                                       return_dense=True, 
                                       return_sparse=True, 
                                       return_colbert_vecs=True)
        return embeddings

    def compute_scores(self, query: str, corpus_texts: list, corpus_embeddings: dict):
        """
        在线检索演示：演示如何分别计算三路的得分
        """
        query_emb = self.model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=True)
        
        # 1. 稠密得分计算 (内积/余弦相似度)
        # dense_vecs 是规则的 numpy 矩阵，直接用矩阵乘法 (@) 瞬间算出 Query 和所有文档的得分
        dense_scores = (query_emb['dense_vecs'] @ corpus_embeddings['dense_vecs'].T)[0].tolist()
        
        # 2. 稀疏得分计算 (Lexical Weights 点积)
        sparse_scores = [self.model.compute_lexical_matching_score(query_emb['lexical_weights'][0], doc_weight) 
                         for doc_weight in corpus_embeddings['lexical_weights']]
        
        # 3. ColBERT 细粒度多向量得分计算 (MaxSim 矩阵点积)
        # 因为每个切片的 Token 数量不同（保存在 list 中），必须逐个取出来与 Query 进行计算
        q_colbert_vec = query_emb['colbert_vecs'][0]
        colbert_scores = [self.model.colbert_score(q_colbert_vec, p_vec) 
                          for p_vec in corpus_embeddings['colbert_vecs']]
        
        return dense_scores, sparse_scores, colbert_scores

# 沙箱实验
if __name__ == "__main__":
    # 强制让沙箱跑在 GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    engine = BGEM3Engine()
    
    # 1. 虚拟两段古籍切片
    corpus = [
        "黄公望晚年结庐富春江，用披麻皴绘就长卷，尽显道家隐逸之风，笔墨虚实相生。",
        "李思训擅长青绿山水，金碧辉煌，线条刚劲挺拔，为北宗山水之祖。"
    ]
    
    # 2. 离线建库：一把梭哈提取三种特征
    corpus_embs = engine.encode_corpus(corpus)
    
    print("\n==== 📊 离线数据特征维度窥探 ====")
    print(f"-> Dense 稠密向量形状: {corpus_embs['dense_vecs'].shape} (每句话被强行压缩成1个1024维向量)")
    print(f"-> Sparse 稀疏权重数量: 句子1激活了 {len(corpus_embs['lexical_weights'][0])} 个词汇权重")
    
    # 彻底揭开 ColBERT 的真面目
    print(f"-> ColBERT 细粒度矩阵结构:")
    print(f"   总共有 {len(corpus_embs['colbert_vecs'])} 个外层切片 (因为输入了2句话)")
    print(f"   🔍 切片 1 的真实 Token 矩阵形状: {corpus_embs['colbert_vecs'][0].shape}")
    print(f"      (这代表第1句话被切成了 {corpus_embs['colbert_vecs'][0].shape[0]} 个独立的 Token，每个 Token 都有自己的 1024 维表示！)")
    
    # 3. 在线检索演示
    query = "请问元代有哪些画家用了披麻皴？"
    print(f"\n==== 🔍 模拟在线检索: '{query}' ====")
    dense_s, sparse_s, colbert_s = engine.compute_scores(query, corpus, corpus_embs)
    
    print("\n[三路打分结果比对]")
    for i, text in enumerate(corpus):
        print(f"\n切片 {i+1}: {text[:20]}...")
        # 修正：直接用 [i] 获取当前切片的得分
        print(f"   🎯 稠密语义得分 (懂不懂意思): {dense_s[i]:.4f}")
        print(f"   🎯 稀疏词频得分 (有没有精准命中'披麻皴'): {sparse_s[i]:.4f}")
        print(f"   🎯 细粒度排版得分 (上下文咬合度): {colbert_s[i]:.4f}")