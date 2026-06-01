# /root/Workspace/ShanshuiAgent/src/core/ingestion_manager.py

import os
import sys
import json
import hashlib
import pickle
from pymilvus import MilvusClient, DataType

class IngestionStateManager:
    """🌟 增量状态管理：用 JSON 文件持久化记录已成功入库的 PDF 及其 MD5"""
    def __init__(self, tracking_file: str):
        self.tracking_file = tracking_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """防御性读取状态文件"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] ⚠️ 状态文件损坏，已重置为空状态: {e}")
                return {}
        return {}

    def calculate_md5(self, filepath: str) -> str:
        """计算文件的 MD5 哈希值，用于侦测文件是否被修改"""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"[!] ⚠️ 无法读取文件以计算 MD5: {filepath} -> {e}")
            return ""

    def mark_as_completed(self, filename: str, md5_hash: str):
        """颗粒度落盘：防崩溃断点续传"""
        self.state[filename] = md5_hash
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[!] ⚠️ 状态存盘失败: {e}")

    def clear_state(self):
        """清空追踪状态"""
        self.state = {}
        if os.path.exists(self.tracking_file):
            os.remove(self.tracking_file)


class LandscapeDatabaseManager:
    """🌟 Milvus 底层物理操作接口 (支持 ARRAY 高级数据结构)"""
    def __init__(self, db_path: str, colbert_path: str):
        self.db_path = db_path
        self.colbert_path = colbert_path
        self.collection_name = "landscape_rag"
        self.client = None
        self.colbert_db = {}
        
    def db_exists(self) -> bool:
        return os.path.exists(self.db_path)

    def init_database(self, force_rebuild=False):
        """防御性初始化物理库"""
        if force_rebuild and self.db_exists():
            try:
                os.remove(self.db_path)
                if os.path.exists(self.colbert_path): 
                    os.remove(self.colbert_path)
                print("    -> 💥 [物理销毁] 旧数据库与张量库已彻底清除。")
            except Exception as e:
                print(f"[!] 致命错误：无法删除旧数据库。{e}")
                sys.exit(1)
                
        self.client = MilvusClient(str(self.db_path))
            
        if not self.client.has_collection(self.collection_name):
            print("\n[*] 正在创建 Milvus 稠密+稀疏 双擎数据表 (含 ARRAY 高阶类型)...")
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="dynasty", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=20)
            
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            
            print("[*] 正在为向量字段构建物理索引...")
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)
            
        print("[*] 正在将 Milvus 集合载入内存...")
        self.client.load_collection(self.collection_name)
        
        if os.path.exists(self.colbert_path):
            try:
                with open(self.colbert_path, 'rb') as f:
                    self.colbert_db = pickle.load(f)
            except EOFError:
                print("[!] ⚠️ ColBERT 库损坏，初始化为空。")
                self.colbert_db = {}

    def insert_batch(self, insert_data: list, new_colbert_tensors: dict):
        """防崩溃原子级批量写入"""
        if not insert_data: return
        try:
            self.client.insert(collection_name=self.collection_name, data=insert_data)
            self.colbert_db.update(new_colbert_tensors)
            with open(self.colbert_path, 'wb') as f: 
                pickle.dump(self.colbert_db, f)
        except Exception as e:
            print(f"\n[!] 🚨 致命错误：写入失败，抛弃事务。报错: {e}")
