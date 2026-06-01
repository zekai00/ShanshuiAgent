# /root/Workspace/ShanshuiAgent/src/agent/memory/memory_manager.py

import sqlite3
import json
from datetime import datetime

from src.config import USER_MEMORY_DB, ensure_runtime_dirs

DB_PATH = USER_MEMORY_DB

class MemoryManager:
    @staticmethod
    def _init_db():
        """初始化数据库表结构，确保支持 JSON 存储"""
        ensure_runtime_dirs()
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT PRIMARY KEY,
                content TEXT,        -- 存储结构化 JSON 字符串
                updated_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def get_memory(user_id: str) -> dict:
        """
        获取用户的长时结构化记忆。
        返回格式: {"preferences": [], "feedback": [], "context": []}
        """
        if not DB_PATH.exists():
            MemoryManager._init_db()
            return {"preferences": [], "feedback": [], "context": []}
        
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM user_memories WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            try:
                # 将数据库中的 JSON 字符串解析为字典
                return json.loads(result[0])
            except json.JSONDecodeError:
                return {"preferences": [], "feedback": [], "context": []}
        return {"preferences": [], "feedback": [], "context": []}

    @staticmethod
    def save_memory(user_id: str, new_insights: dict):
        """
        持久化存储并合并用户记忆。
        new_insights 应包含: preferences (list), feedback (list), context (str)
        """
        MemoryManager._init_db()
        
        # 1. 获取现有记忆
        old_memory = MemoryManager.get_memory(user_id)
        
        # 2. 合并偏好 (Preferences) 与 反馈 (Feedback) - 去重处理
        for key in ["preferences", "feedback"]:
            if key in new_insights and isinstance(new_insights[key], list):
                combined = old_memory.get(key, []) + new_insights[key]
                # 利用 set 去重，保持记忆精炼
                old_memory[key] = list(set(combined))
        
        # 3. 记录带有绝对时间戳的项目上下文 (Context)
        if "context" in new_insights and new_insights["context"]:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            time_stamped_note = f"[{now_str}] {new_insights['context']}"
            
            context_list = old_memory.get("context", [])
            context_list.append(time_stamped_note)
            
            # 只保留最近 10 条项目记录，防止长时记忆库过于臃肿
            old_memory["context"] = context_list[-10:]

        # 4. 写回数据库
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO user_memories (user_id, content, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET 
                content = excluded.content,
                updated_at = excluded.updated_at
        ''', (user_id, json.dumps(old_memory, ensure_ascii=False), now))
        
        conn.commit()
        conn.close()
        print(f"\n[💾 MemoryManager] 已同步更新用户 {user_id} 的结构化认知地图。")
