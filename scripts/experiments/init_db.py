# /root/Workspace/ChineseLandscape/scripts/experiments/init_db.py

import sqlite3
import os

# 确保数据目录存在
DB_DIR = "/root/Workspace/ChineseLandscape/src/agent/memory"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "user_memories.db")

def init_database():
    """初始化 SQLite 数据库表结构"""
    print(f"[*] 正在初始化长时记忆数据库: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建用户画像表
    # user_id: 唯一标识
    # content: 存储用户偏好的文本摘要（如：“喜欢水墨画，对倪瓒感兴趣”）
    # updated_at: 最后更新时间
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memories (
            user_id TEXT PRIMARY KEY,
            content TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 数据库初始化成功！")

if __name__ == "__main__":
    init_database()