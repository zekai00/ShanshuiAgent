import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import USER_MEMORY_DB, ensure_runtime_dirs

DB_PATH = USER_MEMORY_DB

def init_database():
    """初始化 SQLite 数据库表结构"""
    print(f"[*] 正在初始化长时记忆数据库: {DB_PATH}")
    
    ensure_runtime_dirs()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
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
