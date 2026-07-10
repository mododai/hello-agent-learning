"""
文档存储
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import threading
import logging

logger = logging.getLogger(__name__)

class DocumentStore(ABC):
    @abstractmethod
    def add_memory(
            self,
            memory_id: str,
            user_id: str,
            content: str,
            memory_type: str,
            timestamp: int,
            importance: float,
            properties: Dict[str, Any] = None
    ) -> str:
        """添加记忆"""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆"""
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        """更新记忆"""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档"""
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        pass


class SQLiteDocumentStore(DocumentStore):
    # 类变量
    _instances = {}
    _initialized_dbs = set()
    _lock = threading.Lock()
    def __new__(cls, db_path: str = "./memory.db"):
        """同一路径一个实例"""
        abs_path = os.path.abspath(db_path)
        with cls._lock:
            # 加锁避免多个线程同时创建同一个路径的实例
            if abs_path not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[abs_path] = instance

        return cls._instances[abs_path]

    def __init__(self, db_path: str = "./memory.db"):
        # 即使 __new__() 返回的是旧对象,__init__() 仍然可能再次被调用
        if hasattr(self, "_initialized"):
            return

        self.db_path = db_path
        self.local = threading.local()  # 线程本地存储, 保存每个线程自己的 SQLite 连接
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            logger.info(f"SQLite 文档存储初始化完成: {db_path}")

        self._initialized = True


    def add_memory(self,
                   memory_id: str,
                   user_id: str, content: str,
                   memory_type: str,
                   timestamp: int,
                   importance: float,
                   properties: Dict[str, Any] = None
                   ) -> str:
        """

        :param memory_id:
        :param user_id:
        :param content:
        :param memory_type:
        :param timestamp:
        :param importance:
        :param properties:
        :return:
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))

        # 插入记忆
        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            memory_id,
            user_id,
            content,
            memory_type,
            timestamp,
            importance,
            json.dumps(properties) if properties is not None else None,
        ))

        conn.commit()
        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._memories_row_to_dict(row)

    def search_memories(self,
                        user_id: Optional[str] = None,
                        memory_type: Optional[str] = None,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None,
                        importance_threshold: Optional[float] = None,
                        limit: int = 10
                        ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        where_conditions = []
        params = []
        # 构建查询条件
        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)

        if memory_type:
            where_conditions.append("memory_type = ?")
            params.append(memory_type)

        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)

        if importance_threshold:
            where_conditions.append("importance >= ?")
            params.append(importance_threshold)

        # 构建查询sql语句
        where_clause = ""
        if where_conditions and len(where_conditions) > 0:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        cursor.execute(f"""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, params + [limit])

        memories = []
        for row in cursor.fetchall():
            memories.append(self._memories_row_to_dict(row))

        return memories

    def update_memory(self,
                      memory_id: str,
                      content: str = None,
                      importance: float = None,
                      properties: Dict[str, Any] = None
                      ) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        # 构建更新字段
        update_fields = []
        params = []

        if content is not None:
            update_fields.append("content = ?")
            params.append(content)

        if importance is not None:
            update_fields.append("importance = ?")
            params.append(importance)

        if properties is not None:
            update_fields.append("properties = ?")
            params.append(json.dumps(properties))

        if not update_fields:
            return False

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)

        cursor.execute(f"""
            UPDATE memories
            SET {', '.join(update_fields)}
            WHERE id = ?
        """, params)

        conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM memories WHERE id = ?
        """, (memory_id,))
        deleted_count = cursor.rowcount > 0

        conn.commit()

        return deleted_count
    def get_database_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # 统计各表的记录数
        tables = ["users", "memories", "concepts", "memory_concepts", "concept_relationships"]
        for table in tables:
            cursor.execute(f"""
                SELECT COUNT(*) as count FROM {table}
            """)
            stats[f"{table}_count"] = cursor.fetchone()["count"]

        cursor.execute("""
            SELECT memory_type, COUNT(*) as count 
            FROM memories
            GROUP BY memory_type 
        """)
        memory_types = {}
        for memory_type, count in cursor.fetchall():
            memory_types[memory_type] = count

        stats["memory_types"] = memory_types

        # 统计用户分布
        cursor.execute("""
           SELECT user_id, COUNT(*) as count
           FROM memories
           GROUP BY user_id
           ORDER BY count DESC
           LIMIT 10
        """)
        top_users = {}
        for row in cursor.fetchall():
            top_users[row["user_id"]] = row["count"]
        stats["top_users"] = top_users

        stats["store_type"] = "sqlite"
        stats["db_path"] = self.db_path

        return stats

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档"""
        import uuid
        import time

        doc_id = str(uuid.uuid4())
        user_id = metadata.get("user_id", "system")

        return self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {}
        )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.get_memory(memory_id=document_id)

    def _get_connection(self):
        """获取线程本地连接"""

        if not hasattr(self.local, "connection"):
            # 如果线程中没有连接过
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        return self.local.connection

    def close(self):
        if hasattr(self.local, "connection"):
            self.local.connection.close()
            delattr(self.local, "connection")
            logger.info("SQLite 连接已关闭")

    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建记忆表
        cursor.execute("""
               CREATE TABLE IF NOT EXISTS memories (
                   id TEXT PRIMARY KEY,
                   user_id TEXT NOT NULL,
                   content TEXT NOT NULL,
                   memory_type TEXT NOT NULL,
                   timestamp INTEGER NOT NULL,
                   importance REAL NOT NULL,
                   properties TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   FOREIGN KEY (user_id) REFERENCES users (id)
               )
               """)

        # 创建概念表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建记忆-概念关联表
        cursor.execute("""
                   CREATE TABLE IF NOT EXISTS memory_concepts (
                       memory_id TEXT NOT NULL,
                       concept_id TEXT NOT NULL,
                       relevance_score REAL DEFAULT 1.0,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       PRIMARY KEY (memory_id, concept_id),
                       FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                       FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
                   )
               """)

        # 创建概念关系表
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_relationships (
                    from_concept_id TEXT NOT NULL,
                    to_concept_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    properties TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                    FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                    FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
                )
                """)

        # 创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)"
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        conn.commit()
        logger.info("SQLite 数据库表和索引创建完成")

    def _memories_row_to_dict(self, row):

        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"]
        }


if __name__ == '__main__':
    store = SQLiteDocumentStore()
    store.add_document(
        content="test",
        metadata={
            "user_id": "test",
        }
    )
