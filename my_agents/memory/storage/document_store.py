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
            properties: Dict[str, Any] = None,
            semantic_fact: Dict[str, Any] = None,
            fact_cardinality: str = None,
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
        properties: Dict[str, Any] = None,
        semantic_fact: Dict[str, Any] = None,
        fact_cardinality: str = None,
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
                   properties: Dict[str, Any] = None,
                   semantic_fact: Dict[str, Any] = None,
                   fact_cardinality: str = None,
                   ) -> str:
        """

        :param fact_cardinality:
        :param semantic_fact:
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
        try:
            cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))

            # 新增操作使用普通 INSERT。SQLite 的 REPLACE 实际上会先删除旧行再插入，
            # 可能触发外键级联并掩盖重复 ID，因此不适合作为记忆新增语义。
            cursor.execute("""
                INSERT INTO memories
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

            # 结构化事实与 memories 行在同一个 SQLite 事务中提交，避免只写入其中一张表。
            if semantic_fact is not None:
                self._insert_semantic_fact(
                    cursor,
                    memory_id=memory_id,
                    user_id=user_id,
                    fact=semantic_fact,
                    cardinality=fact_cardinality or "multiple",
                )

            conn.commit()
            return memory_id
        except Exception:
            conn.rollback()
            raise

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
                      properties: Dict[str, Any] = None,
                      semantic_fact: Dict[str, Any] = None,
                      fact_cardinality: str = None,
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

        try:
            cursor.execute(f"""
                UPDATE memories
                SET {', '.join(update_fields)}
                WHERE id = ?
            """, params)
            updated = cursor.rowcount > 0

            # metadata 中的事实状态变化必须同步到规范化事实表。
            if updated and semantic_fact is not None:
                self._upsert_semantic_fact(
                    cursor,
                    memory_id=memory_id,
                    user_id=self._get_memory_user_id(cursor, memory_id),
                    fact=semantic_fact,
                    cardinality=fact_cardinality or "multiple",
                )

            conn.commit()
            return updated
        except Exception:
            conn.rollback()
            raise

    def delete_memory(self, memory_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM memories WHERE id = ?
        """, (memory_id,))
        deleted_count = cursor.rowcount > 0

        conn.commit()

        return deleted_count

    def find_semantic_facts(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        object_value: Optional[str] = None,
        status: Optional[str] = "active",
        statuses: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """使用规范化字段精确查询结构化事实，并返回对应的记忆记录。

        查询不再扫描 ``memories.properties`` JSON。多值谓词可以返回多条记录；
        调用方只有在明确需要单条结果时才自行选择或校验数量。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        conditions = [
            "sf.user_id = ?",
            "sf.subject = ?",
            "sf.predicate = ?",
        ]
        params: List[Any] = [user_id, subject, predicate]
        if object_value is not None:
            conditions.append("sf.object = ?")
            params.append(object_value)
        if statuses is not None:
            if not statuses:
                return []
            placeholders = ", ".join("?" for _ in statuses)
            conditions.append(f"sf.status IN ({placeholders})")
            params.extend(statuses)
        elif status is not None:
            conditions.append("sf.status = ?")
            params.append(status)

        limit_clause = ""
        if limit is not None:
            if limit <= 0:
                return []
            limit_clause = "LIMIT ?"
            params.append(limit)

        cursor.execute(
            f"""
            SELECT m.id, m.user_id, m.content, m.memory_type, m.timestamp,
                   m.importance, m.properties, m.created_at
            FROM semantic_facts AS sf
            JOIN memories AS m ON m.id = sf.memory_id
            WHERE {' AND '.join(conditions)}
            ORDER BY
                CASE sf.status
                    WHEN 'active' THEN 0
                    WHEN 'superseded' THEN 1
                    WHEN 'retracted' THEN 2
                    ELSE 3
                END ASC,
                sf.valid_from DESC,
                m.timestamp DESC,
                m.id DESC
            {limit_clause}
            """,
            params,
        )
        return [self._memories_row_to_dict(row) for row in cursor.fetchall()]

    def get_semantic_fact(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """读取一条规范化事实，主要用于测试、诊断和数据库一致性检查。"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT memory_id, user_id, subject, predicate, object, knowledge_type,
                   confidence, source, status, cardinality, valid_from, valid_to,
                   supersedes
            FROM semantic_facts
            WHERE memory_id = ?
            """,
            (memory_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def replace_memory_fact(
        self,
        old_memory_id: str,
        old_properties: Dict[str, Any],
        old_fact: Dict[str, Any],
        new_memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        new_properties: Dict[str, Any],
        new_fact: Dict[str, Any],
        fact_cardinality: str,
    ) -> str:
        """在单个 SQLite 事务中使旧事实失效并写入替代事实。

        先更新旧事实状态，再插入新 active 事实，以满足单值事实的部分唯一索引；
        任意一步失败都会整体回滚，不会留下“旧值失效但新值不存在”的中间状态。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                SELECT user_id, memory_type
                FROM memories
                WHERE id = ?
                """,
                (old_memory_id,),
            )
            old_row = cursor.fetchone()
            if not old_row or old_row["memory_type"] != memory_type:
                raise ValueError(f"未找到待替代的语义记忆: {old_memory_id}")
            if old_row["user_id"] != user_id:
                raise PermissionError("不能跨用户替代语义事实")

            cursor.execute(
                """
                UPDATE memories
                SET properties = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(old_properties), old_memory_id),
            )
            self._upsert_semantic_fact(
                cursor,
                memory_id=old_memory_id,
                user_id=user_id,
                fact=old_fact,
                cardinality=fact_cardinality,
            )

            cursor.execute(
                "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
                (user_id, user_id),
            )
            cursor.execute(
                """
                INSERT INTO memories
                (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    new_memory_id,
                    user_id,
                    content,
                    memory_type,
                    timestamp,
                    importance,
                    json.dumps(new_properties),
                ),
            )
            self._insert_semantic_fact(
                cursor,
                memory_id=new_memory_id,
                user_id=user_id,
                fact=new_fact,
                cardinality=fact_cardinality,
            )
            conn.commit()
            return new_memory_id
        except Exception:
            conn.rollback()
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # 统计各表的记录数
        tables = [
            "users",
            "memories",
            "semantic_facts",
            "concepts",
            "memory_concepts",
            "concept_relationships",
        ]
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
            # SQLite 默认不会启用外键约束，必须对每个线程连接显式打开。
            self.local.connection.execute("PRAGMA foreign_keys = ON")
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

        # 将结构化事实提升为独立表，使 subject/predicate/object/status 可以被索引。
        # memories.properties.fact 同时作为 MemoryItem 的读取镜像保存；当前项目使用
        # 空数据库起步，因此这里不包含任何旧表或旧 JSON 数据迁移逻辑。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_facts (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                knowledge_type TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                source TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'retracted')),
                cardinality TEXT NOT NULL CHECK (cardinality IN ('single', 'multiple')),
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                supersedes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (supersedes) REFERENCES semantic_facts (memory_id)
                    DEFERRABLE INITIALLY DEFERRED
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
            "CREATE INDEX IF NOT EXISTS idx_semantic_facts_lookup "
            "ON semantic_facts (user_id, subject, predicate, status)",
            "CREATE INDEX IF NOT EXISTS idx_semantic_facts_object "
            "ON semantic_facts (user_id, subject, predicate, object, status)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)"
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        # 唯一索引属于当前数据库结构本身，初始化时直接创建，不依赖数据迁移流程。
        self._create_semantic_fact_unique_indexes(cursor)

        conn.commit()
        logger.info("SQLite 数据库表和索引创建完成")

    @staticmethod
    def _get_memory_user_id(cursor, memory_id: str) -> str:
        """在当前事务中取得记忆所属用户，找不到时立即失败。"""
        cursor.execute("SELECT user_id FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"未找到记忆: {memory_id}")
        return row["user_id"]

    @staticmethod
    def _fact_value(fact: Dict[str, Any], field: str, default: Any = None) -> Any:
        """读取事实字段，并为真正缺失的必填字段提供清晰错误。"""
        value = fact.get(field, default)
        if value is None:
            raise ValueError(f"semantic_fact 缺少字段: {field}")
        return value

    @classmethod
    def _insert_semantic_fact(
        cls,
        cursor,
        memory_id: str,
        user_id: str,
        fact: Dict[str, Any],
        cardinality: str,
    ) -> None:
        """在调用方事务中插入一条规范化事实。"""
        cursor.execute(
            """
            INSERT INTO semantic_facts
            (memory_id, user_id, subject, predicate, object, knowledge_type,
             confidence, source, status, cardinality, valid_from, valid_to,
             supersedes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                memory_id,
                user_id,
                cls._fact_value(fact, "subject"),
                cls._fact_value(fact, "predicate"),
                cls._fact_value(fact, "object"),
                cls._fact_value(fact, "knowledge_type", "fact"),
                float(cls._fact_value(fact, "confidence", 0.8)),
                cls._fact_value(fact, "source", "conversation"),
                cls._fact_value(fact, "status", "active"),
                cardinality,
                cls._fact_value(fact, "valid_from"),
                fact.get("valid_to"),
                fact.get("supersedes"),
            ),
        )

    @classmethod
    def _upsert_semantic_fact(
        cls,
        cursor,
        memory_id: str,
        user_id: str,
        fact: Dict[str, Any],
        cardinality: str,
    ) -> None:
        """在调用方事务中插入或同步一条规范化事实。"""
        cursor.execute("SELECT 1 FROM semantic_facts WHERE memory_id = ?", (memory_id,))
        if cursor.fetchone() is None:
            cls._insert_semantic_fact(
                cursor,
                memory_id=memory_id,
                user_id=user_id,
                fact=fact,
                cardinality=cardinality,
            )
            return
        cursor.execute(
            """
            UPDATE semantic_facts
            SET user_id = ?, subject = ?, predicate = ?, object = ?,
                knowledge_type = ?, confidence = ?, source = ?, status = ?,
                cardinality = ?, valid_from = ?, valid_to = ?, supersedes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE memory_id = ?
            """,
            (
                user_id,
                cls._fact_value(fact, "subject"),
                cls._fact_value(fact, "predicate"),
                cls._fact_value(fact, "object"),
                cls._fact_value(fact, "knowledge_type", "fact"),
                float(cls._fact_value(fact, "confidence", 0.8)),
                cls._fact_value(fact, "source", "conversation"),
                cls._fact_value(fact, "status", "active"),
                cardinality,
                cls._fact_value(fact, "valid_from"),
                fact.get("valid_to"),
                fact.get("supersedes"),
                memory_id,
            ),
        )

    @staticmethod
    def _create_semantic_fact_unique_indexes(cursor) -> None:
        """建立只约束当前事实的部分唯一索引。"""
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_semantic_facts_active_value
            ON semantic_facts (user_id, subject, predicate, object)
            WHERE status = 'active'
            """
        )
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_semantic_facts_active_single
            ON semantic_facts (user_id, subject, predicate)
            WHERE status = 'active' AND cardinality = 'single'
            """
        )

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
