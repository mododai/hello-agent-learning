r"""真实环境语义记忆端到端检查。

本脚本连接 ``.env`` 配置的真实 Embedding 服务和 Qdrant，同时使用临时 SQLite。
它不会被 ``unittest discover`` 自动执行；请在项目根目录手动运行：

    .\.venv\Scripts\python.exe tests\real_semantic_memory_check.py

测试数据使用唯一 user_id，并在 ``finally`` 中尽力清理 SQLite 和 Qdrant，避免污染
其他用户或正式数据。脚本不会输出 API Key 等敏感配置。
"""

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime

from dotenv import load_dotenv

# 直接执行 tests 下脚本时，Python 默认只把 tests 加入 sys.path；补入项目根目录。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_agents.memory.base import MemoryConfig, MemoryItem
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.semantic import SemanticMemory
from my_agents.memory.types.semantic_fact import SemanticFact


def build_item(
    memory_id: str,
    user_id: str,
    content: str,
    predicate: str,
    object_value: str,
    knowledge_type: str = "fact",
) -> MemoryItem:
    """构造一条可写入真实语义记忆系统的结构化事实。"""
    return MemoryItem(
        id=memory_id,
        content=content,
        memory_type="semantic",
        user_id=user_id,
        timestamp=datetime.now(),
        importance=0.8,
        metadata={
            "fact": SemanticFact(
                subject="user",
                predicate=predicate,
                object=object_value,
                knowledge_type=knowledge_type,
                source="real_integration_test",
            )
        },
    )


def main() -> None:
    """执行写入、去重、替换、检索、数据库核对和远端清理。"""
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    required = ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"真实环境缺少配置: {', '.join(missing)}")

    run_id = uuid.uuid4().hex
    user_id = f"semantic-real-check-{run_id}"
    ids = {
        "latte": str(uuid.uuid4()),
        "tea": str(uuid.uuid4()),
        "tea_duplicate": str(uuid.uuid4()),
        "hangzhou": str(uuid.uuid4()),
        "shanghai": str(uuid.uuid4()),
        "swimming": str(uuid.uuid4()),
    }
    memory = None

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            memory = SemanticMemory(MemoryConfig(storage_path=temp_dir))

            latte_id = memory.add(
                build_item(
                    ids["latte"],
                    user_id,
                    "用户喜欢喝拿铁咖啡",
                    "drink_preference",
                    "拿铁",
                    "preference",
                )
            )
            tea_id = memory.add(
                build_item(
                    ids["tea"],
                    user_id,
                    "用户也喜欢喝绿茶",
                    "drink_preference",
                    "绿茶",
                    "preference",
                )
            )
            duplicate_id = memory.add(
                build_item(
                    ids["tea_duplicate"],
                    user_id,
                    "再次确认用户喜欢喝绿茶",
                    "drink_preference",
                    "绿茶",
                    "preference",
                )
            )
            memory.add(
                build_item(
                    ids["hangzhou"],
                    user_id,
                    "用户当前居住在杭州",
                    "current_city",
                    "杭州",
                )
            )
            city_id = memory.add(
                build_item(
                    ids["shanghai"],
                    user_id,
                    "用户已经搬到上海居住",
                    "current_city",
                    "上海",
                )
            )
            memory.add(
                build_item(
                    ids["swimming"],
                    user_id,
                    "用户会游泳",
                    "skill",
                    "游泳",
                    "skill",
                )
            )
            memory.retract_fact(ids["swimming"], user_id, "用户否认该信息")

            preferences = memory.find_active_facts(
                user_id=user_id,
                subject="user",
                predicate="drink_preference",
            )
            current_cities = memory.find_active_facts(
                user_id=user_id,
                subject="user",
                predicate="current_city",
            )
            retrieval = memory.retrieve(
                "用户喜欢喝什么饮料",
                user_id=user_id,
                limit=10,
            )
            print("retrieval:", retrieval)
            city_history = memory.retrieve(
                "用户现在住在哪里，以前住在哪里",
                user_id=user_id,
                limit=10,
                predicate="current_city",
                retrieval_mode="timeline",
            )
            skill_audit = memory.retrieve(
                "审计用户的技能记录",
                user_id=user_id,
                limit=10,
                predicate="skill",
                retrieval_mode="audit",
            )
            print("city_history:", city_history)


            assert latte_id == ids["latte"]
            assert tea_id == ids["tea"]
            assert duplicate_id == ids["tea"], "重复绿茶应复用第一次写入的 ID"
            assert city_id == ids["shanghai"]
            assert {item.id for item in preferences} == {ids["latte"], ids["tea"]}
            assert [item.id for item in current_cities] == [ids["shanghai"]]
            assert {ids["latte"], ids["tea"]} <= {item.id for item in retrieval}
            assert [item.id for item in city_history] == [
                ids["shanghai"],
                ids["hangzhou"],
            ]
            assert [item.id for item in skill_audit] == [ids["swimming"]]
            assert memory.get_fact(skill_audit[0]).status == "retracted"

            old_city = memory.doc_store.get_semantic_fact(ids["hangzhou"])
            new_city = memory.doc_store.get_semantic_fact(ids["shanghai"])
            assert old_city and old_city["status"] == "superseded"
            assert new_city and new_city["status"] == "active"
            assert new_city["supersedes"] == ids["hangzhou"]
            assert memory.doc_store.get_memory(ids["tea_duplicate"]) is None

            print(
                json.dumps(
                    {
                        "result": "PASS",
                        "embedding_dimension": memory.vector_store.vector_size,
                        "qdrant_collection": memory.vector_store.collection_name,
                        "preference_count": len(preferences),
                        "active_city": new_city["object"],
                        "retrieved_ids": [item.id for item in retrieval],
                        "history_ids": [item.id for item in city_history],
                        "audit_ids": [item.id for item in skill_audit],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            if memory is not None:
                # clear 按唯一 user_id 删除本次写入的 SQLite 行和 Qdrant points。
                memory.clear(user_id=user_id)
                remaining = memory.find_active_facts(
                    user_id=user_id,
                    subject="user",
                    predicate="drink_preference",
                )
                if remaining:
                    raise RuntimeError("真实环境测试清理失败：SQLite 仍有测试事实")

                # Qdrant 删除使用 wait=True，但仍显式按测试 user_id 反查，确保没有
                # 留下孤立向量。该过滤条件不会读取或影响其他用户的 points。
                cleanup_query = memory._single_vector(
                    memory.embedder.encode("真实环境测试清理检查")
                )
                remaining_vectors = memory.vector_store.search_similar(
                    query_vector=cleanup_query,
                    limit=10,
                    where={"memory_type": "semantic", "user_id": user_id},
                )
                if remaining_vectors:
                    raise RuntimeError("真实环境测试清理失败：Qdrant 仍有测试向量")
                print(
                    json.dumps(
                        {
                            "cleanup": "PASS",
                            "remaining_sqlite_facts": 0,
                            "remaining_qdrant_points": 0,
                        },
                        ensure_ascii=False,
                    )
                )
                memory.doc_store.close()
                abs_path = os.path.abspath(memory.doc_store.db_path)
                SQLiteDocumentStore._instances.pop(abs_path, None)
                SQLiteDocumentStore._initialized_dbs.discard(abs_path)


if __name__ == "__main__":
    main()
