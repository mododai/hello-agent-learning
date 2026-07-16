r"""使用真实 Embedding、Qdrant 和 SQLite 验证统一 MemoryTool。

本脚本不会被 ``unittest discover`` 自动执行。请在项目根目录手动运行：

    .\.venv\Scripts\python.exe tests\real_memory_tool_check.py

脚本为每次运行生成唯一用户，验证 working、episodic、semantic 三类记忆的工具入口，
并在 ``finally`` 中删除本次产生的 SQLite 记录和 Qdrant 向量。
"""

import json
import os
import sys
import tempfile
import uuid

from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_agents.memory import MemoryConfig
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.tools.builtin.memory_tool import MemoryTool
from my_agents.tools.response import ToolStatus


def require_success(response, operation: str):
    """让失败的工具响应立即终止测试，并保留可读错误信息。"""
    if response.status != ToolStatus.SUCCESS:
        raise AssertionError(f"{operation} 失败: {response.to_json()}")
    return response


def main() -> None:
    """执行三类记忆的真实写入、检索、统计和清理验证。"""
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    required = ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"真实环境缺少配置: {', '.join(missing)}")

    user_id = f"memory-tool-real-check-{uuid.uuid4().hex}"
    tool = None
    created_ids = []

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            tool = MemoryTool(
                user_id=user_id,
                memory_config=MemoryConfig(storage_path=temp_dir),
            )

            working = require_success(
                tool.run(
                    {
                        "action": "remember",
                        "memory_type": "working",
                        "content": "当前正在验收完整记忆工具",
                        "importance": 0.7,
                    }
                ),
                "写入工作记忆",
            )
            created_ids.append(working.data["memory_id"])
            episodic = require_success(
                tool.run(
                    {
                        "action": "remember",
                        "memory_type": "episodic",
                        "content": "今天完成了 MemoryTool 的真实环境测试",
                        "session_id": "real-memory-tool-session",
                        "outcome": "等待检索验证",
                        "importance": 0.8,
                    }
                ),
                "写入情景记忆",
            )
            created_ids.append(episodic.data["memory_id"])
            semantic = require_success(
                tool.run(
                    {
                        "action": "remember",
                        "memory_type": "semantic",
                        "content": "用户喜欢喝绿茶",
                        "predicate": "drink_preference",
                        "object_value": "绿茶",
                        "knowledge_type": "preference",
                        "importance": 0.8,
                    }
                ),
                "写入语义记忆",
            )
            created_ids.append(semantic.data["memory_id"])

            working_result = require_success(
                tool.run(
                    {
                        "action": "recall",
                        "memory_type": "working",
                        "query": "验收记忆工具",
                    }
                ),
                "检索工作记忆",
            )
            episodic_result = require_success(
                tool.run(
                    {
                        "action": "recall",
                        "memory_type": "episodic",
                        "query": "真实环境测试",
                        "session_id": "real-memory-tool-session",
                    }
                ),
                "检索情景记忆",
            )
            semantic_result = require_success(
                tool.run(
                    {
                        "action": "recall",
                        "predicate": "drink_preference",
                    }
                ),
                "检索语义记忆",
            )
            stats = require_success(tool.run({"action": "stats"}), "读取统计")

            assert working_result.data["count"] == 1
            assert working_result.data["memories"][0]["memory_type"] == "working"
            assert episodic_result.data["count"] == 1
            assert episodic_result.data["memories"][0]["memory_type"] == "episodic"
            assert semantic_result.data["count"] == 1
            assert semantic_result.data["memories"][0]["fact"]["object"] == "绿茶"
            assert stats.data["total_count"] == 3

            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "user_id": user_id,
                        "enabled_types": stats.data["enabled_types"],
                        "counts": {
                            name: value["count"]
                            for name, value in stats.data["by_type"].items()
                        },
                        "retrieved_types": [
                            working_result.data["memories"][0]["memory_type"],
                            episodic_result.data["memories"][0]["memory_type"],
                            semantic_result.data["memories"][0]["memory_type"],
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            if tool is not None:
                tool.manager.clear()

                # 不能把网络异常误判成清理成功。按已知 ID 直接检查远端点是否仍存在。
                semantic_memory = tool.manager.get_memory("semantic")
                point_ids = [
                    semantic_memory.vector_store._to_point_id(memory_id)
                    for memory_id in created_ids
                ]
                remaining = semantic_memory.vector_store.client.retrieve(
                    collection_name=semantic_memory.vector_store.collection_name,
                    ids=point_ids,
                    with_payload=False,
                    with_vectors=False,
                ) if point_ids else []
                if remaining:
                    raise AssertionError(f"真实测试向量清理失败: {len(remaining)}")

                semantic_memory.doc_store.close()
                abs_path = os.path.abspath(semantic_memory.doc_store.db_path)
                SQLiteDocumentStore._instances.pop(abs_path, None)
                SQLiteDocumentStore._initialized_dbs.discard(abs_path)


if __name__ == "__main__":
    main()
