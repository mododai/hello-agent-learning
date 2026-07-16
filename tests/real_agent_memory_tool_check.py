r"""真实 LLM 通过 SimpleAgent Function Calling 使用 MemoryTool 的验收脚本。

手动运行：

    .\.venv\Scripts\python.exe tests\real_agent_memory_tool_check.py

脚本会连接 ``.env`` 中的 LLM、Embedding 和 Qdrant，使用唯一用户执行“保存偏好→
再次召回”两轮对话，并在结束时清理该用户产生的全部持久化测试数据。
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

# Windows PowerShell 可能使用 GBK；真实模型回答包含 emoji 时必须显式使用 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from my_agents.agents.simple_agent import SimpleAgent
from my_agents.core.llm import AgenticLLM
from my_agents.memory import MemoryConfig
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.tools import MemoryTool, ToolRegistry


SYSTEM_PROMPT = """你是记忆工具验收 Agent，必须严格遵守以下规则：
1. 用户要求记住长期偏好时，必须调用 memory 工具的 remember 操作。
2. 饮品偏好使用 memory_type=semantic、predicate=drink_preference、
   object_value=用户明确说出的饮品、knowledge_type=preference。
3. 用户询问自己的饮品偏好时，必须调用 memory 工具的 recall 操作，
   predicate=drink_preference。
4. 必须根据工具返回结果回答，不能凭对话历史猜测；不要虚构工具结果。
"""


def main() -> None:
    """验证真实模型能够选择工具、保存结构化事实并在下一轮召回。"""
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    required = (
        "LLM_MODEL_ID",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"真实 Agent 测试缺少配置: {', '.join(missing)}")

    user_id = f"agent-memory-real-check-{uuid.uuid4().hex}"
    tool = None

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            tool = MemoryTool(
                user_id=user_id,
                memory_config=MemoryConfig(storage_path=temp_dir),
            )
            registry = ToolRegistry()
            registry.register_tool(tool)
            agent = SimpleAgent(
                name="real-memory-agent",
                llm=AgenticLLM(temperature=0.0),
                system_prompt=SYSTEM_PROMPT,
                tool_registry=registry,
                enable_tool_calling=True,
                max_tool_iterations=3,
            )

            store_answer = agent.run(
                "请使用记忆工具记住：我喜欢喝绿茶。",
                enable_thinking=False,
            )
            semantic = tool.manager.get_memory("semantic")
            stored = semantic.retrieve(
                query=None,
                user_id=user_id,
                predicate="drink_preference",
                object_value="绿茶",
                retrieval_mode="current",
                limit=10,
            )
            if len(stored) != 1:
                raise AssertionError(
                    f"Agent 没有写入预期语义事实，实际数量: {len(stored)}"
                )

            recall_answer = agent.run(
                "请使用记忆工具查询：我喜欢喝什么？",
                enable_thinking=False,
            )
            print(recall_answer)
            print("History 历史对话:")
            history = agent.get_history()
            for record in history:
                print(str(record))
            if "绿茶" not in recall_answer:
                raise AssertionError(f"Agent 最终回答未包含召回事实: {recall_answer}")

            fact = semantic.get_fact(stored[0])
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "model": os.getenv("LLM_MODEL_ID"),
                        "user_id": user_id,
                        "stored_fact": {
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "status": fact.status,
                        },
                        "store_answer": store_answer,
                        "recall_answer": recall_answer,
                        "history_roles": [
                            message.role for message in agent.get_history()
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            if tool is not None:
                semantic = tool.manager.get_memory("semantic")
                documents = semantic.doc_store.search_memories(
                    user_id=user_id,
                    limit=1_000_000,
                )
                persistent_ids = [doc["memory_id"] for doc in documents]
                tool.manager.clear()

                point_ids = [
                    semantic.vector_store._to_point_id(memory_id)
                    for memory_id in persistent_ids
                ]
                remaining = semantic.vector_store.client.retrieve(
                    collection_name=semantic.vector_store.collection_name,
                    ids=point_ids,
                    with_payload=False,
                    with_vectors=False,
                ) if point_ids else []
                if remaining:
                    raise AssertionError(
                        f"真实 Agent 测试向量清理失败: {len(remaining)}"
                    )

                semantic.doc_store.close()
                abs_path = os.path.abspath(semantic.doc_store.db_path)
                SQLiteDocumentStore._instances.pop(abs_path, None)
                SQLiteDocumentStore._initialized_dbs.discard(abs_path)


if __name__ == "__main__":
    main()
