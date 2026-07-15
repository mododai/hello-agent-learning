"""结构化语义事实生命周期验收测试。"""

import os
import tempfile
import unittest
from datetime import datetime

from pydantic import ValidationError

from my_agents.memory.base import MemoryConfig, MemoryItem
from my_agents.memory.storage.document_store import SQLiteDocumentStore
from my_agents.memory.types.semantic import SemanticMemory
from my_agents.memory.types.semantic_change import FactChange
from my_agents.memory.types.semantic_fact import SemanticFact
from my_agents.memory.types.semantic_policy import PredicatePolicy
from tests.support import FakeEmbedder, FakeVectorStore


class SemanticFactLifecycleTest(unittest.TestCase):
    """验证多值并存、单值替代、撤回、历史查询与失败补偿。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.embedder = FakeEmbedder()
        self.vector_store = FakeVectorStore()
        self.memory = SemanticMemory(
            MemoryConfig(storage_path=self.temp_dir.name),
            storage_backend={
                "embedder": self.embedder,
                "vector_store": self.vector_store,
            },
        )

    def tearDown(self):
        # Windows 不允许删除仍被 SQLite 连接占用的临时数据库，因此显式关闭连接，
        # 并清理 SQLiteDocumentStore 针对该路径保存的进程级单例缓存。
        connection = getattr(self.memory.doc_store.local, "connection", None)
        if connection is not None:
            connection.close()
        abs_path = os.path.abspath(self.memory.doc_store.db_path)
        SQLiteDocumentStore._instances.pop(abs_path, None)
        SQLiteDocumentStore._initialized_dbs.discard(abs_path)
        self.temp_dir.cleanup()

    @staticmethod
    def item(memory_id, user_id, content, fact):
        return MemoryItem(
            id=memory_id,
            content=content,
            memory_type="semantic",
            user_id=user_id,
            timestamp=datetime.now(),
            importance=0.8,
            metadata={"fact": fact, "source_message_id": f"source-{memory_id}"},
        )

    @staticmethod
    def fact(predicate, value, knowledge_type="fact"):
        return SemanticFact(
            subject="user",
            predicate=predicate,
            object=value,
            knowledge_type=knowledge_type,
        )

    def stored_fact(self, memory_id):
        doc = self.memory.doc_store.get_memory(memory_id)
        return SemanticFact.model_validate(doc["properties"]["fact"])

    def test_unknown_predicate_is_multiple_and_custom_policy_can_be_registered(self):
        """未知谓词必须保守地允许多值，且当前实例可注册业务策略。"""
        default_policy = self.memory.get_predicate_policy("unknown_relation")
        self.assertEqual(default_policy.cardinality, "multiple")
        self.assertEqual(default_policy.replacement_mode, "explicit_only")

        self.memory.register_predicate_policy(
            "favorite_color",
            PredicatePolicy(cardinality="single", replacement_mode="automatic"),
        )
        policy = self.memory.get_predicate_policy(" favorite_color ")
        self.assertEqual(policy.cardinality, "single")
        self.assertEqual(policy.replacement_mode, "automatic")

        with self.assertRaises(ValidationError):
            PredicatePolicy(cardinality="multiple", replacement_mode="automatic")

    def test_multiple_value_preference_keeps_both_facts_active(self):
        """喜欢拿铁和喜欢绿茶不是冲突，两条偏好应同时有效。"""
        latte = self.item(
            "latte",
            "u1",
            "用户喜欢拿铁",
            self.fact("drink_preference", "拿铁", "preference"),
        )
        tea = self.item(
            "tea",
            "u1",
            "用户也喜欢绿茶",
            self.fact("drink_preference", "绿茶", "preference"),
        )

        self.assertEqual(self.memory.add(latte), "latte")
        self.assertEqual(self.memory.add(tea), "tea")
        self.assertEqual(self.stored_fact("latte").status, "active")
        self.assertEqual(self.stored_fact("tea").status, "active")

    def test_single_value_predicate_supersedes_previous_fact(self):
        """current_city 是单值谓词，新城市应替代旧城市并保留历史链。"""
        hangzhou = self.item(
            "city-hangzhou",
            "u1",
            "用户当前住在杭州",
            self.fact("current_city", "杭州"),
        )
        shanghai = self.item(
            "city-shanghai",
            "u1",
            "用户已经搬到上海",
            self.fact("current_city", "上海"),
        )

        self.memory.add(hangzhou)
        self.memory.add(shanghai)

        old_fact = self.stored_fact("city-hangzhou")
        new_fact = self.stored_fact("city-shanghai")
        self.assertEqual(old_fact.status, "superseded")
        self.assertIsNotNone(old_fact.valid_to)
        self.assertEqual(new_fact.status, "active")
        self.assertEqual(new_fact.supersedes, "city-hangzhou")

    def test_default_retrieval_excludes_history_but_can_include_it(self):
        """默认检索只返回当前事实，include_inactive 才返回历史事实。"""
        self.memory.add(
            self.item("old-city", "u1", "用户住在杭州", self.fact("current_city", "杭州"))
        )
        self.memory.add(
            self.item("new-city", "u1", "用户住在上海", self.fact("current_city", "上海"))
        )

        current = self.memory.retrieve("用户住在哪里", user_id="u1", limit=10)
        history = self.memory.retrieve(
            "用户住在哪里", user_id="u1", limit=10, include_inactive=True
        )

        self.assertEqual([item.id for item in current], ["new-city"])
        self.assertEqual([item.id for item in history], ["new-city", "old-city"])
        self.assertGreater(
            history[0].metadata["retrieval_score"],
            history[1].metadata["retrieval_score"],
        )

    def test_structured_timeline_returns_only_current_and_superseded_predicate(self):
        """城市时间线只返回当前和被替代城市，不得混入 active 饮品或撤回事实。"""
        self.memory.add(
            self.item("latte", "u1", "用户喜欢拿铁", self.fact("drink_preference", "拿铁"))
        )
        self.memory.add(
            self.item("old-city", "u1", "用户住在杭州", self.fact("current_city", "杭州"))
        )
        self.memory.add(
            self.item("new-city", "u1", "用户搬到上海", self.fact("current_city", "上海"))
        )
        self.memory.add(
            self.item("old-skill", "u1", "用户会游泳", self.fact("skill", "游泳"))
        )
        self.memory.retract_fact("old-skill", "u1", "信息录入错误")

        embedding_calls_before = len(self.embedder.calls)
        timeline = self.memory.retrieve(
            "用户现在住在哪里，以前住在哪里",
            user_id="u1",
            predicate="current_city",
            retrieval_mode="timeline",
            limit=10,
        )

        self.assertEqual([item.id for item in timeline], ["new-city", "old-city"])
        self.assertEqual(
            [self.memory.get_fact(item).status for item in timeline],
            ["active", "superseded"],
        )
        # 结构化谓词检索直接查询 SQLite，不应产生额外 embedding 调用。
        self.assertEqual(len(self.embedder.calls), embedding_calls_before)

    def test_retracted_fact_is_only_visible_in_audit_mode(self):
        """retracted 表示撤回记录，不应进入 current 或 timeline，只供 audit 审计。"""
        self.memory.add(
            self.item("swimming", "u1", "用户会游泳", self.fact("skill", "游泳"))
        )
        self.memory.retract_fact("swimming", "u1", "用户否认该信息")

        current = self.memory.retrieve_facts("u1", "user", "skill", "current")
        timeline = self.memory.retrieve_facts("u1", "user", "skill", "timeline")
        audit = self.memory.retrieve_facts("u1", "user", "skill", "audit")

        self.assertEqual(current, [])
        self.assertEqual(timeline, [])
        self.assertEqual([item.id for item in audit], ["swimming"])
        self.assertEqual(self.memory.get_fact(audit[0]).status, "retracted")

        with self.assertRaises(ValueError):
            self.memory.retrieve_facts("u1", "user", "skill", "unknown")

    def test_explicit_retraction_preserves_history(self):
        """撤回偏好不删除记录，但默认检索不再返回它。"""
        self.memory.add(
            self.item(
                "latte",
                "u1",
                "用户喜欢拿铁",
                self.fact("drink_preference", "拿铁", "preference"),
            )
        )

        self.assertTrue(self.memory.retract_fact("latte", "u1", "用户明确表示不再喜欢"))
        retracted = self.stored_fact("latte")
        self.assertEqual(retracted.status, "retracted")
        self.assertIsNotNone(retracted.valid_to)
        self.assertEqual(
            self.memory.doc_store.get_memory("latte")["properties"]["retraction_reason"],
            "用户明确表示不再喜欢",
        )
        self.assertEqual(self.memory.retrieve("拿铁", user_id="u1"), [])
        self.assertEqual(
            [item.id for item in self.memory.retrieve(
                "拿铁", user_id="u1", include_inactive=True
            )],
            ["latte"],
        )

    def test_fact_change_retracts_exact_multiple_value_fact(self):
        """FactChange.retract 只撤回完整匹配的 object，不影响同谓词其他值。"""
        self.memory.add(
            self.item("latte", "u1", "喜欢拿铁", self.fact("drink_preference", "拿铁"))
        )
        self.memory.add(
            self.item("tea", "u1", "喜欢绿茶", self.fact("drink_preference", "绿茶"))
        )
        change = FactChange(
            operation="retract",
            fact=self.fact("drink_preference", "拿铁"),
            reason="用户不再喜欢拿铁",
        )
        request_item = self.item(
            "retract-request", "u1", "不再喜欢拿铁", change.fact
        )

        self.assertEqual(self.memory.apply_fact_change(change, request_item), "latte")
        self.assertEqual(self.stored_fact("latte").status, "retracted")
        self.assertEqual(self.stored_fact("tea").status, "active")

    def test_replacing_multiple_value_fact_requires_explicit_target(self):
        """多值谓词有多个候选值时，replace 不得随机选择替代目标。"""
        self.memory.add(
            self.item("latte", "u1", "喜欢拿铁", self.fact("drink_preference", "拿铁"))
        )
        self.memory.add(
            self.item("tea", "u1", "喜欢绿茶", self.fact("drink_preference", "绿茶"))
        )
        replacement = self.fact("drink_preference", "乌龙茶")
        request_item = self.item("oolong", "u1", "现在用乌龙茶替代拿铁", replacement)

        with self.assertRaises(ValueError):
            self.memory.apply_fact_change(
                FactChange(operation="replace", fact=replacement),
                request_item,
            )

        result = self.memory.apply_fact_change(
            FactChange(
                operation="replace",
                fact=replacement,
                target_memory_id="latte",
                reason="用户明确用乌龙茶替代拿铁",
            ),
            request_item,
        )
        self.assertEqual(result, "oolong")
        self.assertEqual(self.stored_fact("latte").status, "superseded")
        self.assertEqual(self.stored_fact("tea").status, "active")
        self.assertEqual(self.stored_fact("oolong").status, "active")
        old_doc = self.memory.doc_store.get_memory("latte")
        self.assertEqual(
            old_doc["properties"]["replacement_reason"],
            "用户明确用乌龙茶替代拿铁",
        )

    def test_failed_new_vector_keeps_old_single_value_fact_active(self):
        """替代的新向量写入失败时，旧事实不能提前失效。"""
        self.memory.add(
            self.item("old-city", "u1", "住在杭州", self.fact("current_city", "杭州"))
        )
        self.vector_store.fail_next_add = True

        with self.assertRaises(RuntimeError):
            self.memory.add(
                self.item("new-city", "u1", "住在上海", self.fact("current_city", "上海"))
            )

        self.assertEqual(self.stored_fact("old-city").status, "active")
        self.assertFalse(self.memory.has_memory("new-city"))

    def test_retraction_and_replacement_are_isolated_by_user(self):
        """任何生命周期操作都不能跨越 user_id 权限边界。"""
        self.memory.add(
            self.item("u1-city", "u1", "住在杭州", self.fact("current_city", "杭州"))
        )
        self.assertFalse(self.memory.retract_fact("u1-city", "u2"))
        with self.assertRaises(PermissionError):
            self.memory.supersede_fact(
                "u1-city",
                self.item("u2-city", "u2", "住在上海", self.fact("current_city", "上海")),
            )
        self.assertEqual(self.stored_fact("u1-city").status, "active")


if __name__ == "__main__":
    unittest.main()
