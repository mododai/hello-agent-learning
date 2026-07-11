import unittest
from datetime import datetime

from pydantic import ValidationError

from my_agents.memory.types.semantic_fact import SemanticFact


class SemanticFactTest(unittest.TestCase):
    def test_create_fact_and_normalize_text(self):
        fact = SemanticFact(
            subject=" user ",
            predicate=" drink_preference ",
            object=" 无糖拿铁 ",
            knowledge_type="preference",
            confidence=0.9,
        )

        self.assertEqual(fact.subject, "user")
        self.assertEqual(fact.predicate, "drink_preference")
        self.assertEqual(fact.object, "无糖拿铁")
        self.assertEqual(fact.key, ("user", "drink_preference"))
        self.assertEqual(fact.status, "active")
        self.assertIsInstance(fact.valid_from, datetime)

    def test_fact_key_and_value_comparison(self):
        latte = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖拿铁",
        )
        same_latte = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖拿铁",
        )
        green_tea = SemanticFact(
            subject="user",
            predicate="drink_preference",
            object="无糖绿茶",
        )

        self.assertTrue(latte.has_same_value(same_latte))
        self.assertEqual(latte.key, green_tea.key)
        self.assertFalse(latte.has_same_value(green_tea))

    def test_reject_invalid_confidence_and_blank_fields(self):
        with self.assertRaises(ValidationError):
            SemanticFact(subject="user", predicate="likes", object="茶", confidence=1.1)

        with self.assertRaises(ValidationError):
            SemanticFact(subject=" ", predicate="likes", object="茶")

    def test_reject_unknown_fields(self):
        with self.assertRaises(ValidationError):
            SemanticFact(
                subject="user",
                predicate="likes",
                object="茶",
                unsupported_field=True,
            )


if __name__ == "__main__":
    unittest.main()
