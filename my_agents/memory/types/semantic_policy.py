"""语义事实谓词策略。

谓词策略描述同一个 ``subject + predicate`` 在同一时刻允许存在多少个有效值。
它解决了“用户可以同时喜欢拿铁和绿茶，但同一时刻通常只有一个当前城市”这一差异。
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


FactCardinality = Literal["single", "multiple"]
ReplacementMode = Literal["automatic", "explicit_only"]


class PredicatePolicy(BaseModel):
    """控制某类事实是否允许多值，以及是否自动替代旧值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cardinality: FactCardinality = "multiple"
    replacement_mode: ReplacementMode = "explicit_only"

    @model_validator(mode="after")
    def automatic_replacement_requires_single_value(self) -> "PredicatePolicy":
        """拒绝无法确定替代目标的策略组合。

        multiple 谓词可能同时存在多个 object，automatic 无法判断应该替代哪一个；
        这种情况必须使用 explicit_only，并由 FactChange.target_memory_id 指定目标。
        """
        if self.cardinality == "multiple" and self.replacement_mode == "automatic":
            raise ValueError("multiple 谓词不能使用 automatic 替代模式")
        return self


# 未知谓词必须采用保守策略：允许多个值共存，并且只能显式撤回或替代。
# 错误保留一条事实还可以在后续修正；错误覆盖则可能造成不可见的信息丢失。
DEFAULT_PREDICATE_POLICY = PredicatePolicy()


DEFAULT_PREDICATE_POLICIES: Dict[str, PredicatePolicy] = {
    "current_city": PredicatePolicy(
        cardinality="single",
        replacement_mode="automatic",
    ),
    "current_employer": PredicatePolicy(
        cardinality="single",
        replacement_mode="automatic",
    ),
    "preferred_language": PredicatePolicy(
        cardinality="single",
        replacement_mode="automatic",
    ),
    "drink_preference": PredicatePolicy(
        cardinality="multiple",
        replacement_mode="explicit_only",
    ),
    "skill": PredicatePolicy(
        cardinality="multiple",
        replacement_mode="explicit_only",
    ),
    "allergy": PredicatePolicy(
        cardinality="multiple",
        replacement_mode="explicit_only",
    ),
}


class PredicatePolicyRegistry:
    """保存谓词策略的轻量注册表。

    每个 SemanticMemory 默认拥有独立注册表，避免测试或某个 Agent 的自定义策略
    意外影响其他实例。构造时复制默认映射，因此不会修改模块级常量。
    """

    def __init__(self, policies: Optional[Dict[str, PredicatePolicy]] = None):
        self._policies = dict(DEFAULT_PREDICATE_POLICIES)
        for predicate, policy in (policies or {}).items():
            self.register(predicate, policy)

    @staticmethod
    def normalize_predicate(predicate: str) -> str:
        """统一谓词名称并拒绝空字符串。"""
        normalized = predicate.strip()
        if not normalized:
            raise ValueError("predicate 不能为空")
        return normalized

    def register(self, predicate: str, policy: PredicatePolicy) -> None:
        """注册或覆盖一个谓词策略。"""
        name = self.normalize_predicate(predicate)
        self._policies[name] = PredicatePolicy.model_validate(policy)

    def get(self, predicate: str) -> PredicatePolicy:
        """读取策略；未知谓词返回保守的默认多值策略。"""
        name = self.normalize_predicate(predicate)
        return self._policies.get(name, DEFAULT_PREDICATE_POLICY)
