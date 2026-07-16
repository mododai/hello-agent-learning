"""把自然语言问题转换成 :class:`FactQuery` 的确定性规则规划器。"""

from typing import Dict, Iterable, Optional, Tuple

from .types.semantic_query import FactQuery, FactRetrievalMode


DEFAULT_PREDICATE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "current_city": (
        "住在哪里",
        "住哪",
        "居住地",
        "当前城市",
        "以前住",
        "搬到哪里",
        "住过哪里",
    ),
    "current_employer": (
        "在哪里工作",
        "在哪工作",
        "当前公司",
        "就职公司",
        "雇主",
    ),
    "preferred_language": (
        "偏好语言",
        "首选语言",
        "喜欢用什么语言",
    ),
    "drink_preference": (
        "喜欢喝什么",
        "喝什么",
        "喜欢喝",
        "饮品偏好",
        "饮料偏好",
    ),
    "skill": (
        "会什么",
        "擅长什么",
        "技能",
        "能力",
    ),
    "allergy": (
        "对什么过敏",
        "过敏",
    ),
}


class SemanticQueryPlanner:
    """使用谓词别名和意图关键词生成可解释的结构化查询。

    规则规划器的目标不是覆盖所有中文表达，而是提供稳定、可测试的基线。无法唯一
    判断谓词时返回 ``None``，调用方应回退到 Qdrant，不允许规划器随意猜测并过滤掉
    正确结果。
    """

    AUDIT_KEYWORDS = ("审计", "撤回", "否认", "错误记录", "全部记录", "所有记录")
    TIMELINE_KEYWORDS = (
        "以前",
        "曾经",
        "过去",
        "历史",
        "原来",
        "先前",
        "变化",
        "变更",
        "搬过",
    )

    def __init__(self, predicate_aliases: Optional[Dict[str, Iterable[str]]] = None):
        self._predicate_aliases = {
            predicate: tuple(aliases)
            for predicate, aliases in DEFAULT_PREDICATE_ALIASES.items()
        }
        for predicate, aliases in (predicate_aliases or {}).items():
            self.register_aliases(predicate, aliases)

    @staticmethod
    def _normalize_text(value: str, field_name: str) -> str:
        """统一规则匹配文本，并为无效配置提供明确异常。"""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        normalized = value.strip().casefold()
        if not normalized:
            raise ValueError(f"{field_name} 不能为空")
        return normalized

    def register_aliases(self, predicate: str, aliases: Iterable[str]) -> None:
        """为自定义谓词注册或追加自然语言别名。"""
        normalized_predicate = self._normalize_text(predicate, "predicate")
        if isinstance(aliases, str):
            aliases = (aliases,)
        normalized_aliases = tuple(
            self._normalize_text(alias, "alias") for alias in aliases
        )
        if not normalized_aliases:
            raise ValueError("aliases 不能为空")
        existing = self._predicate_aliases.get(normalized_predicate, ())
        # dict.fromkeys 在保留顺序的同时完成去重，便于测试和调试规则优先级。
        self._predicate_aliases[normalized_predicate] = tuple(
            dict.fromkeys((*existing, *normalized_aliases))
        )

    @classmethod
    def detect_retrieval_mode(cls, normalized_query: str) -> FactRetrievalMode:
        """审计意图优先于时间线意图，其他问题默认查询当前事实。"""
        if any(keyword in normalized_query for keyword in cls.AUDIT_KEYWORDS):
            return "audit"
        if any(keyword in normalized_query for keyword in cls.TIMELINE_KEYWORDS):
            return "timeline"
        return "current"

    def detect_predicate(self, normalized_query: str) -> Optional[str]:
        """选择匹配到的最长别名；同长度歧义时拒绝猜测。"""
        matches = []
        for predicate, aliases in self._predicate_aliases.items():
            for alias in aliases:
                if alias in normalized_query:
                    matches.append((len(alias), predicate, alias))
        if not matches:
            return None

        longest = max(length for length, _, _ in matches)
        predicates = {
            predicate for length, predicate, _ in matches if length == longest
        }
        return next(iter(predicates)) if len(predicates) == 1 else None

    def plan(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        subject: str = "user",
    ) -> Optional[FactQuery]:
        """生成事实查询；未识别或存在谓词歧义时返回 ``None``。"""
        normalized_query = self._normalize_text(query, "query")
        predicate = self.detect_predicate(normalized_query)
        if predicate is None:
            return None
        return FactQuery(
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            retrieval_mode=self.detect_retrieval_mode(normalized_query),
            limit=limit,
            source_query=query,
        )
