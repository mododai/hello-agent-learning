"""
工作记忆:
    -记忆系统中最活跃的部分，负责存储当前对话会话中的临时信息
"""
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List
import heapq
from ..base import BaseMemory, MemoryItem, MemoryConfig
import logging
logger = logging.getLogger(__name__)

class WorkingMemory(BaseMemory):
    """
    工作记忆实现。

    特点：
    1. 纯内存存储，适合短期上下文记忆。
    2. 支持容量限制，超过容量时删除优先级最低的记忆。
    3. 支持 TTL 自动过期清理。
    4. 支持混合检索：TF-IDF 向量检索 + 关键词匹配。
    """

    def __init__(self, config: MemoryConfig = None, storage_backend=None):
        super().__init__(config, storage_backend)

        self.config = config or getattr(self, 'config', None)

        self.max_capacity = getattr(self.config, "working_memory_capacity", 50) or 50
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 60) or 60

        self.memories : List[MemoryItem] = []


    def add(self, memory_item: MemoryItem) -> str | None:
        if memory_item.memory_type != self.memory_type:
            return None

        # 清理过期记忆
        self._expire_od_memories()

        self.memories.append(memory_item)

        if len(self.memories) > self.max_capacity:
            self._remove_lowest_priority()

        return memory_item.id


    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> List[MemoryItem]:

        # 清理过期记忆, 可能会出现空
        self._expire_od_memories()
        if not self.memories:
            return []

        filtered_memories: List[MemoryItem] = self.memories

        if user_id:
            filtered_memories = [m for m in filtered_memories if m.user_id == user_id]

        if not filtered_memories:
            return []
        # 计算TF-IDF向量检索
        vector_stores = self._try_tfidf_search(query)

        # 计算综合分数
        scored_memories = []
        for memory in filtered_memories:
            vector_score = vector_stores.get(memory.id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content)

            # 混合评分
            base_relevance = vector_score * 0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score
            # 时间衰减
            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance *= time_decay

            # 重要性权重
            importance_weight = 0.8 + (memory.importance_weight * 0.4)

            final_score = base_relevance * importance_weight

            if final_score > 0.0:
                scored_memories.append((final_score, memory))
        # 按分数排序并返回
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(self,
               memory_id: str,
               content: str = None,
               importance: float = None,
               metadata: Dict[str, Any] = None
               ) -> bool:

        for memory in self.memories:
            if memory.id != memory_id:
                continue

            if content is not None:
                memory.content = content
            if importance is not None:
                memory.importance = importance
            if metadata is not None:
                memory.metadata = metadata

            return True

        return False


    def remove(self, memory_id: str) -> bool:
        """删除指定工作记忆"""
        original_count = len(self.memories)
        self.memories = [
            memory for memory in self.memories
            if memory.id != memory_id
        ]
        removed = len(self.memories) != original_count
        return removed
    def has_memory(self, memory_id: str) -> bool:
        return any(memory_id == memory.id for memory in self.memories)

    def clear(self):
        self.memories.clear()

    def get_stats(self) -> Dict[str, Any]:

        self._expire_od_memories()

        active_memories = self.memories
        return {
            "count": len(active_memories),  # 活跃记忆数量
            "forgotten_count": 0,  # 工作记忆中已遗忘的记忆会被直接删除
            "total_count": len(self.memories),  # 总记忆数量
            #"current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            #"max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            #"session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            #"token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working"
        }

    def _expire_od_memories(self) -> None:
        if not self.memories:
            return
        # 截断时间, timedelta: 表示一段时间长度
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept: List[MemoryItem] = [] # 保留的记忆
        for memory in self.memories:
            if memory.timestamp < cutoff_time:
                # 记忆的时间在截断时间之前, 跳过
                continue
            kept.append(memory)

        if len(kept) == len(self.memories):
            # 没有过期记忆
            return
        # 覆盖
        self.memories = kept



    def _calculate_priority(self, memory_item: MemoryItem) -> float:
        """计算优先级"""
        priority = memory_item.importance

        time_decay = self._calculate_time_decay(memory_item.timestamp)
        priority *= time_decay
        return priority

    def _calculate_time_decay(self, timestamp) -> float:
        """计算时间衰减"""
        if timestamp is None:
            return 1.0
        decay_factor = getattr(self.config, "decay_factor", 0.95) or 0.95
        time_diff = datetime.now() - timestamp
        hours = max(time_diff.total_seconds() / 3600, 0)

        decay_factor = decay_factor ** (hours / 6)
        return decay_factor

    def _try_tfidf_search(self, query: str):
        """
        TF: 全称 Term Frequency, 即词频; 表示某个词在一篇文档中出现的频率。
        IDF: 全称是 Inverse Document Frequency, 即逆文档频率; 衡量一个词在所有文档中是否常见
        TF-IDF: TF 和 IDF 相乘, 一个词在当前文档中出现得多，并且在其他文档中出现得少，那么它的 TF-IDF 值就高。

        :param query:
        :return:
        """
        if not query or not query.strip():
            return {}

        if not self.memories:
            return {}

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # 过滤空记忆
            valid_memories = [
                memory
                for memory in self.memories
                if getattr(memory, "content", None) is not None
                   and str(memory.content).strip() != ""
            ]

            if not valid_memories:
                return {}

            documents = [
                str(memory.content)
                for memory in self.memories
            ]

            # TF-IDF 向量话器
            vectorizer = TfidfVectorizer(
                tokenizer=self._tokenize_for_tfidf,
                token_pattern=None,
                lowercase=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                norm="l2",
            )

            # 先用所有 memory 内容建立词表
            memory_vectors = vectorizer.fit_transform(documents)

            # 再把 query 转成同一个词表下的向量
            query_vector = vectorizer.transform([query])

            # 计算 query 与每条 memory 的余弦相似度
            similarities = cosine_similarity(
                query_vector,
                memory_vectors
            ).ravel()

            # 只返回分数大于 0 的结果
            scores = {}
            for memory, score in zip(valid_memories, similarities):
                if score > 0:
                    scores[memory.id] = float(score)

            return scores
        except Exception as e:
            logger.error(e)
            return {}

    def _tokenize_for_tfidf(self, text: str) -> List[str]:
        """
        给 TF-IDF 使用的分词函数
        :param text: 文本
        :return: 分词列表
        """

        text = (text or "").lower()
        if not text:
            return []

        # 使用jieba分词
        try:
            import jieba
            return [
                word
                for word in jieba.lcut(text)    # jieba.lcut返回的是列表
                if word.strip()
            ]
        except Exception as e:
            logger.error(f"未导入库 jieba 或 jieba分词失败: {e}")
            return []

    def _stopword(self, word: str) -> bool:
        """
        过滤简单词
        :param word:
        :return:
        """

        stopwords = {
             "的", "了", "是", "我", "你", "他", "她", "它",
            "在", "和", "与", "或", "就", "都", "而", "及",
            "a", "an", "the", "is", "are", "was", "were",
            "to", "of", "in", "on", "for", "and", "or"
        }
        return word in stopwords

    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """
        计算 query 和 memory.content 的关键词匹配分数。
        :param query:
        :param content:
        :return:
        """
        query = (query or "").strip().lower()
        content = (content or "").strip().lower()

        if not query or not content:
            return 0.0

        query_tokens = self._tokenize_for_tfidf(query)
        content_tokens = self._tokenize_for_tfidf(content)

        # 计数器
        query_counter = Counter(query_tokens)
        content_counter = Counter(content_tokens)

        # 集合(方便求并)
        query_set = set(query_counter.keys())
        content_set = set(content_counter.keys())

        matched_tokens = query_set & content_set

        if not matched_tokens:
            # 没有匹配的分词
            return 0.0

        # query 分词覆盖率: query 中有多少关键词被 content 命中
        coverage_score = len(matched_tokens) / len(content_tokens)

        # 命中词频分数: 命中的分词在 content 中出现越多，分数越高
        matched_frequency = sum(content_counter[token] for token in matched_tokens)
        total_content_frequency = sum(content_counter.values())
        frequency_score = matched_frequency / total_content_frequency

        # query 完整出现在 content 中
        phrase_bonus = 0.0
        if query in content:
            phrase_bonus = 0.3

        # query 中较长分词直接出现在 content 中，给额外加分
        important_token_bonus = 0.0
        for token in query_set:
            if len(token) > 2 and token in content:
                important_token_bonus += 0.05

        # 这里的常量后续可以改成从 配置(config类) 获取
        score = coverage_score * 0.65 + frequency_score * 0.2 + phrase_bonus + important_token_bonus

        return min(score, 1.0)  # 最大为 1


    def _remove_lowest_priority(self):
        if not self.memories:
            return

        lowest_priority = float('inf')
        lowest_memory = None

        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory
        if lowest_priority:
            self.remove(lowest_memory.id)

