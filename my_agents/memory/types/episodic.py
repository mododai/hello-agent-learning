"""
情景记忆负责存储具体的事件和经历，它的设计重点在于保持事件的完整性和时间序列关系
"""
from ..base import MemoryConfig

class EpisodicMemory:
    """
    情景记忆实现
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        