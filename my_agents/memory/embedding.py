from typing import Union, List, Any
from abc import ABC, abstractmethod
import httpx
import os

class EmbeddingModel(ABC):

    @abstractmethod
    def encode(self, text: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalOllamaEmbeddingModel(EmbeddingModel):
    """适配 Ollama 部署的 embedding 模型"""
    def __init__(self,
                 model: str = "qwen3-embedding:4b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 **kwargs
                 ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._dimension = None


    def encode(self, text: Union[str, List[str]]) -> List[Any]:
        """
        发送请求编码 text

        curl http://localhost:11434/api/embed -d '{
          "model": "all-minilm",
          "input": ["Why is the sky blue?", "Why is the grass green?"]
        }'

        response:
        {
          "model": "all-minilm",
          "embeddings": [
            [
              0.010071029, -0.0017594862, 0.05007221, 0.04692972, 0.054916814,
              0.008599704, 0.105441414, -0.025878139, 0.12958129, 0.031952348
            ],
            [
              -0.0098027075, 0.06042469, 0.025257962, -0.006364387, 0.07272725,
              0.017194884, 0.09032035, -0.051705178, 0.09951512, 0.09072481
            ]
          ]
        }
        :param text: 需要编码得文本
        :return:
        """
        # 将当个字符串转为字符串列表
        if isinstance(text, str):
            text = [text]


        # 发送请求
        response = httpx.post(
            url=f"{self.base_url}/api/embed",
            json= {
                "model": self.model,
                "input": text,
            },
            timeout=self.timeout,
        )

        # 如果出现http异常则抛出
        response.raise_for_status()
        # json格式结果
        data = response.json()

        return data["embeddings"]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vector = self.encode("dimension test")
            self._dimension = len(vector[0])
        return self._dimension



_embedding_model = None
def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        embed_model_type = os.getenv("EMBED_MODEL_TYPE", "Ollama")
        if embed_model_type == "Ollama":
            _embedding_model = LocalOllamaEmbeddingModel(
                model=os.getenv("EMBED_MODEL_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")),
                base_url=os.getenv("EMBED_MODEL_BASE_URL", "http://localhost:11434"),
            )
        else:
            raise ValueError(f"不支持的 embedding 模型类型: {embed_model_type}")
    return _embedding_model

def get_dimension(default: int= 384):
    try:
        return int(getattr(get_embedding_model(), "dimension"))
    except Exception:
        return int(default)
