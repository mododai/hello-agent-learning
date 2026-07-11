import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from my_agents.memory.embedding import LocalOllamaEmbeddingModel


def assert_vector(value: Any, name: str) -> None:
    """检查一个值是不是 embedding 向量。"""
    assert isinstance(value, list), f"{name} 应该是 list，实际是 {type(value)}"
    assert len(value) > 0, f"{name} 不应该是空列表"
    preview_types = [type(item).__name__ for item in value[:5]]
    assert all(
        isinstance(item, (int, float)) for item in value[:10]
    ), f"{name} 的元素应该是数字，前 5 个元素类型是 {preview_types}"


def main() -> None:
    model_name = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    embedding_model = LocalOllamaEmbeddingModel(
        model=model_name,
        base_url=base_url,
        timeout=60,
    )

    print(f"使用模型: {model_name}")
    print(f"Ollama 地址: {base_url}")

    single_vectors = embedding_model.encode("你好，世界")
    assert isinstance(single_vectors, list), "单条输入也应返回向量批次"
    assert len(single_vectors) == 1, "单条输入应该只返回一个向量"
    single_vector = single_vectors[0]
    assert_vector(single_vector, "single_vectors[0]")
    print(f"单条文本向量维度: {len(single_vector)}")
    print(f"单条文本向量前 5 项: {single_vector[:5]}")

    batch_vectors = embedding_model.encode(["你好，世界", "今天适合学习 agent 记忆系统"])
    assert isinstance(batch_vectors, list), "batch_vectors 应该是 list"
    assert len(batch_vectors) == 2, f"批量输入 2 条文本，应该返回 2 个向量，实际返回 {len(batch_vectors)} 个"
    assert_vector(batch_vectors[0], "batch_vectors[0]")
    assert_vector(batch_vectors[1], "batch_vectors[1]")
    assert len(batch_vectors[0]) == len(batch_vectors[1]), "同一个模型返回的向量维度应该一致"
    print(f"批量文本向量数量: {len(batch_vectors)}")

    assert embedding_model.dimension == len(single_vector), (
        f"dimension 应该等于单条向量长度："
        f"dimension={embedding_model.dimension}, len(single_vector)={len(single_vector)}"
    )
    print(f"dimension 属性: {embedding_model.dimension}")

    print("Embedding 测试通过")


if __name__ == "__main__":
    main()
