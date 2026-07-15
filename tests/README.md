# 记忆系统验收测试

该目录保存可重复运行、无需连接外部服务的记忆系统验收测试。

## 本阶段覆盖范围

- 未知谓词默认采用多值策略
- 自定义谓词策略注册
- 多值偏好同时保持 active
- 单值事实自动 supersede
- 默认检索过滤历史事实
- `include_inactive=True` 查询历史
- 显式撤回但不删除历史
- `FactChange` 精确撤回某个多值事实
- 新事实写入失败时旧事实保持 active
- 撤回和替代的用户隔离
- SQLite `semantic_facts` 规范化事实表和部分唯一索引
- 多值事实列表查询及“第二个值再次写入”的精确去重
- SQLite 约束失败时完整回滚，不留下半条记忆
- `current / timeline / audit` 结构化事实检索模式
- 城市时间线按谓词隔离，不混入 active 饮品，撤回事实仅在审计模式出现

## 运行命令

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

同时运行旧测试和新验收测试：

```powershell
.\.venv\Scripts\python.exe -m unittest -v `
  test_semantic_fact.py `
  test_semantic_memory.py `
  tests.test_semantic_fact_lifecycle `
  tests.test_semantic_fact_sqlite
```

## 测试边界

测试使用 `FakeEmbedder` 和 `FakeVectorStore`，重点验证业务规则和 SQLite 持久化。
Ollama 与 Qdrant Cloud 的网络连通性由根目录的 `test_embedding.py` 和 `test.py`
单独验证，避免外部网络波动导致生命周期回归测试不稳定。
