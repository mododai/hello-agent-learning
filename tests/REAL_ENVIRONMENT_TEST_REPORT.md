# 语义记忆真实环境测试报告

测试日期：2026-07-15（Asia/Shanghai）

## 测试环境

- SQLite：临时真实数据库文件，不使用测试替身
- Embedding：项目 `.env` 配置的真实服务
- 向量维度：2560
- Qdrant：项目 `.env` 配置的真实云服务
- Collection：`agents_vectors`
- 数据隔离：每次运行生成唯一 `user_id` 和 UUID memory ID

## 执行命令

```powershell
.\.venv\Scripts\python.exe tests\real_semantic_memory_check.py
```

## 验收结果

以下场景全部通过：

1. 向真实 Embedding 服务请求 2560 维向量。
2. 向真实 Qdrant 写入拿铁、绿茶、杭州和上海事实向量。
3. SQLite `semantic_facts` 保存拿铁和绿茶两条独立 active 多值事实。
4. 第二次添加绿茶时复用原 memory ID，没有新增重复 SQLite 行或 Qdrant point。
5. `current_city` 从杭州变为上海后，杭州为 `superseded`，上海为 `active`。
6. Qdrant 语义检索可以召回当前事实。
7. 规则规划器把“现在和以前住在哪里”转换为 `current_city + timeline`，结果只含上海和杭州。
8. 规则规划器把“审计撤回的技能记录”转换为 `skill + audit` 并返回撤回记录。
9. 测试结束后按唯一用户清理数据。
10. 清理后 SQLite 剩余测试事实为 0。
11. 清理后 Qdrant 剩余测试 points 为 0。

最终结果：`PASS`

## 观察到的网络警告

执行期间出现一次：

```text
SSL: UNEXPECTED_EOF_WHILE_READING
```

该异常发生在 Qdrant collection 初始化检查阶段。当前 `QdrantVectorStore` 会记录错误并
保留已经创建的客户端，本次后续 upsert、query 和 delete 均成功，因此没有导致测试失败
或遗留数据。这仍说明云连接存在偶发 TLS/网络波动，后续可以为初始化检查增加有限次数的
重试，并让持续失败时明确终止初始化。
