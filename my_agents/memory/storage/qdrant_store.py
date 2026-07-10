"""
Qdrant 向量数据库存储封装实现
"""
import logging
import os
import threading
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
# qdrant_client 包
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    Filter, FieldCondition, MatchValue, SearchRequest,
    SearchParams, PointsSelector, PointIdsList, CollectionInfo
)


logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Qdrant Connection Manager"""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls,
                     url: Optional[str] = None,
                     api_key: Optional[str] = None,
                     collection_name: str = "agents_vectors",
                     vector_size: int = 384,
                     distance: str = "cosine",
                     timeout: int = 60,
                     **kwargs) -> 'QdrantVectorStore' :
        key = (url or "local", collection_name)
        if key not in cls._instances:
            with cls._lock:

                if key not in cls._instances:
                    logger.info("创建Qdrant连接")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", vector_size)),
                        distance=os.getenv("QDRANT_DISTANCE", distance),
                        timeout=int(os.getenv("QDRANT_TIMEOUT", timeout)),
                        **kwargs
                    )
                else:
                    logger.debug(f"复用现有Qdrant连接: {collection_name}")
        else:
            logger.debug(f"复用现有Qdrant连接: {collection_name}")

        return cls._instances[key]


class QdrantVectorStore:


    def __init__(self,
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 collection_name: str = "agents_vectors",
                 vector_size: int = 384,
                 distance: str = "cosine",
                 timeout: int = 60,
                 **kwargs):
        """

        :param url: Qdrant云服务URL
        :param api_key: Qdrant云服务API密钥
        :param collection_name: 集合名称
        :param vector_size: 向量维度
        :param distance: 距离度量方式 (cosine, dot, euclidean)
        :param timeout: 连接超时时间
        :param kwargs:
        """

        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        # HNSW 配置
        ## m: 图中每个节点最多连接多少个邻居。
        ## ef_construct: 插入新向量、构建索引时，搜索多少候选邻居。
        ## ef: 查询时在底层搜索时保留和扩展多少候选点。
        self.hnsw_m = int(os.getenv("QDRANT_HNSW_M", "32"))
        self.hnsw_ef_construct = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
        self.hnsw_ef = int(os.getenv("QDRANT_SEARCH_EF", "128"))
        self.search_exact = os.getenv("QDRANT_SEARCH_EXACT", "False") == "True"
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        # 距离计算方式
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        # 初始化客户端
        self.client: Optional[QdrantClient] = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            if self.url and self.api_key:
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
                logger.info(f"成功连接到Qdrant云服务: {self.url}")
            elif self.url:
            # 使用自定义URL（无API密钥）
                self.client = QdrantClient(
                    url=self.url,
                    timeout=self.timeout
                )
                logger.info(f"成功连接到Qdrant服务: {self.url}")
            else:
            # 使用本地服务（默认）
                self.client = QdrantClient(
                    host="localhost",
                    port=6333,
                    timeout=self.timeout
                )
                logger.info("成功连接到本地Qdrant服务: localhost:6333")
            # 创建或获取集合
            self._ensure_collection()
        except Exception as e:
            logger.error(f"Qdrant连接失败: {e}")

    @staticmethod
    def _to_point_id(memory_id: Any) -> Optional[Any]:
        """将业务 memory_id 转换为 Qdrant point ID。

        合法 UUID 直接使用；其他字符串采用确定性的 UUID5。写入和删除都
        必须使用此方法，否则会产生无法删除的孤儿向量。
        """
        if isinstance(memory_id, int):
            return memory_id
        if not isinstance(memory_id, str):
            return None
        try:
            uuid.UUID(memory_id)
            return memory_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        try:

            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            hnsw_cfg = HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.hnsw_ef_construct,
            )

            if self.collection_name in collection_names:
                logger.info(f"使用现有Qdrant集合: {self.collection_name}")
                info: CollectionInfo = self.client.get_collection(self.collection_name)
                try:
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        hnsw_config=hnsw_cfg,
                    )
                except Exception as ie:
                    logger.debug(f"跳过更新HNSW配置: {ie}")
            else:
                logger.info(f"创建Qdrant集合: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    hnsw_config=hnsw_cfg,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
            self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise

    def _ensure_payload_indexes(self):
        """
        提前规定好 payload 可以加快条件匹配 (类似于索引)
        | 类型         | Python 写法                                    | 适合字段                 | 支持的典型过滤   |
        | ----------  | --------------------------------------------   | ---------------------  | -------------- |
        | `keyword`   | `PayloadSchemaType.KEYWORD`                    | ID、标签、枚举、分类      | 精确匹配      |
        | `integer`   | `PayloadSchemaType.INTEGER`                    | 时间戳、数量、等级、轮次   | 精确匹配、范围过滤 |
        | `float`     | `PayloadSchemaType.FLOAT`                      | 价格、评分、相似度、权重   | 范围过滤      |
        | `bool`      | `PayloadSchemaType.BOOL`                       | 是否启用、是否外部数据     | 精确匹配      |
        | `geo`       | `PayloadSchemaType.GEO`                        | 经纬度位置               | 地理范围过滤    |
        | `datetime`  | `PayloadSchemaType.DATETIME`                   | 创建时间、更新时间        | 时间范围过滤    |
        | `text`      | `PayloadSchemaType.TEXT`或`TextIndexParams`    | 长文本字段               | 全文检索      |
        | `uuid`      | `PayloadSchemaType.UUID`                       | UUID 字段               | 精确匹配，节省内存 |

        """
        try:
            index_fields = [
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("user_id", models.PayloadSchemaType.KEYWORD),
                ("memory_id", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.INTEGER),
                ("modality", models.PayloadSchemaType.KEYWORD),  # 感知记忆模态筛选
                ("source", models.PayloadSchemaType.KEYWORD),
                ("external", models.PayloadSchemaType.BOOL),
                ("namespace", models.PayloadSchemaType.KEYWORD),
                # RAG相关字段索引
                ("is_rag_data", models.PayloadSchemaType.BOOL),
                ("rag_namespace", models.PayloadSchemaType.KEYWORD),
                ("data_source", models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, schema_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception as ie:
                    logger.warning(f"索引 {field_name} 已存在或创建失败: {ie}")

        except Exception as e:
            logger.warning(f"创建payload索引时出错: {e}")

    def add_vectors(self,
                    vectors: List[List[float]],
                    metadata: List[Dict[str, Any]],
                    ids: Optional[List[str]] = None,
                    ) -> bool:

        """
        添加向量到Qdrant
        :param vectors: 向量列表
        :param metadata: 元数据列表
        :param ids: 可选的ID列表
        :return: bool 是否成功
        """

        try:
            # 确保各个参数数组的长度相同
            if not vectors:
                logger.warning("向量列表为空")
                return False

            if len(vectors) != len(metadata):
                logger.error(f"vectors 和 metadata 数量不一致: {len(vectors)} != {len(metadata)}")
                return False

            if ids is not None and len(ids) != len(vectors):
                logger.error(f"ids 和 vectors 数量不一致: {len(ids)} != {len(vectors)}")
                return False

            # 生成ID（如果未提供）
            # 优先使用 metadata 中的 memory_id，避免重复添加
            if ids is None:
                ids = [
                    str(meta.get("memory_id", f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}"))
                    for i, meta in enumerate(metadata)
                ]
            # 在Qdrant中存在collection中的数据是Point
            logger.info(f"开始添加向量: n_vectors={len(vectors)} n_meta={len(metadata)} collection={self.collection_name}")
            points: List[PointStruct] = []
            for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
                vector_size = len(vector)
                if vector_size != self.vector_size:
                    logger.warning(f"向量维度不匹配: 期望{self.vector_size}, 实际{vector_size}")
                    continue

                meta_with_timestamp = meta.copy()
                # 保留业务时间戳，另行记录向量索引写入时间。
                meta_with_timestamp.setdefault("indexed_at", int(datetime.now().timestamp()))

                if "external" in meta_with_timestamp and not isinstance(meta_with_timestamp.get("external"), bool):
                    # normalize to bool
                    val = meta_with_timestamp.get("external")
                    meta_with_timestamp["external"] = True if str(val).lower() in ("1", "true", "yes") else False

                safe_id = self._to_point_id(point_id)
                if safe_id is None:
                    logger.info(f"不支持 ID 类型{type(point_id)}")
                    continue

                points.append(
                    PointStruct(
                        id=safe_id,
                        vector=vector,
                        payload=meta_with_timestamp,
                    )
                )

            if not points:
                logger.warning("没有有效的向量点")
                return False

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info("添加完成")
            return True
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False

    def search_similar(self,
                       query_vector: list[float],
                       limit: int = 10,
                       score_threshold: Optional[float] = None,
                       where: Optional[Dict[str, Any]] = None
                       ) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        :param query_vector: 查询向量
        :param limit: 返回结果数量限制
        :param score_threshold: 相似度阈值
        :param where: 过滤条件
        :return:  搜索结果
        """

        try:
            if len(query_vector) != self.vector_size:
                logger.warning(f"查询向量维度错误: 期望{self.vector_size}, 实际{len(query_vector)}")
                return []
            query_filter = None
            if where:
                # 构建过滤
                conditions = []
                for key, value in where.items():
                    if isinstance(value, (str, int, float, bool)):
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value),
                            )
                        )
                if conditions:
                    query_filter = Filter(must = conditions)

            search_params = SearchParams(
                hnsw_ef=self.hnsw_ef,   # 搜索的 ef 配置
                exact=self.search_exact,    # 精确搜索启用配置
            )
            res = self.client.query_points(
                query=query_vector,
                collection_name=self.collection_name,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,    # 给相似度搜索结果设置一个最低相关性门槛，低于这个门槛的结果不返回
                search_params=search_params,
                with_payload=True,
                with_value=False,

            )
            search_points = res.points

            # 转换格式
            results = []
            for point in search_points:
                results.append(
                    {
                        "id": point.id,
                        "score": point.score, # 与 query_vector 的距离
                        "metadata": point.payload,  # 载荷
                    }
                )
            return results
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def delete_vectors(self, ids: List[str]) -> bool:
        """
        删除向量
        :param ids: 要删除的向量ID列表
        :return:
        """
        try:
            if not ids:
                return True

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=ids,
                ),
                wait=True
            )
            logger.info(f"成功删除 {len(ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    def delete_memories(self, memory_ids: List[str]) -> bool:
        try:
            if not memory_ids:
                return True
            point_ids = []
            for memory_id in memory_ids:
                point_id = self._to_point_id(memory_id)
                if point_id is None:
                    logger.warning("跳过不支持的 memory_id 类型: %s", type(memory_id))
                    continue
                point_ids.append(point_id)

            deleted = self.delete_vectors(point_ids)
            logger.info(f"成功按memory_id删除 {len(memory_ids)} 个Qdrant向量")
            return deleted
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
               获取集合信息

               Returns:
                   Dict: 集合信息
               """
        try:
            collection_info: CollectionInfo = self.client.get_collection(self.collection_name)

            info = {
                "store_type": "qdrant",
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": self.vector_size,
                    "distance": self.distance.value,
                }
            }

            return info

        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}

    def __del__(self):
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception as e:
                pass
