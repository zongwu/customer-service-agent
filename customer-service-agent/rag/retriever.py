"""统一检索入口：BM25 + 向量混合召回 → RRF 融合 → Cross-Encoder 重排。

链路：
    query
      ├─ 预处理（分词去停用词，供 BM25）
      ├─ 稠密检索  Chroma.similarity_search(k=5)
      ├─ 稀疏检索  BM25.k=5
      ├─ RRF 融合   k=5
      └─ 重排       top-3

降级设计（每一路都可单独失效而不拖垮整条链路）：
    稠密失败 → 仅 BM25；稀疏失败 → 仅向量；
    融合失败 → 退回向量原始结果；重排失败 → 退回融合结果。
    每次降级都打 warn 日志，否则线上降级了都不知道。

有界自主：本模块**没有**自主循环、无递归、无 while True。
单次 retrieve() 的推理调用次数上限由各段 k 值固定（≤ 5+5 次打分），
天然满足有界要求。
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid

logger = logging.getLogger("rag.retriever")

# 各段召回条数。融合候选 5 条、重排取 top3——候选太多会拖慢 Cross-Encoder，
# 太少又容易在融合阶段就把正确答案挤掉，5 是当前语料规模下的平衡点。
DEFAULT_TOP_K = 3
DENSE_K = 5
SPARSE_K = 5
FUSION_K = 5

# 输入长度上限：超长 query 没有任何检索价值，直接拒绝而非硬塞进模型
MAX_QUERY_LENGTH = 512


class RetrievalError(RuntimeError):
    """检索链路整体不可用（两路召回都失败）。"""


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def dense_retrieve(query: str, k: int = DENSE_K) -> list[str]:
    """稠密检索：Chroma 向量库相似度搜索。"""
    from .vectorstore import get_vectorstore

    store = get_vectorstore()
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs if doc and doc.page_content.strip()]


def sparse_retrieve(query: str, k: int = SPARSE_K) -> list[str]:
    """稀疏检索：BM25 词面匹配。"""
    from .bm25_retriever import bm25_retrieve

    return bm25_retrieve(query, k=k)


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    *,
    trace_id: str | None = None,
) -> list[str]:
    """混合检索主入口。

    Args:
        query: 用户原始问题。
        k: 最终返回条数，默认 3。
        trace_id: 链路追踪 ID，不传则自动生成。用于把一次请求的多段日志串起来。

    Returns:
        Top-K 相关文档文本。

    Raises:
        ValueError: query 为空或超长。
        RetrievalError: 稠密与稀疏检索同时失败。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query 超长({len(query)} > {MAX_QUERY_LENGTH} 字符)，请精简后再试"
        )

    trace_id = trace_id or _new_trace_id()
    started = time.monotonic()

    # 预处理：只喂给 BM25。向量侧用原始 query——
    # Embedding 自带的分词器在完整语义上表现更好，人工分词反而损失信息。
    from .preprocessor import preprocess

    processed = preprocess(query)
    logger.info(
        "[%s] 检索开始: query=%r -> 预处理=%r", trace_id, query, processed
    )

    dense_docs: list[str] = []
    sparse_docs: list[str] = []

    try:
        dense_docs = dense_retrieve(query, k=DENSE_K)
        logger.info("[%s] 稠密召回 %d 篇", trace_id, len(dense_docs))
    except Exception as exc:  # noqa: BLE001 - 检索异常类型不可穷举
        logger.warning(
            "[%s] 稠密检索失败，降级为仅 BM25: %s\n%s",
            trace_id, exc, traceback.format_exc(),
        )

    try:
        sparse_docs = sparse_retrieve(processed, k=SPARSE_K)
        logger.info("[%s] 稀疏召回 %d 篇", trace_id, len(sparse_docs))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] BM25 检索失败，降级为仅向量: %s\n%s",
            trace_id, exc, traceback.format_exc(),
        )

    if not dense_docs and not sparse_docs:
        raise RetrievalError(
            f"[{trace_id}] 稠密与稀疏检索均失败，无法完成检索。"
            f"请检查 chroma_db 索引是否存在，以及 BM25 语料是否为空。"
        )

    # RRF 融合：内部已处理单路为空的情况
    from .fusion import fuse_two

    fused = fuse_two(dense_docs, sparse_docs, k=FUSION_K)
    if not fused:
        # 融合层异常或候选全被过滤 → 退回向量原始结果
        logger.warning("[%s] 融合结果为空，退回稠密原始结果", trace_id)
        fused = dense_docs[:FUSION_K]

    if len(fused) <= k:
        final = fused[:k]
    else:
        try:
            from .reranker import rerank

            final = rerank(query, fused, top_k=k)
        except Exception as exc:  # noqa: BLE001
            # 重排是精度优化而非必需项，失败不应影响召回可用性
            logger.warning(
                "[%s] 重排失败，降级为返回融合结果: %s\n%s",
                trace_id, exc, traceback.format_exc(),
            )
            final = fused[:k]

    logger.info(
        "[%s] 检索完成: dense=%d sparse=%d fused=%d -> %d 篇, 耗时 %.2fs",
        trace_id, len(dense_docs), len(sparse_docs), len(fused), len(final),
        time.monotonic() - started,
    )
    return final


def retrieve_with_trace(query: str, k: int = DEFAULT_TOP_K) -> tuple[list[str], str]:
    """带 trace_id 返回的检索，供上层记录链路。"""
    trace_id = _new_trace_id()
    return retrieve(query, k=k, trace_id=trace_id), trace_id