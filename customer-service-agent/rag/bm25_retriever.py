"""BM25 稀疏检索。

为什么需要 BM25：Embedding 对语义相近但用词不同的表达召回好，但对「订单」「退款」
这类精确关键词的区分度有限；BM25 走词面匹配，两者互补。Dense + Sparse 混合检索
在客服 FAQ 这种「问题表述多样、答案关键词固定」的场景提升最明显。
"""

from __future__ import annotations

import logging
import threading

from rank_bm25 import BM25Okapi

logger = logging.getLogger("rag.bm25")

_index_cache: "BM25Index | None" = None
_index_lock = threading.Lock()


class BM25IndexError(RuntimeError):
    """BM25 索引构建失败。终止类错误。"""


class BM25Index:
    """对一组 chunk 文本建立的 BM25Okapi 索引。

    使用注意：BM25Okapi 对**所有**语料一次性建模，语料变动需重建实例。
    本项目语料是静态知识库文件，一次构建足够。
    """

    def __init__(self, documents: list[str]) -> None:
        if not documents:
            raise BM25IndexError("BM25 索引不能为空")

        self._documents = list(documents)

        # 分词在预处理层做，保证索引侧与查询侧用同一套规则。
        # 关键：不能对空词串建索引，BM25Okapi 计算平均文档长度时会除零。
        from .preprocessor import preprocess

        corpus_tokens: list[list[str]] = []
        kept_positions: list[int] = []
        for idx, text in enumerate(self._documents):
            tokenized = preprocess(text)
            if not tokenized:
                # 整篇都是停用词/标点的文档跳过，避免污染 IDF 统计
                continue
            corpus_tokens.append(tokenized.split())
            kept_positions.append(idx)

        if not corpus_tokens:
            raise BM25IndexError(
                "所有文档经预处理后均为空，无法建立 BM25 索引；"
                "请检查知识库内容或 preprocessor 的停用词表"
            )

        self._kept_positions = kept_positions
        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info("BM25 索引构建完成: %d/%d 篇文档", len(corpus_tokens), len(self._documents))

    @property
    def documents(self) -> list[str]:
        return self._documents

    def get_scores(self, query: str) -> list[float]:
        """返回每篇文档对该 query 的 BM25 分数（顺序与 documents 一致）。"""
        from .preprocessor import preprocess

        tokenized = preprocess(query)
        if not tokenized:
            # query 全是停用词：无法形成有效匹配，返回全 0 而不是抛错，
            # 让上层走「无稀疏候选」的降级路径。
            return [0.0] * len(self._documents)

        raw_scores = self._bm25.get_scores(tokenized.split())
        scores = [0.0] * len(self._documents)
        for score, position in zip(raw_scores, self._kept_positions):
            scores[position] = float(score)
        return scores

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """返回 Top-K 文档文本（按 BM25 分数降序）。

        零分文档不返回：零分意味着没有任何词命中，塞进候选只会污染重排。
        """
        if k <= 0:
            return []

        scores = self.get_scores(query)
        ranked = sorted(
            ((score, idx) for idx, score in enumerate(scores)),
            key=lambda pair: (-pair[0], pair[1]),
        )

        results: list[str] = []
        for score, idx in ranked[:k]:
            if score <= 0.0:
                continue
            results.append(self._documents[idx])

        logger.debug("BM25 召回 %d 篇 (k=%d)", len(results), k)
        return results


def build_index(documents: list[str] | None = None) -> BM25Index:
    """构建 BM25 索引；documents 为空时从 loader 取默认知识库切片。"""
    if documents is None:
        from .loader import get_chunk_texts

        documents = get_chunk_texts()
    return BM25Index(documents)


def get_bm25_index() -> BM25Index:
    """进程内单例索引。知识库变更后调用 reset_bm25_cache()。"""
    global _index_cache
    if _index_cache is None:
        with _index_lock:
            if _index_cache is None:
                _index_cache = build_index()
    return _index_cache


def reset_bm25_cache() -> None:
    global _index_cache
    with _index_lock:
        _index_cache = None


def bm25_retrieve(query: str, k: int = 5) -> list[str]:
    """BM25 检索入口，返回 Top-K 文档文本。

    Args:
        query: 原始 query 文本，内部会做分词与停用词过滤。
        k: 返回条数，默认 5。

    Returns:
        文档文本列表；无命中时返回空列表。
    """
    if not query or not query.strip():
        return []
    return get_bm25_index().retrieve(query, k=k)