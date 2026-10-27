"""RRF（Reciprocal Rank Fusion）融合。

选 RRF 而不是分数加权归一化的原因：
- Chroma 的余弦相似度与 BM25 的分数完全不在同一量纲，BM25 分数甚至无上界，
  硬做 min-max 归一化会被离群值主导；
- RRF 只吃**排名**，天然免疫量纲差异，且不需要调权重；
- k=60 是原论文的经验值，不是随手写的魔数 [待核实: Cormack et al., CIKM 2009]。

本项目 dense 与 sparse 召回数量接近，因此不做配额切分——
配额会强制把低质量候选塞进结果，反而伤害重排效果。
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger("rag.fusion")

# RRF 平滑常数，抑制头部排名的绝对统治力
DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 5


def reciprocal_rank_fusion(
    doc_lists: Sequence[Sequence[str]],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[str]:
    """对多个排序文档列表做 RRF 融合。

    Args:
        doc_lists: 多路召回结果，每项是一个按相关性降序排列的文档列表。
        top_k: 融合后返回条数，默认 5。
        rrf_k: RRF 平滑常数，默认 60。取值越大，头部排名优势越弱。

    Returns:
        融合后按得分降序排列的 Top-K 文档文本。输入为空时返回空列表。

    Note:
        文档以文本内容作为去重键。两路召回即使顺序不同，文本一致即视为同一
        文档，分数累加——这正是 RRF 想要的「多路都认同则排前」的效果。
    """
    if not doc_lists or top_k <= 0:
        return []

    fused: dict[str, float] = {}
    # 记录首次出现的顺序，用于同分时稳定排序（保证结果可复现）
    first_seen: dict[str, int] = {}

    for ranked in doc_lists:
        if not ranked:
            continue
        for rank, doc in enumerate(ranked, start=1):
            if doc is None:
                continue
            key = doc.strip()
            if not key:
                continue
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(key, len(first_seen))

    if not fused:
        logger.warning("RRF 融合没有收到任何有效文档")
        return []

    # 同分时按首次出现顺序排，避免 set/dict 顺序导致结果抖动
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    results = [doc for doc, _ in ordered[:top_k]]
    logger.debug("RRF 融合: 输入 %d 路, 输出 %d 篇", len(doc_lists), len(results))
    return results


def fuse_two(
    list_a: Sequence[str] | None,
    list_b: Sequence[str] | None,
    k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[str]:
    """两路召回的 RRF 融合（dense + sparse 场景的便捷封装）。

    某一路为空时不会报错，直接退化为另一路的结果——
    这是检索层最重要的降级：向量库故障不该让整个 RAG 链路挂掉。
    """
    candidates = [lst for lst in (list_a or [], list_b or []) if lst]
    if not candidates:
        return []
    if len(candidates) == 1:
        logger.warning("仅一路召回有结果，跳过 RRF 融合直接返回（降级路径）")
        return list(candidates[0][:k])
    return reciprocal_rank_fusion(candidates, top_k=k, rrf_k=rrf_k)


# 语义更明确的重命名，指向同一实现
rrf_fusion = reciprocal_rank_fusion