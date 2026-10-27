"""RAG 检索层测试。

验证目标（Phase 5 评测的最小可执行版本）：
1. 首次运行自动构建向量索引；
2. 三个真实客服问法都能召回正确片段；
3. 每层（preprocessor / bm25 / fusion / rerank）可独立调用。

运行方式见 README 或直接：
    python -m tests.test_rag
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# 允许从项目根目录直接 python tests/test_rag.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25_retriever import bm25_retrieve  # noqa: E402
from rag.embeddings import EmbeddingError  # noqa: E402
from rag.fusion import fuse_two  # noqa: E402
from rag.preprocessor import preprocess  # noqa: E402
from rag.retriever import retrieve  # noqa: E402
from rag.vectorstore import VectorStoreError, build_index, index_exists  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

TEST_QUERIES = ["怎么退款", "密码忘了", "会员有什么用"]

# 期望召回的关键词 + 必须不出现的干扰关键词。
#
# 这里刻意断言"不含干扰主题"：如果切片退化成整篇知识库一个 chunk，
# 任何 query 都能"命中"关键词（因为全库都在结果里），测试会假阳性通过，
# 而混合检索实际上已完全失效。加上反向断言才能暴露这种退化。
EXPECTED = {
    "怎么退款": {"must": "退款", "must_not": ("会员", "密码")},
    "密码忘了": {"must": "密码", "must_not": ("退款", "会员")},
    "会员有什么用": {"must": "会员", "must_not": ("退款", "快递")},
}


def ensure_index() -> bool:
    """索引不存在时自动构建。这是幂等的，已存在则跳过。

    Returns:
        索引是否可用。

    构建失败**不中断测试**：本机 torch 可能被安全软件拦截（WinError 126），
    而分词/切片/BM25/RRF 的测试根本不依赖向量索引——
    因为环境问题把整个测试套件挡在门外，等于没有测试。
    """
    if index_exists():
        print("[setup] 检测到已有向量索引，跳过构建")
        return True

    print("[setup] 未检测到向量索引，开始构建（首次运行需下载 Embedding 模型）...")
    started = time.monotonic()
    try:
        build_index()
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"[setup] !! 向量索引构建失败，稠密检索相关用例将走降级路径: {exc}")
        print("[setup]    分词 / 切片 / BM25 / RRF 用例不受影响，继续执行")
        return False
    print(f"[setup] 索引构建完成，耗时 {time.monotonic() - started:.1f}s")
    return True


def test_preprocessor() -> None:
    """分词去停用词：语气词与标点应被剔除，关键业务词保留。"""
    print("\n=== 分词预处理 ===")
    for query in TEST_QUERIES:
        print(f"  {query!r} -> {preprocess(query)!r}")

    tokens = preprocess("我想问一下退款的事情").split()
    assert "退款" in tokens, f"业务关键词丢失: {tokens}"
    assert "我" not in tokens, f"停用词未被过滤: {tokens}"


def test_chunking() -> None:
    """切片粒度检查：这是混合检索能生效的前提。

    若知识库被切成单一大 chunk，BM25/向量/重排会全部退化，
    但端到端断言仍可能假阳性通过——所以这里单独卡一道。
    """
    print("\n=== 切片粒度 ===")
    from rag.loader import CHUNK_SIZE, get_chunk_texts

    chunks = get_chunk_texts()
    print(f"  共 {len(chunks)} 块，最长 {max(len(c) for c in chunks)} 字 "
          f"(chunk_size={CHUNK_SIZE})")
    for i, chunk in enumerate(chunks):
        topics = [kw for kw in ("密码", "快递", "退款", "会员") if kw in chunk]
        print(f"    [{i}] 主题={topics or ['通用']} | {chunk[:34]}...")
        assert len(topics) <= 1, (
            f"chunk[{i}] 混入多个主题 {topics}，切片已退化为整块合并"
        )

    assert len(chunks) >= 4, f"切片数 {len(chunks)} 过少，未按问答对边界拆分"
    assert len(set(chunks)) == len(chunks), "存在重复 chunk"
    print("  切片粒度：通过")


def test_bm25() -> None:
    """BM25 单独可用，且 top-1 精确命中对应主题。"""
    print("\n=== BM25 稀疏检索 ===")
    for query in TEST_QUERIES:
        docs = bm25_retrieve(query, k=3)
        print(f"  {query!r} 召回 {len(docs)} 篇")
        for i, doc in enumerate(docs, 1):
            print(f"    [{i}] {doc[:60]}...")
        assert docs, f"BM25 未召回任何结果: {query}"

        rule = EXPECTED[query]
        assert rule["must"] in docs[0], (
            f"BM25 top1 未命中 {rule['must']!r}: {docs[0][:40]}"
        )
    print("  BM25 精确性：通过")


def test_fusion() -> None:
    """RRF 融合：两路都命中的文档应排在只有一路命中的前面。"""
    print("\n=== RRF 融合 ===")
    a = ["共同文档A", "仅A文档1", "仅A文档2"]
    b = ["共同文档A", "仅B文档1", "仅B文档2"]
    fused = fuse_two(a, b, k=3)
    print(f"  输入 A={a}")
    print(f"  输入 B={b}")
    print(f"  融合结果: {fused}")

    assert "共同文档A" in fused, "双路命中文档应排在首位"
    assert fused[0] == "共同文档A", f"排序错误: {fused}"

    # 单路为空时应降级为直接返回另一路，而不是抛异常
    degraded = fuse_two(a, [], k=3)
    assert degraded == a[:3], f"单路降级行为异常: {degraded}"
    print("  单路降级: 通过")


def test_retrieve() -> None:
    """端到端混合检索：三个真实问法都应命中正确主题。

    稠密路不可用时会自动降级为仅 BM25，此时本用例验证的是降级路径的质量——
    这同样是有价值的断言（用户实际遇到的正是这条路径）。
    """
    print("\n=== 端到端混合检索 ===")
    if not index_exists():
        print("  （向量索引不可用，当前验证的是「仅 BM25」降级路径）")
    failures: list[str] = []

    for query in TEST_QUERIES:
        started = time.monotonic()
        docs = retrieve(query, k=3)
        elapsed = time.monotonic() - started

        print(f"\n  查询: {query!r}  (耗时 {elapsed:.2f}s)")
        for i, doc in enumerate(docs[:3], 1):  # 最多打印 3 个
            print(f"    [{i}] {doc}")
            print(f"        ({len(doc)} 字)")

        assert docs, f"检索返回空: {query}"

        rule = EXPECTED[query]
        top1 = docs[0]
        if rule["must"] not in top1:
            failures.append(f"{query}: top1 缺少 {rule['must']}")
            print(f"    !! top1 未包含期望关键词 {rule['must']!r}")

        # 反向断言：拦截「整篇知识库被当成一个 chunk 召回」的退化情形
        polluted = [kw for kw in rule["must_not"] if kw in top1]
        if polluted:
            failures.append(f"{query}: top1 混入干扰主题 {polluted}")
            print(f"    !! top1 混入干扰主题 {polluted}（疑似切片退化）")

    print("\n=== 结果汇总 ===")
    if failures:
        print(f"  失败项: {failures}")
        raise AssertionError("; ".join(failures))
    print(f"  全部 {len(TEST_QUERIES)} 条查询命中正确主题，且无干扰主题混入")


def run_all() -> None:
    """按顺序跑全部测试，便于直接执行脚本。"""
    tests = [
        test_preprocessor,
        test_chunking,
        test_bm25,
        test_fusion,
        test_retrieve,
    ]
    for test in tests:
        test()
    print("\n全部测试通过。")


def test_rag() -> None:
    """pytest 入口函数。"""
    ensure_index()
    run_all()


if __name__ == "__main__":
    ensure_index()
    try:
        run_all()
    except AssertionError as exc:
        print(f"\n测试失败: {exc}")
        sys.exit(1)