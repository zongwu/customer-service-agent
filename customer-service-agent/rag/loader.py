"""知识库加载与切片。

切片策略：RecursiveCharacterTextSplitter，优先在空行（问答对边界）处断开，
其次在单行、句号、问号、分号处断开，保证一个 chunk 尽量是一个完整问答对，
不把答案拦腰截断——这是客服场景召回质量的关键。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger("rag.loader")

# 项目根目录（rag/ 的上一级），保证在任何工作目录下运行都能定位到 data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_PATH = PROJECT_ROOT / "data" / "knowledge_base.txt"

# 切片参数：chunk_size 300 字符对中文客服 FAQ 足够覆盖一个完整问答对；
# overlap 30 保证答案跨切片边界时不丢上下文。
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SEPARATORS: list[str] = ["\n\n", "\n", "。", "？", "；"]

SOURCE_NAME = "knowledge_base.txt"

# 单文件上限，防止误传入超大文件把内存打爆
MAX_FILE_BYTES = 16 * 1024 * 1024


class KnowledgeBaseError(RuntimeError):
    """知识库文件缺失、不可读或内容为空时抛出。属于终止类错误，不重试。"""


def resolve_kb_path(path: str | os.PathLike[str] | None = None) -> Path:
    """定位知识库文件路径，未显式指定时使用 data/knowledge_base.txt。"""
    if path is not None:
        return Path(path).expanduser().resolve()
    return DEFAULT_KB_PATH


def read_knowledge_base(path: str | os.PathLike[str] | None = None) -> str:
    """读取知识库原始文本。

    Raises:
        KnowledgeBaseError: 文件不存在、过大或内容为空。
    """
    kb_path = resolve_kb_path(path)

    if not kb_path.exists():
        raise KnowledgeBaseError(f"知识库文件不存在: {kb_path}")
    if not kb_path.is_file():
        raise KnowledgeBaseError(f"知识库路径不是文件: {kb_path}")

    size = kb_path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise KnowledgeBaseError(
            f"知识库文件过大({size} bytes > {MAX_FILE_BYTES} bytes): {kb_path}"
        )
    if size == 0:
        raise KnowledgeBaseError(f"知识库文件为空: {kb_path}")

    # 强制 UTF-8：知识库是中文内容，用系统默认编码在 Windows 上会直接抛
    # UnicodeDecodeError，这种失败是确定性的，重试没有意义。
    text = kb_path.read_text(encoding="utf-8")
    if not text.strip():
        raise KnowledgeBaseError(f"知识库内容仅含空白字符: {kb_path}")
    return text


def get_text_splitter():
    """构造切片器。

    兼容处理：langchain>=0.3 把 splitters 拆到 langchain-text-splitters 包，
    老版本则在 langchain.text_splitter 下。两个路径都试一次。
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - 仅老版本触发
        # 兼容 shim：新环境走不到这里，pyright 按当前环境解析报缺包是预期行为
        from langchain.text_splitter import (  # pyright: ignore[reportMissingImports]
            RecursiveCharacterTextSplitter,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )


@lru_cache(maxsize=1)
def load_documents(
    path: str | os.PathLike[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> list[Document]:
    """读取并切片知识库，返回带 metadata 的 Document 列表。

    每个 Document 的 metadata 为:
        {"source": "knowledge_base.txt", "chunk_id": <int>}

    结果做了 lru_cache：同一进程内重复建索引/查 BM25 时避免重复读盘与切片。
    需要强制刷新时调用 load_documents.cache_clear()。

    Args:
        path: 知识库路径，默认 data/knowledge_base.txt。
        chunk_size: 切片长度，默认 300。

    Raises:
        KnowledgeBaseError: 文件缺失或内容为空。
    """
    # cache 的 key 必须可哈希，Path/str 都支持，这里统一转成字符串
    cache_key = str(resolve_kb_path(path)) if path is not None else None
    return _load_documents_uncached(cache_key, chunk_size)


def _split_by_block(text: str) -> list[str]:
    """第一层：按空行切成「问答块」。

    为什么需要这一层：RecursiveCharacterTextSplitter 会贪心地把各级分隔符
    切出的碎片合并到填满 chunk_size 为止。若知识库全文短于 chunk_size
    （短FAQ 库是常态），它会把所有问答对塞进同一个 chunk——
    混合检索退化成「永远返回整篇知识库」，BM25、RRF、Cross-Encoder 全部失效。

    所以先按空行把每个问答对独立成块，再对超长块做第二层递归切分。
    这样既尊重 chunk_size 上限，又保证 chunk 粒度对齐语义边界。
    """
    blocks = [b.strip() for b in text.split("\n\n")]
    return [b for b in blocks if b]


def _load_documents_uncached(path: str | None, chunk_size: int) -> list[Document]:
    text = read_knowledge_base(path)

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - 仅老版本触发
        # 兼容 shim：新环境走不到这里，pyright 按当前环境解析报缺包是预期行为
        from langchain.text_splitter import (  # pyright: ignore[reportMissingImports]
            RecursiveCharacterTextSplitter,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )

    # 第二层：单个问答块超长时，用完整分隔符链继续细切
    raw_chunks: list[str] = []
    for block in _split_by_block(text):
        if len(block) <= chunk_size:
            raw_chunks.append(block)
            continue
        try:
            raw_chunks.extend(splitter.split_text(block))
        except Exception as exc:  # noqa: BLE001
            # 兜底：递归切分失败时整块保留，宁可 chunk 超长也不丢内容
            logger.warning("问答块递归切分失败，整块保留: %s", exc)
            raw_chunks.append(block)

    if not raw_chunks:
        raise KnowledgeBaseError(f"切片结果为空，检查切片参数: chunk_size={chunk_size}")

    documents: list[Document] = []
    seen: set[str] = set()
    for idx, content in enumerate(raw_chunks):
        cleaned = content.strip()
        # 纯空白 chunk 对召回无贡献，丢弃但不影响 chunk_id 连续性设计
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        documents.append(
            Document(
                page_content=cleaned,
                metadata={"source": SOURCE_NAME, "chunk_id": idx},
            )
        )

    if not documents:
        raise KnowledgeBaseError("切片后没有可用文档")

    logger.info("知识库切片完成: %d 个 chunk (chunk_size=%d)", len(documents), chunk_size)
    return documents


def get_chunk_texts(path: str | os.PathLike[str] | None = None) -> list[str]:
    """返回 chunk 纯文本列表，BM25 与 RRF 融合按此内容对齐。"""
    return [doc.page_content for doc in load_documents(path)]