"""Chroma 向量库封装。

两个入口：
- build_index():从知识库构建索引并持久化到 CHROMA_DIR
- load_vectorstore():从已有目录加载索引

注意：不同 chromadb / langchain-chroma 版本在构造函数参数上有差异
（旧版用 persist_directory，新版推荐 collection_name + persist_directory），
这里统一走 langchain_chroma.Chroma 的公共参数，并屏蔽持久化 API 变更。
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger("rag.vectorstore")

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "customer_kb"

# 索引写入是幂等的：同一目录重复 build 会产生重复向量，
# 因此重建前强制要求显式确认，避免线上误调用把库清空。
_write_lock = threading.Lock()

# Chroma 查询超过这个秒数视为异常，避免 API 请求被检索拖死。
DEFAULT_QUERY_TIMEOUT = 5.0


class VectorStoreError(RuntimeError):
    """向量库构建或加载失败。终止类错误，不做自动重试。"""


def get_chroma_dir() -> Path:
    return Path(os.getenv("CHROMA_DIR") or DEFAULT_CHROMA_DIR).expanduser().resolve()


def get_collection_name() -> str:
    return os.getenv("COLLECTION_NAME") or DEFAULT_COLLECTION_NAME


def index_exists() -> bool:
    """判断本地是否已有可用索引目录。

    Chroma 的持久化目录以 subdir_name + .chroma 文件标记完成写入，
    只判断目录存在会把「写了一半」的目录误判为可用索引。
    """
    chroma_dir = get_chroma_dir()
    if not chroma_dir.is_dir():
        return False
    return any(chroma_dir.iterdir())


def _collection_is_populated(collection: Any) -> bool:
    """检查集合里是否已有向量。空集合视为需要重建。"""
    try:
        count = collection.count()
    except Exception as exc:  # noqa: BLE001 - 老版本 API 差异导致
        logger.debug("collection.count() 不可用，改用 peek 判断: %s", exc)
        return True
    return count > 0


def _chroma_cls():
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise VectorStoreError(
            "找不到 langchain_chroma.Chroma，请先安装 langchain-chroma>=0.1.0"
        ) from exc
    return Chroma


def build_index(
    chunks: list[Any] | None = None,
    *,
    force_rebuild: bool = False,
) -> Any:
    """构建并持久化向量索引。

    Args:
        chunks: 已切片的 Document 列表。为 None 时从 loader 读取默认知识库。
        force_rebuild: True 时先删除旧索引目录再重建（破坏性操作，需显式传参）。

    Returns:
        Chroma 实例。

    Raises:
        VectorStoreError: 依赖缺失、构建失败。
    """
    from .embeddings import get_embeddings
    from .loader import load_documents

    with _write_lock:
        chroma_dir = get_chroma_dir()
        collection_name = get_collection_name()

        if index_exists() and not force_rebuild:
            raise VectorStoreError(
                f"索引已存在于 {chroma_dir}。如需重建请传 force_rebuild=True，"
                "该操作会删除并重新生成整个索引目录。"
            )

        if force_rebuild and chroma_dir.exists():
            logger.warning("force_rebuild=True，删除旧索引目录: %s", chroma_dir)
            shutil.rmtree(chroma_dir, ignore_errors=True)

        documents = chunks if chunks is not None else load_documents()
        if not documents:
            raise VectorStoreError("没有可写入的 chunk，知识库为空")

        embeddings = get_embeddings()
        Chroma = _chroma_cls()

        logger.info(
            "开始构建向量索引: %d 个 chunk -> %s (collection=%s)",
            len(documents), chroma_dir, collection_name,
        )

        try:
            # langchain-chroma 的 from_documents 内部即完成持久化，
            # 不需要（也不应）再调用 persist()，那是旧版 API。
            store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(chroma_dir),
            )
        except Exception as exc:
            raise VectorStoreError(f"构建向量索引失败: {exc}") from exc

        try:
            if not _collection_is_populated(store._collection):  # noqa: SLF001
                raise VectorStoreError(
                    f"索引写入后集合为空，请检查嵌入维度与数据是否匹配: {chroma_dir}"
                )
        except AttributeError:  # 非 chromadb 后端实现，跳过校验
            pass

        logger.info("向量索引构建完成: %s", chroma_dir)
        return store


def load_vectorstore() -> Any:
    """从 CHROMA_DIR 加载已有索引。

    Raises:
        VectorStoreError: 索引不存在或加载失败。
    """
    from .embeddings import get_embeddings

    chroma_dir = get_chroma_dir()
    collection_name = get_collection_name()

    if not index_exists():
        raise VectorStoreError(
            f"向量索引不存在: {chroma_dir}。请先调用 rag.vectorstore.build_index()。"
        )

    Chroma = _chroma_cls()

    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(chroma_dir),
        )
    except Exception as exc:
        raise VectorStoreError(
            f"加载向量索引失败: {chroma_dir}, 原因: {exc}"
        ) from exc


_store_cache: Any | None = None
_store_lock = threading.Lock()


def get_vectorstore() -> Any:
    """进程内单例的 Chroma 实例，避免每次检索都重连磁盘索引。"""
    global _store_cache
    if _store_cache is None:
        with _store_lock:
            if _store_cache is None:
                _store_cache = load_vectorstore()
    return _store_cache


def reset_vectorstore_cache() -> None:
    """重建索引后调用，强制下次检索重新加载。"""
    global _store_cache
    with _store_lock:
        _store_cache = None