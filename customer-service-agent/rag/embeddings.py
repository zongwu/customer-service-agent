"""中文 Embedding 模型加载。

默认模型 BAAI/bge-large-zh-v1.5，输出 1024 维向量，检索场景下效果与速度
平衡较好。首次运行会从 HuggingFace 下载约 1.3GB 权重，请提前预留磁盘与时间
[待核实: 实际文件大小以首次运行时 huggingface 缓存目录为准]。

向量归一化（normalize_embeddings=True）是硬要求：
- 余弦相似度检索下，归一化后点积即余弦，Chroma 的默认距离度量才等价；
- RRF 融合只依赖排名而非分数，归一化不影响融合，但影响后续加阈值判断。
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

from dotenv import load_dotenv

logger = logging.getLogger("rag.embeddings")

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# 模型加载很慢（首次可能几分钟），单例 + 锁保证进程内只加载一次，
# 避免 API 多请求时并发重复下载/占用显存。
_load_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """Embedding 模型加载或推理失败。属于终止类错误：换模型前重试无意义。"""


def get_embedding_model_name() -> str:
    """读取 EMBEDDING_MODEL 环境变量，未配置时回退到默认值。"""
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _device() -> str:
    """优先用 GPU，无 GPU 时回退 CPU。torch 不可用时返回 cpu。

    注意：这里捕获的是 Exception 而非 ImportError。torch 加载失败在 Windows 上
    常常抛 OSError(WinError 126)，若只捕 ImportError 会让异常穿透到调用方，
    用户看到的是一整页torch 内部堆栈，而不是"内存不足"这种可执行结论。
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch 是硬依赖，正常不会走到
        logger.warning("未安装 torch，Embedding 回退到 CPU")
        return "cpu"
    except Exception as exc:  # noqa: BLE001 - torch 的 DLL 加载失败类型不可穷举
        from . import diagnose_torch_import_error

        logger.error(
            "torch 导入失败，Embedding 回退到 CPU。\n  - %s",
            diagnose_torch_import_error(exc),
        )
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_embeddings():
    """真正构造 HuggingFaceEmbeddings。只应在锁内调用。"""
    model_name = get_embedding_model_name()

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # pragma: no cover - 取决于安装的 langchain 版本
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise EmbeddingError(
                "找不到 HuggingFaceEmbeddings。请安装 langchain-huggingface "
                "或使用 langchain-community<0.3 版本。"
            ) from exc

    logger.info("加载 Embedding 模型: %s (device=%s)", model_name, _device())

    device = _device()

    # 注意：show_progress_bar **不能**写进 encode_kwargs。
    # langchain_huggingface 的 _embed() 内部调用的是
    #   client.encode(texts, show_progress_bar=self.show_progress, **encode_kwargs)
    # 它自己已经传了一次，我们再传就会
    #   TypeError: got multiple values for keyword argument 'show_progress_bar'
    # 关进度条要走 HuggingFaceEmbeddings 自己的 show_progress 字段。
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            show_progress=False,
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 16,
            },
        )
    except Exception as exc:
        # 典型成因：网络不通导致模型下载失败、显存不足、模型名写错。
        # 这些都不是「再试一次就会好」的错误，所以直接抛出让上层降级。
        #
        # torch DLL 加载失败要单独说清楚：它长得像「找不到文件」，
        # 但实际多数是提交内存耗尽。若不翻译，用户会去反复重装 torch——
        # 而 torch 装得没问题，重装一万次也一样失败。
        from . import diagnose_torch_import_error

        if isinstance(exc, OSError):
            detail = diagnose_torch_import_error(exc)
        else:
            detail = str(exc)

        raise EmbeddingError(
            f"加载 Embedding 模型失败: {model_name}\n原因: {detail}"
        ) from exc


@lru_cache(maxsize=1)
def get_embeddings():
    """进程内单例的 HuggingFaceEmbeddings 实例（已开启归一化）。"""
    with _load_lock:
        return _build_embeddings()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把文本批量编码为归一化向量。"""
    if not texts:
        return []
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """把单条 query 编码为归一化向量。"""
    return get_embeddings().embed_query(query)


# 对外导出别名，语义与用户要求一致
load_embeddings = get_embeddings
get_embedding_model = get_embeddings