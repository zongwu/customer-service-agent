"""Cross-Encoder 重排序。

Bi-Encoder（Embedding）把 query 和文档**分别**编码，文档向量无法针对当前 query
定制；Cross-Encoder 把 (query, doc) 拼在一起送进模型做交互式打分，精度更高，
代价是每对都要一次前向传播，无法预计算。所以它只能用在混合召回后的少量候选上。

模型：BAAI/bge-reranker-v2-m3，多语言、支持中英文混合，输出单个 logit，
用 sigmoid 转成 0~1 相关性分数 [待核实: 该模型的 score 语义为 sigmoid(logit)，
见 HuggingFace 模型卡 bge-reranker-v2-m3 章节]。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Sequence

from dotenv import load_dotenv

logger = logging.getLogger("rag.reranker")

load_dotenv()

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 3

# 单次打分输入的最大 token 数，超出会被截断
MAX_LENGTH = 512
# 批量打分大小。v2-m3 是 568M 参数，CPU 上建议 8~16，GPU 可放大
DEFAULT_BATCH_SIZE = 8
# 整个重排阶段的墙钟上限（秒）。超时返回空列表，由上层降级为「不重排」
DEFAULT_RERANK_TIMEOUT = 20.0

_model_cache = {}
_model_lock = threading.Lock()


class RerankerError(RuntimeError):
    """重排模型加载或打分失败。"""


def get_reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL") or DEFAULT_RERANKER_MODEL


def _device() -> str:
    """优先用 CUDA。与 embeddings._device 同理：torch 的 DLL 加载失败会抛
    OSError 而非 ImportError，必须一并捕获并翻译，否则重排链路会被
    一句「找不到指定的模块」打断，用户误以为文件损坏。"""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch 是硬依赖
        logger.warning("未安装 torch，Reranker 回退到 CPU")
        return "cpu"
    except Exception as exc:  # noqa: BLE001 - torch 的 DLL 加载失败类型不可穷举
        from . import diagnose_torch_import_error

        logger.error(
            "torch 导入失败，Reranker 回退到 CPU。\n  - %s",
            diagnose_torch_import_error(exc),
        )
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """加载 (tokenizer, model) 并缓存。首次运行会下载权重，需预留时间与磁盘。

    bge-reranker 首次加载前会联网下载，可能因网络问题失败；
    这类失败是确定性的，重试无益，直接抛出由上层降级。
    """
    model_name = get_reranker_model_name()
    if model_name in _model_cache:
        return _model_cache[model_name]

    with _model_lock:
        if model_name in _model_cache:  # 双检，避免并发重复加载
            return _model_cache[model_name]

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RerankerError(
                "找不到 transformers，请先安装 transformers>=4.38.0"
            ) from exc

        device = _device()
        logger.info("加载 Reranker 模型: %s (device=%s)", model_name, device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
        except Exception as exc:
            from . import diagnose_torch_import_error

            detail = (
                diagnose_torch_import_error(exc)
                if isinstance(exc, OSError)
                else str(exc)
            )
            raise RerankerError(
                f"加载 Reranker 模型失败: {model_name}\n原因: {detail}"
            ) from exc

        _model_cache[model_name] = (tokenizer, model, device)
        return _model_cache[model_name]


def score_pairs(
    query: str,
    documents: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    """对 (query, doc) 逐对打分，返回与 documents 同序的分数列表。

    分数经 sigmoid 归一到 0~1，便于设阈值与跨请求比较。
    截断策略取 truncation=True + max_length：宁可丢上下文也不能让输入超长
    导致推理 OOM 或结果失真。
    """
    if not documents:
        return []
    if not query or not query.strip():
        raise RerankerError("query 不能为空")

    import torch

    tokenizer, model, device = load_model()

    scores: list[float] = []
    with torch.no_grad():  # 推理不建计算图，省显存也更快
        for start in range(0, len(documents), batch_size):
            batch = list(documents[start : start + batch_size])
            encoded = tokenizer(
                batch,
                [query] * len(batch),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            logits = model(**encoded, return_dict=True).logits.view(-1).float()
            # bge-reranker 输出单 logit，sigmoid 后作为相关性分数；
            # 若未来换多分类模型，这里需按业务定义改写而非直接套用。
            probs = torch.sigmoid(logits).cpu().tolist()
            scores.extend(float(p) for p in probs)

    return scores


def rerank(
    query: str,
    documents: Sequence[str] | None,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[str]:
    """重排序并返回 Top-K 文档文本。

    Args:
        query: 查询文本。
        documents: 候选文档（来自 RRF 融合），一般 5~20 条。
        top_k: 返回条数，默认 3。
        batch_size: 打分批大小。

    Returns:
        按相关性降序的 Top-K 文档文本。候选为空时返回空列表。

    Raises:
        RerankerError: query 为空或打分失败。
    """
    if not documents:
        return []
    if top_k <= 0:
        return []

    try:
        scores = score_pairs(query, documents, batch_size=batch_size)
    except RerankerError:
        raise
    except Exception as exc:  # noqa: BLE001 - 模型推理异常类型不可穷举
        raise RerankerError(f"重排打分失败: {exc}") from exc

    if len(scores) != len(documents):  # pragma: no cover - 防御性校验
        raise RerankerError(
            f"打分数量({len(scores)})与候选数量({len(documents)})不一致"
        )

    ranked: list[tuple[float, int, str]] = [
        (score, idx, doc) for idx, (score, doc) in enumerate(zip(scores, documents))
    ]
    # 同分时保持候选原顺序，保证结果稳定可复现
    ranked.sort(key=lambda item: (-item[0], item[1]))

    logger.debug(
        "重排完成: %d 篇候选 -> top%d, 最高分 %.4f",
        len(documents), top_k, ranked[0][0] if ranked else 0.0,
    )
    return [doc for _, _, doc in ranked[:top_k]]


def rerank_with_scores(
    query: str,
    documents: Sequence[str] | None,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, float]]:
    """与 rerank 相同，但额外返回分数。评测与调试阈值时使用。"""
    if not documents:
        return []

    scores = score_pairs(query, documents)
    ranked = sorted(
        ((score, idx, doc) for idx, (score, doc) in enumerate(zip(scores, documents))),
        key=lambda item: (-item[0], item[1]),
    )
    return [(doc, score) for score, _, doc in ranked[:top_k]]


def get_reranker() -> object:
    """对外暴露加载入口，兼容不同的导入习惯。"""
    return load_model()