"""中文文本预处理：用 jieba 分词并剔除停用词，供 BM25 稀疏检索使用。

为什么需要：BM25Okapi 是词袋模型，标点与虚词会稀释关键词权重。
「我想问下问一下退款的事情」经分词去停用词后接近「退款 事情」，
BM25 才能把正确的 FAQ 排上来。

边界说明：这里的分词结果仅供 BM25 使用。稠密检索（向量）用原始 query，
因为 embedding 模型自带的分词器在原始语义空间上效果更好。
"""

from __future__ import annotations

import re
from typing import Iterable

import jieba

# 中文停用词表：标点、语气词、代词、介词、连词等对检索无区分度的词。
STOPWORDS: set[str] = {
    # 标点与符号
    "，", "。", "、", "；", "：", "？", "！", "…", "—", "～", "·",
    "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "「", "」",
    ",", ".", ";", ":", "?", "!", "\"", "'", "(", ")", "[", "]",
    "<", ">", "/", "\\", "|", "-", "_", "=", "+", "*", "&", "^", "%", "$", "#", "@", "~", "`",
    # 疑问与语气
    "吗", "呢", "吧", "啊", "呀", "哦", "嘛", "啦", "哟", "噢", "嗯", "唉", "哈",
    "请问", "请", "麻烦", "谢谢", "感谢", "你好", "您好",
    # 代词与虚词
    "我", "我们", "你", "你们", "您", "他", "她", "它", "他们", "她们", "它们",
    "这", "那", "这个", "那个", "这些", "那些", "这样", "那样", "什么", "怎么", "怎样", "如何",
    "为什么", "为啥", "哪里", "哪儿", "谁", "多少", "几",
    "的", "了", "着", "地", "得", "之", "与", "和", "或", "及", "以及",
    "是", "在", "有", "就", "都", "也", "还", "又", "并", "而", "但", "不过",
    "因为", "所以", "如果", "虽然", "然后", "接着", "另外", "此外",
    "一个", "一下", "一些", "一点", "一直", "可以", "能", "能否", "是否",
    # 客服语料里高频但无检索价值的词
    "客服", "亲", "您好呢",
}

# 非中文/字母/数字的字符统一剥离
_CLEAN_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
_ASCII_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def _normalize(text: str) -> str:
    return _CLEAN_PATTERN.sub(" ", text or "").strip().lower()


def is_stopword(token: str) -> bool:
    """判断 token 是否应被过滤：停用词表命中、全标点、或单字符无意义组合。"""
    if not token or _ASCII_PUNCT.match(token):
        return True
    return token in STOPWORDS


def preprocess(text: str) -> str:
    """把一段中文文本处理成空格分隔的词串，供 BM25Okapi 建索引/检索使用。

    Args:
        text: 原始文本。

    Returns:
        空格分隔的词串。若清洗后为空则返回空字符串（调用方需处理冷启动）。
    """
    if not text:
        return ""

    normalized = _normalize(text)
    if not normalized:
        return ""

    tokens: list[str] = []
    for raw in jieba.lcut(normalized):
        token = raw.strip()
        if not token or is_stopword(token):
            continue
        tokens.append(token)

    return " ".join(tokens)


# 兼容别名：语义更直白的调用方可以直接用 preprocess_text
preprocess_text = preprocess


def preprocess_batch(texts: Iterable[str]) -> list[str]:
    """批量分词，与 preprocess 语义一致，用于建 BM25 索引时减少开销。"""
    return [preprocess(text) for text in texts]