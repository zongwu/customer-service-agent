"""客服 Agent 的四个节点。

节点契约（LangGraph 约定）：
- 接收完整 state，返回**只包含自己更新的字段**的 dict
- 返回值由 LangGraph 合并回全局 state
- 节点内不应抛出未处理异常——异常会中断整个图

关于 LLM 调用：
DeepSeek 通过 OpenAI 兼容接口访问，因此用 ChatOpenAI + base_url 指向
DeepSeek 端点，无需官方 SDK。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from .state import (
    INTENT_ACCOUNT,
    INTENT_LOGISTICS,
    INTENT_REFUND,
    INTENT_UNKNOWN,
    VALID_INTENTS,
    AgentState,
)

load_dotenv()

logger = logging.getLogger("agent.nodes")

# ------------------------------------------------------------------
# LLM 客户端
# ------------------------------------------------------------------

# 意图分类的容错：模型可能回 "退款" / "refund." / "退款类" 等各种变体，
# 归一化后统一映射到标准枚举。解析失败一律降级为 unknown（转人工），
# 绝不猜——猜错会让用户拿到不相关的答案。
_INTENT_ALIASES: dict[str, str] = {
    "退款": INTENT_REFUND,
    "退货": INTENT_REFUND,
    "退单": INTENT_REFUND,
    "物流": INTENT_LOGISTICS,
    "快递": INTENT_LOGISTICS,
    "配送": INTENT_LOGISTICS,
    "发货": INTENT_LOGISTICS,
    "账户": INTENT_ACCOUNT,
    "账号": INTENT_ACCOUNT,
    "登录": INTENT_ACCOUNT,
    "密码": INTENT_ACCOUNT,
    "其他": INTENT_UNKNOWN,
    "未知": INTENT_UNKNOWN,
}

# 检索无结果时给模型的明确指令语，用于判定"未找到答案"
NO_ANSWER_MARKER = "我没有找到相关信息"

# LLM 报错码 → 处置建议。命中即在日志中给出人话提示，避免把账户问题
# 误判成代码故障（402/401 是自建 demo 阶段最常见的两类）。
_LLM_ERROR_HINTS: dict[str, str] = {
    "402": "DeepSeek 账户余额不足，请到平台充值后再试",
    "401": "DEEPSEEK_API_KEY 无效或已失效，请检查 .env",
    "403": "该 API Key 无权访问此模型，请核对 DEEPSEEK_MODEL 与 key 权限",
    "429": "触发 DeepSeek 限流，请稍后重试或降低并发",
}


def _build_llm():
    """构造 ChatOpenAI 客户端。未配置 API Key 时抛出明确错误。

    延迟构造：import 本模块不应该因为缺配置就失败，
    否则 graph.py 的导入链会被拖垮。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY。请在项目根目录创建 .env 并填入真实的 key，"
            "或设置环境变量 DEEPSEEK_API_KEY。"
        )

    from langchain_openai import ChatOpenAI

    # [待核实] 官方 API 文档当前列出的是 deepseek-flash / deepseek-v4-pro；
    # 旧名 deepseek-chat / deepseek-reasoner 据第三方信息已于 2026-07 下线。
    # 上线前请以 platform.deepseek.com 的模型列表为准，并用 .env 的
    # DEEPSEEK_MODEL 覆盖，不要依赖这里的默认值。
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        # langchain-openai 的类型签名要求 SecretStr：避免 key 以明文出现在
        # repr/日志里，这正是 C6（凭据不落日志）想要的约束
        api_key=SecretStr(api_key),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0,     # 分类与客服回答都要确定性，关闭采样随机性
        timeout=30,        # 显式超时，避免请求无限挂起
        max_retries=2,     # 网络类错误重试，SDK 内建指数退避
    )


# 进程内单例。模型客户端创建开销小但没必要每次调用都新建。
_llm_singleton: Any = None


def get_llm():
    """获取 LLM 单例。首次调用时才真正初始化。"""
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = _build_llm()
    return _llm_singleton


def _diagnose_llm_error(exc: BaseException) -> str:
    """把 LLM 异常翻译成可执行的处置建议。

    区分「账户/鉴权问题」与「临时故障」很关键：前者用户自己就能修，
    后者才是代码该兜的底。混为一谈会让人误以为是链路 bug——
    实测遇到 402 余额不足时，日志只写"调用失败"，排查方向完全跑偏。
    """
    text = str(exc)
    for code, hint in _LLM_ERROR_HINTS.items():
        if code in text:
            return f"{hint}（HTTP {code}）"
    return "临时故障，已按降级路径处理"


def _invoke(messages: list[Any]) -> str:
    """调用 LLM 并返回文本内容。

    统一处理 AIMessage / str 两种返回形态——不同 langchain 版本的
    message.content 可能是 str 也可能是 list。
    """
    response = get_llm().invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # 多模态返回是 list[dict]，这里只取 text 部分
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content).strip()


# ------------------------------------------------------------------
# 节点 a: 意图分类
# ------------------------------------------------------------------

_INTENT_PROMPT = (
    "你是一个客服意图分类器。用户问题属于以下类别之一："
    "refund（退款）、logistics（物流）、account（账户）、unknown（其他）。"
    "只输出类别名称。"
)


def _normalize_intent(raw: str) -> str:
    """把模型返回的意图字符串归一化到标准枚举。

    流程：转小写 → 去掉标点空白 → 优先精确匹配 → 再做别名匹配 →
    都不中则返回 unknown。

    任何异常路径都落到 unknown：分类失败就转人工，
    这是安全的降级方向；反过来（分类失败却继续检索）可能答非所问。
    """
    if not raw:
        return INTENT_UNKNOWN

    text = raw.strip().lower()
    # 去掉包裹的引号、句号、冒号等
    text = re.sub(r"[\s\"'`:：。，,.\-—_]+", "", text)

    if not text:
        return INTENT_UNKNOWN
    if text in VALID_INTENTS:
        return text

    # 关键词包含匹配：模型可能回 "这是refund类问题" 这类冗长表述
    for intent in (INTENT_REFUND, INTENT_LOGISTICS, INTENT_ACCOUNT):
        if intent in text:
            return intent

    # 中文别名匹配
    for alias, intent in _INTENT_ALIASES.items():
        if alias in raw:
            return intent

    logger.warning("无法识别的意图返回值 %r，归类为 unknown", raw)
    return INTENT_UNKNOWN


def classify_intent(state: AgentState) -> dict[str, Any]:
    """节点 a: 判断用户意图。

    Args:
        state: 当前状态，取 query。

    Returns:
        {"intent": <标准意图枚举>}
    """
    query = state.get("query", "").strip()
    if not query:
        return {"intent": INTENT_UNKNOWN}

    try:
        raw = _invoke([
            SystemMessage(content=_INTENT_PROMPT),
            HumanMessage(content=query),
        ])
    except Exception as exc:  # noqa: BLE001 - LLM 异常类型不可穷举
        # 分类失败不能中断整条链路，退化为 unknown 走转人工分支
        logger.warning(
            "意图分类调用失败，降级为 unknown: %s | 诊断: %s",
            exc, _diagnose_llm_error(exc),
        )
        return {"intent": INTENT_UNKNOWN}

    intent = _normalize_intent(raw)
    logger.info("意图分类: %r -> %s", raw, intent)
    return {"intent": intent}


# ------------------------------------------------------------------
# 节点 b: 检索上下文
# ------------------------------------------------------------------

def retrieve_context(state: AgentState) -> dict[str, Any]:
    """节点 b: 从知识库检索相关片段。

    Args:
        state: 当前状态，取 query。

    Returns:
        {"context": 拼接后的上下文文本, "sources": 来源列表}
    """
    query = state.get("query", "").strip()
    if not query:
        return {"context": "", "sources": []}

    try:
        from rag.retriever import retrieve

        documents = retrieve(query, k=3)
    except Exception as exc:  # noqa: BLE001
        # 检索失败不抛：让生成节点看到空上下文，
        # 由模型按提示词回答"没有找到相关信息"
        logger.warning("检索失败，使用空上下文: %s", exc)
        return {"context": "", "sources": []}

    if not documents:
        logger.info("检索无结果")
        return {"context": "", "sources": []}

    # 用换行拼接，模型对分隔清晰的多段文本理解更好
    context = "\n".join(documents)
    # 来源取每段首句，便于人工核对
    sources = [
        (doc.strip().splitlines() or [""])[0][:60] for doc in documents
    ]
    logger.info("检索到 %d 个片段", len(documents))
    return {"context": context, "sources": sources}


# ------------------------------------------------------------------
# 节点 c: 生成回答
# ------------------------------------------------------------------

_ANSWER_PROMPT = (
    "你是一名客服助手，只能根据以下知识库上下文回答问题。"
    "如果上下文中没有相关信息，请说'抱歉，我没有找到相关信息'。\n"
    "上下文：{context}"
)


def generate_answer(state: AgentState) -> dict[str, Any]:
    """节点 c: 基于检索上下文生成回答。

    Args:
        state: 当前状态，取 context 与 query。

    Returns:
        {"answer": 回答文本, "need_human": 是否需转人工}
    """
    context = state.get("context", "")
    query = state.get("query", "").strip()

    if not context:
        # 检索为空时直接走转人工，不必再问一次模型——
        # 空上下文必然答不出东西，白白多一次 LLM 往返。
        logger.info("上下文为空，直接转人工")
        return {
            "answer": "很抱歉，我暂时无法处理这个问题，已为您转接人工客服。",
            "need_human": True,
        }

    prompt = _ANSWER_PROMPT.format(context=context)

    try:
        answer = _invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "生成回答失败，转人工: %s | 诊断: %s", exc, _diagnose_llm_error(exc)
        )
        return {
            "answer": "很抱歉，我暂时无法处理这个问题，已为您转接人工客服。",
            "need_human": True,
        }

    if not answer:
        return {
            "answer": "很抱歉，我暂时无法处理这个问题，已为您转接人工客服。",
            "need_human": True,
        }

    # 模型明确说找不到信息时也应转人工——用户拿不到答案，
    # 停在"我不知道"比转人工更糟。
    need_human = NO_ANSWER_MARKER in answer
    if need_human:
        logger.info("模型未找到相关信息，标记转人工")

    return {"answer": answer, "need_human": need_human}


# ------------------------------------------------------------------
# 节点 d: 转人工
# ------------------------------------------------------------------

def escalate_to_human(state: AgentState) -> dict[str, Any]:
    """节点 d: 转接人工客服。

    Args:
        state: 当前状态（本节点不读取任何字段）。

    Returns:
        {"answer": 转人工话术, "need_human": True}
    """
    logger.info("转接人工客服")
    return {
        "answer": "很抱歉，我暂时无法处理这个问题，已为您转接人工客服。",
        "need_human": True,
    }
