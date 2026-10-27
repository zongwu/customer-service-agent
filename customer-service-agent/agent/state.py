"""客服 Agent 的状态定义。

AgentState 是 LangGraph 各节点之间传递数据的唯一载体。
每个节点只返回**自己负责更新的字段**（partial update），
LangGraph 会自动合并进全局状态——节点不需要、也不应该返回完整 state。

字段说明：
    messages:  对话历史（role/content 结构）
    query:     当前用户问题
    intent:    意图分类结果（refund / logistics / account / unknown）
    context:   RAG 检索到的上下文（拼接后的文本）
    sources:   检索来源列表
    answer:    最终回答
    need_human:是否需要转接人工
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """客服 Agent 全局状态。

    用 total=False：所有字段都是可选的。
    LangGraph 允许节点只返回部分字段，未返回的字段保持原值不变；
    total=False 让类型检查器也接受这种 partial update 写法。
    """

    # ---- 输入 ----
    messages: list[dict[str, Any]]
    query: str

    # ---- 中间产物 ----
    intent: str
    context: str
    sources: list[str]

    # ---- 输出 ----
    answer: str
    need_human: bool


# 意图枚举。放在 state 模块里，nodes 与 graph 共用同一份定义，
# 避免两边各写一套字符串常量导致拼写不一致。
INTENT_REFUND = "refund"
INTENT_LOGISTICS = "logistics"
INTENT_ACCOUNT = "account"
INTENT_UNKNOWN = "unknown"

VALID_INTENTS: frozenset[str] = frozenset({
    INTENT_REFUND, INTENT_LOGISTICS, INTENT_ACCOUNT, INTENT_UNKNOWN,
})


def initial_state(query: str, messages: list[dict[str, Any]] | None = None) -> AgentState:
    """构造初始状态，保证字段齐全且类型正确。

    Args:
        query: 用户当前问题。
        messages: 已有对话历史，默认空列表。

    Returns:
        字段完整的 AgentState。
    """
    return AgentState(
        messages=list(messages or []),
        query=query,
        intent="",
        context="",
        sources=[],
        answer="",
        need_human=False,
    )
