"""客服 Agent 工作流组装。

图结构：
    START -> classify_intent
                ├─ intent == unknown -> escalate_to_human -> END
                └─ 其他             -> retrieve_context -> generate_answer -> END

checkpointer 用 MemorySaver：**进程内存活，重启即丢**。
它只提供跨调用的会话记忆（同一 thread_id 续聊），
生产环境需要持久化时应换成 SqliteSaver / PostgresSaver。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import classify_intent, escalate_to_human, generate_answer, retrieve_context
from .state import INTENT_UNKNOWN, AgentState, initial_state

load_dotenv()

logger = logging.getLogger("agent.graph")


def _route_by_intent(state: AgentState) -> str:
    """条件分支：意图不明确时转人工，否则继续检索。

    路由函数只读 state，返回下一个节点的名字。
    用常量比较而非裸字符串，避免拼写漂移。
    """
    intent = state.get("intent", INTENT_UNKNOWN)
    if intent == INTENT_UNKNOWN:
        return "escalate_to_human"
    return "retrieve_context"


def build_graph():
    """组装并编译工作流图。

    Returns:
        编译后的 CompiledStateGraph（带 MemorySaver checkpointer）。
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("escalate_to_human", escalate_to_human)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "escalate_to_human": "escalate_to_human",
            "retrieve_context": "retrieve_context",
        },
    )
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("escalate_to_human", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# 图结构是静态的，编译一次即可反复调用。
# 模块级对象：FastAPI 进程内共享同一份编译结果，避免每请求重编译。
# build_graph() 只做图结构组装、无外部 IO，因此 import 时执行是安全的。
compiled_graph = build_graph()


def get_graph():
    """获取编译后的图（即 compiled_graph）。"""
    return compiled_graph


def invoke_agent(query: str, thread_id: str = "demo") -> AgentState:
    """便捷入口：单条问题走完整工作流。

    Args:
        query: 用户问题。
        thread_id: 会话 ID。同一 thread_id 共享 checkpointer 记忆，
                   可实现多轮对话上下文延续。

    Returns:
        执行完的最终状态。
    """
    state = initial_state(query)
    # 显式标注 RunnableConfig：裸 dict 在 pyright 下与 LangChain 的参数类型不兼容
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # LangGraph 的类型桩把返回值标成 dict[str, Any] | Any，粒度不够；
    # 运行时它就是完整 AgentState，用 cast 显式声明语义
    return cast(AgentState, get_graph().invoke(state, config=config))


def _require_api_key() -> None:
    """未配置 DeepSeek key 时友好退出。

    单独成函数：__main__ 与外部调用方都应先过这道检查，
    避免跑到一半才因为鉴权失败报一屏难懂的堆栈。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("=" * 56)
        print("未配置 DEEPSEEK_API_KEY，无法运行客服 Agent。")
        print()
        print("请按以下步骤配置后重试：")
        print("  1. 在项目根目录复制环境变量模板：")
        print("       copy .env.example .env      (Windows)")
        print("       cp .env.example .env        (macOS/Linux)")
        print("  2. 编辑 .env，填入真实的 DEEPSEEK_API_KEY")
        print("  3. 重新运行本程序")
        print("=" * 56)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    _require_api_key()

    # 按任务要求的初始状态构造；用 initial_state 保证字段齐全
    state = initial_state("怎么改密码")

    print("=" * 56)
    print("客服 Agent 工作流测试")
    print("  图结构: START -> classify_intent -> retrieve_context"
          " -> generate_answer -> END")
    print("                          \\-> escalate_to_human -> END (intent=unknown)")
    print("=" * 56)

    # total=False 的 TypedDict 不能直接下标访问（pyright 无法证明键存在）
    result = invoke_agent(state.get("query", ""), thread_id="demo-session")

    print()
    print(f"意图      : {result.get('intent')}")
    print(f"检索片段数: {len(result.get('sources', []))}")
    print(f"转人工    : {result.get('need_human')}")
    print()
    print("-" * 56)
    print("回答:")
    print(result.get("answer"))
    print("-" * 56)
    print("来源:")
    for s in result.get("sources", []):
        print(f"  - {s}")
    if not result.get("sources"):
        print("  (无)")

    # 多轮验证 checkpointer：同一 thread_id 再次提问
    print()
    print("=" * 56)
    print("多轮测试（同一 thread_id，验证会话记忆）")
    print("=" * 56)
    result2 = invoke_agent("那退款多久能到账？", thread_id="demo-session")
    print(f"意图      : {result2.get('intent')}")
    print(f"转人工    : {result2.get('need_human')}")
    print("回答:")
    print(result2.get("answer"))