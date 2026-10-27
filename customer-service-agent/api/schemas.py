"""API 数据契约。

用 Pydantic 定义请求/响应结构，FastAPI 据此自动生成 OpenAPI 文档与入参校验。
字段命名与前端解析逻辑一一对应（ui/chat.html 读取 reply / sources / need_human）。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TurnRequest(BaseModel):
    """单轮对话请求。"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户消息",
        examples=["怎么改密码"],
    )
    session_id: str = Field(
        default="demo-session",
        min_length=1,
        max_length=128,
        description="会话 ID。同一 ID 复用 LangGraph 的 checkpointer 记忆，实现多轮上下文延续",
        examples=["demo-session"],
    )


class TurnResponse(BaseModel):
    """单轮对话响应。"""

    reply: str = Field(..., description="客服回复正文")
    sources: list[str] = Field(default_factory=list, description="引用的知识库来源")
    need_human: bool = Field(default=False, description="是否需要转接人工客服")


class HealthResponse(BaseModel):
    """健康检查响应。云端部署时供负载均衡/容器探活使用。"""

    status: str = Field(default="ok", description="服务状态")
    timestamp: str = Field(
        ...,
        description="服务端当前时间，ISO 8601 格式",
        examples=["2026-10-23T11:46:19.123456+08:00"],
    )


def now_iso() -> str:
    """当前本地时间的 ISO 8601 字符串（带时区偏移）。

    单独成函数：/health 与异常响应都用它，避免各处时间格式不一致。
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")
