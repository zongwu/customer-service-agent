"""FastAPI 服务入口（云端部署版）。

对外暴露：
    POST /chat    客服单轮问答（ui/chat.html / ui/workbench.html 调用）
    GET  /health  健康检查（容器探活 / 负载均衡）
    GET  /        服务自述（前端用它探测后端是否在线）
    GET  /docs    OpenAPI 交互文档

设计要点（面向云端部署）：
- **异步包装**：LangGraph 的 `graph.invoke()` 是同步阻塞调用（内部走 httpx 同步客户端）。
  直接放在 async 端点里会占住事件循环、把并发打到 1。用 `asyncio.to_thread`
  丢进线程池，事件循环保持可响应。
- **墙钟上限**：单次请求用 `asyncio.wait_for` 兜底，避免 LLM 侧异常时请求无限挂起。
- **并发闸门**：信号量限制同时在跑的图调用数，防止线程池被打满后请求排队雪崩。
- **可观测**：每请求生成 trace_id，贯穿日志并通过 `X-Trace-Id` 响应头回传，
  便于用一次对话的 id 串起前后端与服务端日志。
- **凭据不落盘**：日志统一过脱敏过滤器，形如 sk-xxx 的字符串会被打码。

异常策略：节点内部已把可预期的失败降级为"转人工"（不抛异常）；
真正未捕获的异常由全局处理器兜住，返回 500 + trace_id，并把 traceback 记入日志。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

# 保证 `python api/main.py` 直接运行时也能导入项目根下的 agent / rag 包
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

# 环境变量必须先加载：agent.graph 在 import 时会读 .env 构造配置
load_dotenv()

from agent.graph import compiled_graph  # noqa: E402
from agent.state import AgentState  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from .schemas import HealthResponse, TurnRequest, TurnResponse, now_iso  # noqa: E402

# ------------------------------------------------------------------
# 日志：INFO 级别 + 脱敏
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

# 形如 sk-xxx 的密钥，以及 api_key=xxx / Authorization: Bearer xxx 这类片段
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(
        r"(?i)((?:api[_-]?key|authorization|bearer)\s*[:=]\s*)([^\s,;'\"]+)"
    ),
)


def _redact(text: str) -> str:
    """把日志文本里的凭据打码。凭据只从环境变量读，但异常信息里可能带出来。"""
    text = _SECRET_PATTERNS[0].sub("sk-***", text)
    return _SECRET_PATTERNS[1].sub(r"\1***", text)


class _RedactingFilter(logging.Filter):
    """日志脱敏过滤器。挂在 api 及其子 logger 上。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - 格式化失败不应影响记录
            return True
        redacted = _redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        if record.exc_text:
            record.exc_text = _redact(record.exc_text)
        return True


for _name in ("api", "agent", "rag", "uvicorn", "uvicorn.error"):
    logging.getLogger(_name).addFilter(_RedactingFilter())

# ------------------------------------------------------------------
# 运行参数（云端部署时通过环境变量覆盖）
# ------------------------------------------------------------------

# 单次请求墙钟上限（秒）。LLM 客户端自身有 timeout=30 + 重试 2 次，
# 因此最坏情况应低于 3×30；这里再兜一层，避免整条链路无限挂起。
AGENT_TIMEOUT_SEC = float(os.getenv("AGENT_TIMEOUT_SEC", "60"))

# 同时在跑的图调用上限。超过则排队——比雪崩后再超时好。
MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "8")))

# 来源标签显示上限：知识库片段很长，全量回传会把前端来源区挤成一大段灰字。
MAX_SOURCES = 2
SOURCE_LABEL_LEN = 24

_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)

# ------------------------------------------------------------------
# 应用实例
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时打印配置自述，便于部署后一眼确认关键项是否就位。"""
    _log_startup()
    yield
    logger.info("服务关闭")


def _log_startup() -> None:
    """打印启动配置。API Key 只打印"是否配置"，绝不打印值。"""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    key_ready = bool(key) and key != "your_deepseek_api_key_here"

    index_ready = False
    try:
        from rag.vectorstore import index_exists

        index_ready = index_exists()
    except Exception as exc:  # noqa: BLE001
        logger.warning("索引状态检查失败: %s", exc)

    logger.info("=" * 60)
    logger.info("客服 Agent 服务启动")
    logger.info("  DeepSeek API Key : %s", "已配置" if key_ready else "未配置（将全部降级为转人工）")
    logger.info("  向量索引         : %s", "已就绪" if index_ready else "不存在（稠密检索降级为仅 BM25）")
    logger.info("  单次超时         : %.0fs", AGENT_TIMEOUT_SEC)
    logger.info("  最大并发         : %d", MAX_CONCURRENCY)
    logger.info("  检查点           : MemorySaver（进程内存，重启即丢）")
    logger.info("=" * 60)


app = FastAPI(
    title="客服Agent Demo",
    description="基于 Chroma + LangGraph + DeepSeek 的智能客服",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS：前端可能以 file:// 打开（origin 为 null），本地联调需放开全部来源。
# 生产部署应收敛为具体域名，并注意 allow_origins=["*"] 与 allow_credentials=True
# 不可同时出现（浏览器会拒绝），因此此处显式关闭 credentials。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ------------------------------------------------------------------
# 异常兜底
# ------------------------------------------------------------------


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """HTTP 已知错误（400/404/422…）：原样返回，但补 trace_id 便于排查。"""
    trace_id = uuid.uuid4().hex[:12]
    logger.warning(
        "[%s] HTTP %d %s %s: %s",
        trace_id, exc.status_code, request.method, request.url.path, exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "trace_id": trace_id,
            "timestamp": now_iso(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """入参校验失败（422）。默认响应不带 trace_id，这里补上以便与前端报错对号入座。"""
    trace_id = uuid.uuid4().hex[:12]
    # 只记 loc/type，不记 input —— 校验错误的 input 就是用户原文，没必要入日志
    summary = [(e.get("loc"), e.get("type")) for e in exc.errors()]
    logger.warning(
        "[%s] 入参校验失败 %s %s: %s",
        trace_id, request.method, request.url.path, summary,
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "trace_id": trace_id,
            "timestamp": now_iso(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局兜底：任何未捕获异常都返回 500 + trace_id，traceback 只进日志。

    响应体不回传异常详情——内部堆栈可能含路径、依赖版本等敏感信息，
    客户端只拿得到 trace_id，凭它到服务端日志里对号入座。
    """
    trace_id = uuid.uuid4().hex[:12]
    logger.error(
        "[%s] 未处理异常 %s %s\n%s",
        trace_id,
        request.method,
        request.url.path,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"服务器内部错误，请联系管理员并提供 trace_id={trace_id}",
            "trace_id": trace_id,
            "timestamp": now_iso(),
        },
    )


# ------------------------------------------------------------------
# 图调用
# ------------------------------------------------------------------


def _invoke_graph(state: AgentState, config: RunnableConfig) -> AgentState:
    """在工作线程里同步执行图。

    单独包一层而不是直接把 lambda 丢给 to_thread：
    invoke 抛出的异常需要原样传回协程侧，包装后语义更清晰，也方便打点。
    """
    # LangGraph 的类型桩把返回值标成 dict[str, Any] | Any，粒度不够；
    # 运行时它就是完整状态，用 cast 显式声明语义
    result = cast(AgentState, compiled_graph.invoke(state, config=config))
    return result


def _source_label(chunk: str) -> str:
    """把知识库片段压成一行来源标签。

    取首行（知识库是「问：…」格式）并截断；格式若变了就退化为整段去换行，
    不能因为格式假设而抛错。
    """
    text = chunk.strip()
    first_line = text.splitlines()[0].strip() if text else ""
    label = first_line or text.replace("\n", " ")
    if len(label) > SOURCE_LABEL_LEN:
        label = label[:SOURCE_LABEL_LEN] + "…"
    return label


def _preview(text: str, limit: int = 80) -> str:
    """日志里的用户输入预览。全文入日志属于不必要的 PII 暴露。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


# ------------------------------------------------------------------
# 接口
# ------------------------------------------------------------------


@app.get("/", tags=["健康检查"])
async def root() -> dict[str, Any]:
    """服务自述。前端打开页面时用 GET 探测此处判断后端是否在线。"""
    return {
        "service": "customer-service-agent",
        "status": "ok",
        "timestamp": now_iso(),
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health() -> HealthResponse:
    """健康检查。容器探活与负载均衡只应依赖这个端点——它不触碰模型与检索。"""
    return HealthResponse(status="ok", timestamp=now_iso())


@app.post("/chat", response_model=TurnResponse, tags=["客服问答"])
async def chat(req: TurnRequest, response: Response) -> TurnResponse:
    """客服单轮问答。

    流程：构造初始状态 → 以 session_id 作 thread_id 调用 LangGraph →
    从最终状态取 answer / sources / need_human。

    Args:
        req: 含 message 与 session_id。
        response: 用于回写 X-Trace-Id 响应头。

    Returns:
        reply 为客服回复；sources 为引用的知识库来源；
        need_human 为 true 时前端应提示"已转接人工客服"。
    """
    trace_id = uuid.uuid4().hex[:12]
    response.headers["X-Trace-Id"] = trace_id
    started = time.monotonic()

    message = req.message.strip()
    session_id = req.session_id.strip() or "demo-session"
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    logger.info(
        "[%s] 收到消息 session=%s 长度=%d: %s",
        trace_id, session_id, len(message), _preview(message),
    )

    # 按 LangGraph 约定构造完整初始状态。
    # 字段写全而不是只给 query：节点里用 state.get(...) 取值，
    # 缺字段时会静默走默认值，排查起来很费劲。
    state: AgentState = {
        "messages": [{"role": "user", "content": message}],
        "query": message,
        "intent": "",
        "context": "",
        "sources": [],
        "answer": "",
        "need_human": False,
    }
    # 显式标注 RunnableConfig：裸 dict 与 LangChain 的 invoke 参数类型不兼容
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    try:
        async with _SEMAPHORE:
            final = await asyncio.wait_for(
                asyncio.to_thread(_invoke_graph, state, config),
                timeout=AGENT_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        logger.error(
            "[%s] 图调用超时（>%.0fs），session=%s",
            trace_id, AGENT_TIMEOUT_SEC, session_id,
        )
        raise

    reply = str(final.get("answer") or "").strip()
    raw_sources = final.get("sources") or []
    need_human = bool(final.get("need_human", False))

    # 节点理论上总会给出非空 answer；这里是最后一道防线，
    # 宁可转人工也不要把空串当成回答返回给前端。
    if not reply:
        logger.warning("[%s] 图未产出 answer，兜底为转人工", trace_id)
        reply = "抱歉，我暂时无法处理这个问题，已为您转接人工客服。"
        need_human = True

    sources = [_source_label(str(s)) for s in raw_sources[:MAX_SOURCES]]

    logger.info(
        "[%s] 完成 intent=%s 来源=%d need_human=%s 耗时=%.2fs",
        trace_id,
        final.get("intent") or "-",
        len(sources),
        need_human,
        time.monotonic() - started,
    )
    return TurnResponse(reply=reply, sources=sources, need_human=need_human)


# ------------------------------------------------------------------
# 本地启动入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        # 开发期想热重载：UVICORN_RELOAD=1 python api/main.py
        reload=os.getenv("UVICORN_RELOAD", "0") == "1",
    )
