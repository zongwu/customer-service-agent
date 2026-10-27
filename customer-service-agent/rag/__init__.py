"""客服 Agent 的 RAG 检索层。

职责边界：
- loader:切分知识库文本
- embeddings / vectorstore:稠密检索
- preprocessor / bm25_retriever:稀疏检索
- fusion:RRF 融合
- reranker:Cross-Encoder 重排
- retriever:统一检索入口

本包内的所有外部依赖（模型下载、向量库读写、推理）都必须带超时与降级，
统一的重试工具放在这里，避免各模块各写一套。
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from functools import wraps
from typing import Any, Callable, Iterable, TypeVar

logger = logging.getLogger("rag")

# ── 环境变量引导（必须早于任何 huggingface_hub 的 import）────────────────
# huggingface_hub 在 **被 import 的那一刻** 就把 HF_ENDPOINT 读成模块常量，
# 之后再往 os.environ 里写值完全无效。放在各子模块里做 load_dotenv 太晚：
# 等 rag.embeddings 被导入时，hub 往往已经随 chromadb/sentence-transformers
# 一起进来了。本包是项目入口，import rag 时它一定还没加载，所以放这里。
from dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

T = TypeVar("T")

# 单次外部调用的墙钟上限（秒）。超过即视为失败并降级，避免链路卡死。
DEFAULT_WALL_CLOCK_LIMIT = 30.0


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """指数退避 + 抖动的重试装饰器（不引入额外依赖）。

    为什么用抖动：多个请求同时失败时，无抖动的固定退避会让重试时刻对齐，
    形成同步风暴，把刚恢复的下游再打挂一次。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - 重试语义需要逐次捕获
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                    delay += random.uniform(0, delay * 0.5)  # 抖动
                    logger.warning(
                        "%s 第 %d/%d 次失败(%s)，%.2fs 后重试",
                        func.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator


class Timer:
    """墙钟计时器。用于给本地模型推理设上限，超时即走降级。"""

    def __init__(self, limit: float = DEFAULT_WALL_CLOCK_LIMIT) -> None:
        self.limit = limit
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def expired(self) -> bool:
        return self.elapsed() > self.limit


def batched(items: Iterable[T], size: int) -> Iterable[list[T]]:
    """把可迭代对象切成固定大小的批，避免一次性把全部输入塞进模型。"""
    if size <= 0:
        raise ValueError("batch size 必须大于 0")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# torch / DLL 加载失败的诊断
# ---------------------------------------------------------------------------

# torch 在 Windows 上的主 DLL 体积参考值（MB），用于判断是否为大文件映射失败
_LARGE_DLL_MB = 200


def _windows_commit_pressure() -> str | None:
    """检测 Windows 提交内存压力。非Windows 或读不到则返回 None。

    为什么重要：WinError126 在这里极易被误读成「文件缺失」，
    实际最常见的原因是提交内存耗尽——大DLL（torch_cpu.dll 有 300MB+）
    映射进地址空间时拿不到提交额度，Windows 报「找不到指定的模块」，
    但文件其实好好地躺在磁盘上。排查方向完全相反，必须在报错里点明。
    """
    if sys.platform != "win32":
        return None

    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None

        total_commit = status.ullTotalPageFile
        avail_commit = status.ullAvailPageFile
        if total_commit == 0:
            return None

        used_ratio = (total_commit - avail_commit) / total_commit
        if used_ratio < 0.85:
            return None

        return (
            f"系统提交内存已用 {used_ratio:.1%}"
            f"（{avail_commit / 1024 ** 3:.2f}GB 可用 / {total_commit / 1024 ** 3:.2f}GB 上限）"
        )
    except Exception:  # noqa: BLE001 - 诊断代码不允许影响主流程
        return None


def _detect_dll_hooks() -> list[str]:
    """检测可能挂钩 DLL 加载的安全软件进程。

    为什么重要：这类 HIPS/应用防护产品会挂钩 LoadLibrary，
    对未签名的大体积新文件直接拒绝，导致 WinError 126。
    此时 GetLastError 往往返回 0（不是真正的"模块缺失"），
    错误信息具有极强的误导性——用户会以为文件损坏去重装，
    而重装多少次都会被同样拦截。
    """
    hooks = []
    # 常见会挂钩 DLL 加载的进程名（小写匹配）
    suspects = {
        "hipsdaemon": "火绒 HIPS 应用行为防护",
        "huorong": "火绒安全",
        "aliprotect": "阿里云安全组件",
        "qqpcmgr": "腾讯电脑管家",
        "qqpcguard": "腾讯电脑管家",
        "360tray": "360 安全卫士",
        "360safemon": "360 安全卫士",
        "savservice": "瑞星安全",
        "avp": "卡巴斯基",
        "mbamservice": "Malwarebytes",
        "crowdstrike": "CrowdStrike",
        "symantec": "Symantec",
    }
    try:
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            for key, label in suspects.items():
                if f'"{key}' in line.lower():
                    hooks.append(label)
                    break
    except Exception:  # noqa: BLE001 - 诊断代码不允许影响主流程
        pass
    return sorted(set(hooks))


def diagnose_torch_import_error(exc: BaseException) -> str:
    """把torch 导入异常翻译成「可执行的下一步」，而不是把原始堆栈甩给用户。

    覆盖四类常见根因：
    1. 提交内存耗尽（WinError 126 + 大 DLL）——最容易被误诊为文件缺失
    2. 安全软件 HIPS 挂钩拦截（GetLastError=0 时高度疑似）
    3. torch 未安装
    4. VC++ 运行库等依赖缺失
    """
    error_text = str(exc)
    hints: list[str] = []

    # WinError 126 的判定不能只看 exc.winerror：手工构造的 OSError、
    # 以及部分包装过的异常都不带该属性。所以异常属性与错误文本双路判定。
    is_dll_not_found = (
        isinstance(exc, OSError)
        and (getattr(exc, "winerror", None) == 126 or "126" in error_text)
    )

    if isinstance(exc, ImportError) and exc.name == "torch":
        hints.append("torch 未安装。请执行: pip install torch")
    elif is_dll_not_found:
        commit_note = _windows_commit_pressure()
        hooks = _detect_dll_hooks()

        if hooks:
            hints.append(
                "检测到安全软件进程（" + "、".join(hooks) + "），"
                "它很可能挂钩了 DLL 加载。这类产品的应用行为防护会拦截"
                "未签名的大体积新文件，并报 WinError 126 —— "
                "但文件其实完好，GetLastError 甚至可能返回 0。"
                "处理：临时退出该安全软件后重试；"
                "或把 Python 解释器目录加入其信任白名单。"
            )
        if commit_note:
            hints.append(
                f"另一可能原因是提交内存不足。{commit_note}。"
                f"torch 需加载 300MB+ 的 DLL，提交率过高时会加载失败。"
                f"处理：关闭占内存的程序，或将页面文件设为系统托管后重启。"
            )
        elif not hooks:
            hints.append(
                "此错误高度疑似内存/提交额度不足，而非文件缺失"
                f"（错误码 126 = 依赖无法加载，文件本身通常完好）。"
                f"请先关闭占内存的程序（浏览器、多个 IDE 实例）后重试。"
            )

        hints.append(
            "排查命令："
            "Get-Process | Sort-Object -Descending -Property WorkingSet64 "
            "| Select-Object -First 10 Name, Id, "
            "@{N='MB';E={[math]::Round($_.WorkingSet64/1MB)}}"
        )
        hints.append(
            "若确认内存充足且无安全软件拦截，检查 VC++ 运行库："
            "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )
    elif isinstance(exc, OSError):
        hints.append(f"系统加载 DLL 失败，可能是依赖缺失。{error_text}")

    # 末尾附上原始错误，除非上面已经引用过，避免同一句话重复两遍
    if hints and error_text not in hints[-1]:
        hints.append(f"原始错误: {error_text}")
    return "\n  - ".join(hints)