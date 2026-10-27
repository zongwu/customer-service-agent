# 客服 Agent Demo

> 基于 **Chroma + LangGraph + FastAPI + DeepSeek** 的中文客服 Agent Demo。
> 用混合检索（BGE 向量 + BM25 + RRF 融合 + Cross-Encoder 重排）召回知识库片段，
> 交给 DeepSeek 生成口语化回答；意图不明或知识库无解时自动转人工。
> 附带客户端聊天页与坐席工作台两个前端，支持 Docker 一键部署。

**这不是自主 Agent，是固定工作流。** 客服问答本质是"检索 + 单次生成"，
引入 planner/reflection 循环只会增加成本、延迟和不确定性，因此这里用一条
有明确边界的 LangGraph 状态图：推理次数恒定（分类 1 次 + 生成 1 次，
转人工路径仅 1 次），没有循环、没有自主张。

---

## 技术栈清单

| 层次 | 组件 | 本项目实测版本 | 版本建议 |
|---|---|---|---|
| 服务框架 | FastAPI | 0.159.1 | `>=0.110,<1.0` |
| ASGI 服务器 | Uvicorn | 0.54.0 | `>=0.27` |
| 数据校验 | Pydantic | 2.14.1 | `>=2.6,<3.0`（必须 v2） |
| 工作流编排 | LangGraph | 1.2.16 | `>=1.0`（0.x 与 1.x API 差异大） |
| 模型接入 | langchain-openai | 1.7.2 | `>=1.0` |
| 模型抽象 | langchain-core | 1.6.13 | 与 langgraph 大版本保持一致 |
| 向量库 | ChromaDB | 1.5.9 | `>=1.0` |
| 向量库接入 | langchain-chroma | 1.1.0 | 与 chromadb 大版本保持一致 |
| Embedding 接入 | langchain-huggingface | 1.2.3 | `>=1.0`（langchain-community 已 sunset，勿再依赖其 HuggingFaceEmbeddings） |
| 中文 Embedding | sentence-transformers | 6.2.0 | `>=3.0` |
| 推理运行时 | PyTorch | 2.14.1 | CPU 版即可；Docker 里已用 CPU wheel |
| Transformer 库 | transformers | 5.20.2 | 由 sentence-transformers 决定，不建议手改 |
| 稀疏检索 | rank-bm25 | 0.2.2 | 无更新，固定即可 |
| 中文分词 | jieba | 0.42.1 | `>=0.42` |
| 配置 | python-dotenv | 1.2.4 | `>=1.0` |
| 运行时 | Python | 3.13.15（本地）/ 3.11（Docker） | `>=3.11` |

> 代码里大量使用 `X | None`、`list[str]` 等 PEP 585/604 语法，**最低要求 Python 3.10**；
> 建议 3.11 及以上。Docker 镜像锁定 3.11（生态 wheel 最全）。

---

## 目录结构

```
customer-service-agent/
├── rag/                检索层：把"问题"变成"相关知识库片段"
│   ├── preprocessor.py     jieba 分词 + 去停用词，供 BM25 使用
│   ├── loader.py           知识库读取与切片（按空行切块 + 超长递归细切）
│   ├── embeddings.py       BGE 中文 Embedding 加载（含 torch 故障诊断）
│   ├── vectorstore.py      Chroma 索引构建 / 加载 / 存在性检查
│   ├── bm25_retriever.py   BM25Okapi 稀疏检索
│   ├── fusion.py           RRF 融合（把两路排名合成一路）
│   ├── reranker.py         Cross-Encoder 重排序
│   ├── retriever.py        统一入口 retrieve(query, k=3) + 各级降级
│   └── __init__.py         重试/计时/批处理工具 + torch 导入故障诊断
│
├── agent/              编排层：把"片段"变成"回答"
│   ├── state.py            AgentState 定义 + 意图常量 + 别名归一化表
│   ├── nodes.py            4 个节点：分类 / 检索 / 生成 / 转人工
│   └── graph.py            图组装、条件边、MemorySaver、compiled_graph、CLI 测试
│
├── api/                服务层：把"回答"变成"HTTP 响应"
│   ├── schemas.py          TurnRequest / TurnResponse / HealthResponse
│   └── main.py             CORS、日志脱敏、异步包装、全局异常处理、uvicorn 入口
│
├── ui/                 前端（单文件 HTML，零外部依赖）
│   ├── chat.html           客户端聊天页
│   └── workbench.html      坐席工作台（导航 / 队列 / 会话 / AI 辅助 四栏）
│
├── data/
│   └── knowledge_base.txt  知识库原文（"问：…\n答：…" 空行分隔）
│
├── tests/
│   └── test_rag.py         检索层端到端测试（含切片粒度断言）
│
├── Dockerfile              生产镜像（python:3.11-slim）
├── docker-compose.yml      一键编排（含 volume 与健康检查）
├── .dockerignore           排除 chroma_db / __pycache__ / .env
├── .env.example            环境变量模板
└── requirements.txt        依赖清单
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> 首次运行会自动从 HuggingFace 下载 `bge-large-zh-v1.5`（约 1.3GB）与
> `bge-reranker-v2-m3`（约 2.2GB）。国内网络慢可在 `.env` 里设
> `HF_ENDPOINT=https://hf-mirror.com`。

### 2. 配置环境变量

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

至少填 `DEEPSEEK_API_KEY`。完整说明见 `.env.example`。

### 3. 启动服务

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
# 或
python api/main.py
```

交互式 API 文档：<http://localhost:8000/docs>

### 4. 打开前端

```bash
start ui/chat.html                        # 直接双击打开
python -m http.server 5500 --directory ui   # 或用本地服务器，访问 :5500/chat.html
```

### 5.（可选）构建向量索引

**不构建也能跑**——会自动降级为仅 BM25 检索。要启用向量检索需先建索引：

```bash
python -c "from rag.vectorstore import build_index; build_index()"
```

### 6. 跑检索层测试

```bash
python tests/test_rag.py
```

---

## 模块详解

### RAG 模块

```
用户问题
   │
   ├─ preprocessor（jieba 分词 + 去停用词）────→ BM25 稀疏检索 k=5
   │
   └─ 原始 query ──────────────────────────→ Chroma 向量检索 k=5
                                                │
                                        fusion（RRF, k=60）→ 5 条
                                                │
                                    reranker（Cross-Encoder）top-3
                                                │
                                          最终召回片段
```

| 环节 | 实现 | 关键参数 | 为什么 |
|---|---|---|---|
| 切片 | `loader.py` | `chunk_size=300`、`overlap=30`、分隔符 `["\n\n","\n","。","？","；"]` | **先按空行切"问答块"**，只有超长块才递归细切。直接用递归切分器会把整个知识库贪婪合并成 1 块，混合检索彻底失效 |
| 向量库 | `vectorstore.py` | Chroma + 余弦距离 | 本地持久化，无需额外服务进程 |
| Embedding | `embeddings.py` | `BAAI/bge-large-zh-v1.5`，`normalize_embeddings=True` | 归一化后点积 ≡ 余弦，与 Chroma 的 cosine 空间一致 |
| 稀疏检索 | `bm25_retriever.py` | BM25Okapi，jieba 分词，k=5 | 补向量检索的短板：精确关键词、专有名词 |
| 融合 | `fusion.py` | RRF，`k=60` | 见下 |
| 重排 | `reranker.py` | `BAAI/bge-reranker-v2-m3`，top-3，`MAX_LENGTH=512` | Cross-Encoder 逐对精算相关度，把融合后的粗排再校准一次 |

**为什么用 RRF 而不是分数加权**：Chroma 的余弦相似度在 [0,1]，BM25 分数无上界，
两者不在同一量纲，硬做归一化会被离群值主导。RRF 只吃排名（`1/(k+rank)`），
天然免疫量纲差异，也不需要调权重。`k=60` 来自 Cormack 等人的原始论文经验值。

**降级矩阵**（每一段都能单独失效而不拖垮整体）：

| 故障 | 行为 |
|---|---|
| 稠密检索失败 | 退回仅 BM25（本项目在本机正处在这个状态） |
| BM25 检索失败 | 退回仅向量检索 |
| 两路都失败 | 抛 `RetrievalError` |
| RRF 无结果 | 退回向量原始结果 |
| 重排失败 | 退回融合结果 |

### Agent 模块

```
                       ┌─ intent == unknown ─→ escalate_to_human ─┐
START ─→ classify_intent ─┤                                        ├─→ END
                       └─ refund/logistics/account ─→ retrieve_context
                                                        └─→ generate_answer ─→ END
```

`AgentState`：`messages / query / intent / context / sources / answer / need_human`
（`total=False`，节点只返回自己更新的字段，由 LangGraph 合并）。

| 节点 | 职责 | 失败降级 |
|---|---|---|
| `classify_intent` | LLM 判意图，归一化到 `refund`/`logistics`/`account`/`unknown` | → `unknown` |
| `retrieve_context` | 调 `rag.retriever.retrieve(query, k=3)` | → 空上下文 |
| `generate_answer` | 带上下文生成；上下文为空时**不再调 LLM**；命中"没有找到相关信息"视为无解 | → `need_human` |
| `escalate_to_human` | 直接产出转人工话术 | — |

**四个节点都不抛异常**——异常会中断整张图，所以任何失败都降级为"转人工"，
这是安全的降级方向；反过来（分类失败却继续检索）可能答非所问。

**意图归一化是必需的**：LLM 分类输出不稳定，同一个意思可能吐出中文、英文、
带标点或带解释（"这是退款类问题"）。`state.py` 里的别名表把这些统一映射，
解析失败一律落 `unknown`，**绝不猜**。

**记忆**：`MemorySaver` checkpointer，按 `thread_id` 隔离会话。
**仅进程内存活，重启即丢**；生产需换 `SqliteSaver` / `PostgresSaver`。

### API 模块

| 端点 | 方法 | 说明 |
|---|---|---|
| `/chat` | POST | 客服问答，见下方契约 |
| `/health` | GET | 探活，返回 `{"status":"ok","timestamp":...}`，不触碰模型与检索 |
| `/` | GET | 服务自述，前端用它探测后端是否在线 |
| `/docs` | GET | OpenAPI 交互文档 |

```jsonc
// POST /chat  请求
{"message": "如何申请退款？", "session_id": "demo-session"}

// 响应
{
  "reply": "进入「我的订单」找到对应订单，点击「申请退款」……",
  "sources": ["问：如何申请退款？答：进入「我的订单」…"],
  "need_human": false
}
```

云端部署相关的设计（均可用环境变量调整）：

- **异步包装**：`graph.invoke()` 是同步阻塞调用，直接放进 async 端点会把并发打到 1。
  用 `asyncio.to_thread` 丢进线程池。实测 6 并发总耗时约 4s。
- **墙钟上限**：`asyncio.wait_for`，`AGENT_TIMEOUT_SEC`（默认 60s）。
- **并发闸门**：`asyncio.Semaphore`，`MAX_CONCURRENCY`（默认 8），防线程池打满后雪崩。
- **CORS**：`allow_origins=["*"]`（本地 `file://` 打开时 origin 为 `null`）。
  注意 `*` 与 `allow_credentials=True` 不能共存，故显式关闭 credentials。生产收敛为具体域名。
- **可观测**：每请求 `trace_id` 贯穿日志 + `X-Trace-Id` 响应头；4xx/5xx 响应体也带。
- **凭据脱敏**：日志过滤器打码 `sk-xxx` / `api_key=xxx`；用户输入只记 80 字预览。
- **全局异常处理器**：未捕获异常 → 500 + `trace_id`，traceback 只进日志，
  响应体不回传内部堆栈。

### UI 模块

两个页面都是**单文件 HTML**，CSS/JS/图标全部内联，零外部依赖，双击即可打开。

`ui/chat.html` 客户端聊天页：

- 用户消息靠右、客服靠左，圆角气泡 + 头像
- `sources` 以「📎 来源：…」形式附在气泡下方；`need_human=true` 时显示红色「已转接人工客服」
- 输入区支持 Enter 发送 / Shift+Enter 换行；**IME 输入中用 `isComposing` 屏蔽回车**，避免中文候选词被误发送
- `AbortController` 30s 超时 + `busy` 标志防重复提交
- 页面加载时 GET 探测后端，离线时给出 CORS/网络相关的明确提示
- 所有用户输入与后端文本走 `textContent` 而非 `innerHTML`，防 XSS
- 图标用内联 SVG 而非 emoji：Windows 上 📎 会退化成 `¶`，跨平台不一致

`ui/workbench.html` 坐席工作台：四栏布局（导航 / 队列 / 会话 / AI 辅助），
队列按标签过滤，AI 续写直接调用真实 `/chat` 接口。

---

## 常见问题

### 如何获取 DeepSeek API Key？

1. 打开 <https://platform.deepseek.com> 注册 / 登录
2. 进入 **API Keys** 页面，点 **创建 API Key**
3. **立即复制**——密钥只完整显示一次
4. 写入 `.env`：`DEEPSEEK_API_KEY=sk-xxxxxx`

本项目的 DeepSeek 走 **OpenAI 兼容接口**，`base_url` 为
`https://api.deepseek.com/v1`（官方也提供不带 `/v1` 的 `https://api.deepseek.com`）。

> [待核实: 以 platform.deepseek.com 的模型列表为准] 官方 API 文档当前列出的
> 模型名是 `deepseek-flash` / `deepseek-v4-pro`；旧名 `deepseek-chat` /
> `deepseek-reasoner` 据第三方信息已于 2026-07 下线。代码默认值已改为
> `deepseek-flash`，但**上线前请到平台确认当前可用模型名**，并在 `.env` 里显式指定。

**余额不足会静默降级**：返回 `402 Insufficient Balance` 时链路不崩，
而是走"转人工"分支——看起来"跑通了"，实际每次都在兜底。看日志里的 `诊断:` 字段：

```
意图分类调用失败，降级为 unknown: Error code: 402 - ... | 诊断: DeepSeek 账户余额不足（HTTP 402）
```

| 码 | 含义 | 处置 |
|---|---|---|
| 402 | 余额不足 | 去平台充值 |
| 401 | Key 无效或失效 | 重新创建 Key |
| 403 | Key 无权访问该模型 | 核对 `DEEPSEEK_MODEL` 与 Key 权限 |
| 429 | 限流 | 降低并发或稍后重试 |

### 如何增加知识库条目？

知识库是 `data/knowledge_base.txt`，格式固定——**空行分隔的「问：…\n答：…」块**：

```
问：如何申请退款？
答：进入"我的订单"找到对应订单，点击"申请退款"，填写原因并提交。审核通过后款项将在 1-3 个工作日退回原支付账户。

问：订单发货后多久能到？
答：标准快递通常 3-5 个工作日，偏远地区可能延长至 7 天。
```

切片逻辑**先按空行切块**，所以一个空行隔开的问答对 = 一个独立切片，不会串味。

**全量重建**（推荐，知识库只有几十 KB 时重建成本极低）：

```python
from rag.vectorstore import build_index, reset_vectorstore_cache
from rag.bm25_retriever import reset_bm25_cache
from rag.loader import load_documents

load_documents.cache_clear()        # 切片结果有缓存，必须清
build_index(force_rebuild=True)     # build_index 默认拒绝覆盖，需显式 force
reset_bm25_cache()
reset_vectorstore_cache()
```

```bash
# 或用命令行（等价于上面那段）
python -c "from rag.vectorstore import build_index; build_index(force_rebuild=True)"
```

**增量添加**：Chroma 支持 `collection.add()`，但 **BM25 必须全量重建**
（BM25Okapi 依赖全语料的 IDF 统计，无法原地增量）。条目少时直接全量重建更省心。

改完后建议跑 `python tests/test_rag.py`，它会断言切片数 ≥ 4 且每块主题单一。

### 如何部署到云服务器？

**第一步：Docker 启动**

```bash
docker compose up -d --build
curl http://<服务器IP>:8000/health
```

详见下一节「Docker 部署」。

**第二步：Nginx 反向代理**

把 8000 端口藏在 Nginx 后面，由 Nginx 负责 HTTPS、域名与限流：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 让 /health 走独立日志，便于探活监控不被业务日志淹没
    location /health {
        proxy_pass http://127.0.0.1:8000;
        access_log off;
    }

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # 大模型生成可能超过 30s，读超时要放宽
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # 单个 IP 限流，避免被刷爆 DeepSeek 额度
    limit_req zone=chat burst=10 nodelay;
}
```

配套的限流区写在 `http {}` 块里：`limit_req_zone $binary_remote_addr zone=chat:10m rate=5r/s;`

HTTPS 用 Certbot：

```bash
sudo certbot --nginx -d your-domain.com
```

**部署前必做三件事**：

1. **换掉 MemorySaver** → `SqliteSaver` 或 `PostgresSaver`。否则重启丢会话、
   多副本时上下文断裂。
2. **收敛 CORS**：把 `allow_origins=["*"]` 改成你的前端域名。
3. **收敛 `HOST`**：容器里 `0.0.0.0` 配合 Docker 端口映射是对的，
   但不要直接把 8000 暴露到公网——走上面的 Nginx。

### 如何更换 Embedding 模型？

改 `.env` 的 `EMBEDDING_MODEL` 即可，代码不写死模型名：

```bash
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5    # 更小更快（约 90MB），精度略降
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5    # 默认
```

然后**必须重建索引**——向量维度变了，旧索引完全失效：

```bash
rm -rf chroma_db
python -c "from rag.vectorstore import build_index; build_index()"
```

重排模型同理，改 `RERANKER_MODEL`（默认 `BAAI/bge-reranker-v2-m3`），
重排不落盘，改完重启服务即可，无需重建索引。

换模型时注意：

- 必须是**中文**模型，且输出维度与 Chroma collection 一致（换维度就要删库重建）
- 换模型后 `tests/test_rag.py` 的命中结果可能变化，那是正常的

### 服务起不来 / 检索全走降级怎么办？

看启动日志的自述块：

```
客服 Agent 服务启动
  DeepSeek API Key : 已配置 / 未配置（将全部降级为转人工）
  向量索引         : 已就绪 / 不存在（稠密检索降级为仅 BM25）
```

两条都是"降级态"时，服务仍可用，但所有回答都会是转人工——先按上面的
402/401 表排查 Key。

---

## Docker 部署

### 先读这段：构建卡在 torch 那一层的原因

`docker build` 是在**容器里**下载依赖，而容器：

- **没有宿主机的代理** —— 容器里 `127.0.0.1` 是容器自己，宿主机的 `127.0.0.1:62975` 够不着
- **没有宿主机的 `pip.ini`** —— 你配的清华镜像在容器里不生效
- torch 是唯一必须走境外源的大包，CPU wheel 也有 200MB+

于是 `RUN pip install torch --index-url https://download.pytorch.org/whl/cpu`
在国内网络下极易超时，表现为：

```
ERROR: failed to build: failed to receive status: rpc error: code = Unavailable
desc = error reading from server: EOF
```

注意这**不是 pip 报错**（没有 `Could not find a version`），也不是磁盘满、
不是守护进程挂了——是 BuildKit 的长连接被掐断。

### 但 EOF 有两种根因，先分清是哪一种

`error reading from server: EOF` 只是"连接断了"的表象。判据是**看构建日志的最后一行**：

| 最后一行长这样 | 根因 | 解法 |
|---|---|---|
| `Downloading torch-2.x...` 卡住几十秒 | 网络：容器够不到境外源 | 用下面的方式一（预下载 wheel） |
| `Installing collected packages: ... torch` | **宿主环境：WSL2 引擎崩了** | 见下节 |

第二种长这样，跟网络、跟 Dockerfile 都没关系：

```
Docker Desktop is unable to start
starting WSL engine: bootstrapping main distribution: running wslexec: ...
exit status 0xc00000fd
```

`0xc00000fd` 是 **STATUS_STACK_OVERFLOW**——Docker Desktop 的 WSL2 虚拟机崩了，
BuildKit 的连接随之断开。此时连 `docker info` 都会失败。

修复顺序（从上到下，代价递增）：

1. 完全退出 Docker Desktop → 在 PowerShell 执行 `wsl --shutdown` → 重新打开 Docker Desktop
2. **临时退出火绒安全软件**（`HipsDaemon`）——本项目已确认火绒的 HIPS 会拦截本机的
   torch DLL 加载，它对 WSL/Docker 的行为同样可能下手。把 `wsl.exe`、
   `vmmem`、`com.docker.backend.exe` 加入白名单后可长期共存
3. `wsl --update` 更新 WSL 内核后重试
4. Docker Desktop → Troubleshoot → **Reset to factory defaults**（会清掉本地镜像）

验证环境是否恢复：`docker info` 能正常返回即说明 WSL 引擎起来了。

### 构建方式一：宿主机预下载 wheel，本地装（推荐）

宿主机有代理，让宿主机去下；容器只负责装，不碰境外源：

```bash
# 宿主机执行：把 py3.11 + linux 的 CPU wheel 拉到本地 wheels/（约 196 MB）
curl -L -o "wheels/torch-2.14.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"

docker build -t customer-service-agent:0.3.0 .
```

`wheels/` 里只要有一个 `.whl`，Dockerfile 就走本地路径
（`pip install --find-links /tmp/wheels torch`），其余依赖仍走国内镜像。

> **这里用 `curl` 而不是 `pip download`，是踩过的坑**：`pip download` 会顺带解析
> torch 的依赖，而 `filelock` / `sympy` / `jinja2` 这些**不在 pytorch cpu 源上**，
> pip 会对每个依赖反复重试直到超时——实测卡住 6 分钟且一个字节都没落盘。
> 只下 torch 本体、依赖交给国内镜像，反而一步到位。

> `wheels/` 已加入 `.gitignore`，不会进版本库。

### 构建方式二：联网装（wheels/ 为空时自动走这条）

```bash
docker build -t customer-service-agent:0.3.0 .
```

容器要能访问 `download.pytorch.org`。若构建环境本身有代理，可以传进去：

```bash
docker build \
  --build-arg HTTP_PROXY=http://host.docker.internal:7890 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:7890 \
  -t customer-service-agent:0.3.0 .
```

`host.docker.internal` 是 Docker Desktop 提供的宿主机别名。
其余依赖已默认走清华镜像（`PIP_INDEX_URL`），国内可直连。

### 构建方式三：跳过 torch（镜像最小）

```bash
docker build --build-arg WITH_TORCH=0 -t customer-service-agent:slim .
```

检索会自动降级为**仅 BM25**（代码已内置该路径，服务照常可用），
代价是语义相近但用词不同的问法会漏召回。

| 构建方式 | torch 来源 | 镜像体积量级 | 失败率 |
|---|---|---|---|
| 一、预下载 wheel | 离线 | 中 | 最低 |
| 二、联网 | 境外源 | 中 | 高（国内网络） |
| 三、WITH_TORCH=0 | 不装 | 小 | 无 |

### 其他构建要点

- 基础镜像 `python:3.11-slim`
- 先复制 `requirements.txt` 单独成层——改代码不会让依赖层缓存失效（装 torch 是最慢的一步）
- `chroma_db`、`__pycache__`、`.env` 由 `.dockerignore` 排除，**不会打进镜像**
- `WITH_TORCH=0` 时会把 `torch` 从 `requirements.txt` 里剔除，
  否则它会被 `sentence-transformers` / `transformers` 的间接依赖重新拉回来（还可能是 CUDA 版）

### 运行（单容器）

```bash
docker run -d --name agent-api \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/chroma_db:/app/chroma_db" \
  customer-service-agent:0.3.0
```

### 运行（docker compose，推荐）

```bash
docker compose up -d --build      # 构建并后台启动
docker compose ps                 # 看健康状态
docker compose logs -f agent-api  # 跟日志
docker compose down               # 停止
```

`docker-compose.yml` 已包含端口映射、环境变量注入、`chroma_db` 持久化卷、
`restart: unless-stopped` 与健康检查。

### 首次构建向量索引

容器内 `chroma_db` 初始为空，需要跑一次建库（会下载 BGE 模型，首次较慢）：

```bash
docker compose run --rm agent-api \
  python -c "from rag.vectorstore import build_index; build_index()"
```

不建也能跑，只是稠密检索降级为仅 BM25。

### 验证

```bash
curl http://localhost:8000/health
# {"status":"ok","timestamp":"2026-10-23T12:00:00+08:00"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"怎么改密码","session_id":"demo-session"}'
```

---

## 参考资料

| 主题 | 链接 |
|---|---|
| LangGraph 官方文档 | <https://docs.langchain.com/langgraph> |
| LangGraph 旧版文档（含大量示例） | <https://langchain-ai.github.io/langgraph/> |
| LangChain Python 文档 | <https://docs.langchain.com/> |
| FastAPI 官方文档 | <https://fastapi.tiangolo.com/> |
| Uvicorn 文档 | <https://www.uvicorn.org/> |
| Chroma 官方文档 | <https://docs.trychroma.com/> |
| DeepSeek API 文档 | <https://api-docs.deepseek.com/> |
| DeepSeek 开放平台（创建 Key） | <https://platform.deepseek.com/> |
| BGE 中文 Embedding 模型卡 | <https://huggingface.co/BAAI/bge-large-zh-v1.5> |
| BGE 重排模型卡 | <https://huggingface.co/BAAI/bge-reranker-v2-m3> |
| RRF 原始论文（Cormack et al., CIKM 2009） | <https://dl.acm.org/doi/10.1145/1571941.1572114> |
| Pydantic v2 文档 | <https://docs.pydantic.dev/> |

---

## 贡献指南

本项目用 **WorkBuddy Vibe Coding** 迭代——用自然语言描述需求，由 AI 助手直接读改代码、
起服务验证、再把结论讲清楚。想在现有基础上继续迭代，照下面的路子走效率最高。

### 推荐的需求描述方式

「做什么 + 约束 + 验收方式」三件套，比单纯说"优化一下"有效得多：

```
在 agent/nodes.py 里给 generate_answer 加上流式输出，
要求：不改动现有 AgentState 字段，失败时仍降级为转人工，
完成后用 curl 验证 /chat 能收到 SSE。
```

### 几个已经踩过的坑（提需求时顺带说明可省一轮返工）

- **本机 torch 被火绒 HIPS 拦截**（`WinError 126`），稠密检索处于降级态。
  涉及向量检索的改动，请先确认 `chroma_db/` 是否已构建。
- **DeepSeek 账户余额为 0**，真实 LLM 调用会返回 402 并走转人工分支。
  要验证"正常回答"路径，需先充值，或用桩替换 `agent.nodes.get_llm`。
- **本机设了 HTTP_PROXY**，curl 访问 `127.0.0.1` 必须加 `--noproxy '*'`，
  否则返回 502。浏览器不受影响。
- **Windows 上 emoji 图标会变形**，前端一律用内联 SVG。

### 改动后的自检清单

```bash
python tests/test_rag.py                       # 检索层：切片粒度 + 三查询命中
python -m agent.graph                          # 工作流：需有效 DEEPSEEK_API_KEY
python -m uvicorn api.main:app --port 8000     # 服务：/health 与 /chat 各打一次
pip install pyright && pyright                 # 类型检查：当前基线 0 errors
```

> 类型检查用 `pyright --pythonpath <你的解释器路径>` 指到装了依赖的环境，
> 否则所有第三方 import 都会报 missing。

**注意**：`test_rag.py` 在向量索引构建失败时（如 torch 被安全软件拦截）
**不会中断**——分词/切片/BM25/RRF 用例照常跑，端到端用例自动切换为验证
「仅 BM25」降级路径，输出里会明确标注当前验证的是哪条路径。

改了 `data/knowledge_base.txt` 记得重建索引；改了 `requirements.txt` 记得同步
`Dockerfile` 的依赖层缓存假设。

### 代码风格约定

- 类型注解用 PEP 585/604（`list[str]`、`X | None`），不用 `typing.List` / `Optional`
- 注释解释 **why**，不解释 **what**
- 外部调用必须有显式超时 + 重试 + 降级路径
- 节点 / 端点里不抛未处理异常
- 凭据只从环境变量读，不进日志、不进代码

---

## 已知限制

- BM25 对零分文档不返回（刻意设计：零分候选会污染重排）
- 语料为静态知识库，BM25Okapi 需全量建模，动态语料重建成本随规模上升
- 知识库按空行（问答对边界）切片；单块超长时才递归细切
- 会话记忆仅存于进程内存（MemorySaver），生产需换持久化 checkpointer
- CORS 当前放通全部来源，生产需收敛
- 超时后工作线程无法被中断，只能等它自己跑完
