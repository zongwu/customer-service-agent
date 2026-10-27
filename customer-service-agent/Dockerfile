# 客服 Agent Demo —— 生产镜像
#
# ── 构建方式（按推荐顺序）───────────────────────────────────────────────
#
# 1) 本地安装 torch（推荐：最快、最稳、不依赖容器网络）
#    先在宿主机把 wheel 拉到 wheels/（宿主机有代理，容器没有）：
#      curl -L -o wheels/torch-2.14.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl \
#        "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
#      docker build -t customer-service-agent:0.3.0 .
#    wheels/ 里只要有一个 .whl，构建就走本地路径。
#
#    注：这里用 curl 而不是 `pip download`。pip 会顺带解析 torch 的依赖，
#    而 filelock / sympy / jinja2 这些不在 pytorch cpu 源上，pip 会逐个重试
#    直到超时（实测卡住 6 分钟不落盘）。只下 torch 本体、依赖交给国内镜像更省心。
#
# 2) 联网安装 torch（wheels/ 为空时自动走这条）
#      docker build -t customer-service-agent:0.3.0 .
#    容器需能访问 download.pytorch.org；国内网络大概率超时，见 README「Docker 构建」。
#
# 3) 完全跳过 torch（镜像最小，检索自动降级为仅 BM25）
#      docker build --build-arg WITH_TORCH=0 -t customer-service-agent:slim .
#
# 运行：docker run --rm -p 8000:8000 --env-file .env customer-service-agent:0.3.0

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# pip 源。容器里直连境外 PyPI 极慢且常被掐断，默认给国内镜像。
# 构建环境若走代理，可覆盖：--build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# torch 占镜像体积的绝大部分，且是唯一必须走境外源的大包——构建失败几乎都出在这一层。
# 设为 0 则完全跳过；rag 检索代码已内置降级路径，会自动退回仅 BM25。
ARG WITH_TORCH=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# 把 pip 配置写进镜像，后续所有 pip install 都继承。
# retries/timeout 是给不稳定网络留的余量：单连接被掐断时 pip 自己重试，而不是整个构建失败。
RUN printf '[global]\nindex-url = %s\ntrusted-host = %s\nretries = 10\ntimeout = 120\n' \
    "$PIP_INDEX_URL" "$PIP_TRUSTED_HOST" > /etc/pip.conf

COPY requirements.txt .

# torch 双路径：优先用预下载 wheel 装，没有才联网拉。
# 两条路径都保留，是因为这一层的失败率远高于其余所有依赖之和。
#
# 刻意不加 --no-index：wheels/ 里只放 torch 本体，
# 它的依赖（filelock / sympy / jinja2 等）不在 pytorch cpu 源上，
# 交给 PIP_INDEX_URL（国内镜像）装比一起打包更省事。
COPY wheels/ /tmp/wheels/
RUN set -eux; \
    if [ "$WITH_TORCH" = "1" ]; then \
        if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
            echo "[build] 命中预下载 wheel，本地安装 torch"; \
            pip install --find-links /tmp/wheels torch; \
        else \
            echo "[build] wheels/ 为空，联网安装 torch（需容器可访问 $TORCH_INDEX_URL）"; \
            pip install torch --index-url "$TORCH_INDEX_URL"; \
        fi; \
    else \
        echo "[build] WITH_TORCH=0，跳过 torch，检索将降级为仅 BM25"; \
    fi

# 其余依赖。WITH_TORCH=0 时把 torch 从清单里剔除，
# 否则它会被 sentence-transformers/transformers 的间接依赖重新拉回来（还可能是 CUDA 版）。
RUN set -eux; \
    if [ "$WITH_TORCH" = "0" ]; then \
        grep -v '^torch$' requirements.txt > /tmp/req.txt; \
    else \
        cp requirements.txt /tmp/req.txt; \
    fi; \
    pip install -r /tmp/req.txt; \
    rm -f /tmp/req.txt

# 装完即删，别把 200MB wheel 留在镜像层里
RUN rm -rf /tmp/wheels

# 复制项目代码。chroma_db / __pycache__ / .env 由 .dockerignore 排除
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
