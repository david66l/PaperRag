# OPTIMIZED_BY_CODEX_STEP_1
ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${PYTHON_BASE_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update \
    && apt-get install -y --fix-missing --no-install-recommends build-essential poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip wheel --wheel-dir /wheels -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# NETWORK_MIRROR_AND_PRE_DOWNLOAD_OPTIMIZED_STEP_2
RUN pip install --no-cache-dir --no-index --find-links=/wheels sentence-transformers \
    && HF_ENDPOINT=https://hf-mirror.com python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${PYTHON_BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update \
    && apt-get install -y --fix-missing --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

COPY . .

EXPOSE 8000 8501
CMD ["python", "scripts/run_api.py"]
# STEP_1_SUMMARY: Multi-stage Docker image with poppler-utils is ready for API and Streamlit runtime.
