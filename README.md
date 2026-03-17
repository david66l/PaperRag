# PaperRAG — AI Paper Knowledge Base Q&A System

基于 arXiv 论文元数据的 RAG（检索增强生成）问答系统。

## 架构概览

```
ArXiv JSON → Loader → Cleaner → Chunker → Embedder → FAISS + BM25
                                                          ↓
User Query → Dense Recall + BM25 Recall → Fusion → Rerank → Context Build → LLM → Answer
```

### RAG 六步流水线

| 步骤 | 模块 | 说明 |
|------|------|------|
| Loading | `app/ingestion/loaders/` | 从 JSONL 加载 arXiv 元数据 |
| Slicing | `app/ingestion/chunkers/` | 按 title+abstract 语义切片 |
| Embedding | `app/embedding/` | sentence-transformers 本地嵌入 |
| Storage | `app/storage/` | FAISS 向量索引 + BM25 关键词索引 |
| Retrieval | `app/retrieval/` | 多路召回 → 融合 → 重排 → 上下文构造 |
| Generation | `app/generation/` | Prompt 构造 → LLM 生成 → 引用格式化 |

### 检索架构（三层设计）

1. **召回层**: Dense (FAISS) + BM25 + Metadata Boost 多路召回
2. **融合层**: RRF / 加权融合，分数归一化后合并
3. **精排层**: CrossEncoder 重排（可选），回退到融合分数排序

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

从 [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) 下载 `arxiv-metadata-oai-snapshot.json`，放入 `data/` 目录。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API key
```

### 4. 构建索引

```bash
# 方式一：命令行脚本（推荐开发调试）
python scripts/build_index.py --data data/arxiv-metadata-oai-snapshot.json --limit 1000 --rebuild

# 方式二：通过 API
curl -X POST http://localhost:8000/ingest/build \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/arxiv-metadata-oai-snapshot.json", "limit": 1000, "rebuild": true}'
```

### 5. 启动 API 服务

```bash
python scripts/run_api.py
# 或
uvicorn app.main:app --reload
```

### 6. 发起查询

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LoRA?", "top_k": 5, "mode": "concise"}'
```

### 7. 启动 Streamlit 前端（可选）

```bash
streamlit run scripts/streamlit_app.py
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ingest/build` | 构建/重建索引 |
| POST | `/query` | 知识库问答 |
| GET | `/health` | 健康检查 |
| GET | `/config` | 查看当前配置 |
| GET | `/papers/{doc_id}` | 查看论文详情 |

## 项目结构

```
app/
├── api/routes/          # FastAPI 路由
├── core/                # 配置、日志、Schema、异常
├── ingestion/           # 数据加载、清洗、切片
├── embedding/           # 嵌入抽象层 + 实现
├── storage/             # 存储层（FAISS、BM25、文档/chunk 仓库）
├── retrieval/           # 召回、融合、重排、上下文构造
├── generation/          # Prompt 构造、LLM 生成、引用格式化
├── services/            # 业务服务层
├── evaluation/          # 评测模块（预留）
└── main.py              # FastAPI 入口
```

## 配置说明

所有配置通过环境变量或 `.env` 文件管理，前缀为 `PAPERRAG_`。详见 `.env.example`。

核心参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `local` | `local` / `api` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 嵌入模型 |
| `FUSION_STRATEGY` | `rrf` | `rrf` / `weighted` |
| `RERANK_ENABLED` | `false` | 是否启用重排 |
| `LLM_PROVIDER` | `openai_compatible` | LLM 供应商 |
| `TOP_K_DENSE` | `30` | Dense 召回数量 |
| `TOP_K_BM25` | `30` | BM25 召回数量 |
| `TOP_N_CONTEXT` | `5` | 最终上下文 chunk 数 |

## 扩展指南

- **新增 Embedding 模型**: 继承 `BaseEmbeddingProvider`，实现 `embed_documents` / `embed_query`
- **新增 LLM**: 继承 `BaseLLMProvider`，实现 `generate`
- **新增 Reranker**: 继承 `BaseRerankerProvider`，实现 `rerank`
- **新增切片策略**: 继承 `BaseChunker`，实现 `chunk`
- **新增融合策略**: 继承 `BaseFusionStrategy`，实现 `fuse`
