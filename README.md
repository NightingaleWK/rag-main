# 本地化RAG知识库服务

这是一个基于本地模型的RAG (Retrieval Augmented Generation) 知识库服务，使用FastAPI提供API接口。

## 功能特点

- 使用本地化的大语言模型，无需依赖云服务
- 支持知识库的建立、查询和管理
- 基于向量检索的相关文档获取
- 提供简洁的RESTful API接口

## 技术栈

- FastAPI: 提供高性能API服务
- LangChain: 构建RAG检索增强生成系统
- llama-cpp-python: 本地运行LLM模型
- FAISS: 高效的向量存储和检索
- Sentence Transformers: 文本向量化

## 目录结构

```
rag-fastapi-service/
├── data/                 # 知识库文档目录
├── models/               # 模型存储目录
├── faiss_index/          # 向量索引存储目录（自动创建）
├── main.py               # FastAPI服务入口
├── rag_core.py           # RAG核心逻辑
├── .env                  # 环境变量配置
└── requirements.txt      # 项目依赖
```

## 安装

1. 克隆仓库并安装依赖：

```bash
git clone https://github.com/yourusername/rag-fastapi-service.git
cd rag-fastapi-service
pip install -r requirements.txt
```

2. 下载模型并放入models目录

您需要下载与llama.cpp兼容的GGUF格式的模型文件，例如Chinese-Llama-2等，并将其放入models目录中。

## 使用方法

1. 启动服务：

```bash
python main.py
```

2. 初始化系统（第一次使用或更新知识库时）：

```bash
curl -X POST "http://localhost:8000/initialize" -H "Content-Type: application/json" -d '{"rebuild_vector_store": true}'
```

3. 查询知识库：

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "什么是人工智能?"}'
```

## API文档

启动服务后，访问 http://localhost:8000/docs 查看完整的API文档。

主要接口：

- `POST /initialize`: 初始化或重新初始化系统
- `GET /status`: 获取系统状态
- `POST /query`: 查询知识库
- `GET /health`: 健康检查

## 配置

在`.env`文件中可配置以下参数：

- `MODEL_PATH`: 模型文件路径
- `EMBEDDINGS_MODEL`: 向量嵌入模型
- `PORT`: 服务端口
- `HOST`: 服务地址
- `DOCS_DIR`: 知识库文档目录

## 知识库管理

1. 将您的文本文档（.txt格式）放入`data`目录
2. 重新初始化系统以更新知识库

## 许可证

MIT 