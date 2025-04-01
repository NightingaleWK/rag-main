# 本地知识库RAG服务

这是一个基于FastAPI的本地知识库问答系统，使用RAG（检索增强生成）技术实现。系统支持多种文档格式，使用本地LLM模型进行问答，并提供RESTful API接口。

## 功能特点

- 支持多种文档格式：PDF、TXT、DOCX、HTML等
- 使用本地LLM模型（默认使用通义千问2.5-7B）
- 支持GPU加速（可选）
- 支持Chroma和FAISS两种向量存储
- 提供RESTful API接口
- 支持动态添加和更新文档
- 支持中文问答

## 系统要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- 至少8GB内存（推荐16GB以上）
- 足够的磁盘空间用于存储向量数据库和模型文件

## 安装步骤

1. 克隆项目并进入项目目录：
```bash
git clone [项目地址]
cd rag-fastapi-service
```

2. 创建并激活虚拟环境：
```bash
python -m venv rag-venv
# Windows
.\rag-venv\Scripts\activate
# Linux/Mac
source rag-venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载并合并模型文件：

   4.1. 下载模型文件
   从 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main) 下载以下文件：
   - qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
   - qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf

   4.2. 编译安装 llama.cpp
   ```bash
   # 克隆 llama.cpp 仓库
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp

   # 编译
   # Windows (使用 Visual Studio)
   cmake -B build
   cmake --build build --config Release

   # Linux/Mac
   make
   ```

   4.3. 合并模型文件
   ```bash
   # 进入 llama.cpp 目录
   cd llama.cpp

   # 合并模型文件
   ./build/bin/Release/llama-gguf.exe -m qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -m qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf -o qwen2.5-7b-instruct-q4_k_m.gguf

   # 将合并后的模型文件移动到项目的 models 目录
   mv qwen2.5-7b-instruct-q4_k_m.gguf ../models/
   ```

5. 准备文档：
   - 将需要索引的文档放入`data`目录
   - 支持的文档格式：PDF、TXT、DOCX、HTML等

## 配置说明

在`.env`文件中配置系统参数：

```env
# 模型配置
MODEL_PATH=models/qwen2.5-7b-instruct-q4_k_m.gguf
MODEL_TYPE=llama
CONTEXT_LENGTH=4096
GPU_LAYERS=0

# 向量数据库配置
VECTOR_DB_TYPE=chroma  # 可选: faiss, chroma
VECTOR_DB_PATH=vector_store

# 文档处理配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# 向量化模型配置
EMBEDDINGS_MODEL=moka-ai/m3e-base

# GPU配置
USE_GPU=True

# 服务配置
PORT=8000
HOST=0.0.0.0

# 数据配置
DOCS_DIR=data
```

## 启动服务

```bash
python main.py
```

服务将在 http://localhost:8000 启动

## API接口

### 1. 初始化系统
```http
POST /initialize
Content-Type: application/json

{
    "rebuild_vector_store": false,
    "model_path": "models/qwen2.5-7b-instruct-q4_k_m.gguf",
    "embeddings_model": "moka-ai/m3e-base",
    "vector_store_type": "chroma",
    "use_gpu": true,
    "gpu_layers": 0
}
```

### 2. 查询接口
```http
POST /query
Content-Type: application/json

{
    "query": "你的问题"
}
```

### 3. 刷新向量存储
```http
POST /refresh
```

### 4. 添加新文档
```http
POST /add_documents
Content-Type: application/json

{
    "file_paths": ["data/new_doc.pdf"]
}
```

### 5. 系统状态查询
```http
GET /status
```

### 6. 健康检查
```http
GET /health
```

## 注意事项

1. 首次使用需要下载模型文件，请确保有足够的磁盘空间
2. 使用GPU加速需要安装CUDA和相应的PyTorch版本
3. 文档处理过程中会占用较大内存，请确保系统有足够的内存
4. 向量数据库会占用较大磁盘空间，请确保有足够的存储空间

## 常见问题

1. 如果遇到内存不足，可以调整`CHUNK_SIZE`和`CHUNK_OVERLAP`参数
2. 如果GPU内存不足，可以调整`GPU_LAYERS`参数
3. 如果向量检索效果不理想，可以调整`CHUNK_SIZE`和`CHUNK_OVERLAP`参数

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

### 许可证说明

MIT 许可证是一个宽松的软件许可证，源自麻省理工学院（Massachusetts Institute of Technology）。该许可证的主要特点是：

- 允许任何人自由使用、修改和分发软件
- 要求在所有副本中包含版权声明和许可证声明
- 不提供任何担保
- 不限制商业使用
- 不要求开源修改后的代码

使用本软件时，您需要：

1. 在您的项目中包含原始的 LICENSE 文件
2. 在您的项目中包含版权声明
3. 在您的项目中包含 MIT 许可证声明

如果您对本许可证有任何疑问，请参考 [LICENSE](LICENSE) 文件或联系项目维护者。 