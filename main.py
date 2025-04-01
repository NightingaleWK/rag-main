import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core import RAGSystem
import logging
import uvicorn
from dotenv import load_dotenv
import torch
from contextlib import asynccontextmanager
from typing import List

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义lifespan上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行的代码
    if not os.path.exists("models"):
        os.makedirs("models")
        logger.warning("创建模型目录，请确保将模型文件放入models目录")
    yield
    # 应用关闭时执行的代码

app = FastAPI(
    title="本地知识库RAG服务", 
    description="基于本地模型的知识库问答系统",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局RAG系统实例
rag_system = None

class QueryRequest(BaseModel):
    query: str

class InitRequest(BaseModel):
    rebuild_vector_store: bool = False
    model_path: str = os.getenv("MODEL_PATH", "models/chinese-llama-2-7b.Q4_K_M.gguf")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "moka-ai/m3e-base")
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "chroma")  # 更改默认值为"chroma"
    use_gpu: bool = os.getenv("USE_GPU", "True").lower() == "true"
    gpu_layers: int = int(os.getenv("GPU_LAYERS", "0").split("#")[0].strip())

class RefreshRequest(BaseModel):
    """刷新向量存储的请求模型"""
    pass

class AddDocumentsRequest(BaseModel):
    """添加文档到向量存储的请求模型"""
    file_paths: List[str]

@app.post("/initialize")
async def initialize(request: InitRequest, background_tasks: BackgroundTasks):
    """初始化或重新初始化RAG系统"""
    global rag_system
    
    logger.info(f"开始初始化RAG系统，使用模型：{request.model_path}，向量存储：{request.vector_store_type}，GPU支持：{request.use_gpu}")
    
    # 检查模型文件是否存在
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {request.model_path}")
    
    # 检查向量存储类型是否支持
    if request.vector_store_type not in ["faiss", "chroma"]:
        raise HTTPException(status_code=400, detail=f"不支持的向量存储类型: {request.vector_store_type}，支持的类型有: faiss, chroma")
    
    # 后台任务初始化系统
    def init_system():
        global rag_system
        try:
            rag_system = RAGSystem(
                model_path=request.model_path,
                embeddings_model=request.embeddings_model,
                vector_store_type=request.vector_store_type,
                use_gpu=request.use_gpu,
                gpu_layers=request.gpu_layers
            )
            success = rag_system.initialize(rebuild_vector_store=request.rebuild_vector_store)
            if success:
                logger.info("RAG系统初始化完成")
            else:
                logger.error("RAG系统初始化失败：没有找到可索引的文档")
                rag_system = None
        except Exception as e:
            logger.error(f"RAG系统初始化失败: {str(e)}")
            rag_system = None
    
    background_tasks.add_task(init_system)
    return {"status": "初始化任务已启动，请稍后查询系统状态"}

@app.get("/status")
async def status():
    """获取系统状态"""
    global rag_system
    if rag_system is None:
        return {"status": "未初始化"}
    else:
        return {
            "status": "已就绪",
            "config": {
                "model_path": rag_system.model_path,
                "embeddings_model": rag_system.embeddings_model,
                "vector_store_type": rag_system.vector_store_type,
                "use_gpu": rag_system.use_gpu,
                "gpu_available": torch.cuda.is_available() if rag_system.use_gpu else False
            }
        }

@app.post("/query")
async def query(request: QueryRequest):
    """查询接口"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="系统未初始化，请先调用/initialize接口")
    
    if not request.query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    try:
        logger.info(f"处理查询: {request.query}")
        result = rag_system.process_query(request.query)
        return {
            "query": request.query,
            "answer": result["result"]
        }
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

@app.post("/refresh")
async def refresh_vector_store(request: RefreshRequest, background_tasks: BackgroundTasks):
    """刷新向量数据库，重新加载并索引所有文档"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="系统未初始化，请先调用/initialize接口")
    
    def refresh_task():
        success = rag_system.refresh_vector_store()
        if success:
            logger.info("向量数据库刷新成功")
        else:
            logger.error("向量数据库刷新失败")
    
    background_tasks.add_task(refresh_task)
    return {"status": "刷新任务已启动，请稍后查询系统状态"}

@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest, background_tasks: BackgroundTasks):
    """将新文档添加到向量数据库"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="系统未初始化，请先调用/initialize接口")
    
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="文件路径列表不能为空")
    
    def add_documents_task():
        success = rag_system.add_documents(request.file_paths)
        if success:
            logger.info(f"成功将文档添加到向量数据库: {request.file_paths}")
        else:
            logger.error(f"添加文档到向量数据库失败: {request.file_paths}")
    
    background_tasks.add_task(add_documents_task)
    return {"status": "添加文档任务已启动，请稍后查询系统状态"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
