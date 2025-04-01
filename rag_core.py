import os
from typing import List, Dict, Any, Optional
import torch
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    BSHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGSystem:
    def __init__(self, 
                 docs_dir: str = "data", 
                 model_path: str = "models/qwen2.5-7b-instruct-q4_k_m.gguf",
                 embeddings_model: str = "moka-ai/m3e-base",
                 vector_store_type: str = "chroma",
                 use_gpu: bool = True,
                 gpu_layers: int = 0):
        """
        初始化RAG系统
        
        参数:
            docs_dir: 文档目录路径
            model_path: LLM模型路径
            embeddings_model: 嵌入模型名称
            vector_store_type: 向量存储类型 (目前只支持 "chroma")
            use_gpu: 是否使用GPU
            gpu_layers: 使用GPU的层数 (0为自动决定)
        """
        self.docs_dir = docs_dir
        self.model_path = model_path
        self.embeddings_model = embeddings_model
        self.vector_store_type = vector_store_type.lower()
        self.use_gpu = use_gpu
        self.gpu_layers = gpu_layers
        self.vector_store = None
        self.qa_chain = None
        
        # 检查是否有可用GPU
        if self.use_gpu:
            if torch.cuda.is_available():
                self.n_gpu_layers = self.gpu_layers if self.gpu_layers > 0 else -1  # -1 means all layers
                print(f"GPU可用: {torch.cuda.get_device_name(0)}")
            else:
                self.use_gpu = False
                self.n_gpu_layers = 0
                print("GPU不可用，将使用CPU模式")
        else:
            self.n_gpu_layers = 0
    
    def load_documents(self) -> List:
        """加载多种格式文档并分割为块"""
        # 配置不同文件类型的加载器
        loaders = {
            "**/*.txt": TextLoader,
            "**/*.pdf": PyPDFLoader,
            "**/*.docx": Docx2txtLoader,
            "**/*.html": BSHTMLLoader,
            "**/*.htm": BSHTMLLoader,
        }
        
        all_documents = []
        for glob_pattern, loader_cls in loaders.items():
            try:
                print(f"尝试加载 {glob_pattern} 类型的文件...")
                loader = DirectoryLoader(
                    self.docs_dir, 
                    glob=glob_pattern, 
                    loader_cls=loader_cls,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                documents = loader.load()
                all_documents.extend(documents)
                print(f"从 {glob_pattern} 加载了 {len(documents)} 个文档")
            except Exception as e:
                print(f"加载 {glob_pattern} 出错: {str(e)}")
                print(f"错误详情: {type(e).__name__}")
        
        if not all_documents:
            print(f"警告: 在 {self.docs_dir} 目录中没有找到支持的文档")
            print(f"当前目录内容: {os.listdir(self.docs_dir)}")
            return []
            
        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(all_documents)
        
        print(f"总共加载了 {len(all_documents)} 个文档，分割为 {len(chunks)} 个块")
        return chunks
    
    def create_vector_store(self, chunks):
        """创建向量存储"""
        # 初始化嵌入模型
        device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={
                'device': device
            }
        )
        
        # 创建Chroma向量存储
        persist_directory = "chroma_db"
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="rag_collection"
        )
        # 新版本的Chroma不再需要显式调用persist()
        print(f"创建并保存Chroma向量索引到 {persist_directory}")
            
        self.vector_store = vector_store
        return vector_store
    
    def load_vector_store(self):
        """从本地加载向量存储"""
        device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={
                'device': device
            }
        )
        
        persist_directory = "chroma_db"
        if os.path.exists(persist_directory):
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name="rag_collection"
            )
            print(f"加载Chroma向量索引从 {persist_directory}")
            return self.vector_store
        else:
            raise FileNotFoundError("Chroma向量存储不存在，请先创建向量存储")
    
    def setup_llm(self):
        """设置语言模型"""
        # 配置LLM
        llm_kwargs = {
            "model_path": self.model_path,
            "temperature": 0.1,
            "max_tokens": 2000,
            "top_p": 0.95,
            "n_ctx": 4096,
            "verbose": False,
            "stop": ["<|im_end|>"],  # Qwen特有的停止词
            "repeat_penalty": 1.1    # 减少重复
        }
        
        # 添加GPU配置
        if self.use_gpu and torch.cuda.is_available():
            llm_kwargs["n_gpu_layers"] = self.n_gpu_layers  # 使用self.n_gpu_layers而不是固定值
            llm_kwargs["n_batch"] = 512
            print(f"使用GPU处理 {self.n_gpu_layers} 层")
        
        # 初始化LLM
        llm = LlamaCpp(**llm_kwargs)
        return llm
    
    def setup_qa_chain(self):
        """设置问答链"""
        if not self.vector_store:
            self.load_vector_store()
            
        llm = self.setup_llm()
        
        # 创建检索器
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 创建提示模板
        template = """<|im_start|>system
你是通义千问，由阿里云开发。你需要基于以下上下文信息回答问题。如果上下文中没有相关信息，请回答不知道，不要编造答案。

上下文信息:
{context}
<|im_end|>

<|im_start|>user
{question}
<|im_end|>

<|im_start|>assistant
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 创建问答链（使用新的LangChain API）
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        self.qa_chain = rag_chain
        
        return self.qa_chain
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """处理用户查询"""
        if not self.qa_chain:
            self.setup_qa_chain()
            
        # 使用新的调用方式
        result = self.qa_chain.invoke(query)
        return {"result": result}

    def initialize(self, rebuild_vector_store: bool = False):
        """初始化RAG系统"""
        vector_index_exists = os.path.exists("chroma_db")
            
        if rebuild_vector_store or not vector_index_exists:
            chunks = self.load_documents()
            if chunks:
                self.create_vector_store(chunks)
            else:
                print("没有文档可索引，请确保数据目录中有支持的文档文件")
                return False
        else:
            self.load_vector_store()
        
        self.setup_qa_chain()
        print(f"RAG系统初始化完成 (向量存储: {self.vector_store_type}, 使用GPU: {self.use_gpu})")
        return True
        
    def refresh_vector_store(self) -> bool:
        """刷新向量数据库，重新加载并索引所有文档"""
        try:
            print("开始刷新向量数据库...")
            chunks = self.load_documents()
            if not chunks:
                print("没有文档可索引，请确保数据目录中有支持的文档文件")
                return False
                
            # 删除旧的向量存储
            if self.vector_store:
                try:
                    self.vector_store._collection.delete(filter={})
                    print("已清空现有向量数据库集合")
                except Exception as e:
                    print(f"清空向量数据库时出错: {str(e)}")
                    
            # 创建新的向量存储
            self.create_vector_store(chunks)
            
            # 重新设置问答链
            self.setup_qa_chain()
            
            print("向量数据库刷新完成")
            return True
        except Exception as e:
            print(f"刷新向量数据库失败: {str(e)}")
            return False
            
    def add_documents(self, file_paths: List[str]) -> bool:
        """添加新文档到向量数据库
        
        参数:
            file_paths: 要添加的文档文件路径列表（相对于工作目录的路径）
        
        返回:
            bool: 是否成功添加文档
        """
        try:
            if not self.vector_store:
                self.load_vector_store()
                
            all_documents = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue
                    
                # 根据文件扩展名选择适当的加载器
                file_ext = os.path.splitext(file_path)[1].lower()
                loader = None
                
                if file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_ext == '.docx':
                    loader = Docx2txtLoader(file_path)
                elif file_ext == '.html' or file_ext == '.htm':
                    loader = BSHTMLLoader(file_path)
                else:
                    print(f"不支持的文件类型: {file_ext}")
                    continue
                
                # 加载文档
                try:
                    documents = loader.load()
                    all_documents.extend(documents)
                    print(f"从 {file_path} 加载了 {len(documents)} 个文档")
                except Exception as e:
                    print(f"加载 {file_path} 出错: {str(e)}")
            
            if not all_documents:
                print("没有成功加载任何文档")
                return False
                
            # 文档分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(all_documents)
            
            # 获取嵌入模型
            device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={
                    'device': device
                }
            )
            
            # 添加到现有向量存储
            self.vector_store.add_documents(documents=chunks)
            print(f"已成功添加 {len(chunks)} 个文档块到向量数据库")
            
            return True
        except Exception as e:
            print(f"添加文档到向量数据库失败: {str(e)}")
            return False
