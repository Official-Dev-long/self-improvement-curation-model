from keyword_extractor import KeywordExtractor 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from openai import OpenAI
from config import Config
import fitz  # PyMuPDF
import glob

import os
os.environ["OPENAI_API_KEY"] = "sk-OBVaImxdTQNZdZbZsiAhlMwmvkvoWSO082HzOYuixVHRCKsE"
os.environ["OPENAI_BASE_URL"] = "https://svip.xty.app/v1"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# deal with ssl error when calling OpenAIEmbeddings
# 专门处理连接公司的ds，不检验ssl
import httpx #chatopenai用的是httpx
import ssl
# 自定义SSL上下文，禁用DH密钥检查
# ssl_ctx = ssl.create_default_context()
# ssl_ctx.check_hostname = False #原生request接口内容
# ssl_ctx.verify_mode = 0 #原生request接口内容
# ssl_ctx.set_ciphers('DEFAULT@SECLEVEL=1')  # 降低安全级别以允许较小的DH密钥
# 创建自定义的httpx.Client实例并传入SSL上下文
# http_client = httpx.Client(verify=ssl_ctx)

class LocalDocumentRetrieverModule:
    
    def __init__(self):
        # self._configure_ssl()
        self.vectorstore = self._initialize_vector_db()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': Config.LOCAL_DOCS_SEARCH_MAX_RESULTS})
        self.keyword_extractor = KeywordExtractor()

    """
    def _configure_ssl(self):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        ssl_ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        self.http_client = httpx.Client(verify=ssl_ctx)
    """

    def _load_pdfs_from_directory(self):
        """Custom PDF loader using PyMuPDF with error handling"""
        documents = []
        pdf_files = glob.glob(os.path.join(Config.DOCS_DIR, "**/*.pdf"), recursive=True)
        
        for pdf_path in pdf_files:
            try:
                with fitz.open(pdf_path) as doc:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text("text")
                        
                        if text.strip():
                            documents.append(Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_path,
                                    "page": page_num + 1
                                }
                            ))
                print(f"Successfully loaded: {pdf_path}")
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {str(e)}")
        
        return documents

    def _initialize_vector_db(self):
        documents = self._load_pdfs_from_directory()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        # Write splits to text file with metadata
        with open("splitted_text.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(splits):
                f.write(f"=== Chunk {i+1} ===\n")
                f.write(f"Source: {chunk.metadata.get('source', 'unknown')}\n")
                f.write(f"Page: {chunk.metadata.get('page', 'N/A')}\n")
                f.write("-" * 50 + "\n")
                f.write(chunk.page_content + "\n\n")

        embeddings = OpenAIEmbeddings()

        return FAISS.from_documents(splits, embeddings)
    
    def retrieve_docs(self, query: str) -> list:
        # keywords = self.keyword_extractor.extract_keywords(query)
        # print(f"Searching local documents using keywords: {', '.join(keywords)}")

        # retrieve_query = " ".join(keywords)

        return self.retriever.invoke(query)