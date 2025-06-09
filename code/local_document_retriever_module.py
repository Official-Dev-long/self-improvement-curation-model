from keyword_extractor import KeywordExtractor 
from config import Config

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # Dummy base class

import faiss
from openai import OpenAI

import fitz  # PyMuPDF
import glob
import numpy as np

import os
os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["OPENAI_BASE_URL"] = "https://svip.xty.app/v1"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DummyEmbedding(Embeddings):
    def embed_documents(self, texts):
        return []
    def embed_query(self, text):
        return []

class LocalDocumentRetrieverModule:
    
    def __init__(self):
        self.vectorstore = self._initialize_vector_db()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': Config.LOCAL_DOCS_SEARCH_MAX_RESULTS})
        self.keyword_extractor = KeywordExtractor()

    def _get_embedding(self, text: str) -> list:
        client = OpenAI(
            api_key = "sk-xxx",
            base_url = "https://svip.xty.app/v1"
        )

        res = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )

        return res.data[0].embedding

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

        texts = [doc.page_content for doc in splits]
        metadatas = [doc.metadata for doc in splits]

        # Embed all texts
        embeddings = [self._get_embedding(text) for text in texts]

        # Use DummyEmbedding as required by FAISS even if not used
        dummy_embedding = DummyEmbedding()

        text_embeddings = list(zip(texts, embeddings))

        # Build FAISS index
        index = FAISS.from_embeddings(text_embeddings=text_embeddings, 
                                      embedding=dummy_embedding,
                                      metadatas=metadatas
        )

        return index
    
    def retrieve_docs(self, query: str) -> list:
        # keywords = self.keyword_extractor.extract_keywords(query)
        # print(f"Searching local documents using keywords: {', '.join(keywords)}")

        # retrieve_query = " ".join(keywords)

        query_embedding = self._get_embedding(query)

        return self.vectorstore.similarity_search_by_vector(query_embedding, k=Config.LOCAL_DOCS_SEARCH_MAX_RESULTS)
