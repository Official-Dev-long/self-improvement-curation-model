from keyword_extractor import KeywordExtractor 
from config import Config

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # Dummy base class

import faiss
from openai import OpenAI

import re
import unicodedata

import fitz  # PyMuPDF
import glob
import numpy as np

import os
os.environ["OPENAI_API_KEY"] = "sk-OBVaImxdTQNZdZbZsiAhlMwmvkvoWSO082HzOYuixVHRCKsE"
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
            api_key = "sk-OBVaImxdTQNZdZbZsiAhlMwmvkvoWSO082HzOYuixVHRCKsE",
            base_url = "https://svip.xty.app/v1"
        )

        res = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )

        return res.data[0].embedding

    def _normalize_characters(self, text: str) -> str:
        """Convert full-width characters to half-width and clean special symbols"""
        # Convert full-width English letters/digits to standard half-width
        text = unicodedata.normalize('NFKC', text)
        
        # Remove special control characters and symbols
        text = re.sub(r'[\x00-\x1f]', '', text)  # Control chars
        text = re.sub(r'[\u200b\u200e\u2028\u2029]', '', text)  # Special whitespace
        text = re.sub(r'[\ue000-\uf8ff]', ' ', text)  # Private use area
        
        # Clean remaining artifacts
        text = re.sub(r'[•●▪]', ' ', text)  # Various bullet characters
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        
        return text

    def _load_pdfs_from_directory(self):
        """Custom PDF loader using PyMuPDF with error handling"""
        #documents = []
        parsed_documents = []
        pdf_files = glob.glob(os.path.join(Config.DOCS_DIR, "**/*.pdf"), recursive=True)
        
        for pdf_path in pdf_files:
            try:
                with fitz.open(pdf_path) as doc:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text("text")

                        # Apply comprehensive character normalization
                        text = self._normalize_characters(text)
                        
                        # Paragraph reconstruction logic
                        lines = text.split('\n')
                        cleaned_lines = []
                        current_paragraph = []
                        
                        for line in lines:
                            stripped = line.strip()
                            if stripped:
                                # Chinese sentence terminators
                                if re.search(r'[。！？…；》\.)\]]$', stripped):
                                    current_paragraph.append(stripped)
                                    cleaned_lines.append(" ".join(current_paragraph))
                                    current_paragraph = []
                                else:
                                    current_paragraph.append(stripped)
                            else:
                                if current_paragraph:
                                    cleaned_lines.append(" ".join(current_paragraph))
                                    current_paragraph = []
                                cleaned_lines.append("")  # Preserve paragraph break
                        
                        if current_paragraph:
                            cleaned_lines.append(" ".join(current_paragraph))
                        
                        text = "\n".join(cleaned_lines)
                        
                        # Final cleanup
                        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive newlines

                        # Save cleaned page text to file
                        with open("parsed_text.txt", "a", encoding="utf-8") as parsed_file:
                            parsed_file.write(f"File: {pdf_path}\n")
                            parsed_file.write(f"--- PAGE {page_num+1} ---\n")
                            parsed_file.write(text)
                            parsed_file.write("\n\n")
                        
                        if text.strip():
                            parsed_documents.append(Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_path,
                                    "page": page_num + 1
                                }
                            ))
                print(f"Successfully loaded: {pdf_path}")
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {str(e)}")
        
        # return documents
        return parsed_documents

    def _initialize_vector_db(self):

        # Check if index exists
        if os.path.exists(Config.FAISS_INDEX_PATH):
            print("Loading existing FAISS index...")
            return FAISS.load_local(
                folder_path=Config.FAISS_INDEX_PATH,
                embeddings=DummyEmbedding(),
                allow_dangerous_deserialization=True  # Necessary for dummy embedding
            )

        # documents = self._load_pdfs_from_directory()
        parsed_documents = self._load_pdfs_from_directory()

        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", "…"]  # Chinese-aware
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
        """

        texts = [doc.page_content for doc in parsed_documents]
        metadatas = [doc.metadata for doc in parsed_documents]

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

        index.save_local(Config.FAISS_INDEX_PATH)
        print(f"Saved new FAISS index to {Config.FAISS_INDEX_PATH}")

        return index
    
    def retrieve_docs(self, query: str) -> list:
        # keywords = self.keyword_extractor.extract_keywords(query)
        # print(f"Searching local documents using keywords: {', '.join(keywords)}")

        # retrieve_query = " ".join(keywords)

        query_embedding = self._get_embedding(query)

        return self.vectorstore.similarity_search_by_vector(query_embedding, k=Config.LOCAL_DOCS_SEARCH_MAX_RESULTS)
