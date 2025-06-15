from keyword_extractor import KeywordExtractor 
from config import Config

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # Dummy base class

import faiss
from openai import OpenAI

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

    def _load_pdfs_from_directory(self):
        """Custom PDF loader using PyMuPDF with error handling"""
        parsed_documents = []
        pdf_files = glob.glob(os.path.join(Config.DOCS_DIR, "**/*.pdf"), recursive=True)
        
        for pdf_path in pdf_files:

            file = client.files.create(
                file=open(pdf_path, "rb"),
                purpose="user_data"
            )

            # pass the pdf file to llm 
            # the llm should return texts in json format
            # write the json formatted texts into parsed_documents

            parsed_documents.append(Document(
                page_content=,
                metadata={
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
