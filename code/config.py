import os

class Config:
    DOCS_DIR = "./docs/"
    VECTOR_DB_PATH = "./vector_db/"
    FAISS_INDEX_PATH = "./saved_FAISS_index" 
    LOCAL_DOCS_SEARCH_MAX_RESULTS = 3

    GOOGLE_SEARCH_API_KEY = "AIzaSyAtepAoejAHntkUqFPsQFpYgoth8vZyf4Y"
    GOOGLE_SEARCH_ENGINE_ID = "50424e5a5536e4b6d"
    WEB_SEARCH_MAX_RESULTS = 10

    LLM_MODEL = "deepseek-r1"

    BOCHA_API_KEY = "sk-xxx"