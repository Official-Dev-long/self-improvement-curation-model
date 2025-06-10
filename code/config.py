import os

class Config:
    DOCS_DIR = "./docs/"
    VECTOR_DB_PATH = "./vector_db/"
    FAISS_INDEX_PATH = "./saved_FAISS_index" 
    LOCAL_DOCS_SEARCH_MAX_RESULTS = 3
    WEB_SEARCH_MAX_RESULTS = 4
    LLM_MODEL = "deepseek-r1"

    #os.environ["OPENAI_API_KEY"] = "sk-xxx"
    #os.environ["OPENAI_BASE_URL"] = "https://svip.xty.app"

    BOCHA_API_KEY = "sk-xxx"