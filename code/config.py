import os

class Config:
    DOCS_DIR = "./docs/"
    VECTOR_DB_PATH = "./vector_db/"
    LOCAL_DOCS_SEARCH_MAX_RESULTS = 4
    WEB_SEARCH_MAX_RESULTS = 4
    LLM_MODEL = "deepseek-r1"

    os.environ["OPENAI_API_KEY"] = "sk-xxx"
    os.environ["OPENAI_BASE_URL"] = "https://svip.xty.app"

    BOCHA_API_KEY = "sk-xxx"