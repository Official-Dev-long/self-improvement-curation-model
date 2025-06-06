import os

class Config:
    DOCS_DIR = "./docs/"
    VECTOR_DB_PATH = "./vector_db/"
    LOCAL_DOCS_SEARCH_MAX_RESULTS = 4
    WEB_SEARCH_MAX_RESULTS = 4
    LLM_MODEL = "deepseek-r1"

    os.environ["OPENAI_API_KEY"] = "sk-OBVaImxdTQNZdZbZsiAhlMwmvkvoWSO082HzOYuixVHRCKsE"
    os.environ["OPENAI_BASE_URL"] = "https://svip.xty.app"

    # os.environ["OPENAI_API_KEY"] = "sk-6qOztfBscRbTIS5HphUOv35PhSgNyj7X5G1IT3l2g9zBfxeU"
    # os.environ["OPENAI_BASE_URL"] = "https://api.key77qiqi.cn"


    BOCHA_API_KEY = "sk-0ca8965443c84eaab07183a37ae9f9a8"