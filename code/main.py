from langchain_openai import ChatOpenAI
from local_document_retriever_module import LocalDocumentRetrieverModule
#from online_search_retriever_module import OnlineSearchRetrieverModule
from prompt_formatting_module import PromptFormatter
from config import Config
import json

import os
os.environ["OPENAI_API_KEY"] = "sk-OBVaImxdTQNZdZbZsiAhlMwmvkvoWSO082HzOYuixVHRCKsE"
os.environ["OPENAI_BASE_URL"] = "https://svip.xty.app"

# decode the unicode escape characters from bocha api response
# import sys
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class RAGSystem:
    def __init__(self):
        self.local_retriever = LocalDocumentRetrieverModule()
        #self.web_retriever = OnlineSearchRetrieverModuleBocha()
        self.llm = ChatOpenAI(model=Config.LLM_MODEL)
        self.print_width = 80  

    def process_query(self, query: str) -> str:
        local_docs = self.local_retriever.retrieve_docs(query)
        print(f"Found {len(local_docs)} relevant document chunks")

        for doc in local_docs:
            print(f"* {doc.page_content} [{doc.metadata}]")

        # web_results = self.web_retriever.search_web(query)
        # print(f"Found {len(web_results)} web results\n")
        # print(json.dumps(web_results, indent=2, ensure_ascii=False))
        ## ensure_ascii=False guarantee chinese characters displayed properly

        print("Synthesizing information...")

        prompt = PromptFormatter.format_prompt(
            query=query,
            local_context=[d.page_content for d in local_docs],
            # web_context=web_results
        )

        response = self.llm.invoke(prompt)

        return response.content

def main():
    # Initialize system once
    rag = RAGSystem()

    while True:
        try:
            query = "福莫特罗单日最大总剂量"
            
            answer = rag.process_query(query)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break

        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            continue

if __name__ == "__main__":
    main()