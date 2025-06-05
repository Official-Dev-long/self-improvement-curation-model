from langchain_openai import ChatOpenAI
#from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from config import Config

class KeywordExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.LLM_MODEL)
        self.prompt_template = ChatPromptTemplate.from_template("""
        Analyze the following query and extract the most important keywords or key phrases 
        for information retrieval. Follow these rules:
        
        1. Identify core concepts and named entities
        2. Include technical terms and proper nouns
        3. Exclude common stop words (the, a, is, etc.)
        4. Maintain original meaning
        5. Return as a comma-separated list
        
        Examples:
        input: "What are the latest developments in transformer architectures?"
        output: "transformer architectures, latest developments"
        
        input: "Explain MCP in computer science"
        output: "MCP, computer science"
        
        Now process this input:
        {query}
        """)

    def extract_keywords(self, query: str) -> List[str]:
        prompt = self.prompt_template.format(query=query)
        response = self.llm.invoke(prompt)
        return [kw.strip() for kw in response.content.split(",")]