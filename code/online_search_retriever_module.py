from keyword_extractor import KeywordExtractor 
from config import Config

from langchain_community.tools import DuckDuckGoSearchResults
import json


class OnlineSearchRetrieverModule:
    
    def __init__(self):
        self.search_tool = DuckDuckGoSearchResults(
            num_results=Config.WEB_SEARCH_MAX_RESULTS,
            output_format="json",
            )

        self.keyword_extractor = KeywordExtractor()

    def search_web(self, query: str) -> list:
        try:
            keywords = self.keyword_extractor.extract_keywords(query)
            search_query = " ".join(keywords)
            #raw_results = self.search_tool.run(search_query)
            raw_results = self.search_tool.run(query)
            #print(raw_results)
            results = json.loads(raw_results)
            return self._parse_json_results(results)

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _parse_json_results(self, results) -> list:
        """Parse list of result dictionaries from DuckDuckGo"""
        parsed = []

        if not results or not isinstance(results, list):
            print("[DEBUG] Unexpected results format")
            return parsed

        for item in results:
            parsed.append({
                'title': item.get('title', 'No Title'),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'category': item.get('category', ''),  
                'icon': item.get('icon', '')           
            })

        return parsed