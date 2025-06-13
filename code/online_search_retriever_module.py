from keyword_extractor import KeywordExtractor
from config import Config
import json
import requests


class OnlineSearchRetrieverModule:

    def __init__(self):
        self.keyword_extractor = KeywordExtr0actor()
        self.api_key = Config.GOOGLE_SEARCH_API_KEY
        self.search_engine_id = Config.GOOGLE_SEARCH_ENGINE_ID
        self.max_results = Config.WEB_SEARCH_MAX_RESULTS

    def search_web(self, query: str) -> list:
        try:
            keywords = self.keyword_extractor.extract_keywords(query)
            search_query = " ".join(keywords)

            # result = self._perform_google_search(search_query)
            result = self._perform_google_search(query)

            return result

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _perform_google_search(self, query: str) -> list:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": self.max_results
        }
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"[ERROR] Google Search API request failed: {response.status_code}, {response.text}")
            return []

        results = response.json().get("items", [])
        return self._parse_google_results(results)

    def _parse_google_results(self, results: list) -> list:
        parsed = []

        for item in results:
            parsed.append({
                "title": item.get("title", "No Title"),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "category": "",  # Google doesn't categorize this way
                "icon": item.get("pagemap", {}).get("cse_thumbnail", [{}])[0].get("src", "")
            })

        return parsed
