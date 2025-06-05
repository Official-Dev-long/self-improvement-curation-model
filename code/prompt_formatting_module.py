from langchain_core.prompts import ChatPromptTemplate
from typing import List
from datetime import date

class PromptFormatter:
    @staticmethod
    def format_prompt(
        query: str,
        local_context: List[str],
    ) -> str:

        # Format local_context as you did previously
        formatted_local_context = "\n- ".join(local_context)

        # Template for the final prompt
        template = """Answer the query using information from these sources:
        
        Local Documents:
        {local_context}
        
        Query: {query}

        please keep the following points in mind:

        - Today is {cur_date}.

        - Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.

        - For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.

        - For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.

        - If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.

        - For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.

        - Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.

        - Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.

        - Unless the user requests otherwise, your response should be in the same language as the user's question.

        
        Now, provide a comprehensive synthesized answer:"""

        # several cautions are referred from official prompt of provided by deepseek
        # https://github.com/deepseek-ai/DeepSeek-R1/pull/399/files
        
        prompt = ChatPromptTemplate.from_template(template)
        return prompt.format(
            query=query,
            local_context=formatted_local_context,
            cur_date=date.today()
        )