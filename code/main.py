import openai
from local_document_retriever_module import LocalDocumentRetrieverModule
from online_search_retriever_module import OnlineSearchRetrieverModule
from prompt_formatting_module import PromptFormatter
from config import Config
import json

import os
import traceback

# decode the unicode escape characters from bocha api response
# import sys
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class RAGSystem:
    def __init__(self):
        self.local_retriever = LocalDocumentRetrieverModule()
        self.web_retriever = OnlineSearchRetrieverModule()
        self.llm = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "sk-OBVaImxdTQNZdZbZsiAhlMwmvkvoWSO082HzOYuixVHRCKsE"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://svip.xty.app")
        )

    def process_query(self, query: str) -> str:
        local_docs = self.local_retriever.retrieve_docs(query)
        print(f"Found {len(local_docs)} relevant document chunks")

        for doc in local_docs:
            print(f"* {doc.page_content} [{doc.metadata}]")

        web_results = self.web_retriever.search_web(query)
        print(f"Found {len(web_results)} web results\n")
        print(json.dumps(web_results, indent=2, ensure_ascii=False))
        # ensure_ascii=False guarantee chinese characters displayed properly

        print("Synthesizing information...")

        prompt = PromptFormatter.format_prompt(
            query=query,
            local_context=[d.page_content for d in local_docs],
            web_context=web_results
        )
        
        # response with RAG
        completion = self.llm.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个医学专家，擅长回答医学相关问题。我会给你一个病例，你要根据病例给出治疗方案。"},
                {"role": "user", "content": prompt}
            ],
        )

        # response without RAG
        """
        completion = self.llm.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个医学专家，擅长回答医学相关问题。我会给你一个病例，你要根据病例给出治疗方案。"},
                {"role": "user", "content": query}
            ]
        )
        """
        
        return completion.choices[0].message.content

def main():
    # Initialize system once
    rag = RAGSystem()

    try:
        query = "一名11岁女性因持续6天发热和咳嗽，以及双侧下肢疼痛1天而入院。她没有显著的病史，家族史不详。\n\n入院时，实验室检查显示白细胞计数为3.84×10^9/L，中性粒细胞比例为78.6%，淋巴细胞比例为16.3%。红细胞计数为4.50×10^12/L，血红蛋白为127 g/L。炎症标志物显示红细胞沉降率（ESR）为8 mm/1h，C反应蛋白（CRP）为33.8 mg/L，降钙素原为2.3 µg/L。血液化学指示肝酶升高：丙氨酸转氨酶（ALT）为343.9 U/L，天冬氨酸转氨酶（AST）为1200 U/L，乳酸脱氢酶（LDH）为1996 U/L，肌酸激酶水平非常高，为52,435 U/L；肾功能和血脂状况均在正常范围内。尿液分析显示红细胞镜检为5-6个每高倍视野，蛋白尿为+2，血尿为+3，酮尿为+2。电解质显示低钾血症，血钾为3.00 mmol/L，低钠血症，血钠为124.30 mmol/L。凝血检查显示D-二聚体为6.62 mg/L，凝血酶原时间（PT）为16.00 s，纤维蛋白原为4.09 g/L，部分凝血活酶时间（APTT）为34.20 s，凝血酶时间为16.1 s。心脏标志物显示肌酸激酶同工酶（CK-MB）为88.86 µg/L，肌红蛋白超过4095.0 µg/L，肌钙蛋白I为0.01 µg/L。免疫学检查显示免疫球蛋白E升高至30,580 IU/mL，白细胞介素6（IL-6）为108.54 ng/L，白细胞介素8（IL-8）为43.39 ng/L。血清铁蛋白为1042.7 µg/L。甲状腺功能检查正常。病原体检测显示肺炎支原体抗体滴度显著，为1:160，肺炎支原体IgM（MPIgM）阳性，而其他呼吸道病原体检测为阴性。进一步的病毒抗原检测显示副流感病毒3型阳性，其他常见呼吸道病毒检测为阴性。T-SPOT.TB、肝炎病毒和EB病毒检测均为阴性。痰液培养显示有少量酵母菌。心电图和心脏超声检查无异常。腹部超声显示脾肿大约为122 mm × 40 mm，左肾结石为3 mm × 4 mm，发现符合胡桃夹综合征涉及左肾静脉。\n\n患者被诊断为重症肺炎和肺炎支原体感染，支持诊断的病原体检测结果显示肺炎支原体的显著抗体滴度和阳性肺炎支原体IgM，同时副流感病毒3型抗原呈阳性。。"
        
        answer = rag.process_query(query)

        print(answer)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

    except Exception as e:
        print(f"\nError processing query: {str(e)}")
        #traceback.print_exc()

if __name__ == "__main__":
    main()