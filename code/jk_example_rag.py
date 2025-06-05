import abc
from typing import List, Dict, Any
import numpy as np

def fake_embed(text: str) -> np.ndarray:
    """
    占位Embedding函数。实际应用中请替换为真实的Embedding模型，如sentence-transformers。
    这里用字符ord和长度简单模拟。
    """
    arr = np.array([ord(c) for c in text])
    if arr.size == 0:
        return np.zeros(10)
    # 取均值和长度，拼成一个10维向量
    mean = arr.mean()
    length = len(text)
    return np.pad(np.array([mean, length]), (0, 8), 'constant')

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class BaseRetriever(abc.ABC):
    """
    检索器基类，定义通用接口，便于后续扩展不同数据源。
    """
    @abc.abstractmethod
    def retrieve(self, query: str, treatment_plan: str, department: str, top_k: int = 5) -> List[str]:
        pass

class LocalListRetriever(BaseRetriever):
    """
    本地文本块列表检索器，基于Embedding的余弦相似度。
    """
    def __init__(self, text_blocks: List[Dict[str, Any]] = None):
        if text_blocks is not None:
            self.text_blocks = text_blocks
        else:
            self.text_blocks = [
                {"department": "内科", "content": "高血压的标准治疗方案包括……"},
                {"department": "外科", "content": "阑尾炎的手术指征为……"},
                {"department": "内科", "content": "糖尿病患者的饮食管理……"},
            ]
        # 预计算文本块的embedding
        for block in self.text_blocks:
            block["embedding"] = fake_embed(block["content"])

    def retrieve(self, query: str, treatment_plan: str, department: str, top_k: int = 5) -> List[str]:
        # 先按department过滤
        filtered = [block for block in self.text_blocks if block["department"] == department]
        if not filtered:
            return []
        # 将query和treatment_plan拼接后做embedding
        query_text = query + " " + treatment_plan
        query_emb = fake_embed(query_text)
        # 计算余弦相似度
        scored = [
            (block["content"], cosine_similarity(query_emb, block["embedding"]))
            for block in filtered
        ]
        # 按相似度排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:top_k]]

# 默认导出本地列表检索器，便于直接使用
Retriever = LocalListRetriever

if __name__ == "__main__":
    # 单元测试
    blocks = [
        {"department": "内科", "content": "高血压的标准治疗方案包括低盐饮食、规律服药等。"},
        {"department": "内科", "content": "糖尿病患者应控制血糖，注意饮食和运动。"},
        {"department": "外科", "content": "阑尾炎的手术治疗方法。"},
        {"department": "内科", "content": "慢性肾脏病的管理包括血压控制和饮食调整。"},
    ]
    retriever = Retriever(blocks)
    query = "患者为高血压，血压控制不佳，如何治疗？"
    treatment_plan = "建议低盐饮食，规律服药，监测血压。"
    department = "内科"
    results = retriever.retrieve(query, treatment_plan, department, top_k=2)
    print("检索结果：")
    for i, r in enumerate(results):
        print(f"{i+1}. {r}")
