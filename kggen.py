# kggen_module.py

"""
模块：kggen_module
功能：独立封装 KG‑Gen 三元组抽取，提供简单接口并支持测试。
依赖：pip install kg-gen
"""

from kg_gen import KGExtractor
from typing import List, Tuple

class KGGenModule:
    """
    封装 KG‑Gen 抽取功能的独立模块。
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        :param model_name: kg-gen 使用的 LLM 模型名
        """
        self.extractor = KGExtractor(model_name=model_name)

    def extract_triples(self, docs: List[str]) -> List[List[Tuple[str, str, str]]]:
        """
        对一系列文档文本进行三元组抽取。
        :param docs: 文档列表，每个元素为纯文本字符串
        :return: 每个文档对应的三元组列表，格式 [(head, relation, tail), ...]
        """
        return self.extractor.extract(docs)


if __name__ == '__main__':
    # ===== 简单测试脚本 =====
    sample_docs = [
        "北京是中国的首都。",
        "华为公司总部位于深圳。"
    ]
    kgmod = KGGenModule(model_name="gpt-4o-mini")
    triples = kgmod.extract_triples(sample_docs)
    for i, tri_list in enumerate(triples, 1):
        print(f"文档 {i} 抽取到的三元组：")
        for h, r, t in tri_list:
            print(f"  {h} —{r}→ {t}")
        print()
