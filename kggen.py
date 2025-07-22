import argparse
from tqdm import tqdm
import pandas as pd
from query_to_answer import Retrieval, Model, encode_emojis
from kg_gen import KGExtractor


def main(args):
    # 1. 初始化检索与模型
    retrieval = Retrieval(
        db_path=args.db_path,
        embedding_model_path=args.embedding_model
    )
    qwen = Model(args.llm_model)

    # 2. 初始化 KG-Gen
    kg_extractor = KGExtractor(model_name=args.kg_model)

    # 3. 读取数据
    df = pd.read_csv(args.csv)
    output = {"questions": [], "answers": [], "LLM_answers": [], "contexts": []}

    # 4. 遍历问答
    for idx in tqdm(range(len(df)), total=len(df)):
        question = df.loc[idx, "questions"]
        gold_answer = df.loc[idx, "answers"]

        # 一级检索
        topic_res = retrieval.query_topic(questions=[question], k_top=1)
        # 二级检索
        docs = retrieval.query_documents(questions=[question], topic_result=topic_res)

        # 文本合并
        merged_docs = ["".join(d["documents"][0]) for d in docs]

        # KG-Gen 抽取三元组
        triples_list = kg_extractor.extract(merged_docs)

        # 对每个文档生成回答
        for doc_text, triples in zip(merged_docs, triples_list):
            # 编码 emoji
            context = encode_emojis(doc_text)
            # 格式化三元组
            kg_block = "\n".join(f"{h} —{r}→ {t}" for h, r, t in triples)

            # 构造 Prompt
            prompt = (
                "你是智能客服，仅根据上下文与知识图回答问题，无需额外解释。\n"
                f"问题：{question}\n\n"
                "Context：\n" + context + "\n\n"
                "知识图谱（三元组）：\n" + kg_block + "\n\n"
                "回答："
            )

            # 生成
            response = qwen.generate_response(prompt)

            # 收集
            output["questions"].append(question)
            output["answers"].append(gold_answer)
            output["LLM_answers"].append(response)
            output["contexts"].append(context)

    # 保存结果
    pd.DataFrame(output).to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, default="QA_with_kggen.csv", help="输出文件路径")
    parser.add_argument("--db_path", type=str, default="chroma_db_bge_gpu", help="ChromaDB 路径")
    parser.add_argument("--embedding_model", type=str, required=True, help="BGE 嵌入模型路径")
    parser.add_argument("--llm_model", type=str, required=True, help="Qwen 模型路径")
    parser.add_argument("--kg_model", type=str, default="gpt-4o-mini", help="KG-Gen 使用的模型名或路径")
    args = parser.parse_args()
    main(args)
