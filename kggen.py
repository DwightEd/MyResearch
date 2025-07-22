import argparse
from tqdm import tqdm
import pandas as pd
from query_to_answer import Retrieval, Model, encode_emojis
from kg_gen import KGExtractor

model_path = "/home/v_zhishunliu/Quen2.5-32B-Instruct"

def main(args):
    retrieval = Retrieval(
        db_path=args.db_path,
        embedding_model_path=args.embedding_model
    )
    qwen = Model(model_path)

    kg_extractor = KGExtractor(model_name=args.kg_model)

    df = pd.read_csv(args.csv)
    output = {"questions": [], "answers": [], "LLM_answers": [], "contexts": []}

    for idx in tqdm(range(len(df)), total=len(df)):
        question = df.loc[idx, "questions"]
        gold_answer = df.loc[idx, "answers"]

        # 一级检索
        topic_res = retrieval.query_topic(questions=[question], k_top=1)
        # 二级检索
        docs = retrieval.query_documents(questions=[question], topic_result=topic_res)

        merged_docs = ["".join(d["documents"][0]) for d in docs]

        triples_list = kg_extractor.extract(merged_docs)

        for doc_text, triples in zip(merged_docs, triples_list):
            context = encode_emojis(doc_text)
            # 格式化三元组
            kg_block = "\n".join(f"{h} —{r}→ {t}" for h, r, t in triples)

            prompt = (
                "你是智能客服，仅根据上下文与知识图回答问题，无需额外解释。\n"
                f"问题：{question}\n\n"
                "Context：\n" + context + "\n\n"
                "知识图谱（三元组）：\n" + kg_block + "\n\n"
                "回答："
            )

            response = qwen.generate_response(prompt)

            output["questions"].append(question)
            output["answers"].append(gold_answer)
            output["LLM_answers"].append(response)
            output["contexts"].append(context)

    pd.DataFrame(output).to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, default="QA_with_kggen.csv", help="输出文件路径")
    args = parser.parse_args()
    main(args)
    args = parser.parse_args()
    main(args)
