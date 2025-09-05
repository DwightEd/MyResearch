import json
import ast
from tqdm import tqdm
import pandas as pd
import jieba
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# ---------- 配置区（按需修改） ----------
INPUT_PATH = "data/sessions.csv"   # 或 data/sessions.jsonl
OUTPUT_CSV = "clustered_qa.csv"
OUTPUT_JSON = "clusters.json"

# Embedding model（可替换为更强的中文模型）
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"  # 或其它中文模型（若可用）

# 中文分词开关（若是中文文本建议 True，用 jieba 做分词以保证 c-TF-IDF 表现好）
USE_CHINESE_TOKENIZER = True

# 文档构造模式： "q+a" / "q" / "a"
DOC_MODE = "q+a"

# BERTopic 参数
MIN_TOPIC_SIZE = 10   # 最小主题大小（HDBSCAN）可调
NR_TOPICS = None      # 若想强制生成固定主题数，可设整数；None 表示自动
# ----------------------------------------

def safe_parse_detail(detail_field):
    """尝试把 detail 字段解析为 Python list（每项为 dict）"""
    if isinstance(detail_field, list):
        return detail_field
    if pd.isna(detail_field):
        return []
    # 若已经是 JSON 字符串
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(detail_field)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            continue
    # fallback: 空列表
    return []

def jieba_tokenizer(text):
    return [tok for tok in jieba.lcut(text) if tok.strip()]

def build_docs_from_df(df):
    """遍历 DataFrame，提取每个 QA 形成单条文档，并保存元信息"""
    docs = []
    metas = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting QA"):
        sid = row.get("session_id", None)
        detail_raw = row.get("detail", None)
        qa_list = safe_parse_detail(detail_raw)
        if not qa_list:
            continue
        # detail 里可能包含多个对象；逐个取出 question/answer
        for i, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                continue
            q = qa.get("question", "")
            a = qa.get("answer", "")
            if not q and not a:
                continue
            if DOC_MODE == "q":
                doc = q.strip()
            elif DOC_MODE == "a":
                doc = a.strip()
            else:
                # q+a 模式，保留分隔符便于后处理
                doc = (q.strip() or "") + " ||| " + (a.strip() or "")
            docs.append(doc)
            metas.append({
                "session_id": sid,
                "detail_index": i,
                "qa_obj": qa,
                "original_detail": qa_list,  # 整个 detail（若想仅保存当前项可改）
                "text": doc
            })
    return docs, metas

def main():
    # 1) 读取数据（支持 CSV 或 JSONL）
    if INPUT_PATH.endswith(".csv"):
        df = pd.read_csv(INPUT_PATH, dtype=str).fillna("")
    elif INPUT_PATH.endswith(".jsonl") or INPUT_PATH.endswith(".ndjson"):
        df = pd.read_json(INPUT_PATH, lines=True, dtype=str).fillna("")
    else:
        raise ValueError("输入文件请使用 .csv 或 .jsonl")

    # 2) 构造文档和元信息
    docs, metas = build_docs_from_df(df)
    if len(docs) == 0:
        print("没有发现任何 QA，可检查 detail 字段格式。")
        return

    # 3) 选择 embedding model 并把文本转成 embeddings（用 sentence-transformers）
    print("加载 embedding 模型：", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    # 4) 配置 CountVectorizer（中文需要 tokenizer）
    if USE_CHINESE_TOKENIZER:
        vectorizer = CountVectorizer(tokenizer=jieba_tokenizer, min_df=2)
    else:
        vectorizer = CountVectorizer(min_df=2)

    # 5) 初始化并训练 BERTopic
    topic_model = BERTopic(embedding_model=None,   # 我们直接传 embeddings，所以 embedding_model=None
                           vectorizer_model=vectorizer,
                           nr_topics=NR_TOPICS,
                           calculate_probabilities=True,
                           verbose=True,
                           min_topic_size=MIN_TOPIC_SIZE)

    topics, probs = topic_model.fit_transform(docs, embeddings)

    # 6) 组织输出：每条记录都含 topic / prob / 元数据
    rows = []
    for i, (topic, prob) in enumerate(zip(topics, probs)):
        row = {
            "topic": int(topic),
            "prob": float(np.max(prob)) if isinstance(prob, (list, np.ndarray)) else float(prob),
            "question": None,
            "answer": None,
            "session_id": metas[i]["session_id"],
            "detail_index": metas[i]["detail_index"],
            "qa_obj": metas[i]["qa_obj"],
            "full_detail": json.dumps(metas[i]["original_detail"], ensure_ascii=False)
        }
        # 从 metas/text 里拆出 question/answer（若模式为 q+a）
        text = metas[i]["text"]
        if "|||" in text:
            q,a = text.split("|||", 1)
            row["question"] = q.strip()
            row["answer"] = a.strip()
        else:
            if DOC_MODE == "q":
                row["question"] = text
                row["answer"] = metas[i]["qa_obj"].get("answer", "")
            elif DOC_MODE == "a":
                row["answer"] = text
                row["question"] = metas[i]["qa_obj"].get("question", "")
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"已保存每条 QA 的聚类结果到：{OUTPUT_CSV}")

    # 7) 按 topic 聚合并导出 JSON（每个 topic 包含主题关键词 & 对应的 QA 列表）
    topic_info = topic_model.get_topic_info()  # 包含 -1/noise 等
    clusters = {}
    for t in topic_info["Topic"].tolist():
        if t == -1:
            # 可选择跳过噪声或者保留
            pass
        keywords = []
        if t != -1:
            keywords = [kw for kw,score in topic_model.get_topic(t)]
        # 所属行
        members = out_df[out_df["topic"]==t].to_dict(orient="records")
        clusters[str(t)] = {
            "topic_id": int(t),
            "size": int(topic_info[topic_info["Topic"]==t]["Count"].values[0]) if t in topic_info["Topic"].values else len(members),
            "keywords": keywords,
            "members": members
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"已保存聚类聚合结果到：{OUTPUT_JSON}")

if __name__ == "__main__":
    main()
