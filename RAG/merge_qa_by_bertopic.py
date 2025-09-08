import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ML libs
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster

# ---------- 参数（可根据需要修改） ----------
INPUT_CSV = "qa_input.csv"      # 输入 CSV 路径（请修改为你的文件）
OUTPUT_FOLDER_BASE = "merged_qa_outputs"
QUESTION_COL = "question"
ANSWER_COL = "answer"
SESSION_COL = "session_id"
TEXT_COL = "original_text"
QA_ID_COL = "qa_id"             # 若输入没有此列，脚本会自动生成 uuid
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 可改为更大模型
SIMILARITY_THRESHOLD = 0.82    # 在同主题内，question 相似度阈值（0-1），>阈值则合并为一组
LARGE_GROUP_THRESHOLD = 1200   # 如果单主题内样本数超过此值，使用 KMeans 降级处理
KMEANS_PER_CLUSTER = 200       # 降级时每个 KMeans 聚类目标大小（大致）
MIN_QUESTION_LEN = 3           # 用于可能的过滤（可设为1）
RANDOM_STATE = 42
# ------------------------------------------------

def ensure_columns(df: pd.DataFrame):
    for c in [QUESTION_COL, ANSWER_COL, SESSION_COL, TEXT_COL]:
        if c not in df.columns:
            raise ValueError(f"输入文件缺少必要列: {c}")
    if QA_ID_COL not in df.columns:
        df[QA_ID_COL] = [str(uuid.uuid4()) for _ in range(len(df))]

def make_output_dir(base: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"{base}_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    return out

def dedupe_exact(df: pd.DataFrame) -> pd.DataFrame:
    # 去掉完全重复（question+answer+original_text 相同）
    before = len(df)
    df = df.drop_duplicates(subset=[QUESTION_COL, ANSWER_COL, TEXT_COL])
    print(f"Exact dedupe: {before} -> {len(df)}")
    return df

def fit_bertopic(questions: List[str], embedding_model_name: str):
    embedder = SentenceTransformer(embedding_model_name)
    # 直接把 embedding 传给 BERTopic 更稳定可控
    embeddings = embedder.encode(questions, show_progress_bar=True, convert_to_numpy=True)
    topic_model = BERTopic(embedding_model=embedder, calculate_probabilities=False, verbose=False)
    topics, probs = topic_model.fit_transform(questions, embeddings)
    return topic_model, topics, embeddings

def cluster_within_topic(embeddings: np.ndarray, sim_thr: float, large_group_threshold: int):
    """
    输入：embeddings (n, d) for questions in a single topic
    输出：cluster_labels array length n（同一簇代表应合并）
    算法：
      - 若 n <= large_group_threshold：用层次聚类（average linkage）依据余弦距离，阈值 = 1 - sim_thr
      - 否则：先 PCA 降维 + KMeans 划分为若干簇，然后对每个簇再做层次聚类（避免 O(n^2) 的距离矩阵太大）
    """
    n = embeddings.shape[0]
    if n == 1:
        return np.array([1], dtype=int)

    if n <= large_group_threshold:
        # pairwise cosine similarity -> distance = 1 - sim
        sim = cosine_similarity(embeddings)
        # convert to condensed distance for linkage? We'll pass full matrix via linkage on pdist-like:
        # Using linkage on distances requires condensed form. We'll compute condensed distances from similarity.
        # But simpler: compute 1 - sim and then use linkage on the square matrix by flattening upper triangle.
        from scipy.spatial.distance import squareform
        dist = 1.0 - sim
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        # fcluster with threshold on distance:
        cluster_ids = fcluster(Z, t=1.0 - sim_thr, criterion='distance')
        return cluster_ids
    else:
        # 大组的降级策略：先用 PCA -> KMeans 分成若干小块，再分别聚类
        n_clusters = max(1, int(np.ceil(n / KMEANS_PER_CLUSTER)))
        pca = PCA(n_components=min(50, embeddings.shape[1], max(2, int(np.sqrt(n)))), random_state=RANDOM_STATE)
        emb2 = pca.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        coarse_labels = kmeans.fit_predict(emb2)
        final_labels = np.zeros(n, dtype=int)
        next_label = 1
        for lbl in np.unique(coarse_labels):
            idx = np.where(coarse_labels == lbl)[0]
            sub = embeddings[idx]
            if sub.shape[0] == 1:
                final_labels[idx[0]] = next_label
                next_label += 1
                continue
            # perform hierarchical inside this small block
            sim = cosine_similarity(sub)
            from scipy.spatial.distance import squareform
            dist = 1.0 - sim
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            sub_cluster_ids = fcluster(Z, t=1.0 - sim_thr, criterion='distance')
            # normalize sub_cluster_ids to global labels
            for sc in np.unique(sub_cluster_ids):
                members = np.where(sub_cluster_ids == sc)[0]
                for m in members:
                    final_labels[idx[m]] = next_label
                next_label += 1
        return final_labels

def choose_representative_entry(indices: List[int], qa_embeddings: np.ndarray, df_group: pd.DataFrame):
    """
    从 indices（群内条目索引）中选择代表性条目：
    - 计算这些条目的 Q+A embedding centroid，然后挑选距离中心最近的那条作为代表
    返回代表所在的 DataFrame 行 index（相对于 df_group）
    """
    sub_emb = qa_embeddings[indices]
    centroid = np.mean(sub_emb, axis=0, keepdims=True)
    sims = cosine_similarity(sub_emb, centroid).flatten()
    best_local_idx = int(np.argmax(sims))  # 最大相似度
    return indices[best_local_idx]

def merge_cluster(indices: List[int], df_group: pd.DataFrame):
    """
    根据给定索引合并：
    - merged_questions_all: 去重拼接所有 question（用 " ||| " 分隔）
    - merged_answers_all: 去重拼接所有 answer（用 " \n---\n " 分隔）
    - sources: list of dicts with qa_id, question, answer, session_id, original_text
    """
    qs = list(dict.fromkeys(df_group.iloc[indices][QUESTION_COL].astype(str).tolist()))
    as_ = list(dict.fromkeys(df_group.iloc[indices][ANSWER_COL].astype(str).tolist()))
    sources = []
    for i in indices:
        row = df_group.iloc[i]
        sources.append({
            "qa_id": row[QA_ID_COL],
            "question": row[QUESTION_COL],
            "answer": row[ANSWER_COL],
            "session_id": row[SESSION_COL],
            "original_text": row[TEXT_COL]
        })
    merged = {
        "merged_questions_all": " ||| ".join(qs),
        "merged_answers_all": "\n---\n".join(as_),
        "sources": sources,
        "source_count": len(indices),
    }
    return merged

def main():
    # 1) 读取数据
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"载入 {len(df)} 行数据")

    ensure_columns(df)
    df = df.fillna("")
    df = df[df[QUESTION_COL].str.len() >= MIN_QUESTION_LEN].reset_index(drop=True)
    df = dedupe_exact(df)
    out_dir = make_output_dir(OUTPUT_FOLDER_BASE)
    # 保存初始清洗后的表
    df.to_csv(out_dir / "input_cleaned.csv", index=False)

    # 2) 用 BERTopic 聚类 questions
    questions = df[QUESTION_COL].astype(str).tolist()
    print("正在训练 BERTopic，并生成 topic 分配（可能耗时）...")
    topic_model, topics, questions_embeddings = fit_bertopic(questions, EMBEDDING_MODEL_NAME)
    df["topic"] = topics
    # 将 topic 名称尝试查出
    topic_info = []
    unique_topics = sorted(list(set(topics)))
    for t in unique_topics:
        try:
            words = topic_model.get_topic(t)
            # get_topic 返回 [(word, score), ...]
            name = ", ".join([w for w, _ in words[:6]]) if words else ""
        except Exception:
            name = ""
        topic_info.append({"topic": t, "topic_name": name})
    pd.DataFrame(topic_info).to_csv(out_dir / "topics_overview.csv", index=False)
    df.to_csv(out_dir / "input_with_topics.csv", index=False)

    # 3) 在每个 topic 内进一步合并
    merged_rows = []
    detail_rows = []  # 每个原始qa到合并结果的映射（可选）
    qa_plus_embeddings = []  # embedding for Q + " ||| " + A (for representative selection)
    # 预先计算 Q+A embedding（用于代表性选择）
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("计算 Q+A embedding（用于代表性选择）...")
    combined_texts = (df[QUESTION_COL].astype(str) + " ||| " + df[ANSWER_COL].astype(str)).tolist()
    qa_plus_embeddings = embedder.encode(combined_texts, show_progress_bar=True, convert_to_numpy=True)

    # 逐主题处理
    topics_sorted = sorted(df["topic"].unique(), key=lambda x: (x is None, x))
    pbar = tqdm(topics_sorted, desc="Processing topics")
    global_merged_id = 0
    for t in pbar:
        sub_idx = df.index[df["topic"] == t].tolist()
        df_group = df.loc[sub_idx].reset_index(drop=True)
        if df_group.empty:
            continue
        # question embeddings for this group (use the previously computed question-level embeddings)
        q_emb = questions_embeddings[[df.index.get_loc(i) if i in df.index else idx for idx, i in enumerate(sub_idx)]]
        # But above indexing may be tricky; simpler: recompute for df_group questions
        q_emb = embedder.encode(df_group[QUESTION_COL].astype(str).tolist(), show_progress_bar=False, convert_to_numpy=True)

        # cluster within topic
        try:
            cluster_labels = cluster_within_topic(q_emb, SIMILARITY_THRESHOLD, LARGE_GROUP_THRESHOLD)
        except Exception as e:
            print(f"主题 {t} 内聚类出错，退回到每条独立：{e}")
            cluster_labels = np.arange(1, len(df_group) + 1)

        # map cluster id -> indices in df_group
        clusters = {}
        for local_idx, cl in enumerate(cluster_labels):
            clusters.setdefault(int(cl), []).append(local_idx)

        # 对每个 cluster 做合并
        for clid, indices in clusters.items():
            global_merged_id += 1
            merged_meta = merge_cluster(indices, df_group)
            # 选择代表性条目：使用 Q+A embedding 在整个 df 的索引中找到对应位置
            # 我们 need indices mapped to global positions in original df to pick embeddings from qa_plus_embeddings
            global_indices = [sub_idx[i] for i in indices]  # original df indices
            local_rep_idx = choose_representative_entry(list(range(len(indices))), 
                                                        qa_plus_embeddings[global_indices], df_group)
            # local_rep_idx is index in indices list
            rep_global_idx = global_indices[local_rep_idx]
            rep_row = df.loc[rep_global_idx]

            merged_row = {
                "merged_id": f"m{global_merged_id}",
                "topic": int(t),
                "topic_name": next((x["topic_name"] for x in topic_info if x["topic"] == t), ""),
                "representative_qa_id": rep_row[QA_ID_COL],
                "question_rep": rep_row[QUESTION_COL],
                "answer_rep": rep_row[ANSWER_COL],
                "merged_questions_all": merged_meta["merged_questions_all"],
                "merged_answers_all": merged_meta["merged_answers_all"],
                "source_count": merged_meta["source_count"],
                "sources_json": json.dumps(merged_meta["sources"], ensure_ascii=False),
            }
            merged_rows.append(merged_row)
            # 记录明细：每个原始 qa -> merged_id
            for gid in global_indices:
                detail_rows.append({
                    "qa_id": df.loc[gid, QA_ID_COL],
                    "orig_index": int(gid),
                    "merged_id": merged_row["merged_id"],
                    "topic": int(t)
                })

    # 保存结果
    df_merged = pd.DataFrame(merged_rows)
    df_detail = pd.DataFrame(detail_rows)
    df_merged.to_csv(out_dir / "merged_qa.csv", index=False)
    df_detail.to_csv(out_dir / "merge_mapping_detail.csv", index=False)
    print(f"完成！输出文件在：{out_dir.resolve()}")
    print(f"合并后条目数：{len(df_merged)}，映射行数：{len(df_detail)}")

if __name__ == "__main__":
    main()
