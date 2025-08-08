import os
import random
import pandas as pd
import csv

class Record:
    def __init__(self, buckets):
        # buckets: list of (lo, hi)
        self.buckets = buckets
        self.counts = {b: 0 for b in buckets}
        self.overall_min = float('inf')
        self.overall_max = 0
        self.min_sid = None
        self.max_sid = None

    def update(self, sid, length):
        # 更新全局 min/max
        if length < self.overall_min:
            self.overall_min = length
            self.min_sid = sid
        if length > self.overall_max:
            self.overall_max = length
            self.max_sid = sid
        # 更新区间计数
        for b in self.buckets:
            lo, hi = b
            if lo <= length <= hi:
                self.counts[b] += 1
                break

    def report(self):
        print(f"全局最短 session → ID={self.min_sid}, len={self.overall_min}")
        print(f"全局最长 session → ID={self.max_sid}, len={self.overall_max}\n")
        print("各长度区间统计：")
        for b, cnt in self.counts.items():
            print(f"  区间[{b[0]},{b[1]}] → {cnt} 条")

    def save_stats(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        stat_path = os.path.join(output_dir, 'length_stats.csv')
        with open(stat_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['min_sid', self.min_sid])
            writer.writerow(['min_len', self.overall_min])
            writer.writerow(['max_sid', self.max_sid])
            writer.writerow(['max_len', self.overall_max])
            for b, cnt in self.counts.items():
                writer.writerow([f"{b[0]}-{b[1]}", cnt])
        print(f"统计结果已保存到 {stat_path}")

class Sampler:
    def __init__(self, total_samples, seed=42):
        self.total_samples = total_samples
        self.seed = seed
        self.ids = []

    def collect_ids(self, directory):
        # 收集所有 session_id
        for fn in os.listdir(directory):
            if not fn.endswith('.csv'):
                continue
            path = os.path.join(directory, fn)
            for chunk in pd.read_csv(path, usecols=['session_id'], chunksize=10000):
                self.ids.extend(chunk['session_id'].astype(str).tolist())

    def sample(self):
        random.seed(self.seed)
        if len(self.ids) <= self.total_samples:
            return self.ids[:]  # 全量返回
        return random.sample(self.ids, self.total_samples)

    def save(self, sampled_ids, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, 'sampled_sessions.csv')
        pd.DataFrame({'session_id': sampled_ids}).to_csv(
            sample_path, index=False, encoding='utf-8'
        )
        print(f"随机抽样 {len(sampled_ids)} 条结果已保存到 {sample_path}")

def compute_global_range(directory):
    overall_min = float('inf')
    overall_max = 0
    for fn in os.listdir(directory):
        if not fn.endswith('.csv'):
            continue
        path = os.path.join(directory, fn)
        for chunk in pd.read_csv(path, usecols=['session_id','detail'], chunksize=10000):
            chunk = chunk.dropna(subset=['detail'])
            chunk['detail'] = chunk['detail'].astype(str)
            lengths = chunk['detail'].str.len()
            if lengths.empty:
                continue
            # 针对每行同时更新全局
            min_idx, max_idx = lengths.idxmin(), lengths.idxmax()
            overall_min = min(overall_min, int(lengths[min_idx]))
            overall_max = max(overall_max, int(lengths[max_idx]))
    return overall_min, overall_max

def generate_buckets(min_len, max_len, num_buckets):
    span = max_len - min_len + 1
    width = span // num_buckets
    buckets = []
    start = min_len
    for i in range(num_buckets):
        lo = start
        hi = start + width - 1 if i < num_buckets - 1 else max_len
        buckets.append((lo, hi))
        start = hi + 1
    return buckets

def analyze_and_sample(data_dir, num_buckets, total_samples, output_dir):
    # 1. 全局范围
    min_len, max_len = compute_global_range(data_dir)
    print(f"全量 Session 字符长度范围: {min_len} ~ {max_len}")

    # 2. 生成区间 & 统计
    buckets = generate_buckets(min_len, max_len, num_buckets)
    print(f"自动生成 {num_buckets} 个长度区间: {buckets}\n")

    rec = Record(buckets)
    # 遍历更新统计
    for fn in os.listdir(data_dir):
        if not fn.endswith('.csv'): continue
        path = os.path.join(data_dir, fn)
        for chunk in pd.read_csv(path, usecols=['session_id','detail'], chunksize=10000):
            chunk = chunk.dropna(subset=['detail'])
            chunk['detail'] = chunk['detail'].astype(str)
            for idx, row in chunk.iterrows():
                sid = row['session_id']
                length = len(row['detail'])
                rec.update(sid, length)
    rec.report()
    rec.save_stats(output_dir)

    # 3. 随机抽样
    sampler = Sampler(total_samples)
    sampler.collect_ids(data_dir)
    sampled_ids = sampler.sample()
    sampler.save(sampled_ids, output_dir)

def main():
    # ---------- 配置开始 ----------
    data_dir = 'v_lejunliu_data'    # 源 CSV 文件夹
    num_buckets = 5                 # 长度区间数量
    total_samples = 5000            # 随机抽样总数
    output_dir = 'analysis_output'  # 结果输出目录
    # ---------- 配置结束 ----------

    analyze_and_sample(data_dir, num_buckets, total_samples, output_dir)

if __name__ == '__main__':
    main()
