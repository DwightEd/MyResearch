import os
import csv
import pandas as pd

class Record:
    def __init__(self, buckets, sample_size=5):
        self.buckets = buckets
        self.sample_size = sample_size
        self.counts = {b: 0 for b in buckets}
        self.samples = {b: [] for b in buckets}
        self.overall_min = float('inf')
        self.overall_max = 0
        self.min_sid = None
        self.max_sid = None

    def update(self, sid, length):
        # 全局最小/最大
        if length < self.overall_min:
            self.overall_min = length
            self.min_sid = sid
        if length > self.overall_max:
            self.overall_max = length
            self.max_sid = sid
        # 区间更新
        for b in self.buckets:
            lo, hi = b
            if lo <= length <= hi:
                self.counts[b] += 1
                if len(self.samples[b]) < self.sample_size:
                    self.samples[b].append(sid)
                break

    def report(self):
        return {
            'overall_min': (self.min_sid, self.overall_min),
            'overall_max': (self.max_sid, self.overall_max),
            'buckets': {
                f"{lo}-{hi}": {
                    'count': self.counts[(lo,hi)],
                    'samples': self.samples[(lo,hi)]
                } for lo,hi in self.buckets
            }
        }

    def save_samples(self, output_path):
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['bucket','session_id'])
            for b, sids in self.samples.items():
                label = f"{b[0]}-{b[1]}"
                for sid in sids:
                    writer.writerow([label, sid])


def compute_global_range(directory):
    overall_min = float('inf')
    overall_max = 0
    for fn in os.listdir(directory):
        if not fn.endswith('.csv'): continue
        path = os.path.join(directory, fn)
        for chunk in pd.read_csv(path, usecols=['detail'], chunksize=10000):
            chunk = chunk.dropna(subset=['detail'])
            lengths = chunk['detail'].astype(str).str.len()
            if lengths.empty: continue
            overall_min = min(overall_min, int(lengths.min()))
            overall_max = max(overall_max, int(lengths.max()))
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


def compute_and_sample(directory, buckets, sample_size, output_csv):
    rec = Record(buckets, sample_size)
    for fn in os.listdir(directory):
        if not fn.endswith('.csv'): continue
        path = os.path.join(directory, fn)
        for chunk in pd.read_csv(path, usecols=['session_id','detail'], chunksize=10000):
            chunk = chunk.dropna(subset=['detail'])
            chunk['detail'] = chunk['detail'].astype(str)
            lengths = chunk['detail'].str.len()
            for idx, length in lengths.items():
                sid = chunk.at[idx, 'session_id']
                rec.update(sid, int(length))
    stats = rec.report()
    print(f"最短 session → ID={stats['overall_min'][0]}, len={stats['overall_min'][1]}")
    print(f"最长 session → ID={stats['overall_max'][0]}, len={stats['overall_max'][1]}
")
    print("分段统计与示例:")
    for bucket_label, info in stats['buckets'].items():
        print(f"  [{bucket_label}] count={info['count']}, samples={info['samples']}")
    rec.save_samples(output_csv)
    print(f"采样结果已保存到 {output_csv}")


def main():
    data_dir = 'v_lejunliu_data'
    sample_size = 10
    num_buckets = 5  # 自动分成5个长度区间
    output_csv = 'sampled_sessions.csv'

    # 第一步：计算全局最小/最大长度
    min_len, max_len = compute_global_range(data_dir)
    print(f"全量 session 长度范围: {min_len} ~ {max_len}")

    # 第二步：根据分布自动生成区间
    buckets = generate_buckets(min_len, max_len, num_buckets)
    print(f"自动生成的区间: {buckets}
")

    # 第三步：统计并采样
    compute_and_sample(data_dir, buckets, sample_size, output_csv)

if __name__ == '__main__':
    main()
