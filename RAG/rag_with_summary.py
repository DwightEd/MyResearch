import argparse
import csv
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

EMBED_MODEL_PATH = "/home/v_zhishunliu/bge-large-zh-v1.5"
LLM_MODEL_PATH = "/home/v_zhishunliu/bge-large-zh-v1.5"
SIM_THRESHOLD = 0.85
LENGTH_THRESHOLD = 25
BATCH_SIZE = 16  # 根据硬件资源调整

class SessionCleaner:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.marketing_embs = []

    def split_turns(self, text):
        # 支持中文全角冒号和空格+半角冒号+空格的标记
        parts = re.split(r'(?=(坐席：|客户：|坐席 : |客户 : ))', text)
        return [p.strip() for p in parts if p.strip().startswith(('坐席：','客户：','坐席 : ','客户 : '))]

    def remove_greetings(self, turn):
        # 匹配新的标记形式
        m = re.match(r'^(坐席：|客户：|坐席 : |客户 : )', turn)
        label = m.group(1) if m else ''
        content = turn[len(label):]
        # 删除问候语
        content = re.sub(r'^(?:\S{0,10}?好|你好(?:在吗)?|在吗)[，,]?\s*', '', content)
        return label + content

    def is_marketing(self, turn):
        # 同样支持新标记形式
        if not turn.startswith(('坐席：','坐席 : ')):
            return False
        # 统一提取内容
        m = re.match(r'^(坐席：|坐席 : )', turn)
        label = m.group(1)
        content = turn[len(label):].strip()
        if len(content) < LENGTH_THRESHOLD:
            return False
        emb = self.embed_model.encode(content, normalize_embedding=True)
        if self.marketing_embs:
            sims = np.dot(self.marketing_embs, emb)
            if np.max(sims) > SIM_THRESHOLD:
                return True
        self.marketing_embs.append(emb)
        return False

    def clean_session(self, detail):
        turns = self.split_turns(detail)
        cleaned, has_customer = [], False
        for turn in turns:
            turn = self.remove_greetings(turn)
            if self.is_marketing(turn):
                continue
            if turn.startswith(('客户：','客户 : ')):
                has_customer = True
            cleaned.append(turn)
        if not has_customer:
            return None
        return "\n".join(cleaned)

class Model:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, device_map='auto'
        )

    def generate_batch(self, prompts):
        texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "你是智能客服，擅长提取对话摘要和知识要点。"},
                {"role": "user", "content": prompt},
            ]
            texts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        batch_outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
        summaries = []
        for inp_ids, out_ids in zip(inputs.input_ids, batch_outputs):
            gen_ids = out_ids[len(inp_ids):]
            summaries.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return summaries


def process_file(input_path, output_path, cleaner, generator, prompt_template):
    rows, cleaned_prompts, session_ids = [], [], []
    with open(input_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            sid, detail = row['session_id'], row['detail']
            cleaned = cleaner.clean_session(detail)
            if not cleaned:
                continue
            full_prompt = prompt_template.format(cleaned_text=cleaned)
            rows.append(row)
            cleaned_prompts.append(full_prompt)
            session_ids.append(sid)

    all_summaries = []
    for i in range(0, len(cleaned_prompts), BATCH_SIZE):
        batch = cleaned_prompts[i:i+BATCH_SIZE]
        all_summaries.extend(generator.generate_batch(batch))

    if all_summaries:
        with open(output_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=['session_id','detail'])
            writer.writeheader()
            for sid, summary in zip(session_ids, all_summaries):
                writer.writerow({'session_id': sid, 'detail': summary})


def main(input_dir, output_dir, prompt_path):
    os.makedirs(output_dir, exist_ok=True)
    embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    cleaner = SessionCleaner(embed_model)
    generator = Model(LLM_MODEL_PATH)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.csv'):
            continue
        inp = os.path.join(input_dir, fname)
        outp = os.path.join(output_dir, fname)
        process_file(inp, outp, cleaner, generator, prompt_template)
        print(f"Processed {fname} → saved to {outp}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG 对话摘要批量提取与清洗')
    parser.add_argument('--input_dir', required=True, help='输入文件夹路径，包含按月 CSV')
    parser.add_argument('--output_dir', required=True, help='输出摘要文件夹')
    parser.add_argument('--prompt', required=True, help='prompt.txt')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.prompt)
