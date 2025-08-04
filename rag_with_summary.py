import argparse
import csv
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

# 本次对话场景：银行业务咨询，请关注业务相关的专有名词，并准确提取问答要点。

# 请将以下对话提取为结构化摘要，需满足以下要求：

# (1) 句子简化：将复合句拆分为单句，每条一个信息点。  
# (2) 实体与描述分离：拆分命名实体及其属性。  
# (3) 指代消解：将代词替换为明确实体。  
# (4) 删除问候和营销模板。  
# (5) 输出格式：CSV，每行一条知识单元。

# 对话内容如下：
# {cleaned_text}

EMBED_MODEL_PATH = "/home/v_zhishunliu/bge-large-zh-v1.5"
LLM_MODEL_PATH = "/home/v_zhishunliu/bge-large-zh-v1.5"

SIM_THRESHOLD = 0.85
LENGTH_THRESHOLD = 25

class SessionCleaner:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.marketing_embs = []

    def split_turns(self, text):
        parts = re.split(r'(?=(坐席：|客户：))', text)
        return [p.strip() for p in parts if p.strip().startswith(('坐席：','客户：'))]

    def remove_greetings(self, turn):
        m = re.match(r'^(坐席：|客户：)', turn)
        label = m.group(1) if m else ''
        content = turn[len(label):]
        content = re.sub(r'^(?:\S{0,10}?好|你好(?:在吗)?|在吗)[，,]?\s*', '', content)
        return label + content

    def is_marketing(self, turn):
        if not turn.startswith('坐席：'): return False
        content = turn.split('：',1)[1].strip()
        if len(content) < LENGTH_THRESHOLD: return False
        emb = self.embed_model.encode(content, normalize_embedding=True)
        if self.marketing_embs:
            sims = np.dot(self.marketing_embs, emb)
            if np.max(sims) > SIM_THRESHOLD: return True
        self.marketing_embs.append(emb)
        return False

    def clean_session(self, detail):
        turns = self.split_turns(detail)
        cleaned, has_customer = [], False
        for turn in turns:
            turn = self.remove_greetings(turn)
            if self.is_marketing(turn): continue
            if turn.startswith('客户：'): has_customer = True
            cleaned.append(turn)
        if not has_customer: return None
        return "\n".join(cleaned)

class Model:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, device_map='auto'
        )
        # self.pipeline = TextGenerationPipeline(
        #     model=self.model, tokenizer=self.tokenizer,
        #     device=self.model.device, max_new_tokens=512, do_sample=False
        # )

    def generate(self, prompt_text):
        messages = [
            {"role": "system", "content": "你是智能客服，擅长提取对话摘要和知识要点。"},
            {"role": "user", "content": prompt_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        response = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return response.strip()

def process_file(input_path, output_path, cleaner, generator, prompt_template):
    summaries = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            sid, detail = row['session_id'], row['detail']
            cleaned = cleaner.clean_session(detail)
            if not cleaned: continue
            full_prompt = prompt_template.format(cleaned_text=cleaned)
            summary = generator.generate(full_prompt)
            summaries.append({'session_id': sid, 'detail': summary})
    if summaries:
        with open(output_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=['session_id','detail'])
            writer.writeheader()
            writer.writerows(summaries)

def main(input_dir, output_dir, prompt_path):
    os.makedirs(output_dir, exist_ok=True)
    embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    cleaner = SessionCleaner(embed_model)
    generator = Model(LLM_MODEL_PATH)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.csv'): continue
        inp = os.path.join(input_dir, fname)
        outp = os.path.join(output_dir, fname)
        process_file(inp, outp, cleaner, generator, prompt_template)
        print(f"Processed {fname} → saved to {outp}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG 对话摘要批量提取与清洗')
    parser.add_argument('--input_dir', required=True, help='输入文件夹路径，包含按月 CSV')
    parser.add_argument('--output_dir', required=True, help='输出摘要文件夹')
    parser.add_argument('--prompt', required=True, help='Prompt 模板文件路径')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.prompt)
