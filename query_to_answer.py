class BGEEmbeddingFunctionGPU(EmbeddingFunction):
    def __init__(self,model_name,device='cuda'):
        self,device=device
        self.model = SentenceTransformer(model_name).to(self,device)
        self.model.eval()

    def __call__(self,input:Documents)->Embeddings:
        #BGE添加搜索指令
        instructions = ["为这个句子生成表示用于检索相关文章："* doc for doc in input]
        with torch.no_grad():
            embeddings = self.model.encode(
                instructions,
                convert_to_tensor=True,
                normalize_embedding=True,
                device=self.device
    )

        return embeddings.cpu().numpy().tollst()
    
class Model:
    def __init__(self, model_path):
    #加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto",device_map="auto")

def generate_response(self, prompt):
    messages = [
    {"role": "system", "content": "你是智能客服，擅长回答客户的问题."},
    {"role": "user", "content": prompt},
]
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

    generated_ids = self.model.generate(
        **model_inputs,
        max_new_tokens=1024*4,
    )

    generated_ids = (
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    )

    response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

class Retrieval:
    def __init__(self,db_path,embedding_model_path):
        self.client = chromadb.PersistentClient(path=db_path)
        self.bge_ef_gpu = BGEEmbeddingFunctionGPU(model_name=embedding_model_path,device='cuda')

    def query(self,db_name,questions,k_top):
        try:
            collection = self.client.get_collection(name=db_name,embedding_function=self.bge_ef_gpu)
            result=collection.query(
                # query_texts=[{"微业贷提前结算是否可以在手机上操作？"}],
                query_texts=questions,
                n_results=k_top,
                include=["documents","distances","metadatas"]
                )
            return result
        except:
            print("{} 不存在".format(db_name))
            return None
    
    def query_topic(self,questions,k_top=5,db_name="documents_topic"):
        # collection = self.client.get_collection(name="documents_topic",embedding_function-self.bge_ef_gpu)
        # result=collection.query(
            # query_texts=[{"静止段模糊结果是否可以在手机上操作？"}],
            # query_texts=questions,
            # n_results=[{"documents","distance","metadata"]
        # )
        return self.query(db_name,questions,k_top)

    def query_documents(self,questions,topic_result,k_top=5):
        # documents = topic_result['documents']
        all_topic_ids = topic_result['metadatas']
        # distances = topic_id['distances']

        results={}
        for i,question in enumerate(questions):
            topic_ids=[]

            for topic_dict in all_topic_ids[i]:
                if topic_dict["topicid"] not in topic_ids:
                    topic_ids.append(topic_dict["topicid"])

            for topic_id in topic_ids:
                db_name = "topic_{}".format(topic_id)

                response = self.query(db_name,questions,k_top)
                results.append(response)
        return results
    


# 将表情符号编码为Unicode转义字符
def encode_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和图标
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "]",
        flags=re.UNICODE,
    )
    encoded_text = emoji_pattern.sub(
        lambda match: match.group().encode('unicode-escape').decode('ascii'),
        text
    )
    return encoded_text

# 将Unicode转义字符解码为表情符号
def decode_emojis(encoded_text):
    emoji_pattern = re.compile(r'\\U[0-9a-fA-F]{8}|\\u[0-9a-fA-F]{4}')
    decoded_text = emoji_pattern.sub(
        lambda match: match.group().encode('ascii').decode('unicode-escape'),
        encoded_text
    )
    return decoded_text

model_path = "/home/v_zhishunliu/Quen2.5-32B-Instruct"
Quen_model = Model(model_path)
