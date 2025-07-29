import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch
import os
import csv
from tqdm import tqdm
import ast


# 自定义BGE嵌入函数
class BGEEmbeddingFunctionGPU(EmbeddingFunction):
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model = SentenceTransformer(model_name).to(self.device)
        self.model.eval()
    
    def __call__(self, input: Documents) -> Embeddings:
        # BGE添加检索指令
        instructions = ["为这个句子生成表示用于检索相关文章：" + doc for doc in input]
        with torch.no_grad():
            embeddings = self.model.encode(
                instructions,
                convert_to_tensor=True,
                normalize_embedding=True,
                device=self.device
            )
        return embeddings.cpu().numpy().tolist()
    
class Create_vector_collection:
    def __init__(self, client, embedding_model):
        self.client = client
        self.embedding = embedding_model
    
    # 创建文档主题的collection
    def create_topic_collection(self, file_path, batch_size=10):
        key_str = []
        topic_id = []
        
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key in ast.literal_eval(row['Representation']):
                    if key != "":
                        key_str.append(key)
                        topic_id.append({"topic": row['Topic']})
        
        #要加self吗
        collection = self.client.create_collection(
            name="documents_topic",
            embedding_function=self.embedding,
            metadata={"hnsw:space": "ip"}
        )
        
        for i in tqdm(range(0, len(key_str), batch_size)):
            batch_key_str = key_str[i:i+batch_size]
            batch_ids = topic_id[i:i+batch_size]
            if i+batch_size < len(key_str):
                ids = [str(j) for j in range(i,i+batch_size)]
            else:
                ids = [str(j) for j in range(i,len(key_str))]
            
            #添加文档
            collection.add(
                documents=batch_key_str,
                ids=ids,
                metadatas=batch_ids
            )
    
    # 创建文档的collection
    def create_documents_collection(self, directory):
        # 获取所有文档文件
        file_names = [f for f in os.listdir(directory) if f.endswith(".csv")]
        file_paths = [os.path.join(directory, f) for f in file_names]
        
        for file_path, file_name in tqdm(zip(file_paths, file_names), total=len(file_names)):
            documents = []
            topic_id = file_name.split("_")[1].split(".")[0]  # 从文件名提取主题ID
            
            with open(file_path, mode="r", encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    documents.append(row['document'])
            
            try:
                # 为每个主题创建单独的集合
                collection = self.client.create_collection(
                    name=file_name.replace(".csv", ""),
                    embedding_function=self.embedding,
                    metadata={"hnsw:space": "ip"}
                )
                
                # 添加文档到集合
                collection.add(
                    documents=documents,
                    ids=[str(i) for i in range(len(documents))],
                    metadatas=[{"topic_id": topic_id}] * len(documents)
                )
            
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {str(e)}")



if __name__ == "__main__":
    # 初始化Chroma客户端
    client = chromadb.PersistentClient(path='chroma_db_bge_gpu')
    
    # embedding模型路径
    model_path = "/home/v_zhishunliu/bge-large-zh-v1.5"
    bge_ef_gpu = BGEEmbeddingFunctionGPU(model_name=model_path, device='cuda')
    
    # 创建集合
    # collection = client.get_or_create_collection(
    #     name="bge_gpu_collection",
    #     embedding_function=bge_ef_gpu,
    #     metadata={"hnsw:space": "ip"}
    # )
    
    # 创建集合管理器
    Collection = Create_vector_collection(client, bge_ef_gpu)
    # 创建主题集合
    # Collection.create_topic_collection("topic_keywords.csv")
    # 创建文档集合
    Collection.create_documents_collection("docs")



import chromadb

# 初始化Chroma客户端
client = chromadb.PersistentClient(path='chroma_db_bge_gpu')

# embedding模型路径
model_path = "/home/v_zhishunliu/bge-large-zh-v1.5"
bge_ef_gpu = BGEEmbeddingFunctionGPU(model_name=model_path, device='cuda')

# 列出所有集合（可选）
client.list_collections()

# 获取主题集合
collection = client.get_collection(
    name="documents_topic",
    embedding_function=bge_ef_gpu
)

# 执行查询
result = collection.query(
    query_texts=["微业贷提前结清是否可以在手机上操作？"],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)

print(result)
