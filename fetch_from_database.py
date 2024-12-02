import weaviate
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
# 设置 Matplotlib 的缓存目录 
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib' 
# 设置 Hugging Face Transformers 的缓存目录 
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' 
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True) 
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True) 
auth_config = weaviate.AuthApiKey(api_key="8wNsHV3Enc2PNVL8Bspadh21qYAfAvnK2ux3")

# 初始化 Weaviate 客户端
#####################################################################
database_client = weaviate.Client(
  url="", 				#use your own url
  auth_client_secret=auth_config
)
class_name="Data" #
#####################################################################

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")


def encode(sentences, model, tokenizer):
    # 使用 BERT 计算每个句子的向量
    model.eval()  # 切换到评估模式
    embeddings = []

    with torch.no_grad():  # 禁用梯度计算
        for sentence in sentences:
            # 对输入句子进行编码
            print(sentence)
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            print(inputs)
            inputs = {key: value for key, value in inputs.items()}
            # print(inputs)
            # 获取模型输出
            outputs = model(**inputs)
            # 计算句子的向量（可以尝试其他聚合方法，如mean或pooling）
            embedding = outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')

            embeddings.append(embedding)
    # 将结果合并为一个二维数组
    return np.vstack(embeddings)


def insert_keywords_to_weaviate(database_client, class_name, keywords, summaries, avg_embeddings):
    # 批量插入数据到 Weaviate
    with database_client.batch(batch_size=100) as batch:
        for i, (keyword, summary, avg_embedding) in enumerate(zip(keywords, summaries, avg_embeddings)):
            vector = avg_embedding.tolist()
            properties = {
                'keywords': keyword,
                'summary': summary  # 存储的属性
            }
            print(f'Inserting: {keyword} with summary: {summary}')  # Debug info
            batch.add_data_object(
                properties,
                class_name=class_name,
                vector=vector
            )
    print('Insertion completed')

def init_database(database_client, class_name):
    #读取数据文件[[]]
    dataset = []
    with open('train_2000_modified.json', 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    keywords=[item['content'] for item in dataset if 'content' in item]
    summaries=[item['summary'] for item in dataset if 'summary' in item]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")

    keywords_avg_embeddings =[]
    #计算每一组关键词的平局embedding
    for lst in keywords:
        lst = lst.split(',')
        embeddings = encode(lst, model, tokenizer)
        avg_embedding = embeddings.mean(axis=0)
        keywords_avg_embeddings.append(avg_embedding)

    insert_keywords_to_weaviate(database_client, class_name, keywords, summaries, keywords_avg_embeddings)


def fetch_summary_from_database(query_keywords,classname):

    keyword_embeddings=[]
    for keyword in query_keywords:
        keyword_embedding=encode([keyword], model, tokenizer)
        keyword_embeddings.append(keyword_embedding)

    avg_embedding = np.mean(keyword_embeddings, axis=0)
    response = (
        database_client.query
        .get(class_name, ['keywords', 'summary'])  # 查询返回的字段
        .with_near_vector({'vector': avg_embedding})  # 使用向量进行检索
        .with_limit(1)  # 返回前5个结果
        .with_additional(['distance'])  # 返回距离信息
        .do()
    )
    print(response)

    top_distance = response['data']['Get'][class_name][0]['_additional']['distance']
    top_keywords_list=response['data']['Get'][class_name][0]['keywords']
    top_summary = response['data']['Get'][class_name][0]['summary']
    
    return top_distance,top_keywords_list,top_summary

if __name__ == '__main__':
    init_database()
