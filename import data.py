import weaviate
import pandas as pd
import os
import subprocess 
import weaviate.classes as wvc
import torch
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm


# APIKEY = "GSrb56ihk23QWvxpeNVC2u10OpXk0yiNnW40"
# URL = "https://mbnypastqjuv4u9f2v7jzw.c0.us-west3.gcp.weaviate.cloud"
APIKEY = "7NyxjWBxOPaPYdEOy6JVVL5lSf2hoGABwzBw"
URL = "https://b1tlsbgqqjq3rftfz7a5jw.c0.europe-west3.gcp.weaviate.cloud"
# Connect to a WCS instance
# client = weaviate.connect_to_wcs(
#     cluster_url=URL,
#     auth_credentials=weaviate.auth.AuthApiKey(APIKEY))
auth_config = weaviate.AuthApiKey(api_key=APIKEY)

client = weaviate.Client(
  url=URL,
  auth_client_secret=auth_config
)


# Connection to Weaviate
print(client.is_ready())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name="BAAI/bge-large-zh-v1.5"
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

WEAVIATE_API_KEY = APIKEY
WEAVIATE_URL = URL

weaviate_auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
weaviate_client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate_auth_config)

def embed_keyword_tolist(keyword):
    inputs = tokenizer(keyword, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten().tolist()

keywords_data = []
summary_data = []
keywords_embeddings = []

file_path = 'merged_data_single_line.json'
insert_num=2000
with open(file_path, 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        if count >= insert_num:
            break
        try:
            data = json.loads(line.strip())
            keywords = data.get('keywords', '')
            summary= data.get('summary', '')
            keywords_data.append(keywords)
            summary_data.append(summary)

            count += 1

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue
    keywords_data = [keywords.split(',') for keywords in keywords_data]        
# print(f"keywords_data: {keywords_data}")
# print(f"summary_data: {summary_data}")

# keywords_embedding2D=[]
# for keyword_list in keywords_data:
#     keywords_list_embeddings=[]
#     #embed每一个词并拼接在一起
#     for keyword in keyword_list:
#         keyword_embedding=embed_keyword_tolist(keyword)
#         keywords_list_embeddings.append(keyword_embedding)
#     keywords_embedding2D.append(keywords_list_embeddings)

#keywords数据条数*每条关键词个数*关键词embedding维度

#重新按照关键词的embedding展开原有数据，每个数据都要重复对应自己的每一个关键词
expanded_data = []
for i, keyword_list in enumerate(tqdm(keywords_data)):
    for keyword in keyword_list:
        expanded_data.append({
            'group_id': i,
            'original_summary': summary_data[i],
            'keyword_list':','.join(keyword_list),
            'keyword': keyword,
            'embedding': embed_keyword_tolist(keyword)
        })
  

df = pd.DataFrame(expanded_data)
df=df.astype({'group_id': 'int'})

print(df[:15])
with client.batch(batch_size=200) as batch:
    for i in tqdm(range(df.shape[0])):
        try:
            print(f'importing data: {i + 1}/{df.shape[0]}')
            properties = {
                'data_id': i + 1,
                'group_id': int(df.group_id[i]),
                'keyword_list':df.keyword_list[i],
                'keyword': df.keyword[i],
                'summary': df.original_summary[i]
            }
            custom_vector =np.array(df.embedding[i])
            client.batch.add_data_object(
                properties,
                class_name='Ad_DB10',
                vector=custom_vector
            )
        except Exception as e:
            print(f"Error importing data: {e}")
            continue

print('import completed')

name='lk'
