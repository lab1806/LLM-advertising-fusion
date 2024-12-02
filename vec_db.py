import weaviate
import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModel
import subprocess
import os
# 设置 Matplotlib 缓存目录为可写的目录
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
# 设置 Hugging Face Transformers 缓存目录为可写的目录
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

class_name = 'Lhnjames123321'
auth_config = weaviate.AuthApiKey(api_key="8wNsHV3Enc2PNVL8Bspadh21qYAfAvnK2ux3")
client = weaviate.Client(
  url="https://3a8sbx3s66by10yxginaa.c0.asia-southeast1.gcp.weaviate.cloud",
  auth_client_secret=auth_config
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("bert-base-chinese").to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def encode_sentences(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


def init_weaviate():


    file_path = 'data.json'
    sentence_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                sentence1 = data.get('response', '')
                sentence_data.append(sentence1)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue

    sentence_embeddings = encode_sentences(sentence_data, model, tokenizer, device)

    data = {'sentence': sentence_data,
            'embeddings': sentence_embeddings.tolist()}
    df = pd.DataFrame(data)

    with client.batch(batch_size=100) as batch:
        for i in range(df.shape[0]):
            print(f'importing data: {i + 1}/{df.shape[0]}')
            properties = {
                'sentence_id': i + 1,
                'sentence': df.sentence[i],
            }
            custom_vector = df.embeddings[i]
            client.batch.add_data_object(
                properties,
                class_name=class_name,
                vector=custom_vector
            )
    print('import completed')


def use_weaviate(input_str):
    query = encode_sentences([input_str], model, tokenizer, device)[0].tolist()
    nearVector = {
        'vector': query
    }

    response = (
        client.query
        .get(class_name, ['sentence_id', 'sentence'])
        .with_near_vector(nearVector)
        .with_limit(5)
        .with_additional(['distance'])
        .do()
    )
    print(response)
    results = response['data']['Get'][class_name]
    text_list = [result['sentence'] for result in results]
    return text_list

if __name__ == '__main__':
    init_weaviate()
    input_str = input("请输入查询的文本：")
    ans = use_weaviate(input_str)
    print("查询结果：", ans)
