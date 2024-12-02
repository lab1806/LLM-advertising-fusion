import gradio as gr
from huggingface_hub import InferenceClient
import json
import random
import re
from load_data import load_data
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import weaviate
import os
import torch
from tqdm import tqdm
import numpy as np
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 设置缓存目录
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

# Weaviate 连接配置

# 预训练模型配置
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# OpenAI 客户端
openai_client = None

def initialize_openai_client(api_key):
    global openai_client
    openai_client = OpenAI(api_key=api_key)

def extract_keywords(text):
    prompt = """
    你的任务是从用户的输入中提取关键词，特别是名词和形容词，输出关键词之间用空格分隔。例如：苹果 电脑 裤子 蓝色 裙。
    注意:
    1.不要重复输出关键词，如果输入内容较短，你可以输出少于五个关键词，但至少输出两个
    2.对于停用词不要进行输出，停用词如各类人称代词，连词等
    3.关键词应该严格是名词和形容词，不要输出动词等其他词性
    4.输出格式为关键词之间用空格分隔，例如：苹果 电脑 裤子 蓝色 裙
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"从下面的文本中提取五个名词或形容词词性的关键词，以空格分隔：例子：她穿着蓝色的裙子，坐在电脑前，一边吃苹果一边看着裤子的购物网站。 输出：苹果 电脑 裤子 蓝色 裙\n\n 文本：{text}"}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        # model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )

    keywords = response.choices[0].message.content.split(' ')
    return ','.join(keywords)

def initialize_weaviate_client():
    global weaviate_client
    retry_strategy = Retry(
        total=3,  # 总共重试次数
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # 需要重试的方法
        backoff_factor=1  # 重试间隔时间的倍数
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    timeout = 5
###########################################################
    WEAVIATE_API_KEY = ""
    WEAVIATE_URL = ""
###########################################################
    
    
    weaviate_auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
    
    def create_client():
        return weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate_auth_config,
            timeout_config=(timeout, timeout) 
        )
    
    try:
        weaviate_client = create_client()
    except Exception as e:
        print(f"连接超时，重新连接")
        weaviate_client = create_client()


def encode_keywords_to_avg(keywords, model, tokenizer, device):
    embeddings = []
    for keyword in tqdm(keywords):
        inputs = tokenizer(keyword, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
    avg_embedding = sum(embeddings) / len(embeddings)
    return avg_embedding

def encode_keywords_to_list(keywords, model, tokenizer, device):
    start_time = time.time()
    embeddings = []
    model.to(device)  
    for keyword in tqdm(keywords):
        inputs = tokenizer(keyword, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().tolist())
    end_time=time.time()
    print(f"Time taken for encoding: {end_time - start_time}")
    return embeddings


def get_response_from_db(keywords_dict, class_name):
    avg_vec = encode_keywords_to_avg(keywords_dict.keys(), model, tokenizer, device).numpy()
    response = (
        weaviate_client.query
        .get(class_name, ['keywords', 'summary'])
        .with_near_vector({'vector': avg_vec})
        .with_limit(1)
        .with_additional(['distance'])
        .do()
    )

    if class_name.capitalize() in response['data']['Get']:
        result = response['data']['Get'][class_name.capitalize()][0]
        return result['_additional']['distance'], result['summary'], result['keywords']
    else:
        return None, None, None
    
def get_candidates_from_db(keywords_dict, class_name,limit=3):
    embeddings= encode_keywords_to_list(keywords_dict.keys(), model, tokenizer, device)
    candidate_list=[]
    for embedding in embeddings:
        response = (
            weaviate_client.query
            .get(class_name, ['group_id','keyword_list','keyword', 'summary'])
            .with_near_vector({'vector': embedding})
            .with_limit(limit)
            .with_additional(['distance'])
            .do()
        )
        class_name=class_name[0].upper()+class_name[1:]

        if class_name in response['data']['Get']:
            results = response['data']['Get'][class_name]
            for result in results:
                candidate_list.append({
                    'distance': result['_additional']['distance'],
                    'group_id': result['group_id'],
                    'keyword_list':result['keyword_list'],
                    'summary': result['summary'],
                    'keyword': result['keyword']
                    
                })
    return candidate_list


triggered_keywords = {}

def first_keyword_match(keywords_dict,keyword_match_threshold=2):
    if not keywords_dict:
        return None,None
    data=load_data("train_2000_modified.json",2000)
    keywords=[dt['content'] for dt in data]
    max_matches=0
    index=0
    for i, lst in enumerate(keywords):
        list=lst.split(',')
        matches=sum(any(ad_keyword in keyword for keyword in keywords_dict.keys()) for ad_keyword in list)
        if matches>max_matches:
            max_matches=matches
            index=i
    if max_matches<=keyword_match_threshold:
        return None,None
    
    return data[index]['summary'],keywords[index]


######################################
# 创建和管理历史记录的辅助函数
def make_history_cache():
    history = gr.State([])  # 后端状态，用于存储历史记录
    history_cache = gr.Textbox(visible=False, elem_id="history_cache")  # 隐藏的前端缓存
    history_cache_update = gr.Button("", elem_id="elem_update_history", visible=False).click(
        lambda cache: json.loads(cache), inputs=[history_cache], outputs=[history]
    )
    return history, history_cache, history_cache_update

def chatbot_response(message, history, window_size, threshold, score_threshold,user_weight, triggered_weight,candidate_length,api_key):
    #初始化openai client
    initialize_openai_client(api_key)
    initialize_weaviate_client()
    #更新轮次，获取窗口历史
    current_turn = len(history) + 1

    combined_user_message = message
    combined_assistant_message = ""
    partial_response = ""
    for i in range(1, window_size + 1):
        if len(history) >= i:
            if i % 2 == 1:  # 奇数轮次，添加 assistant 的内容
                combined_assistant_message = " ".join([history[-i][1], combined_assistant_message]).strip()
            else:  # 偶数轮次，添加 user 的内容
                combined_user_message = " ".join([history[-i][0], combined_user_message]).strip()

    #提取关键词
    user_keywords = extract_keywords(combined_user_message).split(',')
    #获取关键词字典
    keywords_dict = {keyword: user_weight for keyword in user_keywords}
    
    # triggered_keywords = [] 
    
    #根据上下文轮数更新关键词列表长度
    max_size = 6 * window_size
    if len(keywords_dict) > max_size:
        keywords_dict = dict(list(keywords_dict.items())[-max_size:])

    if combined_assistant_message:
        assistant_keywords = extract_keywords(combined_assistant_message).split(',')
        for keyword in assistant_keywords:
            keywords_dict[keyword] = keywords_dict.get(keyword, 0) + 1

    for keyword in list(keywords_dict.keys()):
        if keyword in triggered_keywords and current_turn - triggered_keywords[keyword] < window_size:
            keywords_dict[keyword] = triggered_weight

    start_time = time.time()
    ad_summary,ad_keywords=first_keyword_match(keywords_dict)
    #关键词匹配命中
    end_time = time.time()
    print(f"Time taken for first keyword match: {end_time - start_time}")

    if ad_summary:
       
        brands=['腾讯','阿里巴巴','百度','京东','华为','小米','苹果','微软','谷歌','亚马逊']
        brand=random.choice(brands)
        ad_message = f"{message} <sep>品牌{brand}<sep>{ad_summary}"
        print(f"ad_sumamry: {ad_summary}")
        messages = [{"role": "system", "content": "请你将生活化、原汁原味的语言提炼出来，具有亲切感，类似于拉家常的方式推销商品，具有融洽的氛围和口语化的语言。请直接输出融合的对话文本。"}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": ad_message})

        for keyword in keywords_dict.keys():
            if any(ad_keyword in keyword for ad_keyword in ad_keywords.split(',')):
                triggered_keywords[keyword] = current_turn    
                  
    #关键词不中
    else:
        start_time=time.time()
        # distance, ad_summary, ad_keywords = get_response_from_db(keywords_dict, class_name="ad_DB02")
        #数据库索引，数据库关键词平均方式
        candidates=get_candidates_from_db(keywords_dict, class_name="Ad_DB10",limit=candidate_length)

        candidates.sort(key=lambda x:x['distance'])
        candidates=[candidate for candidate in candidates if candidate['distance']<threshold]

        print("----------------------------------------------------------------------")
        print(f"keywords:{keywords_dict.keys()}")
        print(f"candidates:{candidates[:5]}")

        #此时的候选集中所有元素都至少有一个关键词命中了
        #筛选后的候选集进行投票，选出被投票最多的一条
        #投中第一个元素加双倍权重
        
        group_scores={}
        if(candidates):
            for candidate in candidates:
                group_id=candidate['group_id']
                keyword = candidate['keyword']
                keyword_list = candidate['keyword_list'].split(',')

                # 检查 keyword 是否是 keyword_list 中的第一个元素
                if keyword in user_keywords:
                    if keyword == keyword_list[0]:
                        score = 6
                    else:
                        score = 2
                else:
                    if keyword == keyword_list[0]:
                        score = 3
                    else:
                        score = 1

                if keyword in triggered_keywords and current_turn - triggered_keywords[keyword] < window_size:
                    if(keyword == keyword_list[0]):
                        score = triggered_weight*3
                    else:
                        keywords_dict[keyword] = triggered_weight

                # 更新 group_scores 字典中的分数
                if group_id in group_scores:
                    group_scores[group_id] += score
                else:
                    group_scores[group_id] = score

    

        distance=1000
        if group_scores:
            max_group_id = max(group_scores, key=group_scores.get)
            max_score = group_scores[max_group_id]
            if(max_score>=score_threshold):
                distance,ad_summary,ad_keywords=[(candidate['distance'],candidate['summary'],candidate['keyword_list']) for candidate in candidates if candidate['group_id']==max_group_id][0]
                #触发->标记触发词
                for keyword in keywords_dict.keys():
                    if any(ad_keyword in keyword for ad_keyword in ad_keywords.split(',')):
                        triggered_keywords[keyword] = current_turn 

                print("ad_keywords: ", ad_keywords)
        if group_scores:
            sorted_group_scores = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
            print(f"group_scores: {sorted_group_scores}")
        
        end_time=time.time()
        print(f"Time taken for vecDB: {end_time - start_time}")

        if distance < 1000:
           pass

        else:
            messages = [{"role": "system", "content": "你是一个热情的聊天机器人。"}]
            for val in history:
                if val[0]:
                    messages.append({"role": "user", "content": val[0]})
                if val[1]:
                    messages.append({"role": "assistant", "content": val[1]})
            messages.append({"role": "user", "content": message})       
 

    if ad_summary:
        raw_initial_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            # model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
        )
        
        # 修改为通过属性访问内容
        initial_response = raw_initial_response.choices[0].message.content
    
        brands = ['腾讯', '阿里巴巴', '百度', '京东', '华为', '小米', '苹果', '微软', '谷歌', '亚马逊']
        brand = random.choice(brands)
        fusion_message = f"用户输入(上下文):\n{message}\n\n原始回复:\n{initial_response}\n\n广告信息:\n来自{brand}品牌：{ad_summary}"
        
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    
        print(f"fusion_message:   {fusion_message}")
    
        fusion_messages = [{"role": "system", "content": system_prompt}]
        fusion_messages.append({"role": "user", "content": fusion_message})
    
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            # model="gpt-3.5-turbo",
            messages=fusion_messages,
            stream=True  # 启用流式响应
        )
    else:
        messages = [{"role": "system", "content": "你是一个热情的聊天机器人。你的所有回复应该是简短的一段式回答，不要过于冗长。"}]
        # for val in history:
        #     if val[0]:
        #         messages.append({"role": "user", "content": val[0]})
        #     if val[1]:
        #         messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})
    
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            # model="gpt-3.5-turbo",
            messages=messages,
            stream=True  # 启用流式响应
        )

    
    # 处理流式响应
    print(f"triggered_keywords: {triggered_keywords}")
    partial_response = ""
    
        # Stream the response from OpenAI API
    for stream_response in response:
        try:
            choices = stream_response.choices
            if choices and hasattr(choices[0], 'delta'):
                delta = choices[0].delta
                if hasattr(delta, 'content'):
                    token = delta.content
                    partial_response += token
                    yield partial_response, history
                else:
                    print("Delta does not contain content attribute.")
            else:
                print("No valid choices or delta in stream response.")
        except Exception as e:
            print(f"Error processing stream_response: {e}")
    
    history.append([message, partial_response])
    yield partial_response, history

# Gradio UI:
def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("""
        <h1>大模型广告植入</h1>
        """)

        # Create history management
        history = gr.State([])  # Backend state for storing conversation history

        # User input and example questions
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(lines=5, label="Ask the AI:")
                gr.Examples(examples=[
                    "Write a poem in Cockney accent about why West Ham is great.",
                    "Write a poem about love.",
                    "Describe a sunset over the ocean."
                ], inputs=question)
                btn = gr.Button(value="Get Response")

            # AI real-time response box
            with gr.Column():
                answer = gr.Textbox(lines=15, label="AI Response:", interactive=False)

            # Chat history window (using chatbot component)
            with gr.Column():
                chat_history = gr.Chatbot(label="Conversation History")

        # Advanced settings section (unchanged)
        with gr.Accordion("Advanced Settings", open=False):
            window_size_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Window Size")
            distance_threshold_slider = gr.Slider(minimum=0.01, maximum=0.3, value=0.25, step=0.01, label="Distance Threshold")
            score_threshold_slider = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Score Threshold")
            user_weight_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="User Keyword Weight")
            triggered_weight_slider = gr.Slider(minimum=0, maximum=2, value=0.5, step=0.5, label="Triggered Keyword Weight")
            candidate_length_slider = gr.Slider(minimum=0, maximum=100, value=30, step=5, label="Number of Candidates")
            api_key_box = gr.Textbox(label="API Key", placeholder="Enter your OpenAI API key here")

        # Button click event
        btn.click(
            fn=chatbot_response,
            inputs=[
                question,
                history,  # Pass the current history state
                window_size_slider,  # Advanced settings inputs
                distance_threshold_slider,
                score_threshold_slider,
                user_weight_slider,
                triggered_weight_slider,
                candidate_length_slider,
                api_key_box  # Pass the API key
            ],
            outputs=[answer, chat_history]  # Outputs: AI response and updated chat history
        )

        # Queue to handle streaming
        demo.queue()
        demo.launch(share=True, debug=True)

# Launch Gradio UI
gradio_ui()

