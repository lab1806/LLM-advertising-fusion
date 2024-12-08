---
title: LLM Advertising fusion
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

# LLM Advertising Embedding Framework


## **Abstract**  

In this paper, we propose a large model advertisement embedding framework, which aims to solve the problem of how to better embed advertisements into real-time conversations driven by large language models (LLMs) by dynamically integrating personalized advertisements into the conversations. Unlike traditional recommender systems that rely on static user behaviour data, the system proposed in this paper uses advanced semantic analysis to extract contextual features from user inputs, and employs the BAAI/bge-large-zh-v1.5 model for high-dimensional vectorisation to capture precise semantic relationships. At the same time, we use a vector database based on Weaviate for efficient and accurate ad retrieval, and design a multi-level scoring mechanism to ensure that the selected ads are contextually relevant and match user intent. We also seamlessly integrated adverts into session responses, maintaining fluidity and consistency without disrupting the user experience.





## **How to Run**

This project integrates OpenAI's GPT-4 API and Weaviate Cloud's vector database for intelligent advertisement embedding. Follow the steps below to set up and run the application.


### Step 1: Clone the Repository
Download the project to your local machine:
```bash
git clone https://github.com/lab1806/LLM-advertising-fusion.git
cd LLM-advertising-fusion
```


### Step 2: Install Dependencies
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Weaviate Cloud
1. Go to the [Weaviate Cloud Console](https://console.weaviate.cloud).
2. Click **Create a Cluster** and choose the **Sandbox** environment.
3. Provide a **Cluster Name** and select a version (any version is acceptable).
4. Click **Create** and wait for the cluster to initialize.
5. Once the cluster is ready, copy the **Cluster URL**.

### Step 4: Upload Data to Weaviate
0. Install anyio and weaviate:
   ```python
   pip install anyio
   pip install weaviate-client
   ```
1. Paste the copied **Cluster URL and API key** into the `import_data.py` script:
   ```python
   URL = ""
   APIKEY = ""
   ```
   Retrieve your Weaviate API Key and Weaviate Instance URL from the Weaviate Cloud Console.
2. **Run import_data.py**
   Models are required for encoding when uploading data. GPUs are recommended.

3. Update the WEAVIATE_API_KEY and WEAVIATE_URL variables in the app.py file (lines 81–82) with your API key and instance URL:
    ```python
    WEAVIATE_API_KEY = ""
    WEAVIATE_URL = ""
    ```
4. Run the following command to upload the provided dataset into the Weaviate Cloud database:
   ```bash
   python fetch_from_database.py
   ```

### Step 5: Run the Application
Start the main application:
   ```bash
   python app.py
   ```
The entire program is built by gradio and requires a simple cpu to support the run.

### Step 6: Modify Parameters
After launching the application, the front-end interface allows you to enter text, view response history and adjust advanced settings. These settings can be modified directly through the interface to customise the behaviour of the application:

#### Adjustable parameters
1. **Window Size**: Determines the size of the context window used to analyse user input. 
  Default value: 3 (range: 1-5).
  
2. **Distance Threshold**: Sets the minimum similarity score required to retrieve adverts. 
  Default value: 0.25 (range: 0.01-0.3).

3. **Score Threshold**: Specifies the minimum score to consider an advert as a match. 
  Default value: 3 (range: 1-20).

4. **User Keyword Weight**: Adjusts the weight of user-supplied keywords in the scoring process. 
  Default: 2 (Range: 1-5).

5. **Trigger Keyword Weight**: Controls the weight of previously triggered keywords to avoid duplicates. 
  Default: 0.5 (Range: 0-2).

6. **Number of Candidate Ads**: Set the maximum number of candidate ads to retrieve. 
  Default value: 30 (Range: 0-100).

7. **API Key**: Text box to enter the OpenAI API key. This allows the application to connect to GPT-4. To be on the safe side, we have protected the api key from being violated, so you will need to refill the api key each time the interface is refreshed. api key is necessary for this code to support the program calls as well as run. **Please be sure to place the api key in the box ahead of time. **

For further help, consult the documentation or check the log files for debugging.
