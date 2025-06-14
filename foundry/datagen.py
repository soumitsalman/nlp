import asyncio
from datetime import datetime, timedelta
from itertools import chain
import json
import logging
import os
import random
import re
import torch
from gliner import GLiNER
from pymongo import MongoClient
from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from tqdm import tqdm
from azure.storage.blob import ContainerClient, BlobProperties
from datagen_prompts import *
from utils import *
from datacleaning import *

log = logging.getLogger("app")

load_dotenv()

#############################################
## COLLECT DATA FROM LARGER LANGUAGE MODEL ##
#############################################

# class LocalLLM:
#     model = None
#     tokenizer = None

#     def __init__(self, model_id: str):
#         from unsloth import FastLanguageModel
#         from transformers import AutoProcessor

#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_id,
#             max_seq_length=8192,
#             load_in_4bit=True
#         )
#         self.model = FastLanguageModel.for_inference(model)
#         self.tokenizer = AutoProcessor.from_pretrained(model_id)

#     def batch_generate_highlights(self, beans: list):
#         beans = batch_truncate(beans)

#         input_tokens = self.tokenizer.apply_chat_template(
#             [create_highlights_prompt(bean['text']) for bean in beans], 
#             add_generation_prompt=True, tokenize=True, padding=True, max_length=2512, truncation=True,
#             return_dict=True, return_tensors="pt").to("cuda")

#         outputs = self.tokenizer.batch_decode(
#             self.model.generate(**input_tokens, max_new_tokens=512, temperature=0.5, use_cache=True, do_sample=True))
        
#         for bean, output in zip(beans, outputs):
#             # <start_of_turn>model <end_of_turn>
#             # start = output.find("<start_of_turn>model")
#             # end = output[start:].find("<end_of_turn>")
#             # output = output[start:end].strip()
#             start = output.rfind("```json")
#             end = output.rfind("```")
#             bean['highlight_response'] = output[start:end+2]

#         return beans
    
#     def batch_generate_summaries(self, beans: list):
#         beans = batch_truncate(beans)

#         input_tokens = self.tokenizer.apply_chat_template(
#             [create_highlights_prompt(bean['text']) for bean in beans], 
#             add_generation_prompt=True, tokenize=True, padding=True, max_length=2512, truncation=True,
#             return_dict=True, return_tensors="pt").to("cuda")

#         outputs = self.tokenizer.batch_decode(
#             self.model.generate(**input_tokens, max_new_tokens=512, temperature=0.5, use_cache=True, do_sample=True))
        
#         for bean, output in zip(beans, outputs):
#             bean['summary_response'] = output

#         return beans

create_summary_prompt = lambda text: [
    {
        "role": "system",
        "content": SYSTEM_INST
    },
    {
        "role": "user",
        "content": SUMMARY_INST.format(input_text=text)
    }
]

create_short_summary_prompt = lambda text: [
    {
        "role": "system",
        "content": SYSTEM_INST
    },
    {
        "role": "user",
        "content": SHORT_SUMMARY_INST.format(input_text=text)
    }
]

class RemoteLLM:
    client = None

    def __init__(self, api_key: str, base_url: str, model_id: str):
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    # def generate_extracts(self, bean: dict):
    #     res = self.client.chat.completions.create(
    #         model=self.model_id, 
    #         messages=create_extracts_prompt(bean),
    #         max_tokens=1024,
    #         temperature=0.2,
    #         seed=666,
    #         response_format={"type": "json_object"}
    #     ).choices[0].message.content.strip()
        
    #     try:
    #         start, end = res.find("{"), res.rfind("}")
    #         if start >= 0 and end >= 0:
    #             res = res[start:end+1]
    #         bean.update(json.loads(res))
    #         bean['extract_response'] = res
    #         bean['collected'] = int(datetime.now().timestamp())
    #     except json.JSONDecodeError as e:
    #         ic(bean['_id'], res)
    #     return bean

    # @retry(tries=3, delay=DELAY)
    def generate_summary(self, bean: dict):
        res = self.client.chat.completions.create(
            model=self.model_id, 
            messages=create_summary_prompt(bean["text"]),
            max_tokens=512,
            temperature=0.4,
            frequency_penalty=0.3,
            seed=666
        )
        bean['summary_response'] = res.choices[0].message.content.strip()
        bean['collected'] = int(datetime.now().timestamp())
        return bean

    # @retry(tries=3, delay=DELAY)
    # def generate_highlights(self, bean: dict):
    #     res = self.client.chat.completions.create(
    #         model=self.model_id, 
    #         messages=create_highlights_prompt(bean["text"]),
    #         max_tokens=384,
    #         temperature=0.5,
    #         response_format={"type": "json_object"}
    #     )
    #     bean['highlight_response'] = res.choices[0].message.content
    #     bean['collected'] = int(datetime.now().timestamp())
    #     return bean

    # @measure_output
    # async def batch_generate_extracts_async(self, beans: list):
    #     beans = batch_truncate(beans)
    #     tasks = [asyncio.to_thread(self.generate_extracts, bean) for bean in beans]
    #     return await asyncio.gather(*tasks) 
    
    # @measure_output
    # async def batch_generate_highlights_async(self, beans: list):
    #     beans = batch_truncate(beans)
    #     tasks = [asyncio.to_thread(self.generate_highlights, bean) for bean in beans]
    #     return await asyncio.gather(*tasks)
    
    @measure_output
    async def batch_generate_summaries_async(self, beans: list):
        beans = batch_truncate(beans)
        tasks = [asyncio.to_thread(self.generate_summary, bean) for bean in beans]
        return await asyncio.gather(*tasks)

get_news_and_blogs = lambda db, skip, limit: list(db.beans.find(filter={"kind": {"$ne": "post"}}, skip=skip, limit=limit, projection={"url": 1, "text": 1}))
get_posts = lambda db, skip, limit: list(db.beans.find(filter={"kind": "post"}, skip=skip, limit=limit, projection={"url": 1, "text": 1}))

def download_raw_data(limit, dir):
    db = MongoClient(os.getenv("DB_CONNECTION_STRING"))["beansackV2"]
    beans = list(db.beans.find(
        filter = {
            "$expr": { "$gt": [{ "$strLenCP": "$text" }, 1200] }
        },
        projection = {
            "_id": 1,
            "text": 1,
            "created": {
                "$dateToString": { "format": "%Y-%m-%d %H:%M:%S", "date": "$created" }
            },
        },
        sort = { "collected": -1 },
        limit = limit
    ))
    save_data_to_directory(beans[:limit], dir, "raw")

# def generate_highlights_data(start_at, end_at, batch_size):
#     db = MongoClient(os.getenv("DB_CONNECTION_STRING"))["beansackV2"]
#     llm = RemoteLLM(model_id="Qwen/QwQ-32B")
    
#     for i in range(start_at, end_at, batch_size):
#         beans = get_news_and_blogs(db, i, batch_size)
#         beans = llm.batch_generate_highlights(beans)
#         utils.save_data_to_db(beans, "highlights")

# async def generate_highlights_data_async(start_at, limit, batch_size):
#     db = MongoClient(os.getenv("DB_CONNECTION_STRING"))["trainingdata"]
#     llm = RemoteLLM(api_key=os.getenv("DEEPINFRA_API_KEY"), base_url=os.getenv("DEEPINFRA_BASE_URL"), model_id="Qwen/QwQ-32B")
    
#     for i in range(0, limit, batch_size):
#         beans = beans = db.cleaned.find(
#             {
#                 "summary": { "$exists": True },
#                 "title": { "$exists": False }
#             },
#             projection={
#                 "_id": 1,
#                 "text": 1
#             },
#             skip=start_at+i, 
#             limit=batch_size
#         )
#         beans = await llm.batch_generate_highlights_async(beans)
#         save_data_to_file(beans, "highlights")

# async def generate_summary_data_async(start_at, limit, batch_size):
#     model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct" # "meta-llama/Llama-4-Scout-17B-16E-Instruct" # "Gryphe/MythoMax-L2-13b" # "nvidia/Llama-3.1-Nemotron-70B-Instruct" # "google/gemma-3-27b-it" # "meta-llama/Llama-4-Scout-17B-16E-Instruct" 
#     db = MongoClient(os.getenv("DB_CONNECTION_STRING"))["beansackV2"]
#     llm = RemoteLLM(api_key=os.getenv("DEEPINFRA_API_KEY"), base_url=os.getenv("DEEPINFRA_BASE_URL"), model_id=ic(model_id))
    
#     all_beans = list(db.beans.find(
#         filter={
#             "text": { "$regex": ".{1250,}" },
#             "collected": { "$gte": (datetime.now() - timedelta(days=30)) }
#         },
#         projection={
#             "_id": 1,
#             "text": 1
#         },
#         skip=start_at, 
#         limit=limit
#     ))

#     os.makedirs(".raw_data", exist_ok=True)
#     for i in range(0, ic(len(all_beans)), batch_size):
#         beans = await llm.batch_generate_summaries_async(all_beans[i:i+batch_size])
#         for bean in beans:
#             bean['summary'] = cleanup_summary(bean)
#         save_data_to_file_path(beans, f".raw_data/summaries-{i+start_at}-{i+start_at+len(beans)}.json")
#         print_results(beans, "summary")
#         ic(i+start_at, i+start_at+len(beans))

# 'NovaSky-AI/Sky-T1-32B-Preview' --> GOOD | $0.12
# "meta-llama/Llama-4-Scout-17B-16E-Instruct" --> Very GOOD | $0.10
# 'nvidia/Llama-3.1-Nemotron-70B-Instruct' --> GOOD | $0.12
# google/gemma-3-27b-it --> GOOD | $0.10
# Qwen/QwQ-32B --> terrible
async def run_generate_summaries_async(from_file_pattern: str, to_directory: str, to_prefix: str):
    llm = RemoteLLM(api_key=os.getenv("DEEPINFRA_API_KEY"), base_url=os.getenv("DEEPINFRA_BASE_URL"), model_id="google/gemma-3-27b-it")
    all_beans = random.sample(load_data_from_directory(from_file_pattern, lambda bean: "www.reddit.com/" in bean['_id']), 3)

    batch_size = 3
    for i in range(0, len(all_beans), batch_size):
        try:
            beans = await llm.batch_generate_summaries_async(all_beans[i:i+batch_size])
            print_results(beans, "summary_response")
            save_data_to_directory(beans, to_directory, to_prefix)
        except Exception as e:
            ic(e)
            print("failed", i, i+batch_size)


# 'NovaSky-AI/Sky-T1-32B-Preview' --> GOOD | $0.12
# "meta-llama/Llama-4-Scout-17B-16E-Instruct" --> Very GOOD | $0.10
# 'nvidia/Llama-3.1-Nemotron-70B-Instruct' --> Very GOOD | $0.12
# google/gemma-3-27b-it --> GOOD | $0.10
# Qwen/QwQ-32B --> GOOD
# async def run_generate_extracts_async(from_file_pattern: str, to_directory: str, to_prefix: str):
#     llm = RemoteLLM(api_key=os.getenv("DEEPINFRA_API_KEY"), base_url=os.getenv("DEEPINFRA_BASE_URL"), model_id="Qwen/QwQ-32B")
#     all_beans = random.sample(load_data_from_directory(from_file_pattern), 3)

#     batch_size = 3
#     for i in range(0, len(all_beans), batch_size):
#         try:
#             beans = await llm.batch_generate_extracts_async(all_beans[i:i+batch_size])
#             # print_results(beans, "extract_response")
#             save_data_to_directory(beans, to_directory, to_prefix)
#         except Exception as e:
#             ic(e)
#             print("failed", i, i+batch_size)

  
def run_blob_tagging():
    container_client = ContainerClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING"), 
        "trainingdata"
    )
    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob.name)
        if 'cleaned' in blob.name: 
            blob_client.set_blob_metadata({"type": "CLEANED"})
            blob_client.set_blob_tags({"type": "CLEANED"})
        elif 'dataset' in blob.name: 
            blob_client.set_blob_metadata({"type": "DATASET"})
            blob_client.set_blob_tags({"type": "DATASET"})
        else: 
            blob_client.set_blob_metadata({"type": "RAW"})
            blob_client.set_blob_tags({"type": "RAW"})


def run_field_transform(beans, field: str, transform_func = lambda x: x):
    for bean in beans:
        if field in bean:
            if field in bean: bean[field] = transform_func(bean[field])
    return beans

def port_training_data_from_prod():
    from concurrent.futures import ThreadPoolExecutor

    prod = MongoClient(os.getenv('MONGODB_CONN_STR'))["test"]
    local = MongoClient("mongodb://localhost:27017/")["trainingdata"]

    filter = {
        "gist": { 
            "$exists": True,
            "$regex": re.compile(r'^[UP]:')
        }
    }
    projection = {"_id": 1, "gist": 1, "content": 1, "entities": 1, "regions": 1}    

    BATCH_SIZE = 1000
    def port(skip: int):
        beans = list(prod.beans.find(filter=filter, projection=projection, skip=skip, limit=BATCH_SIZE))
        if not beans: return

        for bean in beans:
            bean['gist'] = bean['gist'].replace('\n', '').strip()
            if not bean['gist'].endswith(";"): bean['gist'] += ";"
            bean['gist'] = re.sub(r'[UCS]:[^;]+;', '', bean['gist'])
            bean['ped_digest'] = re.sub(r'[NR]:[^;]+;', '', bean['gist']).strip()

            er_digest = ""
            if bean.get('entities'): er_digest += "N:"+("|".join(bean['entities']) if isinstance(bean['entities'], list) else bean['entities'])+";"
            if bean.get('regions'): er_digest += "R:"+("|".join(bean['regions']) if isinstance(bean['regions'], list) else bean['regions'])+";"
            if er_digest: bean['er_digest'] = er_digest

        local.beans.insert_many(beans, ordered=False)
        ic(skip)
       
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(port, range(0, 449000, BATCH_SIZE))

# nuner_model = None
def load_gliner_model(model_name = "numind/NuNerZero_span"):
    # global nuner_model
    model = GLiNER.from_pretrained(model_name, torch_dtype=torch.qint8, max_length=384)
    # if ic(torch.cuda.is_available()): model = model.cuda()
    return model

NAMES = ['organization', 'company', 'person', 'product', 'initiative', 'project', 'programming_language', 'vulnerability', 'disease'] 
REGIONS = ['geographic_region', 'city', 'country', 'continent', 'state', 'park']
LABELS = NAMES+REGIONS
get_names = lambda result: list({res['text'] for res in result if res['label'] in NAMES})
get_regions = lambda result: list({res['text'] for res in result if res['label'] in REGIONS})
def extract_entities(model: GLiNER, texts: list[str]):
    with torch.no_grad():
        results = model.batch_predict_entities(texts, LABELS, threshold=0.6)
    names = list(map(get_names, results))
    regions = list(map(get_regions, results))
    return names, regions

def fix_names_and_regions_in_training_data():
    model = load_gliner_model("numind/NuNerZero_span")
    db = MongoClient("mongodb://localhost:27017/")["trainingdata"]
    filter = {
        "ped_digest": { "$exists": True },
        "er_digest": { "$exists": True },
        'gist_v2': { "$exists": False }
    }
    projection = {
        "_id": 1, 
        "er_digest": 1, 
        "ped_digest": 1
    }

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))

    with tqdm(total=db.beans.count_documents(filter=filter), desc="Fixed", unit="bean") as pbard:
        while db.beans.count_documents(filter=filter):
            beans = list(db.beans.find(filter=filter, projection=projection, limit=BATCH_SIZE))
            entities, regions = extract_entities(model, [bean['er_digest'] for bean in beans])

            updates = []
            for bean, ns, rs in zip(beans, entities, regions):
                update = {
                    'entities_v2': ns if ns else None,
                    'regions_v2': rs if rs else None,
                }
                er_digest = ""
                if ns: er_digest += "N:"+("|".join(ns))+";"
                if rs: er_digest += "R:"+("|".join(rs))+";"
                update['er_digest'] = er_digest if er_digest else None
                update['gist_v2'] = bean['ped_digest'] + er_digest
                
                updates.append(UpdateOne(filter={"_id": bean["_id"]}, update={"$set": update}))
            pbard.update(db.beans.bulk_write(updates, ordered=False).modified_count)

def clean_compressed_digest(digest: str):
    if not digest: return digest

    results = {"P:": [], "E:": [], "D:": [], "N:": [], "R:": []}
    current_pos = 0
    while current_pos < len(digest):
        key = digest[current_pos: current_pos+2]
        if key in results:
            next_key_pos = [digest.find(";"+next_key, current_pos+2) for next_key in results.keys()]
            end = min([pos for pos in next_key_pos if pos>-1], default=len(digest))
            if digest[end-1] == ';': ext = digest[current_pos+2: end-1]
            else: ext = digest[current_pos+2: end]

            results[key].extend(chain(*(item.strip().split(';') for item in ext.strip().split("|"))))
            current_pos = end

        current_pos += 1

    digest = ""    
    for key, value in results.items():
        if not value: continue
        digest += key+"|".join(v.strip() for v in value)+";"
        
    return digest

def clean_compressed_digests_in_training_data():
    FILTER = {
        # 'gist_v2': { "$exists": True }
    }
    PROJECT = {
        "_id": 1,
        "gist": 1,
        "gist_v2": 1,
        "ped_digest": 1       
    }
    BATCH_SIZE = 1000

    db = MongoClient("mongodb://localhost:27017/")["trainingdata"]
    beans = list(db.beans.find(filter=FILTER, projection=PROJECT))
    ic(len(beans))
    for bean in beans:
        bean['gist_v2'] = clean_compressed_digest(bean.get('gist_v2') or bean.get('gist'))
        bean['ped_digest'] = clean_compressed_digest(bean['ped_digest'])
    # def clean_chunk(start: int):
    #     beans = list(db.beans.find(filter=FILTER, skip=start, limit=BATCH_SIZE, projection=PROJECT))
        # for bean in beans:
        #     bean['gist_v2'] = clean_compressed_digest(bean.get('gist_v2') or bean.get('gist'))
        #     bean['ped_digest'] = clean_compressed_digest(bean['ped_digest'])
    
    db.beans.bulk_write(
        [
            UpdateOne(
                {"_id": bean['_id']}, 
                {
                    "$set": {
                        'gist_v2': bean['gist_v2'], 
                        'ped_digest': bean['ped_digest']
                    }
                }
            ) for bean in beans
        ],
        ordered=False
    )
    # with ThreadPoolExecutor(max_workers=os.cpu_count()<<2) as exec:
    #     exec.map(clean_chunk, range(0, ic(db.beans.count_documents(FILTER)), BATCH_SIZE))


def create_data_from_compressed_digests():
    FILTER = {
        'gist_v2': { "$exists": True }
    }
    PROJECT = {
        "_id": 1,
        "content": 1,   
        "ped_digest": 1,
        "er_digest": 1,
        "gist_v2": 1,
        "entities_v2": 1,
        "regions_v2": 1
    }

    db = MongoClient("mongodb://localhost:27017/")["trainingdata"]
    beans = list(db.beans.find(filter=FILTER, projection=PROJECT))
    ic(len(beans))
    for bean in beans:
        bean['gist'] = bean.pop('gist_v2')
        bean['entities'] = bean.pop('entities_v2', None) or None
        bean['regions'] = bean.pop('regions_v2', None) or None

    save_data_to_directory(
        beans, 
        "./foundry/data/compressed-digests",
        "compressed"
    )
    
def create_dataset_from_compressed_digests():
    # FILTER = {
    #     'gist_v2': { "$exists": True }
    # }
    # PROJECT = {
    #     "_id": 1,
    #     "content": 1,    
    #     "gist_v2": 1        
    # }

    # db = MongoClient("mongodb://localhost:27017/")["trainingdata"]
    # beans = list(db.beans.find(filter=FILTER, projection=PROJECT))
    # for bean in beans:
    #     bean['input'] = bean.pop('content')
    #     bean['output'] = bean.pop('gist')
    beans = utils.load_data_from_directory("./foundry/data/compressed-digests/*.json")
    beans = list(map(lambda bean: {'input': bean['content'], 'output': bean['gist']}, beans))
    ic(len(beans))
    save_jsonl_to_directory(beans, "./foundry/.dataset", "dataset")

if __name__ == "__main__":
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect() 
    # clean_compressed_digests_in_training_data()
    # create_data_from_compressed_digests()
    create_dataset_from_compressed_digests()
    # digests = [
    #     "P:Capital One miles can be redeemed for gift cards and cash back, but travel redemptions offer the most value;5 best ways to use Capital One Miles: Flights, Hotels, Other Travel Rewards, Cash Back, Gift Cards;Transferring miles to travel partners can yield more value than standard redemption rates;Capital One occasionally offers transfer bonuses to travel partners;Redeeming miles for cash back, gift cards, online shopping, or experiences typically yields less value than travel redemptions;Capital One miles transfer at a one-to-one ratio to most hotel and airline partners;E:Capital One updated fact-checking process;Ben Walker, CEPF, CFEI® authored the article;Mindy Woodall edited the article;Article updated June 6, 2025;Capital One miles can be used for flights, hotels, and car rentals;Amusement park tickets may be redeemable as travel through third-party sites like Undercover Tourist;D:Standard Capital One Travel redemption value is $0.01 per mile;Business class flight from New York to Paris costs $4,286.31 or 259,000 miles;Wyndham-Vacasa partnership charges 15,000 or 30,000 points per bedroom per night;Cash back redemption rate is 0.5 cent per mile;Gift card redemption rates range from 0.8 to 1 cent per mile;Capital One Venture Rewards Credit Card offers a 75,000 miles welcome bonus;Capital One Venture X Rewards Credit Card offers a $300 Capital One Travel credit annually;Paris;New York;Wyndham;Vacasa;Airbnb;VRBO;Universal Studios;Disneyland;Amazon;Apple;British Airways;Air Canada;Air France KLM;TAP Air Portugal;Singapore Airlines;JetBlue;Emirates;Qantas;EVA Air;Virgin Red;Accor;Choice;Turkish Airlines;Cathay Pacific;Finnair;Avianca;Ben Walker;Mindy Woodall;Capital One Travel;Expedia;Priceline;Air Canada;Air France KLM;British Airways;Wyndham Rewards;Undercover Tourist;Bank of America;N:Mindy Woodall|Ben Walker;R:Paris;",
    #     """P:Potential EU youth mobility scheme|Starmer signals openness to deal|Scheme not freedom of movement|Concerns over uncapped migration|Labour's "red line" on freedom of movement|Agreement expected in principle only|EU seeks up to 4yr stays, UK 1-2yr|Prior Brexit offered similar rights|Existing YMS with Australia, Japan etc|Visas subject to quotas|Tory govt previously rejected EU offer|Khan wants consistent messaging|Lib Dems welcome scheme|Brexit made UK study less attractive|Fishing rights also on agenda|Starmer focuses on economic benefits;E:Summit between UK & EU on Monday|Badenoch criticizes scheme as "back door" free movement|Tice calls scheme "thin end" of free movement|Thomas-Symonds confirms consideration of scheme|Rejection of previous EU offer by Tory govt|Khan comments on mixed messaging;D:18-30 age range considered in previous EU offer|18-35 age range for UK-Australia YMS|100-42000 visa quotas for existing schemes|Since Aug 2021, EU students face international fees;N:Keir Starmer|Laura Kuenssberg|Times|BBC News|EU|James MacCleary|Kemi Badenoch|Richard Tice|Sadiq Khan|Nick Thomas-Symonds;R:Australia|Uruguay|UK|South Korea|Japan|London;""",
    #     """P:Costco shopping tips;P:Smart spending;P:Bulk buying;P:Pantry staples;P:Rewards card;E:Article update;E:Fact-checking process;E:Card application;E:Savings on groceries;E:Summer deals;E:Father's Day promotions;D:2% cash rewards;D:$200 cash bonus;D:0% intro APR;D:$14.99 (Goldfish Crackers);D:$15.99 (Kirkland Trail Mix);D:$13.39 (Kirkland Pizza);D:$17.99 (Kirkland Tuna);D:$14.99 (Seeds of Change Quinoa);D:$6.19 (Cosmic Crisp Apples);D:$5.84 (Rotisserie Chicken);D:$11.99 (Kirkland Shampoo);N:Goldfish Crackers|Jenny Cohen|Costco|Sara Albertelli;R:United States;"""
    # ]   
    # list(map(clean_compressed_digest, digests))
    
    # fix_names_and_regions_in_training_data()
    # port_training_data_from_prod()
    # download_raw_data(100000, "raw_data")
    # asyncio.run(run_generate_extracts_async(".raw_data/raw-*.json", ".generated", ".extracts"))
    # asyncio.run(run_generate_summaries_async("raw_data/raw-*.json", ".generated", "summaries"))

    # run_cleanup()
    # run_datarow_creation()
    # run_blob_tagging()
    # generate_highlights_data(0, 10000, 200)
    # asyncio.run(generate_highlights_data_async(0, 5000, 200))
    # run_cleanup()
       
    # beans = utils.load_data_from_directory("data/cleaned-*.json")
    # beans = list(filter(lambda bean: all(field in bean for field in ['summary', 'title', 'names']), beans))
    # run_field_transform(beans, 'summary', datacleaning.cleanup_markdown)
    # save_data_to_db(beans, "trainingdata")
    # run_datarow_creation("data/cleaned-*.json", ".dataset", "dataset")
    
    # asyncio.run(generate_summary_data_async(0, 5000, 200))
    # asyncio.run(v2_generate_summary_data_async())
    # v2_generate_summary_data_async()
    # run_port_data("data/cleaned-*.json", ".merged", "cleaned")
    # run_port_data(".raw_data/raw-*.json", "data", "cleaned")
