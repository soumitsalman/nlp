import glob
import logging
from retry import retry
import tiktoken
from datetime import datetime
import re
import os
import json
from icecream import ic
from pymongo import UpdateOne
from pymongo import MongoClient

log = logging.getLogger("app")

MAX_TOKENS = 6144
# db = MongoClient(os.getenv("LOCAL_DB"), retryWrites=True)["trainingdata"]
encoding = tiktoken.get_encoding("cl100k_base")

current_time = lambda: datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
truncate = lambda text: encoding.decode(encoding.encode(text)[:MAX_TOKENS])

def batch_truncate(beans: list):
    tokenlist = encoding.encode_batch([bean['text'] for bean in beans], num_threads=os.cpu_count())
    tokenlist = [tokens[:MAX_TOKENS] for tokens in tokenlist]
    texts = encoding.decode_batch(tokenlist, num_threads=os.cpu_count())
    for bean, text in zip(beans, texts):
        bean["text"] = text
    return beans

def pad_current_time(items: list[dict], use_str = True):
    now = current_time() if use_str else datetime.now()
    [item.update({"collected": now}) for item in items]
    return items

# def save_data_to_db(items: list[dict], what: str):
#     if not items: return
#     updates = [UpdateOne(
#         filter = {"_id": item["_id"]},
#         update = {
#             "$set": item,
#             "$setOnInsert": {"collected_at": datetime.now()}
#         },
#         upsert = True
#     ) for item in items]
#     result = db[what].bulk_write(updates, ordered=False)
#     log.info(f"{result.upserted_count} upserted, {result.modified_count} modified")

# def port_db_to_file(what):
#     items = list(db[what].find())
#     save_data_to_file(items, what)

# def port_file_to_db(what):
#     for filename in os.listdir(".data"):
#         if what in filename:
#             with open(os.path.join(".data", filename), "r") as file:
#                 save_data_to_db(json.load(file), what)

def save_data_to_db(beans, db_name: str = "trainingdata"):
    client = MongoClient(os.getenv("LOCAL_DB_CONNECTION_STRING"))
    client.drop_database(db_name)
    db = client[db_name]
    for bean in beans:
        bean['url'] = bean['_id']
        bean['created'] = bean['updated'] = datetime.now()
    db.beans.insert_many(beans, ordered=False)

def save_data_to_file(items, what):
    if not items: return
    
    os.makedirs(".data", exist_ok=True)
    filename = ".data/"+re.sub(r'[^a-zA-Z0-9]', '-', f"{what}-{current_time()}")+".json"
    with open(filename, "w") as file:
        json.dump(pad_current_time(items), file)

def load_data_from_file_path(file_path):
    with open(file_path, "r") as file:
        items = json.load(file)
    return items

def load_data_from_directory(file_name_prefix, filter_func = lambda bean: True):
    beans = []
    for file_path in glob.glob(file_name_prefix):
        beans.extend(filter(filter_func, load_data_from_file_path(file_path)))
    return beans
 
def save_data_to_file_path(items, file_path):
    if not items: return
    with open(file_path, "w") as file:
        json.dump(items, file)

def save_data_to_directory(beans: list[dict], directory: str, file_name_prefix: str):
    os.makedirs(directory, exist_ok=True)
    batch_size = 1000
    for i in range(0, len(beans), batch_size):
        to_write = beans[i:i+batch_size]
        save_data_to_file_path(to_write, f"{directory}/{file_name_prefix}-{i}-{i+len(to_write)}.json")

def save_jsonl_to_directory(data: list[dict], directory: str, file_name_prefix: str):
    os.makedirs(directory, exist_ok=True)
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        with open(f"{directory}/{file_name_prefix}-{i}-{i+batch_size}.jsonl", "w") as file:
            file.writelines([json.dumps(row)+"\n" for row in data[i:i+batch_size]])

def port_data(from_file_pattern: str, to_directory: str, to_prefix: str):
    beans = load_data_from_directory(from_file_pattern)
    collected = int(datetime.now().timestamp())
    for bean in beans:
        bean['collected'] = collected
    save_data_to_directory(beans, to_directory, to_prefix)

def print_results(items, what):
   if not items: return
   for item in items:
       print("===========NEW ITEM===============")
       print(item['_id'])
       print(item.get(what))
       print("===========END ITEM===============")



total_beans = 0
def measure_output(func):
    async def wrapper(*args, **kwargs):
        global total_beans

        start = datetime.now()
        result = await func(*args, **kwargs)
        duration = datetime.now() - start
        
        total_beans += (len(result) if isinstance(result, list) else 0)
        print(f"[{datetime.now()}] {duration} | {total_beans}")
        return result
    return wrapper

