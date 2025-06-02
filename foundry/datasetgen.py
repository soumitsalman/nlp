import json
import utils
from icecream import ic

##########################################
######### PREPARE DATASET/JSONL ##########
##########################################

SUMMARY_MARKDOWN_TEMPLATE = """TASK: Rewrite the article/post text using less than 250 words.

ARTICLE/POST:
{input_text}
"""

INSTRUCTION_TEMPLATE = """TASKS:
{tasks}

RESPONSE FORMAT: 
Response MUST be a json object of the following structure
```json
{{
{fields}
}}
```

ARTICLE/POST:
{input_text}
"""

TASKS_MAP = {
    "summary": "Rewrite the article/post using less than 250 words. This will be the 'summary'.",
    "highlights": "Extract top 3 sentences that are most relevant to the main points and core meaning of the entire article/post. These will be the 'highlights'.",
    "title": "Create a one sentence gist of the article/post. This will be the 'title'.",
    "names": "Extract names of the top 1 - 4 people, products, companies, organizations, or stock tickers mentioned in the article/post that influence the content. These will be the 'names'.",
    "domains": "Identify 1 or 2 domains that the subject matter of the article/post aligns closest to, such as: Cybersecurity, Business & Finance, Health & Wellness, Astrophysics, Smart Automotive, IoT and Gadgets, etc. These will be the 'domains'."
}
FIELDS_MAP = {
    "summary": "\"summary\": string",
    "title": "\"title\": string",
    "highlights": "\"highlights\": [string, string, string]",
    "names": "\"names\": [string, string, string, string]",
    "domains": "\"domain\": [string, string]"
}

def create_summary_markdown_row(bean:dict, field = 'summary'):
    return {
        "_id": bean['_id'],
        "input": bean['text'],
        "fields": ['summary_markdown'],
        "output": bean[field],
        "messages": [
            {
                "role": "system",
                "content": "response_format:markdown"
            },
            {
                "role": "user",
                "content":  SUMMARY_MARKDOWN_TEMPLATE.format(input_text=bean['text'])
            },
            {
                "role": "assistant",
                "content": bean[field]
            }
        ],
        "collected": bean['collected']
    }

def create_one_row(bean:dict, fields: list[str]):
    # create the instruction
    instruction = INSTRUCTION_TEMPLATE.format(
        tasks="\n".join(["  - "+TASKS_MAP[field] for field in fields]),
        fields=",\n".join(["    "+FIELDS_MAP[field] for field in fields]),
        input_text=bean['text']
    )
    output = json.dumps({field: bean[field] for field in fields}, indent=4)

    # create the datarow
    return {
        "_id": bean['_id'],
        "input": bean['text'],
        "fields": fields,
        "output": output,
        "messages": [
            {
                "role": "system",
                "content": "response_format:json_object"
            },
            {
                "role": "user",
                "content":  instruction
            },
            {
                "role": "assistant",
                "content": output
            }
        ],
        "collected": bean['collected']
    }
    

MAX_SUMMARY_LENGTH = 250
within_summary_length = lambda summary, text: summary and (len(summary) < len(text)) and len(summary.split()) <= MAX_SUMMARY_LENGTH
def create_datarows(beans):
    datarows = []   
    for bean in beans:
        # if 'summary' in bean:
        #     datarows.append(create_one_row(bean, ['summary']))
        #     datarows.append(create_summary_markdown_row(bean, 'summary'))

        # NOTE: temporarily disabling this
        # if 'highlights' in bean:
        #     datarows.append(create_one_row(bean, ['highlights']))

        # if 'title' in bean:
        #     datarows.append(create_one_row(bean, ['title']))

        # if 'names' in bean:
        #     datarows.append(create_one_row(bean, ['names']))

        # if 'domains' in bean:
        #     datarows.append(create_one_row(bean, ['domains']))

        # if all(key in bean for key in ['summary', 'highlights', 'title', 'names', 'domains']):
        #     datarows.append(create_one_row(bean, ['summary', 'highlights', 'title', 'names', 'domains']))

        if all(key in bean for key in ['summary', 'title', 'names', 'domains']):
            datarows.append(create_one_row(bean, ['summary', 'title', 'names', 'domains']))

        # if all(key in bean for key in ['highlights', 'names', 'domains']):
        #     datarows.append(create_one_row(bean, ['highlights', 'names', 'domains']))

        # if all(key in bean for key in ['title', 'names', 'domains']):
        #     datarows.append(create_one_row(bean, ['title', 'names', 'domains']))

        # NOTE: temporarily disabling this
        # if all(key in bean for key in ['names', 'domains']):
        #     datarows.append(create_one_row(bean, ['names', 'domains']))

        # if all(key in bean for key in ['title', 'names']):
        #     datarows.append(create_one_row(bean, ['title', 'names']))

        # if all(key in bean for key in ['title', 'domains']):
        #     datarows.append(create_one_row(bean, ['title', 'domains']))

        # if all(key in bean for key in ['summary', 'title']):
        #     datarows.append(create_one_row(bean, ['summary', 'title']))

        # if all(key in bean for key in ['summary', 'highlights']):
        #     datarows.append(create_one_row(bean, ['summary', 'highlights']))

        # if all(key in bean for key in ['summary', 'names']):
        #     datarows.append(create_one_row(bean, ['summary', 'names']))

        # if all(key in bean for key in ['summary', 'domains']):
        #     datarows.append(create_one_row(bean, ['summary', 'domains']))

    return datarows

def run_datarow_creation(from_file_pattern: str, to_directory: str, to_prefix: str):
    beans = utils.load_data_from_directory(from_file_pattern, lambda bean: all(field in bean for field in ['summary', 'title', 'names', 'domains']))
    ic(len(beans))
    datarows = create_datarows(beans)
    ic(len(datarows))
    utils.save_jsonl_to_directory(datarows, to_directory, to_prefix)