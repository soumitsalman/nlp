from datetime import datetime
import os
import re
import utils
import glob
from icecream import ic

ENDING_CHARS = ['.', '?', '!']
START_ITEM = "===========NEW ITEM==============="
END_ITEM = "===========END ITEM==============="
URL_PREFIXES = ["https://", "http://"]


def check_summary(items):
    total, correct = 0, 0
    for item in items:
        summary = item.get("summary")
        if not summary: continue
        total += 1
        summary = summary.strip()
        if any(summary.endswith(char) for char in ENDING_CHARS):
            correct += 1
            print(START_ITEM)
            print(item["_id"])
            print(summary)
            print(END_ITEM)
    return total, correct    

def run_check_summary():
    sum_total, sum_correct = 0, 0
    for file in glob.glob("data/cleaned-*.json"):
        items = utils.load_data_from_file_path(file)
        total, correct = check_summary(items)
        sum_total += total
        sum_correct += correct
    print(f"Total: {sum_total}, Correct: {sum_correct}")


##########################################
######## CLEANUP COLLECTED DATA ##########
##########################################

remove_quote = lambda items: [items.replace("\"", "") for items in items if isinstance(items, str)]
unique_items = lambda items: list({item.strip().lower(): item for item in items}.values())
def parse_to_list(val: str, sep: str = ",") -> list[str]:
    if isinstance(val, list): 
        return [v.strip() for v in val if isinstance(v, str)]
    if isinstance(val, str): 
        if sep: return [v.strip() for v in val.split(sep)]
        return [val.strip()]
    

# cleaning up names
# take name or names
# split string by comma
# if there is only 1 name it is most likely an error to deprecate it
# don't include names that start with lower case letter
# if text contains less than 250 words, take 1 name
# else: take the first 4 names
def cleanup_names(item: dict) -> list[str]:
    names = parse_to_list(item.get('name') or item.get('names'))
    if not names: return 
    names = remove_quote(names)   
    names = unique_items(names)
    names = [name for name in names if name[0].isupper()]
    if len(item['text'].split()) < 250: return names[:1]
    else: return names[:4]

# take domain or domains
# for posts (less than 250 words) take the first domain
# else: take the first 2 domains
def cleanup_domains(item: dict) -> list[str]:
    domains = parse_to_list(item.get('domain') or item.get('domains'))
    if not domains: return
    domains = remove_quote(domains)
    domains = unique_items(domains)
    return domains[:2]

# take gist or highlight_summary or highlight or highlights
# if it is a string then return it.
# if there are none take highlight or the first item of highlights
# replace all "
def cleanup_title(item: dict) -> str:
    gist = item.get('title') or item.get('gist') or item.get('highlight_summary') or item.get('highlight') or item.get('highlights')
    if not gist: return
    if isinstance(gist, list): gist = gist[0]
    return gist.replace("\"", "").strip()

# take highlight or highlights 
# make it a list
# replace all ""
def cleanup_highlights(item: dict) -> list[str]:
    highlights = parse_to_list(item.get('highlight') or item.get('highlights'), sep=None)
    if not highlights: return
    highlights = remove_quote(highlights)
    return highlights

def remove_before(text: str, sub: str) -> str:
    index = text.find(sub)
    if index > 0: return text[index:]
    return text

def remove_after(text: str, sub: str) -> str:
    index = text.find(sub)
    if index > 0: return text[:index]
    return text

MARKDOWN_PREFIX, MARKDOWN_SUFFIX = "```markdown", "```"
MARKDOWN_HEADERS = ["# ", "## ", "### ", "#### ", "**"]
def replace_header_tag(match):
        header_content = match.group(2).strip()  # The content after "# " or "## "
        newline = match.group(3)  # Preserve the newline or end of string
        return f"\n**{header_content}**{newline}"

# replace all " with _
# remove any line that matches the regex Here is a, here is a
# remove ``` or ```markdown from the beginning
# if the summary starts with "The article", or "# ", "In this article", "The text describes" cancel it
# wrap all sentences starting with #, ##, ###, #### with *
# NO_BUENO = ["The article", "This article", "In this article", "The text describes"]
NO_BUENO_SUFFIX = ["This summary was automatically generated"]
TEXT_TO_REMOVE = ["**Rewritten Summary**"]

def cleanup_summary(item: dict) -> str:
    resp = item.get('summary')
    if not resp: return
    # remove the part before ```markdown
    if MARKDOWN_PREFIX in resp: resp = resp[resp.find(MARKDOWN_PREFIX) + len(MARKDOWN_PREFIX):]
    # remove the part after ```
    if MARKDOWN_SUFFIX in resp: resp = resp[:resp.rfind(MARKDOWN_SUFFIX)]
    # sometimes the response starts with ```. In this case remove the part before that
    if MARKDOWN_PREFIX in resp: resp = resp[resp.find(MARKDOWN_PREFIX) + len(MARKDOWN_PREFIX):]
    resp = resp.replace("\"", "_").strip()

    if any(text.startswith(tag) for tag in MARKDOWN_HEADERS):
        resp = remove_before(resp, "\n")
    resp = re.sub(r"(#+ )(.*?)(\n|$)", replace_header_tag, resp)

    for text in TEXT_TO_REMOVE:
        resp = resp.replace(text, "")

    for suffix in NO_BUENO_SUFFIX:
        resp = remove_after(resp, suffix)
    
    return resp.strip()

def cleanup_markdown(text: str) -> str:
    # remove all \t with
    ic('\t' in text)
        # text = text.replace("\t", " ")
        # text = re.sub(r"\s+", " ", text)
        # text = re.sub(r"\s+", "\n", text)
    text = text.replace("\t", "")
    
    # # removing the first line if it looks like a header
    # text = text.strip()
    # if any(text.startswith(tag) for tag in MARKDOWN_HEADERS):
    #     text = remove_before(text, "\n") 

    # replace remaining headers with "**"
    # text = re.sub(r"(#+ )(.*?)(\n|$)", replace_header_tag, text)
    # Replace "\n(any number of spaces)\n" with "\n\n"
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Remove any space after "\n"
    # text = re.sub(r"\n\s+", "\n", text)
    # Replace "\n\n\n" with "\n\n"
    # text = re.sub(r"\n\n\n", "\n\n", text)
    # remove > right after \n
    # text = re.sub(r"\n>", "\n", text)
    # # replace every single \n with \n\n
    # text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text)
    # # Add a space after every "+" if there is no space
    # text = re.sub(r'\+(?!\s)', '+ ', text)

    return text.strip()


def cleanup_items(items: list[dict]) -> list[dict]:
    collected = int(datetime.now().timestamp())
    for i in range(len(items)):
        items[i]['highlights'] = cleanup_highlights(items[i])
        items[i]['title'] = cleanup_title(items[i])
        items[i]['names'] = cleanup_names(items[i])
        items[i]['domains'] = cleanup_domains(items[i])
        items[i]['summary'] = cleanup_summary(items[i])
        items[i]['collected'] = collected
        items[i] = {key:value for key, value in items[i].items() if value}  

    return items

def run_cleanup():
    os.makedirs(".cleaned", exist_ok=True)
    start_counter = 81500
    for file_path in glob.glob(".raw_data/summaries-*.json"):
        items = utils.load_data_from_file_path(ic(file_path))
        items = cleanup_items(items)
        utils.save_data_to_file_path(items, f"data/cleaned-{start_counter}-{start_counter+len(items)}.json")
        start_counter += len(items)


def add_to_summary(bean: dict, line: str):
    if 'summary' not in bean: bean['summary'] = ""
    bean['summary'] += line

TO_DROP = ["Tags:", "Sources:", "Related:", "Author:"]
def scrape_clean_summary(file_path: str):
    beans = {}
    with open(file_path, "r") as file:
        _id = None
        new_item = False
        for line in file:
            if line.startswith(START_ITEM):
                new_item = True
            elif line.startswith(END_ITEM):
                if _id: beans[_id] = beans[_id].strip()
                new_item = False
                _id = None
            elif new_item:
                if _id: 
                    beans[_id] += line
                else: 
                    _id = line.strip()
                    beans[_id] = ""

    beans = {_id: summary.strip() for _id, summary in beans.items() if not any(drop in summary for drop in TO_DROP)}
    
    ic(len(beans))
    return beans

def reassign_summary(to_beans: list[dict], from_beans: dict):
    for bean in to_beans:
        bean['summary'] = from_beans.get(bean['_id'])
        if not bean['summary']:
            del bean['summary']
    return to_beans

def run_reassign_summary(cleaned_summary_file: str):
    from_beans = scrape_clean_summary(cleaned_summary_file)
    # os.makedirs(".cleaned", exist_ok=True)
    for file_path in glob.glob("data/cleaned-*.json"):
        beans = utils.load_data_from_file_path(file_path)
        beans = reassign_summary(beans, from_beans)
        # utils.print_results(beans, "summary")
        utils.save_data_to_file_path(beans, file_path)

def count_attributes(beans: list[dict]):
    total_titles, total_summaries = 0, 0
    for bean in beans:
        if 'title' in bean: total_titles += 1
        if 'summary' in bean: total_summaries += 1
    return total_titles, total_summaries

def run_attribute_count(files = "data/cleaned-*.json"):
    total_titles, total_summaries = 0, 0
    for file_path in glob.glob(files):
        beans = utils.load_data_from_file_path(file_path)
        titles, summaries = count_attributes(beans)
        total_titles += titles
        total_summaries += summaries
    print(f"Total titles: {total_titles}, Total summaries: {total_summaries}")

    # total_titles, total_summaries = 0, 0
    # for file_path in glob.glob(".cleaned/cleaned-*.json"):
    #     beans = utils.load_data_from_file_path(file_path)
    #     titles, summaries = count_attributes(beans)
    #     total_titles += titles
    #     total_summaries += summaries
    # print(f"Total titles: {total_titles}, Total summaries: {total_summaries}")

def run_cleandata_merging():
    beans = utils.load_data_from_directory("data/cleaned-*.json")
    merged = {}
    for bean in beans:
        if len(bean['text'].split()) < 250: 
            continue
        if bean['_id'] not in merged:
            merged[bean['_id']] = bean
        elif merged[bean['_id']]['collected'] < bean['collected']:
            merged[bean['_id']].update(bean)

    ic(len(beans), len(merged))
    utils.save_data_to_directory(list(merged.values()), ".merged")



if __name__ == "__main__":
    # scrape_clean_summary(".raw_data/summary-dump.txt")
    # run_check_summary()    
    # run_reassign_summary(".raw_data/summary-dump-cleaned.txt")
    # run_cleanup()
    run_cleandata_merging()
    run_attribute_count(".merged/cleaned-*.json")
    