import os
import re
import tiktoken
import math
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

LLAMACPP_PREFIX = "llamacpp://"
ONNX_PREFIX = "onnx://"
OPENVINO_PREFIX = "openvino://"
API_URL_PREFIX = "https://"
NUM_THREADS = os.cpu_count()

_encoding = tiktoken.get_encoding("cl100k_base")

def chunk(input: str, context_len: int) -> list[str]:
    tokens = _encoding.encode(input)
    num_chunks = math.ceil(len(tokens) / context_len)
    chunk_size = math.ceil(len(tokens) / num_chunks)
    return [_encoding.decode(tokens[start : start+chunk_size]) for start in range(0, len(tokens), chunk_size)]

def combine_texts(texts: list[str], batch_size: int, delimiter: str = "```") -> list[str]:
    if count_tokens(texts) > batch_size:
        half = len(texts) // 2
        return combine_texts(texts[:half], batch_size, delimiter) + combine_texts(texts[half:], batch_size, delimiter)
    else:
        return [delimiter.join(texts)]
    
def chunk_tokens(input: str, context_len: int, encode_fn) -> list[str]:
    tokens = encode_fn(input)
    num_chunks = math.ceil(len(tokens) / context_len)
    chunk_size = math.ceil(len(tokens) / num_chunks)
    return [tokens[start : start+chunk_size] for start in range(0, len(tokens), chunk_size)]

truncate = lambda input, n_ctx: _encoding.decode(_encoding.encode(input, allowed_special=_ALLOWED_SPECIAL_TOKENS)[:n_ctx]) 
count_tokens = lambda input: len(_encoding.encode(input))

_ALLOWED_SPECIAL_TOKENS = {
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>", 
    "<|system|>",
    "<|human|>"
}

def truncate_batch(input_texts: list[str], n_ctx):
    tokenlist = _encoding.encode_batch(input_texts, num_threads=os.cpu_count(), allowed_special=_ALLOWED_SPECIAL_TOKENS)
    tokenlist = [tokens[:n_ctx] for tokens in tokenlist]
    return _encoding.decode_batch(tokenlist, num_threads=os.cpu_count())

def run_batch(func: Callable, items, num_threads: int = os.cpu_count()):
    results = None
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(func, items))
    return results  

split_parts = lambda text, sep=r'[,]+': [part.strip() for part in re.split(sep, text) if part.strip()]

def remove_before(text: str, sub: str) -> str:
    index = text.find(sub)
    if index >= 0: return text[index+len(sub):]
    return text

def remove_after(text: str, sub: str) -> str:
    index = text.find(sub)
    if index >= 0: return text[:index]
    return text

def clear_gpu_cache():
    import torch
    import gc
    """Clear GPU memory by running garbage collection and clearing CUDA cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


