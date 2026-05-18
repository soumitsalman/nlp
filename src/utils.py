import os
import re
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

LLAMACPP_PREFIX = "llamacpp://"
ONNX_PREFIX = "onnx://"
OPENVINO_PREFIX = "openvino://"
VLLM_PREFIX = "vllm://"
OPENAI_PREFIX = "openai://"
API_URL_PREFIX = "https://"
NUM_THREADS = os.cpu_count()

_ALLOWED_SPECIAL_TOKENS = {
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|system|>",
    "<|human|>",
}

def run_batch(func: Callable, items, num_threads: int = os.cpu_count()):
    results = None
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(func, items))
    return results

split_parts = lambda text, sep=r"[,]+": [
    part.strip() for part in re.split(sep, text) if part.strip()
]

def remove_before(text: str, sub: str) -> str:
    index = text.find(sub)
    if index >= 0:
        return text[index + len(sub) :]
    return text

def remove_after(text: str, sub: str) -> str:
    index = text.find(sub)
    if index >= 0:
        return text[:index]
    return text

def clear_gpu_cache():
    import torch
    import gc

    """Clear GPU memory by running garbage collection and clearing CUDA cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
