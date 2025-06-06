import logging
import threading
import os
import numpy as np
from abc import ABC, abstractmethod
from llama_index.core.text_splitter import SentenceSplitter
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from retry import retry
from .utils import *
import torch

logger = logging.getLogger(__name__)

class Embeddings(ABC):
    splitter: SentenceSplitter = None

    def __init__(self, context_len: int):
        self.splitter = SentenceSplitter.from_defaults(
            chunk_size=context_len-32, # NOTE: this is a hack to accommodate for different tokenizer used by the splitter vs the model 
            chunk_overlap=0, 
            paragraph_separator="\n", 
            include_metadata=False, 
            include_prev_next_rel=False
        )

    def _split(self, text: str):
        splits = self.splitter.split_text(text)
        # return random.sample(splits, k=min(len(splits), MAX_CHUNKS)) 
        return splits       

    def _create_chunks(self, texts: list[str]) -> tuple[list[str], list[int], list[int]]:
        texts = texts if isinstance(texts, list) else [texts]
        
        chunks = list(map(self._split, texts)) # NOTE: batch running will mess this up
        counts = list(map(len, chunks))
        start_idx = [0]*len(chunks)
        for i in range(1,len(counts)):
            start_idx[i] = start_idx[i-1]+counts[i-1]
        return list(chain(*chunks)), start_idx, counts
    
    def _merge_chunks(self, embeddings, start_idx: list[int], counts: list[int]):
        merged_embeddings = lambda start, count: np.mean(embeddings[start:start+count], axis=0).tolist()
        with ThreadPoolExecutor(max_workers=os.cpu_count(), thread_name_prefix="merge_chunks") as exec:
            embeddings = list(exec.map(merged_embeddings, start_idx, counts))
        return embeddings

    @abstractmethod
    def _embed(self, texts: list[str]):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def embed(self, texts: str|list[str]):
        chunks, start_idx, counts = self._create_chunks(texts)
        embeddings = self._embed(chunks)
        embeddings = self._merge_chunks(embeddings, start_idx, counts)
        return embeddings[0] if isinstance(texts, str) else embeddings
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if texts: return self.embed(texts)

    def embed_query(self, query: str) -> list[float]:
        if query: return self.embed("query: "+query)
        
    def __call__(self, texts: str|list[str]):
        if texts: return self.embed(texts)


# local embeddings from llama.cpp
class LlamaCppEmbeddings(Embeddings):
    model_path = None
    context_len = None
    model = None
    lock = None
    def __init__(self, model_path: str, context_len: int):  
        from llama_cpp import Llama

        super().__init__(context_len)
        self.lock = threading.Lock()
        self.model_path = model_path
        self.context_len = context_len
        self.model = Llama(model_path=self.model_path, n_ctx=self.context_len, n_batch=self.context_len, n_threads_batch=os.cpu_count(), n_threads=os.cpu_count(), embedding=True, verbose=False)
    
    def _embed(self, texts: list[str]):
        with self.lock:
            embeddings = self.model.create_embedding(texts)
        return [data['embedding'] for data in embeddings['data']]

class RemoteEmbeddings(Embeddings):
    openai_client = None
    model_name: str
    context_len: int

    def __init__(self, model_name: str, base_url: str, api_key: str, context_len: int): 
        from openai import OpenAI

        super().__init__(context_len)
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=10)
        self.model_name = model_name
        self.context_len = context_len    
       
    @retry(tries=2, delay=5, logger=logger)
    def _embed(self, texts):
        embeddings = self.openai_client.embeddings.create(model=self.model_name, input=texts, encoding_format="float")
        return [data.embedding for data in embeddings.data]
    
class TransformerEmbeddings(Embeddings):
    model = None

    def __init__(self, model_path: str, context_len: int):
        from sentence_transformers import SentenceTransformer

        super().__init__(context_len)
        tokenizer_kwargs = {
            "truncation": True,
            "max_length": context_len,
            "padding": True
        }
        self.model = SentenceTransformer(model_path, cache_folder=os.getenv('HF_HOME'), tokenizer_kwargs=tokenizer_kwargs)

    def _embed(self, texts: str|list[str]):
        with torch.no_grad():
            embs = self.model.encode(texts, batch_size=len(texts), convert_to_numpy=True)
        return embs
    
class OVEmbeddings(Embeddings):
    model = None
    tokenizer = None
    context_len = None

    def __init__(self, model_path: str, context_len: int):
        from optimum.intel.openvino import OVModelForFeatureExtraction
        from transformers import AutoTokenizer

        super().__init__(context_len)        
        self.model = OVModelForFeatureExtraction.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.context_len = context_len

    def _embed(self, texts: str|list[str]):
        input_tokens = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=self.context_len)
        with torch.no_grad():
            output_tokens = self.model(**input_tokens)
            vecs = output_tokens.last_hidden_state.mean(axis=1)
        return vecs
    
class ORTEmbeddings(Embeddings):
    model = None
    tokenizer = None
    context_len = None

    def __init__(self, model_path: str, context_len: int):
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        super().__init__(context_len)        
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.context_len = context_len

    def _embed(self, texts: str|list[str]):
        input_tokens = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=self.context_len)
        with torch.no_grad():
            output_tokens = self.model(**input_tokens)
            vecs = output_tokens.last_hidden_state.mean(axis=1)   
        return vecs

def from_path(
    embedder_path: str, 
    context_len: int = 512,
    base_url: str = None,
    api_key: str = None
) -> Embeddings:
    # initialize digestor
    if base_url: return RemoteEmbeddings(embedder_path, base_url, api_key, context_len)
    if embedder_path.startswith(LLAMACPP_PREFIX): return LlamaCppEmbeddings(embedder_path.removeprefix(LLAMACPP_PREFIX), context_len)
    if embedder_path.startswith(OPENVINO_PREFIX): return OVEmbeddings(embedder_path.removeprefix(OPENVINO_PREFIX), context_len)
    if embedder_path.startswith(ONNX_PREFIX): return ORTEmbeddings(embedder_path.removeprefix(ONNX_PREFIX), context_len)
    return TransformerEmbeddings(embedder_path, context_len)

    




    

