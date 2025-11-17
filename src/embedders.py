import logging
import threading
import os
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from retry import retry
from .utils import *

logger = logging.getLogger(__name__)

_MAX_CHUNKS = 8

class EmbedderBase(ABC):
    splitter = None
    context_len: int = None

    def __init__(self, context_len: int):
        self.context_len = context_len        

    def _split(self, text: str):
        # NOTE: moving the import inside the function so that there is no need to install llama-index if the embedder is used only for small texts
        from llama_index.core.text_splitter import SentenceSplitter
        if not self.splitter: 
            self.splitter = SentenceSplitter.from_defaults(
            chunk_size=self.context_len-32, # NOTE: this is a hack to accommodate for different tokenizer used by the splitter vs the model 
            chunk_overlap=0, 
            paragraph_separator="\n", 
            include_metadata=False, 
            include_prev_next_rel=False
        )
        return self.splitter.split_text(text)[:_MAX_CHUNKS]       

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
        return list(map(merged_embeddings, start_idx, counts))

    @abstractmethod
    def _embed(self, texts: str|list[str]):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Takes a list of strings/large documents as input and chunks them into smaller pieces as needed based on the context length.
        For each document it returns a mean of the embeddings of the chunks."""
        if not texts: return
        
        chunks, start_idx, counts = self._create_chunks(texts)
        embeddings = self._embed(chunks)
        embeddings = self._merge_chunks(embeddings, start_idx, counts)
        return embeddings[0] if isinstance(texts, str) else embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """Embeds a single string as a query. It prepends `query: ` to the input. 
        It processes the string without chunking or truncation for faster response."""
        if not query: return
        vec = self._embed("query: "+query)
        return vec if isinstance(vec, list) else vec.tolist()        
   
    def __call__(self, texts: str|list[str]):
        """This takes a string or an list of strings as an input.
        This calls the embedder directly without chunking or truncation for faster response"""
        if texts: return self._embed(texts).tolist()   

    @abstractmethod
    def _unload_model(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._unload_model()
        return False

class RemoteEmbeddings(EmbedderBase):
    model_client = None
    model_name: str
    context_len: int

    def __init__(self, model_name: str, base_url: str, api_key: str, context_len: int): 
        from openai import OpenAI

        super().__init__(context_len)
        self.model_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=10)
        self.model_name = model_name
        self.context_len = context_len    
       
    @retry(tries=2, delay=5, logger=logger)
    def _embed(self, texts):
        embeddings = self.model_client.embeddings.create(model=self.model_name, input=texts, encoding_format="float")
        return [data.embedding for data in embeddings.data]    

# local embeddings from llama.cpp
class LlamaCppEmbeddings(EmbedderBase):
    model_path = None
    context_len = None
    _model = None
    lock = None
    def __init__(self, model_path: str, context_len: int): 
        super().__init__(context_len)
        self.lock = threading.Lock()
        self.model_path = model_path
        self.context_len = context_len

    def _embed(self, texts):
        with self.lock:
            embeddings = self.model.create_embedding(texts)
        if isinstance(texts, str): return embeddings['data'][0]['embedding']
        return [data['embedding'] for data in embeddings['data']]
    
    @property
    def model(self):
        if not self._model:
            from llama_cpp import Llama
            n_threads = min(1, os.cpu_count()-1)
            self._model = Llama(model_path=self.model_path, n_ctx=self.context_len, n_threads_batch=n_threads, n_threads=n_threads, embedding=True, verbose=False)
        return self._model

    def _unload_model(self):
        if not self._model: return
        del self._model
        self._model = None        
        
class TransformerEmbeddings(EmbedderBase):
    _model = None
    model_path = None
    tokenizer_kwargs = None

    def __init__(self, model_path: str, context_len: int):
        import torch

        super().__init__(context_len)
        self.model_path = model_path
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": context_len,
            "padding": True
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _embed(self, texts: str|list[str]):
        import torch
        with torch.inference_mode(), torch.no_grad():
            embs = self.model.encode(texts, batch_size=len(texts), convert_to_numpy=True)
        return embs
    
    @property
    def model(self):
        if not self._model:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path, cache_folder=os.getenv('HF_HOME'), tokenizer_kwargs=self.tokenizer_kwargs, device=self.device)
        return self._model

    def _unload_model(self):
        if not self._model: return
        del self._model
        self._model = None
        clear_gpu_cache()
    
class OVEmbeddings(EmbedderBase):
    _model = None
    model_path = None
    context_len = None

    def __init__(self, model_path: str, context_len: int):
        super().__init__(context_len)
        self.model_path = model_path
        self.context_len = context_len

    def _embed(self, texts: str|list[str]):
        import torch
        with torch.no_grad(), torch.inference_mode():
            embs = self.model.encode(texts, batch_size=len(texts), convert_to_numpy=True)
        return embs
    
    @property
    def model(self):
        if not self._model:
            from optimum.intel.openvino import OVSentenceTransformer
            self._model = OVSentenceTransformer.from_pretrained(self.model_path, compile={"num_threads": os.cpu_count()-1})
        return self._model

    def _unload_model(self):
        if not self._model: return
        del self._model
        self._model = None
    
class ORTEmbeddings(EmbedderBase):
    _model = None
    model_path = None
    context_len = None

    def __init__(self, model_path: str, context_len: int):
        super().__init__(context_len)
        self.model_path = model_path
        self.context_len = context_len
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": context_len,
            "padding": True
        }

    def _embed(self, texts: str|list[str]):
        import torch
        with torch.inference_mode(), torch.no_grad():
            embs = self.model.encode(texts, batch_size=len(texts), convert_to_numpy=True)
        return embs

    @property
    def model(self):
        if not self._model:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path,  cache_folder=os.getenv('HF_HOME'), tokenizer_kwargs=self.tokenizer_kwargs, backend="onnx", model_kwargs={'file_name': "model.onnx", 'provider': 'CPUExecutionProvider'})
        return self._model

    def _unload_model(self):
        if not self._model: return
        del self._model
        self._model = None

def from_path(
    model_path: str, 
    context_len: int = 512,
    base_url: str = None,
    api_key: str = None
) -> EmbedderBase:
    # initialize digestor
    if base_url: return RemoteEmbeddings(model_path, base_url, api_key, context_len)
    if model_path.startswith(LLAMACPP_PREFIX): return LlamaCppEmbeddings(model_path.removeprefix(LLAMACPP_PREFIX), context_len)
    if model_path.startswith(OPENVINO_PREFIX): return OVEmbeddings(model_path.removeprefix(OPENVINO_PREFIX), context_len)
    if model_path.startswith(ONNX_PREFIX): return ORTEmbeddings(model_path.removeprefix(ONNX_PREFIX), context_len)
    return TransformerEmbeddings(model_path, context_len)

    




    

