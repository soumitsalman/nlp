import base64
import os
import logging
from typing import Callable
from retry import retry
from abc import ABC, abstractmethod
from .utils import *
from icecream import ic

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_COMPLETION_TOKENS = 512
BATCH_SIZE = int(os.getenv('BATCH_SIZE', os.cpu_count()))

log = logging.getLogger(__name__)

class DigestorBase(ABC):
    model_path: str = None
    system_prompt: str = None
    output_parser: Callable = None

    def __init__(self, model_path: str, system_prompt: str, output_parser: Callable):
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.output_parser = output_parser    

    def make_prompt(self, input_msg: str):
        if self.system_prompt: return f"{self.system_prompt}: {input_msg}"
        return input_msg   

    @abstractmethod
    def _run(self, prompt: str):
        raise NotImplementedError("Subclass must implement abstract method") 
    
    @abstractmethod
    def _run_batch(self, prompts: list[str]):
        return list(map(self._run, prompts))
    
    @abstractmethod
    def _unload_model(self):
        pass

    def run(self, input_msg: str):
        response = self._run(self.make_prompt(input_msg))
        if self.output_parser: return self.output_parser(response)
        return response
    
    def run_batch(self, input_messages: list[str]):
        prompts = list(map(self.make_prompt, input_messages))
        responses = self._run_batch(prompts)
        if self.output_parser: responses = list(map(self.output_parser, responses))
        return responses

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._unload_model()
        return False    

class LocalTokenizer:
    tokenizer = None
    max_input_tokens = None
    max_output_tokens = None
    device = None 

    def __init__(self, model_id, max_input_tokens: int, max_output_tokens: int = None, device: str = None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, use_fast=True)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.device = device
        if not self.tokenizer.pad_token: self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_prompts(self, prompts: str|list[str]):
        tokens = self.tokenizer(prompts, padding="max_length", truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        if self.device: tokens = tokens.to(self.device)
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def batch_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
class TransformerDigestor(DigestorBase):    
    device: str = None
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    _tokenizer = None
    _model = None

    def __init__(self, model_path: str, max_input_tokens: int, max_output_tokens: int, output_parser: Callable):
        import torch
        super().__init__(model_path=model_path, system_prompt=None, output_parser=output_parser)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16        
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def _run(self, prompt):
        import torch
        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self.tokenizer.tokenize_prompts(prompt)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3
            )
            generated_text = self._tokenizer.decode(output_tokens[0])
        return generated_text

    def _run_batch(self, prompts, **kwargs):
        import torch
        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self.tokenizer.tokenize_prompts(prompts)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
            generated_texts = self._tokenizer.batch_decode(output_tokens)
        return generated_texts
    
    @property
    def model(self):
        if not self._model:
            from transformers import AutoModelForSeq2SeqLM
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, dtype=self.dtype, device_map=self.device).to(self.device)
        return self._model
    
    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = LocalTokenizer(self.model_path, self.max_input_tokens, self.max_output_tokens, self.device)
        return self._tokenizer
    
    def _unload_model(self):
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        clear_gpu_cache()

class OVDigestor(TransformerDigestor):        
    def _run(self, prompt):
        input_tokens = self.tokenizer.tokenize_prompts(prompt)
        output_tokens = self.model.generate(
            **input_tokens, 
            max_new_tokens=self.max_output_tokens,
            repetition_penalty=1.3,
        )
        return self.tokenizer.decode(output_tokens[0])

    def _run_batch(self, prompts):
        input_tokens = self.tokenizer.tokenize_prompts(prompts)
        output_tokens = self.model.generate(
            **input_tokens, 
            max_new_tokens=self.max_output_tokens,
            repetition_penalty=1.3,
        )
        return self.tokenizer.batch_decode(output_tokens)
    
    @property
    def model(self):
        if not self._model:
            from optimum.intel.openvino import OVModelForSeq2SeqLM
            self._model = OVModelForSeq2SeqLM.from_pretrained(self.model_path)
        return self._model
    
    def _unload_model(self):
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

class ORTDigestor(TransformerDigestor):
    def run(self, prompt):
        import torch
        with torch.no_grad():
            input_tokens = self.tokenizer.tokenize_prompts(prompt)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(output_tokens[0])
        return generated_text

    def _run_batch(self, prompts):
        import torch
        with torch.no_grad():
            input_tokens = self.tokenizer.tokenize_prompts(prompts)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_texts = self.tokenizer.batch_decode(output_tokens)
        return generated_texts
    
    @property
    def model(self):
        if not self._model:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                provider_options={
                    'CPUExecutionProvider': {
                        'arena_extend_strategy': 'kSameAsRequested',
                        'cpu_threads': os.cpu_count()-1,  # Use all available CPU cores
                        'enable_parallel_execution': True,
                        'execution_mode': 'parallel'  # or 'parallel' for some models
                    }
                },
                provider="CPUExecutionProvider"
            )
        return self._model
    
    def _unload_model(self):
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

def from_path(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    max_input_tokens: int = None, 
    max_output_tokens: int = None,
    system_prompt: str = None,
    output_parser: Callable = None,
    json_mode: bool = False
) -> DigestorBase:

    if base_url: NotImplementedError("Remote digestor not supported")
    if model_path.startswith(OPENVINO_PREFIX): return OVDigestor(model_path, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens, output_parser=output_parser)
    elif model_path.startswith(ONNX_PREFIX): return ORTDigestor(model_path, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens, output_parser=output_parser)
    else: return TransformerDigestor(model_path, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens, output_parser=output_parser)