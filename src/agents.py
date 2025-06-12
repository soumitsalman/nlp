import os
import logging
from typing import Callable
from retry import retry
from abc import ABC, abstractmethod
from .utils import *

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_COMPLETION_TOKENS = 512
BATCH_SIZE = int(os.getenv('BATCH_SIZE', os.cpu_count()))

log = logging.getLogger(__name__)

class TextGenerationClient(ABC):
    @abstractmethod
    def run(self, prompt: list[dict[str, str]]) -> str:
        raise NotImplementedError("Subclass must implement abstract method")

    def run_batch(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        return list(map(self.run, prompts))

class RemoteClient(TextGenerationClient):
    openai_client = None
    model_name: str = None
    max_output_tokens: int = None
    temperature: float = None

    def __init__(self, 
        model_name: str,
        base_url: str, 
        api_key: str,
        max_output_tokens: int,
        temperature: float,
        json_mode: bool
    ):
        from openai import OpenAI

        self.openai_client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.json_mode = json_mode

    @retry(tries=2, delay=5, logger=log)
    def run(self, prompt: list[dict[str, str]]) -> str:
        return self.openai_client.chat.completions.create(
            messages=prompt,
            model=self.model_name,
            max_completion_tokens=self.max_output_tokens,
            response_format={ "type": "json_object" } if self.json_mode else None,
            temperature=self.temperature,
            seed=666
        ).choices[0].message.content
    
    def run_batch(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        return batch_run(self.run, prompts, BATCH_SIZE)
    
class LlamaCppClient(TextGenerationClient):
    model = None
    max_output_tokens = None
    model = None
    lock = None
    
    def __init__(self, model_path: str, max_input_tokens: int, max_output_tokens: int, temperature: float, json_mode: bool):
        import threading
        from llama_cpp import Llama

        self.lock = threading.Lock()
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.model = Llama(
            model_path=model_path, n_ctx=max_input_tokens<<1, # this extension is needed to accommodate occasional overflows
            n_threads_batch=NUM_THREADS, n_threads=NUM_THREADS, 
            embedding=False, verbose=False
        )             
  
    def run(self, prompt: str) -> str:
        with self.lock:
            resp = self.model.create_chat_completion(
                messages=prompt,
                max_tokens=self.max_output_tokens,
                response_format={ "type": "json_object" } if self.json_mode else None,
                temperature=self.temperature,
                seed=666
            )['choices'][0]['message']['content'].strip()      
        return resp
    
    def run_batch(self, prompts: list[str]) -> list[str]:
        with self.lock:
            results = [self.run(text) for text in prompts]
        return results
    
def _generate_input_tokens(tokenizer, prompts: list[dict[str, str]]|list[list[dict]], context_len, device: str=None):
    append_contents = lambda prompt: "\n".join(p['content'] for p in prompt)
    if tokenizer.chat_template: tokens = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, padding=True, padding_side='left', truncation=True, max_length=6144, return_dict=True, return_tensors="pt")
    elif isinstance(prompts[0], dict): tokens = tokenizer(append_contents(prompts), padding=True, padding_side='left', truncation=True, max_length=context_len, return_tensors="pt")
    else: tokens = tokenizer(list(map(append_contents, prompts)), padding=True, padding_side='left', truncation=True, max_length=context_len, return_tensors="pt")
    
    if device: tokens = tokens.to(device)
    return tokens
    
DEFAULT_RESPONSE_START = "<|im_start|>assistant\n"
DEFAULT_RESPONSE_END = "<|im_end|>"
class TransformerClient(TextGenerationClient):
    model = None
    tokenizer = None
    device = None
    context_len = None

    def __init__(self, 
        model_id: str,
        context_len: int,
        max_output_tokens: int,
        temperature: float,
        response_start: str,
        response_end: str
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
        import torch

        self.context_len = context_len # this is a buffer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.model =  AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device).to(self.device)
        self.model =  AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=self.device).to(self.device)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.response_start = response_start
        self.response_end = response_end

    def _extract_response(self, generated: str) -> str:
        generated = remove_before(generated, self.response_start)
        return remove_after(generated, self.response_end)

    def run(self, prompt):
        import torch
        input_tokens = _generate_input_tokens(self.tokenizer, prompt, self.context_len, self.device)
        with torch.no_grad():
            output_tokens = self.model.generate(**input_tokens, max_new_tokens=self.max_output_tokens, temperature=self.temperature)
            generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        return self._extract_response(generated_text)

    def run_batch(self, prompts):
        import torch
        input_tokens = _generate_input_tokens(self.tokenizer, prompts, self.context_len, self.device)
        with torch.no_grad():
            output_tokens = self.model.generate(**input_tokens, max_new_tokens=self.max_output_tokens)
            generated_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return batch_run(self._extract_response, generated_texts, BATCH_SIZE)
    
class OVClient(TextGenerationClient):
    model = None
    tokenizer = None
    context_len = None

    def __init__(self, 
        model_id: str,
        context_len: int,
        max_output_tokens: int,
        temperature: float,
        response_start: str,
        response_end: str
    ):
        from transformers import AutoTokenizer
        from optimum.intel.openvino import OVModelForCausalLM, OVModelForSeq2SeqLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.model =  OVModelForCausalLM.from_pretrained(model_id)
        self.model = OVModelForSeq2SeqLM.from_pretrained(model_id)

        self.context_len = context_len
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.response_start = response_start
        self.response_end = response_end

    def _extract_response(self, generated: str) -> str:
        generated = remove_before(generated, self.response_start)
        return remove_after(generated, self.response_end)

    def run(self, prompt):
        input_tokens = _generate_input_tokens(self.tokenizer, prompt, self.context_len)
        output_tokens = self.model.generate(**input_tokens, max_new_tokens=self.max_output_tokens, temperature=self.temperature)
        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        return self._extract_response(generated_text)

    def run_batch(self, prompts):
        input_tokens = _generate_input_tokens(self.tokenizer, prompts, self.context_len)
        output_tokens = self.model.generate(**input_tokens, max_new_tokens=self.max_output_tokens)
        generated_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return batch_run(self._extract_response, generated_texts, BATCH_SIZE)

    
class SimpleAgent:
    client: TextGenerationClient
    max_input_tokens: int = None
    system_prompt: str = None
    output_parser: Callable = None

    def __init__(self, client, max_input_tokens: int, system_prompt: str, output_parser: Callable):
        self.client = client
        self.max_input_tokens = max_input_tokens
        self.system_prompt = system_prompt
        self.output_parser = output_parser

    def _make_prompt(self, input_msg: str):
        if self.system_prompt: return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": input_msg
            }
        ]
        else: return [
            {
                "role": "user",
                "content": input_msg
            }
        ]

    def run(self, input_msg: str):
        if self.max_input_tokens: input_msg = truncate(input_msg, self.max_input_tokens)
        response = self.client.run(self._make_prompt(input_msg))
        if self.output_parser: return self.output_parser(response)
        return response
    
    def run_batch(self, input_messages: list[str]):
        if self.max_input_tokens: input_messages = batch_truncate(input_messages, self.max_input_tokens)
        prompts = batch_run(self._make_prompt, input_messages, BATCH_SIZE)
        responses = self.client.run_batch(prompts)
        if self.output_parser: return batch_run(self.output_parser, responses, BATCH_SIZE)
        return responses

def from_path(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    max_input_tokens: int = None, 
    max_output_tokens: int = None,
    system_prompt: str = None,
    output_parser: Callable = None,
    temperature: float = DEFAULT_TEMPERATURE,
    json_mode: bool = False
) -> SimpleAgent:
    context_len = max_input_tokens+(max_output_tokens<<1)
    if base_url: client = RemoteClient(model_path, base_url, api_key, max_output_tokens, temperature, json_mode)
    elif model_path.startswith(LLAMACPP_PREFIX): client = LlamaCppClient(model_path.removeprefix(LLAMACPP_PREFIX), context_len, max_output_tokens, temperature, json_mode)
    elif model_path.startswith(OPENVINO_PREFIX): client = OVClient(model_path.removeprefix(OPENVINO_PREFIX), context_len, max_output_tokens, temperature, DEFAULT_RESPONSE_START, DEFAULT_RESPONSE_END)
    elif model_path.startswith(ONNX_PREFIX): raise NotImplementedError()
    else: client = TransformerClient(model_path, context_len, max_output_tokens, temperature, DEFAULT_RESPONSE_START, DEFAULT_RESPONSE_END)
    
    return SimpleAgent(client, max_input_tokens, system_prompt, output_parser)