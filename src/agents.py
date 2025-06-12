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
    def __init__(self, tokenizer = None, max_input_tokens: int = None, max_output_tokens: int = None, response_start: str = None, response_end: str = None):
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.response_start = response_start 
        self.response_end = response_end

    def _tokenize_prompts(self, prompts: list[dict[str, str]]|list[list[dict]], device: str = None):
        append_contents = lambda prompt: "\n".join(p['content'] for p in prompt)
        if self.tokenizer.chat_template: tokens = self.tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, padding=True, truncation=True, max_length=self.max_input_tokens, return_tensors="pt", return_dict=True)
        elif isinstance(prompts[0], dict): tokens = self.tokenizer(append_contents(prompts), padding=True, truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        else: tokens = self.tokenizer(list(map(append_contents, prompts)), padding=True, truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        
        if device: tokens = tokens.to(device)
        return tokens

    def _extract_response(self, generated: str) -> str:
        if self.response_start: generated = remove_before(generated, self.response_start)
        if self.response_end: generated = remove_after(generated, self.response_end)
        return generated

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
    
DEFAULT_RESPONSE_START = "<|im_start|>assistant\n"
DEFAULT_RESPONSE_END = "<|im_end|>"
class TransformerClient(TextGenerationClient):
    model = None
    device = None

    def __init__(self, 
        model_id: str,
        max_input_tokens: int,
        max_output_tokens: int,
        response_start: str,
        response_end: str
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # super().__init__(
        #     tokenizer=AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, padding_side='left', use_fast=True),
        #     max_input_tokens=max_input_tokens,
        #     response_start=response_start,
        #     response_end=response_end
        # )
        # self.model =  AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device).to(self.device)
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, use_fast=True),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            response_start=response_start,
            response_end=response_end
        )
        self.model =  AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16).to(self.device)

    def run(self, prompt):
        import torch
        with torch.no_grad():
            input_tokens = self._tokenize_prompts(prompt, self.device)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=bool(self.response_start))
        return self._extract_response(generated_text) if self.response_start else generated_text

    def run_batch(self, prompts):
        import torch
        with torch.no_grad():
            input_tokens = self._tokenize_prompts(prompts, self.device)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=bool(self.response_start))
        return batch_run(self._extract_response, generated_texts, BATCH_SIZE) if self.response_start else generated_texts

    
class OVClient(TextGenerationClient):
    model = None
    max_output_tokens = None

    def __init__(self, 
        model_id: str,
        max_input_tokens: int,
        max_output_tokens: int,
        response_start: str,
        response_end: str
    ):
        from transformers import AutoTokenizer
        from optimum.intel.openvino import OVModelForCausalLM, OVModelForSeq2SeqLM
        # super().__init__(
        #     tokenizer=AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, padding_side='left', use_fast=True),
        #     max_input_tokens=max_input_tokens,
        #     response_start=response_start,
        #     response_end=response_end
        # )
        # self.model =  OVModelForCausalLM.from_pretrained(model_id)
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, use_fast=True),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            response_start=response_start,
            response_end=response_end
        )
        self.model = OVModelForSeq2SeqLM.from_pretrained(model_id)
        
    def run(self, prompt):
        input_tokens = self._tokenize_prompts(prompt)
        output_tokens = self.model.generate(
            **input_tokens, 
            max_new_tokens=self.max_output_tokens, 
            num_beams=1
        )
        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=bool(self.response_start))
        return self._extract_response(generated_text) if self.response_start else generated_text

    def run_batch(self, prompts):
        input_tokens = self._tokenize_prompts(prompts)
        output_tokens = self.model.generate(
            **input_tokens, 
            max_new_tokens=self.max_output_tokens,
            num_beams=1
        )
        generated_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=bool(self.response_start))
        return batch_run(self._extract_response, generated_texts, BATCH_SIZE) if self.response_start else generated_texts

class ONNXClient(TextGenerationClient):
    model = None

    def __init__(self, 
        model_id: str,
        max_input_tokens: int,
        max_output_tokens: int,
        response_start: str,
        response_end: str
    ):
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForCausalLM      
        # super().__init__(
        #     tokenizer=AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, padding_side='left', use_fast=True),
        #     max_input_tokens=max_input_tokens,
        #     response_start=response_start,
        #     response_end=response_end
        # )
        # self.model = ORTModelForCausalLM.from_pretrained(
        #     model_id,
        #     provider_options={
        #         'CPUExecutionProvider': {
        #             'arena_extend_strategy': 'kSameAsRequested',
        #             'cpu_threads': os.cpu_count(),  # Use all available CPU cores
        #             'enable_parallel_execution': True,
        #             'execution_mode': 'parallel'  # or 'parallel' for some models
        #         }
        #     },
        #     provider="CPUExecutionProvider"
        # )  
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(model_id, max_length=max_input_tokens, use_fast=True),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            response_start=response_start,
            response_end=response_end
        )
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_id,
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

    def run(self, prompt):
        import torch
        with torch.no_grad():
            input_tokens = self._tokenize_prompts(prompt)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=bool(self.response_start))
        return self._extract_response(generated_text) if self.response_start else generated_text

    def run_batch(self, prompts):
        import torch
        with torch.no_grad():
            input_tokens = self._tokenize_prompts(prompts)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_texts = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=bool(self.response_start))
        return batch_run(self._extract_response, generated_texts, BATCH_SIZE) if self.response_start else generated_texts
    
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
    context_len = max_input_tokens+(len(system_prompt) if system_prompt else 0) # this is an approximation
    if base_url: client = RemoteClient(model_path, base_url, api_key, max_output_tokens, temperature, json_mode)
    elif model_path.startswith(LLAMACPP_PREFIX): client = LlamaCppClient(model_path.removeprefix(LLAMACPP_PREFIX), context_len, max_output_tokens, temperature, json_mode)
    elif model_path.startswith(OPENVINO_PREFIX): client = OVClient(model_path.removeprefix(OPENVINO_PREFIX), context_len, max_output_tokens, DEFAULT_RESPONSE_START, DEFAULT_RESPONSE_END)
    elif model_path.startswith(ONNX_PREFIX): client = ONNXClient(model_path.removeprefix(ONNX_PREFIX), context_len, max_output_tokens, DEFAULT_RESPONSE_START, DEFAULT_RESPONSE_END)
    else: client = TransformerClient(model_path, context_len, max_output_tokens, DEFAULT_RESPONSE_START, DEFAULT_RESPONSE_END)
    
    return SimpleAgent(client, max_input_tokens, system_prompt, output_parser)