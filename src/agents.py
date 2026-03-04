# DEPRECATED

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

class LMClientBase(ABC):
    @abstractmethod
    def run(self, prompt: list[dict[str, str]], **kwargs) -> str:
        raise NotImplementedError("Subclass must implement abstract method")

    def run_batch(self, prompts: list[list[dict[str, str]]], **kwargs) -> list[str]:
        return list(map(self.run, prompts))

class LMAgentBase(ABC):
    client: LMClientBase
    system_prompt: str = None
    output_parser: Callable = None

    def __init__(self, client, system_prompt: str, output_parser: Callable):
        self.client = client
        self.system_prompt = system_prompt or ""
        self.output_parser = output_parser or (lambda x: x)

    def make_prompt(self, input_msg: str):
        if self.system_prompt: return f"{self.system_prompt}: {input_msg}"
        return input_msg

    @abstractmethod
    def run(self, prompt: list[dict[str, str]]) -> str:
        raise NotImplementedError("Subclass must implement abstract method")

    def run_batch(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        return list(map(self.run, prompts))

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
    
class TransformerText2TextClient(LMClientBase):
    model = None
    device = None
    tokenizer = None
    model = None

    def __init__(self, model_id: str, max_input_tokens: int, max_output_tokens: int):
        import torch
        from transformers import AutoModelForSeq2SeqLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.max_output_tokens = max_output_tokens
        self.tokenizer = LocalTokenizer(model_id, max_input_tokens, max_output_tokens, self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=self.dtype, device_map=self.device).to(self.device)

    def run(self, prompt, **kwargs):
        import torch
        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self.tokenizer.tokenize_prompts(prompt)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3
            )
            generated_text = self.tokenizer.decode(output_tokens[0])
        return generated_text

    def run_batch(self, prompts, **kwargs):
        import torch
        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self.tokenizer.tokenize_prompts(prompts)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
            generated_texts = self.tokenizer.batch_decode(output_tokens)
        return generated_texts

    
class OVText2TextClient(LMClientBase):
    tokenizer = None
    model = None
    max_output_tokens = None

    def __init__(self, 
        model_id: str,
        max_input_tokens: int,
        max_output_tokens: int
    ):
        from optimum.intel.openvino import OVModelForSeq2SeqLM
        self.max_output_tokens = max_output_tokens
        self.tokenizer = LocalTokenizer(model_id, max_input_tokens, max_output_tokens)
        self.model = OVModelForSeq2SeqLM.from_pretrained(model_id)
        
    def run(self, prompt, **kwargs):
        input_tokens = self.tokenizer.tokenize_prompts(prompt)
        output_tokens = self.model.generate(
            **input_tokens, 
            max_new_tokens=self.max_output_tokens,
            repetition_penalty=1.3,
        )
        return self.tokenizer.decode(output_tokens[0])

    def run_batch(self, prompts, **kwargs):
        input_tokens = self.tokenizer.tokenize_prompts(prompts)
        output_tokens = self.model.generate(
            **input_tokens, 
            max_new_tokens=self.max_output_tokens,
            reprepetition_penalty=1.3,
        )
        return self.tokenizer.batch_decode(output_tokens)

class ONNXText2TextClient(LocalTokenizer):
    tokenizer = None
    model = None

    def __init__(self, 
        model_id: str,
        max_input_tokens: int,
        max_output_tokens: int
    ):
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        self.max_output_tokens = max_output_tokens
        self.tokenizer = LocalTokenizer(model_id, max_input_tokens, max_output_tokens)
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

    def run(self, prompt, **kwargs):
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

    def run_batch(self, prompts, **kwargs):
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

class Text2TextAgent(LMAgentBase):
    def __init__(self, client, task: str, output_parser: Callable):
        super().__init__(client, task, output_parser)

    def run(self, input_msg: str):
        response = self.client.run(self.make_prompt(input_msg))
        if self.output_parser: return self.output_parser(response)
        return response
    
    def run_batch(self, input_messages: list[str]):
        responses = self.client.run_batch(run_batch(self.make_prompt, input_messages, BATCH_SIZE))
        if self.output_parser: return run_batch(self.output_parser, responses, BATCH_SIZE)
        return responses
    
# DEFAULT_RESPONSE_START = "<|im_start|>assistant\n"
# DEFAULT_RESPONSE_END = "<|im_end|>"
class TransformerTextGeneratorClient(LMClientBase):
    def __init__(self, tokenizer = None, max_input_tokens: int = None, max_output_tokens: int = None, response_start: str = None, response_end: str = None):
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.response_start = response_start 
        self.response_end = response_end        

    def _tokenize_prompts(self, prompts: list[dict[str, str]]|list[list[dict]], device: str = None):
        append_contents = lambda prompt: "\n".join(p['content'] for p in prompt)
        if self.tokenizer.chat_template: tokens = self.tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, padding="max_length", truncation=True, max_length=self.max_input_tokens, return_tensors="pt", return_dict=True)
        elif isinstance(prompts[0], dict): tokens = self.tokenizer(append_contents(prompts), padding="max_length", truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        else: tokens = self.tokenizer(list(map(append_contents, prompts)), padding="max_length", truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        
        if device: tokens = tokens.to(device)
        return tokens

    def _extract_response(self, generated: str) -> str:
        if self.response_start: generated = remove_before(generated, self.response_start)
        if self.response_end: generated = remove_after(generated, self.response_end)
        return generated


class RemoteTextGeneratorClient(LMClientBase):
    openai_client = None
    model_name: str = None
    max_output_tokens: int = None

    def __init__(self, 
        model_name: str,
        base_url: str, 
        api_key: str,
        max_output_tokens: int
    ):
        from openai import OpenAI

        self.openai_client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=3)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
    
    def run(self, prompt: list[dict[str, str]], **kwargs) -> str:
        response_format = { "type": "json_object" } if kwargs.get("json_mode") else None
        return self.openai_client.chat.completions.create(
            messages=prompt,
            model=self.model_name,
            max_completion_tokens=self.max_output_tokens,
            response_format=response_format,
            seed=666
        ).choices[0].message.content
    
    def run_batch(self, prompts: list[list[dict[str, str]]], **kwargs) -> list[str]:
        return run_batch(lambda x: self.run(x, **kwargs), prompts, BATCH_SIZE)
    
class LlamaCppTextGeneratorClient(LMClientBase):
    model = None
    max_output_tokens = None
    model = None
    lock = None
    
    def __init__(self, model_path: str, max_input_tokens: int, max_output_tokens: int):
        import threading
        from llama_cpp import Llama

        self.lock = threading.Lock()
        self.max_output_tokens = max_output_tokens
        self.temperature = 0.5
        self.model = Llama(
            model_path=model_path, n_ctx=max_input_tokens<<1, # this extension is needed to accommodate occasional overflows
            n_gpu_layers=-1,
            embedding=False, verbose=False
        )             
  
    def run(self, prompt: str, **kwargs) -> str:
        response_format = { "type": "json_object" } if kwargs.get("json_mode") else None
        resp = self.model.create_chat_completion(
            messages=prompt,
            max_tokens=self.max_output_tokens,
            response_format=response_format,
            temperature=self.temperature,
            seed=666
        )['choices'][0]['message']['content'].strip()      
        return resp
    
    def run_batch(self, prompts: list[str], **kwargs) -> list[str]:
        results = [self.run(text, **kwargs) for text in prompts]
        return results
        
class TextGeneratorAgent(LMAgentBase):
    json_mode: bool = False

    def __init__(self, client, max_input_tokens: int, system_prompt: str, output_parser: Callable, json_mode: bool = False):
        super().__init__(client, system_prompt, output_parser)        
        self.max_input_tokens = max_input_tokens
        self.json_mode = json_mode

    def make_prompt(self, input_msg: str):
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

    @retry(tries=3, delay=5, logger=log)
    def run(self, input_msg: str):
        if self.max_input_tokens: input_msg = truncate(input_msg, self.max_input_tokens)
        response = self.client.run(self.make_prompt(input_msg), json_mode=self.json_mode)
        if self.output_parser: return self.output_parser(response)
        return response
    
    def run_batch(self, input_messages: list[str]):
        if self.max_input_tokens: input_messages = truncate_batch(input_messages, self.max_input_tokens)
        prompts = run_batch(self.make_prompt, input_messages, BATCH_SIZE)
        responses = self.client.run_batch(prompts, json_mode=self.json_mode)
        if self.output_parser: return run_batch(self.output_parser, responses, BATCH_SIZE)
        return responses

_NEGATIVE_PROMPT = "text, human face, blurred edges, watermark"

class DiffuserImageGenerationAgent(LMAgentBase):

    def __init__(
        self,
        model_id: str,
        output_parser: Callable = None,
        num_inference_steps: int = 25,
        height: int = 1024,
        width: int = 512
    ):
        import torch
        from diffusers import DiffusionPipeline
        
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.pipe = DiffusionPipeline.from_pretrained(model_id, dtype=torch.float16, variant="fp16")
        if torch.cuda.is_available(): self.pipe = self.pipe.to("cuda")
        super().__init__(None, None, output_parser)

    def run(self, user_msg: str):
        import torch
        with torch.no_grad(), torch.inference_mode():
            result = self.pipe(user_msg, negative_prompt=_NEGATIVE_PROMPT, guidance_scale=5, num_inference_steps=self.num_inference_steps,  height=self.height, width=self.width)
        if self.output_parser: return self.output_parser(result.images[0])
        return result.images[0]

    def run_batch(self, user_msgs: list[str]):
        import torch
        batch_size = os.cpu_count() # this is a rough estimate
        images = []
        for i in range(0, len(user_msgs), batch_size):
            with torch.no_grad(), torch.inference_mode():            
                results = self.pipe(user_msgs[i:i+batch_size], negative_prompt=_NEGATIVE_PROMPT, guidance_scale=5, num_inference_steps=self.num_inference_steps, height=self.height, width=self.width)
            if self.output_parser: images.extend(run_batch(self.output_parser, results.images, len(results.images)))
            else: images.extend(results.images)
        return images

class RemoteImageGenerationAgent(LMAgentBase):
    model_name = None
    client = None

    def __init__(self, 
        model_name: str, 
        base_url: str, 
        api_key: str, 
        output_processor: Callable = None,  
        num_inference_steps: int = 25,
        height: int = 1024,
        width: int = 512
    ):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=30)                
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        super().__init__(None, None, output_processor)
    
    def run(self, user_msg: str):
        response = self.client.images.generate(
            model=self.model_name,
            prompt=user_msg,
            n=self.num_inference_steps,
            size=f"{self.width}x{self.height}",
            output_format="png",
            style="vivid",
            quality="high",
            response_format="b64_json"
        )
        return self.output_processor(base64.b64decode(response.data[0].b64_json))
    
    def run_batch(self, user_msgs: list[str]):
        return run_batch(self.run, user_msgs)

def text2text_client_from_path(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    max_input_tokens: int = None, 
    max_output_tokens: int = None,
    json_mode: bool = False
):
    context_len = max_input_tokens
    if base_url: return RemoteTextGeneratorClient(model_path, base_url, api_key, max_output_tokens, json_mode)
    elif model_path.startswith(LLAMACPP_PREFIX): return LlamaCppTextGeneratorClient(model_path.removeprefix(LLAMACPP_PREFIX), context_len, max_output_tokens, json_mode)
    elif model_path.startswith(OPENVINO_PREFIX): return OVText2TextClient(model_path.removeprefix(OPENVINO_PREFIX), context_len, max_output_tokens)
    elif model_path.startswith(ONNX_PREFIX): return ONNXText2TextClient(model_path.removeprefix(ONNX_PREFIX), context_len, max_output_tokens)
    else: return TransformerText2TextClient(model_path, context_len, max_output_tokens)

def text2text_agent_from_path(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    max_input_tokens: int = None, 
    max_output_tokens: int = None,
    system_prompt: str = None,
    output_parser: Callable = None,
    json_mode: bool = False
) -> Text2TextAgent:

    client = text2text_client_from_path(model_path, base_url, api_key, max_input_tokens, max_output_tokens, json_mode)    
    if model_path.startswith(OPENVINO_PREFIX): return Text2TextAgent(client, system_prompt, output_parser)
    elif model_path.startswith(ONNX_PREFIX): return Text2TextAgent(client, system_prompt, output_parser)
    else: return Text2TextAgent(client, system_prompt, output_parser)
  
def text_generator_agent(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    max_input_tokens: int = None, 
    max_output_tokens: int = None,
    system_prompt: str = None,
    output_parser: Callable = None,
    json_mode: bool = False
):
    client = text_generator_client(model_path, base_url, api_key, max_input_tokens, max_output_tokens)
    return TextGeneratorAgent(client, max_input_tokens, system_prompt, output_parser, json_mode)


def text_generator_client(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    max_input_tokens: int = None, 
    max_output_tokens: int = None
):
    context_len = max_input_tokens<<1
    if base_url: return RemoteTextGeneratorClient(model_path, base_url, api_key, max_output_tokens)
    elif model_path.startswith(LLAMACPP_PREFIX): return LlamaCppTextGeneratorClient(model_path.removeprefix(LLAMACPP_PREFIX), context_len, max_output_tokens)
    else: return TransformerTextGeneratorClient(model_path, context_len, max_output_tokens)

def image_agent_from_path(
    model_path: str,
    base_url: str = None, 
    api_key: str = None,
    output_parser: Callable = None,
    num_inference_steps: int = 25,
    height: int = 1024,
    width: int = 512
):
    if base_url: return RemoteImageGenerationAgent(model_path, base_url, api_key, output_parser, num_inference_steps, height, width)
    else: return DiffuserImageGenerationAgent(model_path, output_parser, num_inference_steps, height, width)
    