import base64
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random
from .models import *
from .utils import *
from icecream import ic

log = logging.getLogger("digestor")

try: import torch
except: log.warning("PyTorch Not Available", extra={'source': __file__, 'num_items': 1})

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.15,
    "max_tokens": 2048
}
DEFAULT_CONTEXT_LEN = 32768

class MicroAgentBase(ABC):
    def __init__(
        self,
        model_name: str,
        context_len: int,
        instruction: str,
        input_template: str,
        output_model: Type[BaseModel],
        **sampling_params,
    ):
        self.model_name = model_name
        self.context_len = context_len
        self.instruction = instruction
        self.input_template = input_template
        self.output_model = output_model
        self.response_mode = "json" if output_model else None
        self.sampling_params = {
            **DEFAULT_SAMPLING_PARAMS,
            **sampling_params,
        }
        self._llm = None
        self._sampling_params = None

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._llm:
            del self._llm
            self._llm = None
            del self._sampling_params
            self._sampling_params = None
            clear_gpu_cache()
        return False

    def create_prompt(self, msg: str):
        prompt = []
        if self.instruction: prompt.append({"role": "system", "content": self.instruction})
        prompt.append({"role": "user", "content": self.input_template.format(input_text=msg) if self.input_template else msg})
        return prompt

    def parse_output(self, response: str):
        response = _strip_fences(response)
        try:
            if self.response_mode == "json":
                return self.output_model.model_validate_json(response)
            if self.response_mode == "compressed":
                return parse_compressed(response)
            if self.response_mode == "markdown":
                return parse_markdown(response)
            if self.response_mode == "tool_call":
                raise NotImplementedError
            return response
        except:
            log.warning("failed parsing: %s", response, exc_info=True)

    @abstractmethod
    def run_batch(self, input_messages: list[str]) -> list[BaseModel]:
        raise NotImplementedError()


class LocalTokenizer:
    tokenizer = None
    context_len = None
    device = None

    def __init__(
        self,
        model_id,
        context_len: int,
        device: str = None,
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, max_length=context_len, use_fast=True
        )
        self.context_len = context_len
        self.device = device
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_prompts(self, prompts):
        tokens = self.tokenizer.apply_chat_template(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.context_len,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if self.device:
            tokens = tokens.to(self.device)
        return tokens

    def decode(self, tokens, input_tokens = None):
        if input_tokens is not None:
            tokens = tokens[len(input_tokens):]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def batch_decode(self, tokens, input_tokens = None):
        if input_tokens is not None:
            tokens = [out_tokens[len(in_tokens):] for out_tokens, in_tokens in zip(tokens, input_tokens)]
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id


# needs 1.3 for repetition penalty support, and max_new_tokens = 384
# "no_repeat_ngram_size": 3,
class TransformerMicroAgent(MicroAgentBase):
    _tokenizer = None
                
    def __enter__(self):
        if not self._llm:
            from transformers import AutoModelForCausalLM

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16
            self._llm = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype, device_map=device).to(device)
            self._tokenizer = LocalTokenizer(self.model_name, self.context_len, device)
            self._sampling_params = {(k if k!= "max_tokens" else "max_new_tokens"): v for k, v in self.sampling_params.items()}

            if self.response_mode == "json":
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
                
                parser = JsonSchemaParser(self.output_model.model_json_schema())
                self._sampling_params["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(
                        self._tokenizer.tokenizer, 
                        parser
                    )
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _run_batch(self, prompts, **kwargs):
        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self._tokenizer.tokenize_prompts(prompts)
            output_tokens = self._llm.generate(input_tokens, do_sample=True, **self._sampling_params)
            generated_texts = self._tokenizer.batch_decode(output_tokens, input_tokens)
        return generated_texts

    def run_batch(self, input_messages: list[str]) -> list[Digest | None]:
        if not self._llm: self.__enter__()

        generated_texts = self._run_batch([self.create_prompt(msg) for msg in input_messages])
        return [self.parse_output(text) for text in generated_texts]


class VLLMMicroAgent(MicroAgentBase):
    def __enter__(self):
        if not self._llm:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            self._llm = LLM(
                model=self.model_name,
                max_model_len=self.context_len,
                gpu_memory_utilization=0.88,
                enforce_eager=True,
                language_model_only=True,
                attention_config={"backend": "TRITON_ATTN"},
            )
            self._sampling_params = SamplingParams(
                **self.sampling_params,
                stop=["}\n", "\n\n", "\t\t", "\n \n", "\n\t\n"],
                structured_outputs=StructuredOutputsParams(
                    json=self.output_model.model_json_schema()
                ),
            )
        return self

    def run_batch(self, input_messages: list[str]) -> list[BaseModel]:
        responses = self._llm.chat([self.create_prompt(msg) for msg in input_messages], sampling_params=self._sampling_params, use_tqdm=False)
        return [self.parse_output(resp.outputs[0].text) if resp.outputs else None for resp in responses]


class RemoteMicroAgent(MicroAgentBase):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        context_len: int,
        instruction: str,
        input_template: str,
        output_model: Type[BaseModel],
        **sampling_params,
    ):
        super().__init__(
            model_name=model_name,
            context_len=context_len,
            instruction=instruction,
            input_template=input_template,
            output_model=output_model,
            **sampling_params,
        )
        self.base_url = base_url
        self.api_key = api_key
        self.sampling_params.pop('top_k', None)
        self.sampling_params.pop('repetition_penalty', None)
        self._sampling_params = {k:v for k, v in self.sampling_params.items() if k not in ["top_k", "repetition_penalty"] or not v}

    def __enter__(self):
        if not self._llm:
            from openai import OpenAI
            self._llm = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180, max_retries=3)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._llm:
            del self._llm
            self._llm = None
        return False

    @retry(stop=stop_after_attempt(REMOTE_RETRY_COUNT), wait=wait_random(*REMOTE_RETRY_JITTER), reraise=True)
    def _run_single(self, msg: str) -> BaseModel:
        response = self._llm.chat.completions.parse(
            model=self.model_name,
            messages=self.create_prompt(msg),
            response_format=self.output_model,
            **self._sampling_params
        )
        return response.choices[0].message.parsed
        
    def run_batch(self, input_messages: list[str]) -> list[BaseModel]:
        if not self._llm: self.__enter__()
        with ThreadPoolExecutor(max_workers=len(input_messages)) as exec:
            results = list(exec.map(self._run_single, input_messages))
        return results


class EntityExtractor:
    model_path: str
    confidence = 0.5
    _splitter = None

    _LABELS = [
        "person",
        "people",
        "organization",
        "company",
        "institution",
        "business",
        "city",
        "state",
        "country",
        "location",
        "stock",
        "ticker",
        "stockticker",
        "product",
    ]
    _LABEL_FIELD_MAPPINGS = {
        "person": "people",
        "people": "people",
        "organization": "companies",
        "company": "companies",
        "institution": "companies",
        "business": "companies",
        "city": "regions",
        "state": "regions",
        "country": "regions",
        "location": "regions",
        "stock": "stock_tickers",
        "ticker": "stock_tickers",
        "stockticker": "stock_tickers",
        "product": "products",
    }   
    _GLINER_BATCH_SIZE = int(os.getenv("GLINER_BATCH_SIZE", 16))
    _MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 4))
    _TOKEN_MARGIN = 16
    _MIN_SIZE = 100

    def __init__(self, model_path: str, context_len: int = 4096, threshold=0.5) -> None:
        # super().__init__(model_name=model_path, context_len=context_len, instruction=None, input_template=None, output_model=None)
        self.model_name = model_path
        self.context_len = context_len
        self.threshold = threshold
        self._llm = None
        self._label_embeddings = None        
        self._splitter = None
    
    def __enter__(self):
        if not self._llm:
            import torch
            from gliner import GLiNER
            from llama_index.core.text_splitter import TokenTextSplitter

            # config.fx_graph_cache = True
            self._llm = GLiNER.from_pretrained(
                self.model_name,
                max_length=self.context_len,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._label_embeddings = self._llm.encode_labels(
                self._LABELS, batch_size=len(self._LABELS)
            )
            self._splitter = TokenTextSplitter(
                chunk_size=self.context_len - self._TOKEN_MARGIN,
                chunk_overlap=self._TOKEN_MARGIN,
                include_metadata=False,
                include_prev_next_rel=False,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._llm:
            del self._llm
            self._llm = None
            del self._label_embeddings
            self._label_embeddings = None        
            del self._splitter
            self._splitter = None        
        clear_gpu_cache()
        return False    

    def parse_output(self, response):
        res = defaultdict(list)
        for ent in response:
            res[self._LABEL_FIELD_MAPPINGS[ent["label"]]].append(ent["text"])
        for k, v in res.items():
            res[k] = list({item.lower(): item for item in v}.values())
        return Digest(**res)

    def _split(self, text: str):
        chunks = self._splitter.split_text(text)[:self._MAX_CHUNKS]
        if len(chunks) > 1 and len(chunks[-1]) < self._MIN_SIZE: chunks = chunks[:-1]
        return chunks

    def _create_chunks(self, texts: list[str]) -> tuple[list[str], list[int], list[int]]:
        texts = texts if isinstance(texts, list) else [texts]
        
        chunks = list(map(self._split, texts))
        counts = list(map(len, chunks))
        start_idx = [0]*len(chunks)
        for i in range(1,len(counts)):
            start_idx[i] = start_idx[i-1]+counts[i-1]
        return list(chain(*chunks)), start_idx, counts

    def _merge_chunks(self, digests: list[Digest]):
        to_merge = [digest for digest in digests if digest]
        if not to_merge: return

        people, companies, regions, stock_tickers, products = set(), set(), set(), set(), set()
        for digest in to_merge:
            if digest.regions: regions.update(digest.regions)
            if digest.people: people.update(digest.people)
            if digest.products: products.update(digest.products)
            if digest.companies: companies.update(digest.companies)
            if digest.stock_tickers: stock_tickers.update(digest.stock_tickers)
        return Digest(
            people=list(people), 
            companies=list(companies), 
            regions=list(regions), 
            stock_tickers=list(stock_tickers), 
            products=list(products)
        )

    def run_batch(self, input_messages: list[str]):
        chunks, start_idx, counts = self._create_chunks(input_messages)
        entities = self._llm.batch_predict_with_embeds(
            chunks,
            labels_embeddings=self._label_embeddings,
            labels=self._LABELS,
            threshold=self.threshold,
            batch_size=self._GLINER_BATCH_SIZE,
        )
        digests = [self.parse_output(group) if group else None for group in entities]        
        return [self._merge_chunks(digests[start:start+count]) for start, count in zip(start_idx, counts)]


def create_micro_agent(model_path: str, context_len: int = DEFAULT_CONTEXT_LEN, instruction: str = None, input_template: str = None, output_model: Type[BaseModel] = Digest, **kwargs) -> MicroAgentBase:
    if model_path.startswith(VLLM_PREFIX):
        return VLLMMicroAgent(
            model_path.removeprefix(VLLM_PREFIX),
            context_len=context_len,
            instruction=instruction,
            input_template=input_template,
            output_model=output_model,
            **kwargs,
        )
    elif kwargs.get("base_url") and kwargs.get("api_key"):
        return RemoteMicroAgent(
            model_path,
            base_url=kwargs.pop("base_url"),
            api_key=kwargs.pop("api_key"),
            context_len=context_len,
            instruction=instruction,
            input_template=input_template,
            output_model=output_model,
            **kwargs,
        )
    else:
        return TransformerMicroAgent(
            model_path, 
            context_len=context_len, 
            instruction=instruction,
            input_template=input_template,
            output_model=output_model, 
            **kwargs
        )

# ---------------------
# PARSING UTILITIES
# ---------------------

def _strip_fences(text: str) -> str:
    return (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```markdown")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

M_GIST = "# GIST"
M_CATEGORIES = "# DOMAINS"
M_ENTITIES = "# ENTITIES"
M_TOPIC = "# TOPIC"
M_REGIONS = "# REGIONS"
M_SUMMARY = "# SUMMARY"
M_KEYPOINTS = "# KEY POINTS"
M_KEYEVENTS = "# KEY EVENTS"
M_DATAPOINTS = "# KEY POINTS"
M_INSIGHT = "# ACTIONABLE INSIGHT"
M_FIELDS = [
    M_GIST,
    M_CATEGORIES,
    M_ENTITIES,
    M_TOPIC,
    M_REGIONS,
    M_SUMMARY,
    M_KEYPOINTS,
    M_KEYEVENTS,
    M_DATAPOINTS,
    M_INSIGHT,
]
M_START = "```markdown"
M_END = "```"
MARKDOWN_HEADERS = ["# ", "## ", "### ", "#### ", "**"]


def parse_markdown(response: str):
    digest = Digest(raw=response)
    response = response.strip().removeprefix(M_START).removesuffix(M_END).strip()
    last = None
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        if any(field in line for field in M_FIELDS):
            last = line
        elif M_GIST in last:
            digest.gist = line
        elif M_CATEGORIES in last:
            digest.categories = split_parts(line)
        elif C_ENTITIES in last:
            digest.entities = split_parts(line)
        elif M_TOPIC in last:
            digest.topic = line
        elif C_REGIONS in last:
            digest.regions = split_parts(line)
        elif M_SUMMARY in last:
            digest.summary = (digest.summary + "\n" + line) if digest.summary else line
        elif C_KEYPOINTS in last:
            if not digest.keypoints:
                digest.keypoints = []
            digest.keypoints.append(line.removeprefix("- ").removeprefix("* "))
        elif M_INSIGHT in last:
            digest.insight = line

    return digest


C_KEYPOINTS = "P:"
C_KEYEVENTS = "E:"
C_DATAPOINTS = "D:"
C_REGIONS = "R:"
C_ENTITIES = "N:"
C_CATEGORIES = "C:"
C_SENTIMENTS = "S:"
COMPRESSED_FIELDS = [
    C_KEYPOINTS,
    C_KEYEVENTS,
    C_DATAPOINTS,
    C_REGIONS,
    C_ENTITIES,
    C_CATEGORIES,
    C_SENTIMENTS,
]


def parse_compressed(response: str):
    if not response:
        return response

    results = {"P:": [], "E:": [], "D:": [], "N:": [], "R:": []}
    current_pos = 0
    while current_pos < len(response):
        key = response[current_pos : current_pos + 2]
        if key in results:
            next_key_pos = [
                response.find(";" + next_key, current_pos + 2)
                for next_key in results.keys()
            ]
            end = min([pos for pos in next_key_pos if pos > -1], default=len(response))
            if response[end - 1] == ";":
                ext = response[current_pos + 2 : end - 1]
            else:
                ext = response[current_pos + 2 : end]

            results[key].extend(
                chain(*(item.strip().split(";") for item in ext.strip().split("|")))
            )
            current_pos = end

        current_pos += 1

    response = ""
    for key, value in results.items():
        if not value:
            continue
        response += key + "|".join(v.strip() for v in value) + ";"

    return Digest(
        raw=response,
        keypoints=results.get("P:") or None,
        keyevents=results.get("E:") or None,
        datapoints=results.get("D:") or None,
        entities=results.get("N:") or None,
        regions=results.get("R:") or None,
    )
