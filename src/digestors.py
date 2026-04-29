import base64
from itertools import chain
import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Type
from pydantic import BaseModel
from .models import *
from .utils import *
from icecream import ic

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
}
DEFAULT_CONTEXT_LEN = 32768

log = logging.getLogger("digestor")

class DigestorBase(ABC):
    def __init__(
        self,
        model_name: str,
        context_len: int = DEFAULT_CONTEXT_LEN,
        output_model: Type[BaseModel] = Digest,
        response_mode: str = "json",
        **sampling_params,
    ):
        self.model_name = model_name
        self.context_len = context_len
        self.output_model = output_model
        self.response_mode = response_mode
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
    
    def _create_prompts(self, input_messages: list[str]):
        return input_messages
    
    def _parse_output(self, response: str):
        response = _strip_fences(response)
        try:
            if self.response_mode == "json": return self.output_model.model_validate_json(response)
            if self.response_mode == "compressed": return parse_compressed(response)
            if self.response_mode == "markdown": return parse_markdown(response)
            if self.response_mode == "tool_call": raise NotImplementedError
            return response
        except: 
            log.warning("failed parsing: %s", response, exc_info=True)
    
    @abstractmethod
    def run_batch(self, input_messages: list[str]) -> list[Digest | None]:
        raise NotImplementedError()


class LocalTokenizer:
    tokenizer = None
    max_input_tokens = None
    max_new_tokens = None
    device = None

    def __init__(
        self,
        model_id,
        context_len: int,
        max_new_tokens: int = None,
        device: str = None,
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, max_length=context_len, use_fast=True
        )
        self.max_input_tokens = context_len
        self.max_new_tokens = max_new_tokens
        self.device = device
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_prompts(self, prompts: str | list[str]):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        if self.device:
            tokens = tokens.to(self.device)
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def batch_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

# needs 1.3 for repetition penalty support, and max_new_tokens = 384
class TransformerDigestor(DigestorBase):
    device: str = None
    max_output_tokens: int = 0
    _tokenizer = None
    _model = None

    def __init__(
        self,
        model_path: str,
        context_len: int,
        output_model: Type[BaseModel] = Digest,
        response_mode: str = "json",
        **sampling_params,
    ):
        import torch

        super().__init__(
            model_name=model_path,
            context_len=context_len,
            output_model=output_model,
            response_mode=response_mode,
            **sampling_params,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.max_new_tokens = sampling_params.get("max_new_tokens", None)
        self._tokenizer = None

    def __enter__(self):
        if not self._llm:
            from transformers import AutoModelForSeq2SeqLM

            self._llm = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, dtype=self.dtype, device_map=self.device
            ).to(self.device)
        
            self._sampling_params = {
                **{k: v for k, v in self.sampling_params.items() if k != "max_tokens"},
                "no_repeat_ngram_size": 3,
            }
        
            self._tokenizer = LocalTokenizer(
                self.model_name,
                self.context_len,
                self.max_new_tokens,
                self.device,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _run_batch(self, prompts, **kwargs):
        import torch

        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self.tokenizer.tokenize_prompts(prompts)
            output_tokens = self.model.generate(**input_tokens, **self._sampling_params)
            generated_texts = self.tokenizer.batch_decode(output_tokens)
        return generated_texts

    def run_batch(self, input_messages: list[str]) -> list[Digest | None]:
        if not self._llm:
            self.__enter__()

        generated_texts = self._run_batch(self._create_prompts(input_messages))
        return [self._parse_output(text) for text in generated_texts]    

class OVDigestor(TransformerDigestor):
    def __enter__(self):
        if not self._llm:
            from optimum.intel.openvino import OVModelForSeq2SeqLM

            self._llm = OVModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self._tokenizer = LocalTokenizer(
                self.model_name,
                self.context_len,
                self.max_new_tokens,
                self.device,
            )

            self._sampling_params = {
                **{k: v for k, v in self.sampling_params.items() if k != "max_tokens"},
                "max_new_tokens": self.max_output_tokens,
            }

        return self

    def _run_batch(self, prompts):
        input_tokens = self.tokenizer.tokenize_prompts(prompts)
        output_tokens = self.model.generate(**input_tokens, **self._sampling_params)
        return self.tokenizer.batch_decode(output_tokens)


class ORTDigestor(TransformerDigestor):
    def __enter__(self):
        if not self._llm:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM

            self._llm = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                provider_options={
                    "CPUExecutionProvider": {
                        "arena_extend_strategy": "kSameAsRequested",
                        "cpu_threads": os.cpu_count() - 1,
                        "enable_parallel_execution": True,
                        "execution_mode": "parallel",
                    }
                },
                provider="CPUExecutionProvider",
            )

            self._tokenizer = LocalTokenizer(
                self.model_name,
                self.context_len,
                self.max_new_tokens,
                self.device,
            )     

            self._sampling_params = {
                **{k: v for k, v in self.sampling_params.items() if k != "max_tokens"},
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

        return self

    def _run_batch(self, prompts):
        input_tokens = self.tokenizer.tokenize_prompts(prompts)
        output_tokens = self.model.generate(**input_tokens, **self._sampling_params)
        return self.tokenizer.batch_decode(output_tokens)


class NamedEntityExtractor(DigestorBase):
    model_path: str
    confidence = 0.5
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

    def __init__(self, model_path: str, context_len: int = 4096, threshold=0.5) -> None:
        super().__init__(model_name=model_path, context_len=context_len)
        self.threshold = threshold
        self._label_embeddings = None

    def __enter__(self):
        if not self._llm:
            import torch
            from gliner import GLiNER

            self._llm = GLiNER.from_pretrained(
                self.model_name,
                max_length=self.context_len,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._label_embeddings = self._llm.encode_labels(self._LABELS, batch_size=len(self._LABELS))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._label_embeddings is not None:
            del self._label_embeddings
            self._label_embeddings = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _parse_output(self, response):
        res = defaultdict(list)
        for ent in response:
            res[self._LABEL_FIELD_MAPPINGS[ent["label"]]].append(ent["text"])
        for k, v in res.items():
            res[k] = list({item.lower(): item for item in v}.values())
        return Digest(**res)

    def run_batch(self, input_messages: list[str]):
        entities = self._llm.batch_predict_with_embeds(
            input_messages, 
            labels_embeddings=self._label_embeddings, 
            labels=self._LABELS,
            threshold=self.threshold,
        )
        return [self._parse_output(group) if group else None for group in entities]

 
class VLLMDigestor(DigestorBase):
    _STRUCTURED_SYS_MSG = """RETURN=JSON object matching schema
    EXCLUDE=unspecified data, implied assessments, assumptions
    REMOVE=N/A,null values, empty fields
    AVOID=markdown, prose, code fences, null placeholders, implied information, assumptions""" 

    _INST_MSG = """DETERMINE {fields} FROM content IF specified\n=== content ===\n{text}"""

    def _create_prompts(self, input_messages: list[str]):
        prompt = lambda msg: [
            {"role": "system", "content": self._STRUCTURED_SYS_MSG},
            {"role": "user", "content": self._INST_MSG.format(fields=",".join(self.output_model.model_fields.keys()), text=msg[:self.context_len>>1])},
        ]
        return [prompt(msg) for msg in input_messages]        

    def _parse_output(self, text: str):
        try: return self.output_model.model_validate_json(_strip_fences(text))
        except Exception: log.warning("failed parsing: %s", text, exc_info=True)
    
    def __enter__(self):
        if not self._llm:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(
                **self.sampling_params,
                max_tokens=2048,
                stop=["}\n", "\n\n", "\t\t", "\n \n", "\n\t\n"],
                structured_outputs=StructuredOutputsParams(
                    json=self.output_model.model_json_schema()
                ),
            )
        return self

    def run_batch(self, input_messages: list[str]) -> list[Digest|None]:
        responses = self._llm.chat(self._create_prompts(input_messages), sampling_params=self._sampling_params, use_tqdm=False)
        return [self._parse_output(resp.outputs[0].text) if resp.outputs else None for resp in responses]


class VLLMDigestorToolCall(VLLMDigestor):
    TOOL_NAME = "extract_fields"
    def __enter__(self):
        if not self._llm:
            from vllm import LLM, SamplingParams
            
            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(**self.sampling_params)
        return self

    def _create_prompts(self, input_messages: list[str]):
        prompt = lambda msg: [
            {"role": "system", "content": f"List of tools: {json.dumps([self._build_tool_schema(self.output_model)])}"},
            {"role": "user", "content":  self._INST_MSG.format(fields=",".join(self.output_model.model_fields.keys()), text=msg[:self.context_len >> 2])},
        ]
        return [prompt(msg) for msg in input_messages]

    def _parse_output(self, text: str):
        cleaned = _strip_fences(text)
        payload = json.loads(cleaned)

        if isinstance(payload, list):
            if len(payload) != 1:
                raise ValueError(f"Expected exactly one tool call, got {len(payload)}")
            payload = payload[0]

        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected tool call payload type: {type(payload).__name__}")

        if payload.get("name") == TOOL_NAME:
            arguments = payload.get("arguments", {})
        elif payload.get("function", {}).get("name") == TOOL_NAME:
            arguments = payload["function"].get("arguments", {})
        elif payload.get("tool_name") == TOOL_NAME:
            arguments = payload.get("arguments", {})
        else:
            raise ValueError(f"Could not find {TOOL_NAME} tool call in payload: {payload}")

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return self.output_model.model_validate(arguments)
    
    def run_batch(self, input_messages: list[str]) -> list[Digest|None]:
        responses = self._llm.chat(self._create_prompts(input_messages), sampling_params=self._sampling_params, use_tqdm=False)
        return [self._parse_output(resp.outputs[0].text) if resp.outputs else None for resp in responses]


    @classmethod
    def _build_tool_schema(cls, model_type: Type[BaseModel]):
        return {
            "name": "extract_fields",
            "description": "Extracts specified fields and contents from input.",
            "parameters": model_type.model_json_schema(),
        }

def from_path(
    model_path: str,
    context_len: int = None,
    **kwargs
) -> DigestorBase:

    if model_path.startswith(OPENVINO_PREFIX):
        return OVDigestor(
            model_path.removeprefix(OPENVINO_PREFIX),
            context_len=context_len,
            output_model=Digest,
            **kwargs,
        )
    elif model_path.startswith(ONNX_PREFIX):
        return ORTDigestor(
            model_path.removeprefix(ONNX_PREFIX),
            context_len=context_len,
            output_model=Digest,
            **kwargs,
        )
    elif model_path.startswith(VLLM_PREFIX):
        return VLLMDigestor(
            model_path.removeprefix(VLLM_PREFIX),
            context_len=context_len,
            output_model=Digest,
            **kwargs
        )
    else:
        return TransformerDigestor(
            model_path,
            context_len=context_len,
            output_model=Digest,
            **kwargs
        )

def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```markdown").removeprefix("```").removesuffix("```").strip()

def _safe_model_dump(item: Optional[Digest]):
    if item is None: return None
    return item.model_dump(mode="json", exclude_none=True, exclude_unset=True, exclude_defaults=True)

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
            digest.summary = (
                (digest.summary + "\n" + line) if digest.summary else line
            )
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
            end = min(
                [pos for pos in next_key_pos if pos > -1], default=len(response)
            )
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