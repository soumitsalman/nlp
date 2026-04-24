import base64
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional, Type, Union

from icecream import ic
from retry import retry
from pydantic import BaseModel

from .models_old import Digest
from .utils import *

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.3,
    "top_k": 60,
    "repetition_penalty": 1.05,
}
DEFAULT_CONTEXT_LEN = 32768
BATCH_SIZE = int(os.getenv("BATCH_SIZE", os.cpu_count()))

log = logging.getLogger(__name__)


class DigestorBase(ABC):
    def __init__(
        self,
        model_name: str,
        context_len: int = DEFAULT_CONTEXT_LEN,
        output_model: Type[BaseModel] = Digest,
        **sampling_params,
    ):
        self.model_name = model_name
        self.output_model = output_model
        self.context_len = context_len
        self.sampling_params = {
            **DEFAULT_SAMPLING_PARAMS,
            **sampling_params,
            "max_tokens": context_len,
        }
        self._llm = None
        self._sampling_params = None

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._llm:
            del self._llm
            self._llm = None
            del self._sampling_params
            self._sampling_params = None
            clear_gpu_cache()
        return False

    def make_prompt(self, input_msg: str):
        system_prompt = getattr(self, "system_prompt", None)
        if system_prompt:
            return f"{system_prompt}: {input_msg}"
        return input_msg
    
    def _create_prompts(self, input_messages: list[str]):
        return input_messages
    
    def _parse_output(self, response) -> Optional[Digest]:
        return response
    
    @abstractmethod
    def run_batch(self, input_messages: list[str]) -> list[Digest | None]:
        raise NotImplementedError

    # def run(self, input_msg: str):
    #     if hasattr(self, "_run"):
    #         response = self._run(self.make_prompt(input_msg))
    #         output_parser = getattr(self, "output_parser", None)
    #         if output_parser:
    #             return output_parser(response)
    #         return response

    #     responses = self.run_batch([input_msg])
    #     return responses[0] if responses else None

    # def run_batch(self, input_messages: list[str]) -> list[Digest | None]:
    #     if self._llm:
    #         responses = self._llm.chat(
    #             self._create_prompts(input_messages),
    #             sampling_params=self._sampling_params,
    #             use_tqdm=False,
    #         )
    #         return [self._parse_output(resp.outputs[0].text) if resp.outputs else None for resp in responses]

    #     if hasattr(self, "_run_batch"):
    #         prompts = list(map(self.make_prompt, input_messages))
    #         responses = self._run_batch(prompts)
    #         output_parser = getattr(self, "output_parser", None)
    #         if output_parser:
    #             responses = list(map(output_parser, responses))
    #         return responses

    #     raise NotImplementedError


class LocalTokenizer:
    tokenizer = None
    max_input_tokens = None
    max_output_tokens = None
    device = None

    def __init__(
        self,
        model_id,
        context_len: int,
        max_output_tokens: int = None,
        device: str = None,
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, max_length=context_len, use_fast=True
        )
        self.max_input_tokens = context_len
        self.max_output_tokens = max_output_tokens
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


class TransformerDigestor(DigestorBase):
    device: str = None
    max_output_tokens: int = 0
    _tokenizer = None
    _model = None

    def __init__(
        self,
        model_path: str,
        context_len: int,
        max_output_tokens: int,
        output_parser: Callable,
    ):
        import torch

        super().__init__(model_name=model_path, context_len=context_len)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.max_output_tokens = max_output_tokens
        self.output_parser = output_parser

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._unload_model()
        return False

    def _create_prompts(self, input_messages: list[str]):
        raise NotImplementedError

    def _parse_output(self, text: str):
        raise NotImplementedError

    def _run(self, prompt):
        import torch

        with torch.inference_mode(), torch.amp.autocast(self.device, self.dtype):
            input_tokens = self.tokenizer.tokenize_prompts(prompt)
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.max_output_tokens,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
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

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, dtype=self.dtype, device_map=self.device
            ).to(self.device)
        return self._model

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = LocalTokenizer(
                self.model_name,
                self.context_len,
                self.max_output_tokens,
                self.device,
            )
        return self._tokenizer

    def _unload_model(self):
        if not self._model:
            return
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

            self._model = OVModelForSeq2SeqLM.from_pretrained(self.model_name)
        return self._model

    def _unload_model(self):
        if not self._model:
            return
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
                eos_token_id=self.tokenizer.eos_token_id,
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
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_texts = self.tokenizer.batch_decode(output_tokens)
        return generated_texts

    @property
    def model(self):
        if not self._model:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM

            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                provider_options={
                    "CPUExecutionProvider": {
                        "arena_extend_strategy": "kSameAsRequested",
                        "cpu_threads": os.cpu_count()
                        - 1,  # Use all available CPU cores
                        "enable_parallel_execution": True,
                        "execution_mode": "parallel",  # or 'parallel' for some models
                    }
                },
                provider="CPUExecutionProvider",
            )
        return self._model

    def _unload_model(self):
        if not self._model:
            return
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None


class NamedEntityExtractor(DigestorBase):
    _model = None
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
        "organization": "organizations",
        "company": "organizations",
        "institution": "organizations",
        "business": "organizations",
        "city": "regions",
        "state": "regions",
        "country": "regions",
        "location": "regions",
        "stock": "stock_tickers",
        "ticker": "stock_tickers",
        "stockticker": "stock_tickers",
        "product": "products",
    }

    def __init__(self, model_path: str, context_len: int = 4096, confidence=0.5) -> None:
        super().__init__(model_name=model_path, context_len=context_len)
        self.threshold = confidence
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
        if self._model:
            del self._model
            self._model = None
        if self._label_embeddings:
            del self._label_embeddings
            self._label_embeddings = None
        clear_gpu_cache()

    def _create_prompts(self, input_messages: list[str]):
        return input_messages

    def _parse_output(self, response):
        res = defaultdict(list)
        for ent in response:
            res[self._LABEL_FIELD_MAPPINGS[ent["label"]]].append(ent["text"])
        for k, v in res.items():
            res[k] = list({item.lower(): item for item in v}.values())
        return Digest(**res, raw="")

    def run_batch(self, input_messages: list[str]):
        entities = self._llm.batch_predict_with_embeds(
            input_messages, 
            labels_embeddings=self._label_embeddings, 
            labels=self._LABELS,
            threshold=self.threshold,
        )
        return [self._parse_output(group) if group else None for group in entities]

_INST_MSG = """
EXTRACT {fields} FROM content IF specified
=== content ===
{text}
"""
 
class vLLMDigestorStructuredOutput(DigestorBase):
    _STRUCTURED_SYS_MSG = """RETURN=JSON object matching schema
    EXCLUDE=unspecified data, implied assessments, assumptions
    REMOVE=N/A,null values, empty fields
    AVOID=markdown, prose, code fences, null placeholders, implied information, assumptions""" 

    def _create_prompts(self, input_messages: list[str]):
        prompt = lambda msg: [
            {"role": "system", "content": self._STRUCTURED_SYS_MSG},
            {"role": "user", "content": _INST_MSG.format(fields=",".join(self.output_model.model_fields.keys()), text=msg[:self.context_len>>2])},
        ]
        return [prompt(msg) for msg in input_messages]        

    def _parse_output(self, text: str):
        try: return self.output_model.model_validate_json(_strip_json_fences(text))
        except Exception: log.warning("failed parsing: %s", text, exc_info=True)
    
    def __enter__(self):
        if not self._llm:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(
                **self.sampling_params,
                structured_outputs=StructuredOutputsParams(
                    json=self.output_model.model_json_schema(),
                    disable_any_whitespace=True,
                ),
            )
        return self

    def run_batch(self, input_messages: list[str]) -> list[Digest|None]:
        responses = self._llm.chat(self._create_prompts(input_messages), sampling_params=self._sampling_params, use_tqdm=False)
        return [self._parse_output(resp.outputs[0].text) if resp.outputs else None for resp in responses]


class vLLMDigestorToolCall(DigestorBase):
    def __enter__(self):
        if not self._llm:
            from vllm import LLM, SamplingParams
            
            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(**self.sampling_params)
        return self

    def _create_prompts(self, input_messages: list[str]):
        prompt = lambda msg: [
            {"role": "system", "content": f"List of tools: {json.dumps([self._build_tool_schema(self.output_model)])}"},
            {"role": "user", "content":  _INST_MSG.format(fields=",".join(self.output_model.model_fields.keys()), text=msg[:self.context_len >> 2])},
        ]
        return [prompt(msg) for msg in input_messages]

    def _parse_output(self, text: str):
        cleaned = _strip_json_fences(text)
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

        return model_type.model_validate(arguments)
    
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


def _strip_json_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

def _safe_model_dump(item: Optional[Digest]):
    if item is None: return None
    return item.model_dump(mode="json", exclude_none=True, exclude_unset=True, exclude_defaults=True)


def from_path(
    model_path: str,
    base_url: str = None,
    api_key: str = None,
    context_len: int = None,
    max_output_tokens: int = None,
    system_prompt: str = None,
    output_parser: Callable = None,
    json_mode: bool = False,
) -> DigestorBase:

    if base_url:
        NotImplementedError("Remote digestor not supported")
    if model_path.startswith(OPENVINO_PREFIX):
        return OVDigestor(
            model_path,
            context_len=context_len,
            max_output_tokens=max_output_tokens,
            output_parser=output_parser,
        )
    elif model_path.startswith(ONNX_PREFIX):
        return ORTDigestor(
            model_path,
            context_len=context_len,
            max_output_tokens=max_output_tokens,
            output_parser=output_parser,
        )
    else:
        return TransformerDigestor(
            model_path,
            context_len=context_len,
            max_output_tokens=max_output_tokens,
            output_parser=output_parser,
        )
