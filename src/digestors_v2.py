from pydantic import BaseModel
from typing import Type, Optional, Any
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer
from abc import ABC, abstractmethod

from .utils import clear_gpu_cache
from .models_v2 import Digest

from icecream import ic

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.05
}
DEFAULT_CONTEXT_LEN = 32768

class DigestorBase(ABC):
    def __init__(self, model_name: str, output_model: Type[BaseModel] = Digest, context_len: int = DEFAULT_CONTEXT_LEN, **sampling_params):
        self.model_name = model_name
        self.output_model = output_model
        self.context_len = context_len
        self.sampling_params = {**DEFAULT_SAMPLING_PARAMS, **sampling_params, "max_tokens": context_len}
        self._llm = None
        self._sampling_params = None

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        if self._llm:
            del self._llm
            self._llm = None
            del self._sampling_params
            self._sampling_params = None
            clear_gpu_cache()

    @abstractmethod
    def _create_prompts(self, input_messages: list[str]):
        raise NotImplementedError

    @abstractmethod
    def _parse_output(self, text: str):
        raise NotImplementedError

    def run_batch(self, input_messages: list[str]):
        outputs = self._llm.chat(self._create_prompts(input_messages), sampling_params=self._sampling_params, use_tqdm=False)
        results = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            try:
                results.append(self._parse_output(text))
            except Exception as e:
                ic(e)
                results.append(None)
        return results

_INST_MSG = """
EXTRACT {fields} FROM content IF specified
=== content ===
{text}
"""
 
class DigestorStructuredOutput(DigestorBase):
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
        return self.output_model.model_validate_json(_strip_json_fences(text))
    
    def __enter__(self):
        if not self._llm:
            self._llm = LLM(model=self.model_name)
            self._sampling_params = SamplingParams(
                **self.sampling_params,
                structured_outputs=StructuredOutputsParams(
                    json=self.output_model.model_json_schema(),
                    disable_any_whitespace=True,
                ),
            )
        return self


class DigestorToolCall(DigestorBase):
    def __enter__(self):
        if not self._llm:
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

def _serialize_outputs(outputs: list[Optional[Digest]]) -> list[Any]:
    return [_safe_model_dump(item) for item in outputs]
