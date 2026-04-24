from pydantic import BaseModel
from typing import Type, Optional, Any
import json
from abc import ABC, abstractmethod
import logging
from .utils import clear_gpu_cache
from .models import Digest
from icecream import ic

log = logging.getLogger("digestor")

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.3,
    "top_k": 60,
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
    def _parse_output(self, text: str) -> Digest|None:
        raise NotImplementedError

    def run_batch(self, input_messages: list[str]) -> list[Digest|None]:
        responses = self._llm.chat(self._create_prompts(input_messages), sampling_params=self._sampling_params, use_tqdm=False)
        return [self._parse_output(resp.outputs[0].text) if resp.outputs else None for resp in responses]

