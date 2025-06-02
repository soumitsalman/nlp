from sentence_transformers import SentenceTransformer
import json
import random
from icecream import ic
import os
from datetime import datetime
from llama_cpp import Llama
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def convert_to_onnx(model_id: str, dir):
    ort_model = ORTModelForCausalLM.from_pretrained(model_id, trust_remote_code = True, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
    ort_model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)


def test_embedder(xformer_path: str, llama_cpp_path: str, data):
    xformer = SentenceTransformer(xformer_path, backend="onnx", model_kwargs={"file_name": "model_quantized.onnx"}, tokenizer_kwargs={"truncation": True, "max_length": 512}, trust_remote_code=True)
    llamamodel = Llama(model_path=llama_cpp_path, n_ctx=512, embedding=True, verbose=False)

    start = datetime.now()
    vecs = xformer.encode(data, batch_size=os.cpu_count(), convert_to_numpy=False, convert_to_tensor=False)
    vecs = [v.tolist() for v in vecs]
    ic(datetime.now() - start)    

    start = datetime.now()
    result = llamamodel.create_embedding(data)
    vecs2 = [data['embedding'] for data in result['data']]
    ic(datetime.now() - start)

    # [ic(v1[0] - v2[0]) for v1, v2 in zip(vecs, vecs2)]

if __name__ == "__main__":
    # with open("embedder-test-data.json", "r") as file:
    #     data = json.load(file)
    # test_embedder("avsolatorio/GIST-small-Embedding-v0", "/root/pycoffeemaker/.models/gist-small-embedding-v0-q8_0.gguf", [item['digest'] for item in data])
    convert_to_onnx("soumitsr/SmolLM2-360M-Instruct-article-digestor", ".models/SmolLM2-360M-Instruct-article-digestor/onnx")