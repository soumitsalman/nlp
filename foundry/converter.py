import os

def convert_to_onnx(model_id: str, save_directory):
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer, AutoQuantizationConfig
    from transformers import AutoTokenizer

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True, cache_dir=os.getenv('HF_HOME'))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    os.makedirs(save_directory, exist_ok=True)
  
    quantizer = ORTQuantizer.from_pretrained(model)
    quantizer.quantize(save_dir=save_directory, quantization_config=AutoQuantizationConfig.avx512(is_static=False))
    tokenizer.save_pretrained(save_directory)

def convert_to_openvino(model_id: str, save_directory: str, model_class):
    from optimum.intel.openvino.quantization import OVQuantizer
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # Load the model and tokenizer
    model = model_class.from_pretrained(model_id, export=True, cache_dir=os.getenv('HF_HOME'))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Quantize the model to INT8 and save it
    os.makedirs(save_directory, exist_ok=True)
    quantizer = OVQuantizer.from_pretrained(model)
    quantizer.quantize(save_directory=save_directory)
    tokenizer.save_pretrained(save_directory)

if __name__ == "__main__":
    from optimum.intel.openvino import OVModelForFeatureExtraction, OVModelForCausalLM, OVModelForSeq2SeqLM
    models = ["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "google/long-t5-tglobal-base", "google/long-t5-local-base"]
    model_classes = [OVModelForCausalLM, OVModelForCausalLM, OVModelForSeq2SeqLM, OVModelForSeq2SeqLM]
    for path, mc in zip(models, model_classes):
        convert_to_openvino(path, ".models/"+path.split("/")[-1]+"-openvino", mc)
    # convert_to_openvino("soumitsr/SmolLM2-360M-Instruct-article-digestor", "models/smollm2-360M-instruct-article-digestor-ovquant")
    # with open("embedder-test-data.json", "r") as file:
    #     data = json.load(file)
    # test_embedder("avsolatorio/GIST-small-Embedding-v0", "/root/pycoffeemaker/.models/gist-small-embedding-v0-q8_0.gguf", [item['digest'] for item in data])
    # convert_to_onnx("soumitsr/SmolLM2-360M-Instruct-article-digestor", ".models/SmolLM2-360M-Instruct-article-digestor/onnx")