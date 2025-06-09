from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

model_id = "avsolatorio/GIST-small-Embedding-v0"
onnx_path = Path("models/GIST-small-Embedding-v0")

# Create ONNX directory
onnx_path.parent.mkdir(parents=True, exist_ok=True)

# Load and convert model to ONNX
model = ORTModelForFeatureExtraction.from_pretrained(model_id, cache_dir=".models", subfolder="onnx", file_name="model_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save ONNX model and tokenizer
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)