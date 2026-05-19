# Coffeemaker — Embeddings & Digest Utilities

Professional, lightweight utilities for two common NLP pipelines:

- Embedding generation (vectorize text for retrieval and semantic search)
- Digest generation (produce structured JSON with title, summary, highlights, named entities, and domains)

The package exposes the core modules at the top level (imports work as `from nlp import embedders, digestors, models`). Examples in this README use the package-level imports.

## Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

Import the modules from the package and create clients using the `from_path(...)` factories.

### Embedding example

```python
from nlp import embedders

texts = [
    "AI will change how developers build software.",
    "Open-source models enable local experimentation."
]

# Create an embedder. Provide a remote base_url/api_key for API-backed embeddings
# or a local model identifier/path for on-device embeddings.
embedder = embedders.from_path(model_path="sentence-transformers/all-MiniLM-L6-v2", context_len=512)
with embedder:
    vectors = embedder.embed_documents(texts)        # -> list[list[float]]
    qvec = embedder.embed_query("What will change in developer tooling?")  # -> list[float]

print(f"Generated {len(vectors)} vectors; first vector length: {len(vectors[0])}")
```

### Digest example

```python
from nlp import digestors, models

article = "Long article text or markdown content to summarize and extract entities from..."

# Create a digestor and provide an output parser to convert raw model output into a typed model
digestor = digestors.from_path(
    model_path="soumitsr/led-base-article-digestor",
    max_input_tokens=1024,
    max_output_tokens=256,
    output_parser=models.Digest.parse_markdown
)

with digestor:
    result = digestor.run(article)

if isinstance(result, models.Digest):
    print(result.raw)
    print(result.keypoints)
else:
    print(result)
```

#### Batching

```python
inputs = [article, article]
with digestor:
    digests = digestor.run_batch(inputs)  # -> list[models.Digest | str]
```

## Backend selection

The `from_path(...)` factories automatically select a backend based on the model path prefix. Supported backends:

| Prefix | Backend | Use case |
|--------|---------|----------|
| (none, default) | HuggingFace Transformer (or Sentence Transformer for embedders) | Local model from HuggingFace Hub or local path |
| `onnx://` | ONNX Runtime | Optimized inference on CPU/GPU with ONNX models |
| `openvino://` | OpenVINO | Intel-optimized models for CPU inference |
| `llamacpp://` | llama.cpp | Quantized models for lightweight local inference |
| `infinity://` | Infinity in-process (`SyncEngineArray`) | High-throughput local embeddings via `infinity_emb` |
| `https://` | OpenAI-compatible API | Remote API endpoints (e.g., OpenAI, custom servers) |

Examples:

```python
# HuggingFace Transformer (default)
embedder = embedders.from_path("sentence-transformers/all-MiniLM-L6-v2", context_len=512)

# ONNX backend
embedder = embedders.from_path("onnx://./model.onnx", context_len=512)

# OpenVINO backend
embedder = embedders.from_path("openvino://./model_ir.xml", context_len=512)

# llama.cpp backend
embedder = embedders.from_path("llamacpp://./model.gguf", context_len=512)

# Infinity in-process backend
embedder = embedders.from_path("infinity://BAAI/bge-small-en-v1.5", context_len=512)

# OpenAI-compatible remote API
embedder = embedders.from_path(
    model_path="text-embedding-3-small",  # model name or ID
    context_len=512,
    base_url="https://api.openai.com/v1",
    api_key="sk-..."
)
```

## API summary

- `embedders.from_path(model_path, context_len, base_url=None, api_key=None)` -> returns an `EmbedderBase` implementation
  - `embed_documents(text_or_list)` -> `list[list[float]]` or `list[float]` for a single string
    - When input text exceeds the model's context window, the backend automatically chunks the input into smaller pieces, embeds each chunk separately, and then computes the **mean** of all chunk embeddings to produce a single vector representing the full document.
  - `embed_query(query)` -> `list[float]`
  - supports context manager usage (`with embedder:`)

- `digestors.from_path(model_path, max_input_tokens, max_output_tokens, base_url=None, api_key=None, output_parser=None)` -> returns a `DigestorBase` implementation
  - `run(input_msg)` -> `str` (raw) or the `output_parser` return value (commonly `models.Digest` or a `dict`)
  - `run_batch(list_of_inputs)` -> `list` of same return items as `run`

## Return types

- Embeddings: `list[float]` (vector) or `list[list[float]]` (multiple vectors)
- Digests: `str` (raw text) or structured Pydantic models such as `models.Digest` / `models.Metadata` when using parser callables

## Implementation notes

- The package exposes backends for remote and local models. See `nlp/src/embedders.py` for available embedder implementations (`RemoteEmbeddings`, `LlamaCppEmbeddings`, `TransformerEmbeddings`, `OVEmbeddings`, `ORTEmbeddings`) and how `from_path` dispatches by prefix.
- See `nlp/src/digestors.py` for the `TransformerDigestor`, `OVDigestor`, and `ORTDigestor` implementations and how `output_parser` is applied to results.
- For structured outputs prefer using parser callables from `nlp/src/models.py` (`Digest.parse_markdown`, `Digest.parse_json`, `Metadata.parse_json`).

## Contribution

- Keep digests concise and faithful to the source tone.
- Tune batch sizes for embeddings according to available memory and backend limits.

See the `src` package for full implementation details and advanced configuration.
