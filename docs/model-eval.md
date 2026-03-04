# Digest & Content Generation Models Evaluation

## Digest & Summary Generation

| Model | Rating | Cost | Key Notes |
|-------|--------|------|-----------|
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | ⭐ Very Good | ~$0.10/M | Balanced size & quality, well-structured summaries, maintains tone |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct` | ⭐ Very Good | ~$0.12/M | High-quality digests, strong instruction following |
| `NovaSky-AI/Sky-T1-32B-Preview` | ⭐ Good | ~$0.12/M | Decent performance, good for structured outputs |
| `google/gemma-3-27b-it` | ⭐ Good | ~$0.10/M | Bullet points, translates well, tone preservation (high cost relative to quality) |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ⭐ Good | ~$0.03/M | Bullet points, some filler text, best cost-to-quality ratio |
| `mistralai/Mistral-Small-24B-Instruct-2501` | ❌ Poor | – | Verbose, poor tone/style preservation |
| `mistralai/Mistral-Nemo-Instruct-2407` | ❌ Poor | – | Excessive verbosity, poor translation |
| `microsoft/Phi-4-multimodal-instruct` | ❌ Poor | – | Low quality output |
| `Qwen/Qwen2.5-7B-Instruct` | ❌ Poor | – | Poor translation, weak multilingual support |
| `google/gemini-1.5-flash` | ❌ Poor | – | No bullet points, poor tone preservation |
| `meta-llama/Llama-3.2-3B-Instruct` | ❌ Poor | – | Excessively verbose |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | ❌ Poor | – | Excessively verbose |
| `Gryphe/MythoMax-L2-13b` | ❌ Poor | – | Low quality output |

### Local Models (Self-Hosted)

| Model Category | Recommendation | Notes |
|---|---|---|
| `soumitsr/led-base-article-digestor` (Seq2Seq) | ⭐ Preferred | Better efficiency, control over format, ideal for on-device |
| `soumitsr/SmolLM2-360M-Instruct-article-digestor` (Decoder) | ⭐ Preferred | Better efficiency, control over format, ideal for on-device |

## Article & Content Generation

| Model | Rating | Cost | Key Notes |
|-------|--------|------|-----------|
| `o3-mini` / `o4-mini` | ⭐ Excellent | API | Strong instruction following, occasional API edge cases |
| `deepseek-ai/DeepSeek-R1` (or `0528`) | ⭐ Good | – | Well-structured articles, good compliance |
| `microsoft/WizardLM-2-8x22B` | ⭐ Good | – | Solid article generation |
| `NovaSky-AI/Sky-T1-32B-Preview` | ⚠️ Mixed | ~$0.12/M | Underwhelming performance |
| `Sao10K/L3.1-70B-Euryale-v2.2` | ⚠️ Mixed | – | Acceptable but inconsistent instruction following |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct` | ⚠️ Mixed | ~$0.12/M | Poor instruction following for articles |
| `gpt-4.1-nano` / `gpt-4-mini` | ❌ Poor | API | Not suitable for content generation |

## Image Generation

| Model | Rating | Cost | Key Notes |
|-------|--------|------|-----------|
| `black-forest-labs/FLUX-1-schnell` | ⭐ Good | ~$0.0005 | Fast, low cost |
| `black-forest-labs/FLUX-1-dev` | ⭐ Good | ~$0.009 | Better quality |
| `run-diffusion/Juggernaut-Lightning-Flux` | ⭐ Good | ~$0.009 | Strong quality/speed balance |
| `run-diffusion/Juggernaut-Flux` | ⭐ Good | ~$0.009 | Higher quality, slower |
| `stabilityai/sdxl-turbo` | ❌ Poor | ~$0.0002 | Poor quality |
| `stabilityai/sd3.5-medium` | ❌ Poor | ~$0.03 | Mediocre output, high cost |
