# Fine-Tuning for Extracting Digests from News, Blogs, Articles, and Social Media Posts
This repo contains code for generating dataset and fine-tuning a smaller language models (< 1B) so that they can perform at a similar maturity level of a larger language models when it comes to simpler tasks such as:
- Generating titles and summaries
- Extracting highlights, named entities, and content domains
- Responding using the `json` response format.

## Dataset Generation
The input content consists of news articles, blogs, and social media posts collected from various RSS feeds, subreddits, and Hacker News (YCombinator). The input content was converted to markdown format (to reduce verbosity) and truncated to 2048 tokens (not trying to summarize a whole book here).

The output content was generated using a combination of larger language models such as `microsoft/WizardLM-8x22`, `google/gemma-3-27b`, and `Qwen/QwQ-32B` as an initial pass. Post-generation, these went through selective human intervention for cleanup. The output is always in `json` format. The outputs can have different combinations of the following fields:
- title
- summary
- highlights
- names
- domains
Labels for each data row indicate which fields are in the output of that row.

The primary focus during data generation was the following:
- Ability to preserve the tonality and the "voice" the original content was written in
- Discouraging click-bait title generation
- Translating content from other languages
- Adherence to `json` as a response format
- Ability to generate different combinations of multiple fields (title, summary, highlights, named entities, and content domain) in one shot without compromising content quality
- Generating an actionable summary text in markdown format even if the overall output is in `json`

## Fine-tuning:
Current Date: 03-31-2025
**HW**: RTX A5000 (24 GB VRAM) + 12vCPU + 25 GB RAM
**GPU Cloud**: [runpod.io](https://www.runpod.io)
**Model**: 135M parameters + 4096 tokens context window

**Lessons Learned**:
Undocumented crap that doesn't make sense but yet here we are - 
- `os.cpu_count()` will return the number of CPUs in the bare-metal under the hood and not what is specified in your container instance. e.g. 128 CPUs instead of 12. This will cause your threading to act like a dumb ass.
- `per_device_train_batch_size` <= 96 and `gradient_accumulation_steps` <= 64 otherwise the system will die (OOM).
- Use less that 400,000 rows of dataset at a time. Anything more than that will make your machine shit its pants. If the dataset is longer than 400K run multiple rounds of iterative training.
- Unless the `chat_template` is one of the ones supported by `Unsloth`, `train_on_responses_only` will fail. For example - it dies on `SmolLM2-*-Instruct` models.
- `SFTTrainer` and/or `Unsloth` will throw a fit if the dataset has a column named `labels`. If you do have it, use `dataset.remove_columns(["labels"])`