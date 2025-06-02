from datetime import datetime
import random
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import os
from icecream import ic
from dotenv import load_dotenv

load_dotenv()

DATA_NUM_PROC = ic(int(os.getenv("DATA_NUM_PROC", os.cpu_count()))) 
MAX_SEQ_LENGTH = ic(int(os.getenv("MAX_SEQ_LENGTH", 4096)))
PER_DEVICE_TRAIN_BATCH_SIZE = ic(int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 64)))
GRADIENT_ACCUMULATION_STEPS = ic(int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 64)))
NUM_TRAIN_EPOCHS = ic(int(os.getenv("NUM_TRAIN_EPOCHS", 1)))

SRC_MODEL_ID = ic(os.getenv("SRC_MODEL_ID", "soumitsr/SmolLM2-135M-Instruct-article-digestor-lora")) # "HuggingFaceTB/SmolLM2-135M-Instruct" # "HuggingFaceTB/SmolLM2-360M-Instruct" # "unsloth/SmolLM2-360M-Instruct",
TRAINED_MODEL_ID = os.getenv("TRAINED_MODEL_ID", "soumitsr/SmolLM2-135M-Instruct-article-digestor")
DATASET_ID = os.getenv("DATASET_ID", "soumitsr/article-digests-improved")

INSTRUCTION_START = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
EOS = "<|im_end|>"

def load_data(dataset_id: str):
    # FIELDS = [        
    #     ['title', 'names', 'domains'],
    #     ['summary_markdown']
    # ]   
    data = load_dataset(dataset_id, split='train', num_proc=DATA_NUM_PROC) #.filter(lambda example: example['fields'] in FIELDS, num_proc=DATA_NUM_PROC)
    # NOTE: checkpoint 04-03-2025 | Trained on the first 87819 ["title", "names", "domains"] + ["summary", "highlights", "title", "names", "domains"]
    # NOTE: checkpoint 04-05-2025 | Trained on the first 114093 ["title", "names", "domains"] + ["summary_markdown"]
    # NOTE: checkpoint 04-06-2025 | Trained on the first 112939 ["title", "names", "domains"] + ["summary_markdown"]
    # NOTE: checkpoint 04-06-2025 | Trained on the first 42196 ["summary", "title", "names", "domains"]
    # NOTE: checkpoint 04-08-2025 | Trained on the first 104358 ["summary_markdown"], ["title", "names", "domains"]
    return data

def load_model(model_id: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer
    
def train_model(model, tokenizer, data):
    # NOTE: lesson learnt - having a 'labels' column is a bad idea
    data = data.map(
        lambda example: {"text": tokenizer.apply_chat_template(example['messages'], tokenize = False, add_generation_prompt = False)}, 
        num_proc=DATA_NUM_PROC,
        batched=True,
    ).shuffle(seed=3407)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = data, 
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = DATA_NUM_PROC,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE, 
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, 
            warmup_steps = 2,
            num_train_epochs = NUM_TRAIN_EPOCHS, # Set this for 1 full training run.
            # max_steps = 5,
            learning_rate = 2e-3,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = ".outputs",
            report_to = "none"
        )
    )
    trainer = train_on_responses_only(
        trainer, 
        instruction_part=INSTRUCTION_START, 
        response_part=RESPONSE_PART
    )

    print(tokenizer.decode(trainer.train_dataset[random.randint(0, len(data))]["input_ids"]))
    print(tokenizer.decode(trainer.train_dataset[random.randint(0, len(data))]["input_ids"]))
    
    trainer.train()
    return model, tokenizer

def save_model(model, tokenizer, model_id: str):
    local_name = ".outputs/trained-"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model.save_pretrained_merged(local_name+"-lora", tokenizer, save_method = "lora") # Local saving
    model.save_pretrained_merged(local_name, tokenizer, save_method = "merged_16bit")
    model.push_to_hub_merged(model_id+"-lora", tokenizer, save_method = "lora") # Online saving
    model.push_to_hub_merged(model_id, tokenizer, save_method = "merged_16bit")
    # model.push_to_hub_merged(model_id+"-4bit", tokenizer, save_method = "merged_4bit_forced")
    # model.push_to_hub_gguf(
    #     model_id+"-gguf",
    #     tokenizer,
    #     quantization_method = ["q4_k_m", "q8_0",]
    # )

def run_training(src_model_id, trained_model_id, dataset_id):
    data = load_data(dataset_id)
    model, tokenizer = load_model(src_model_id)
    model, tokenizer = train_model(model, tokenizer, data)
    save_model(model, tokenizer, trained_model_id)

if __name__ == "__main__":
    run_training(SRC_MODEL_ID, TRAINED_MODEL_ID, DATASET_ID)
