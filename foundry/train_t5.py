import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SOURCE_MODEL_ID = os.getenv('SOURCE_MODEL_ID', "google/long-t5-tglobal-base")
TRAINED_MODEL_ID = os.getenv('TRAINED_MODEL_ID', "soumitsr/long-t5-sm-article-digestor")
DATASET_ID = os.getenv('DATASET_ID', "/home/soumitsr/codes/pycoffeemaker/coffeemaker/nlp/foundry/.dataset")
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE', 1))
EVAL_BATCH_SIZE = int(os.getenv('EVAL_BATCH_SIZE', 1))
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', 2))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '1'))
MAX_STEPS = 3 # Optional[int](os.getenv('MAX_STEPS'))  # None means train for full epochs
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-3))

def load_model(model_id: str):
    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    return model, tokenizer

def prepare_dataset(dataset, tokenizer):
    # 3. Preprocess Dataset
    def tokenize_data(examples):
        inputs = tokenizer(
            ["summarize: " + inp for inp in examples["input"]],
            max_length=5000,  # Long-T5 supports up to 16384, but we use 4096
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        outputs = tokenizer(
            examples["output"],
            max_length=512,  # Factoids are short
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": outputs["input_ids"].squeeze(),
        }

    tokenized_dataset = dataset.map(
        tokenize_data, 
        batched=True, 
        num_proc=os.cpu_count(),
        remove_columns=["input", "output"]
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Split dataset (80% train, 20% validation)
    train_val_split = tokenized_dataset.train_test_split(test_size=0.1)
    return train_val_split["train"], train_val_split["test"]

def train_model(model: AutoModelForSeq2SeqLM, tokenizer, training_data, eval_data):
    # 5. Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=TRAIN_BATCH_SIZE,  # Small batch size for long sequences
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # Effective batch size = 1 * 8 = 8
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,  # Mixed precision for speed
        gradient_checkpointing=True,  # Save memory
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        output_dir="./.output",
        report_to="none"
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    )

    # 7. Fine-Tune Model
    trainer.train()
    return model, tokenizer

def save_model(model: AutoModelForSeq2SeqLM, tokenizer, model_id: str):
    # 8. Save Fine-Tuned Model
    local_path = f"./.output/{model_id}"
    os.makedirs(local_path, exist_ok=True)
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    model.push_to_hub(model_id)
    tokenizer.push_to_hub(model_id)

def run_training():
    dataset = load_dataset(DATASET_ID, split="train", num_proc=os.cpu_count()).select(range(100))
    model, tok = load_model(SOURCE_MODEL_ID)
    tr_data, eval_data = prepare_dataset(dataset, tok)
    model, tok = train_model(model, tok, tr_data, eval_data)
    save_model(model, tok, TRAINED_MODEL_ID)

if __name__ == "__main__":
    run_training()
