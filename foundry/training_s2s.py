import os
import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from dotenv import load_dotenv

load_dotenv()

SOURCE_MODEL_ID = os.getenv('SOURCE_MODEL_ID', "allenai/led-base-16384")
TRAINED_MODEL_ID = os.getenv('TRAINED_MODEL_ID', "soumitsr/led-base-article-digestor")
DATASET_ID = os.getenv('DATASET_ID', "soumitsr/article-digests-compressed")
BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE', 4))
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', 8))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 1))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-4))
OUTPUT_DIR = "./.outputs"
MAX_INPUT_LENGTH = 4096
MAX_OUTPUT_LENGTH = 512

def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, gradient_checkpointing=True, use_cache=False)
    return model, tokenizer

def prepare_dataset(dataset, tokenizer):
    def tokenize_data(batch):
        inputs = tokenizer(
            batch["article"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        )
        outputs = tokenizer(
            batch["summary"],
            max_length=MAX_OUTPUT_LENGTH,
            truncation=True,
            padding="max_length"
        )
        result = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "global_attention_mask": [[0]*len(inputs.input_ids[0])]*len(inputs.input_ids),
            "labels": list(map(lambda label_ids: [-100 if token==tokenizer.pad_token_id else token for token in label_ids], outputs.input_ids))
        }
        result["global_attention_mask"][0][0] = 1
        return result

    tokenized_dataset = dataset.map(
        tokenize_data,
        batched=True,
        # batch_size=os.cpu_count(),
        num_proc=os.cpu_count(),  # Reduced from TRAIN_BATCH_SIZE
        remove_columns=["article", "summary"],
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])
    return tokenized_dataset.shuffle(seed=666), tokenized_dataset.shuffle(seed=666).select(range(100))
    # train_val_split = tokenized_dataset.train_test_split(test_size=0.02)
    # return train_val_split["train"], train_val_split["test"]

def train_model(model: AutoModelForSeq2SeqLM, tokenizer, training_data, eval_data):
    rouge = load("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_output = rouge.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            rouge_types=["rouge2"],
            use_aggregator=True
        )
        rouge2_scores = rouge_output["rouge2"]
        precision = np.mean([s.precision for s in rouge2_scores]) if rouge2_scores else 0.0
        recall = np.mean([s.recall for s in rouge2_scores]) if rouge2_scores else 0.0
        fmeasure = np.mean([s.fmeasure for s in rouge2_scores]) if rouge2_scores else 0.0
        return {
            "rouge2_precision": round(precision, 4),
            "rouge2_recall": round(recall, 4),
            "rouge2_fmeasure": round(fmeasure, 4),
        }

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=MAX_INPUT_LENGTH,
        eval_strategy="steps",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        output_dir=OUTPUT_DIR,
        logging_steps=5,
        # eval_accumulation_steps=10,
        eval_steps=10000,
        save_steps=10,
        save_total_limit=2,
        report_to="none",
        push_to_hub=True,
        hub_model_id=TRAINED_MODEL_ID
    )

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding=True),
        args=training_args,
        train_dataset=training_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()
    trainer.push_to_hub()
    return model, tokenizer

def save_model(model: AutoModelForSeq2SeqLM, tokenizer, model_id: str):
    # 8. Save Fine-Tuned Model
    local_path = f"{OUTPUT_DIR}/{model_id}"
    os.makedirs(local_path, exist_ok=True)
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    model.push_to_hub(model_id)
    tokenizer.push_to_hub(model_id)

def run_training():
    dataset = load_dataset(DATASET_ID, split="train", num_proc=os.cpu_count())
    model, tok = load_model(SOURCE_MODEL_ID)
    tr_data, eval_data = prepare_dataset(dataset, tok)
    model, tok = train_model(model, tok, tr_data, eval_data)
    save_model(model, tok, TRAINED_MODEL_ID)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    run_training()
