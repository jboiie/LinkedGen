"""
Fine-tune distilGPT2 on train.csv for LinkedIn post generation.

Required packages:
- torch
- transformers
- datasets
- pandas

Install with:
    pip install torch transformers datasets pandas
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Paths
TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), 'distilgpt2-finetuned')

# Load data
df = pd.read_csv(TRAIN_PATH, names=["text", "category", "tone"], header=None)

# Combine columns for prompt (customize as needed)
def build_prompt(row):
    return f"Category: {row['category']} | Tone: {row['tone']} | Post: {row['text']}"

df['prompt'] = df.apply(build_prompt, axis=1)

# Prepare Hugging Face dataset
dataset = Dataset.from_pandas(df[['prompt']])

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained('distilgpt2')

training_args = TrainingArguments(
    output_dir=MODEL_OUT,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
