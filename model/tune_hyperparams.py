"""
Hyperparameter tuning for distilGPT2 using Optuna and validation split.
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import optuna
from optuna.trial import TrialState

# Paths
TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
VAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'val.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), 'distilgpt2-best')

# Load data
df_train = pd.read_csv(TRAIN_PATH, names=["text", "category", "tone"], header=None)
df_val = pd.read_csv(VAL_PATH, names=["text", "category", "tone"], header=None)

def build_prompt(row):
    return f"Category: {row['category']} | Tone: {row['tone']} | Post: {row['text']}"

df_train['prompt'] = df_train.apply(build_prompt, axis=1)
df_val['prompt'] = df_val.apply(build_prompt, axis=1)

dataset_train = Dataset.from_pandas(df_train[['prompt']])
dataset_val = Dataset.from_pandas(df_val[['prompt']])


tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
# Fix: Set pad_token to eos_token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128)

tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_val = dataset_val.map(tokenize_function, batched=True)

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)

    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    training_args = TrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        report_to="none",
        fp16=False,
        logging_steps=100
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer
    )
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


