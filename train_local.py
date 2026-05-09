import pandas as pd
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def main():
    csv_path = "data/malay_dataset_translated.csv"
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please run prepare_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Use the translated text for training, but drop rows where translation might have failed (NaN)
    df = df.dropna(subset=['text_ms', 'status'])
    
    # Map the labels to match the backend expectation and handle SUICIDAL properly
    # We will use 4 classes.
    label2id = {'Anxiety': 0, 'Depression': 1, 'Suicidal': 2, 'Normal': 3}
    id2label = {0: 'Anxiety', 1: 'Depression', 2: 'Suicidal', 3: 'Normal'}
    
    df['label'] = df['status'].map(label2id)
    df = df.rename(columns={'text_ms': 'text'}) # rename for tokenizer
    df = df[['text', 'label']]
    
    print(f"Total rows: {len(df)}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # We use a multilingual model since it's Malay text
    MODEL_NAME = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        )
        
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    
    print("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset   = val_dataset.map(tokenize_fn, batched=True)
    
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=id2label,
        label2id=label2id
    )
    
    OUTPUT_DIR = 'backend/local_model'
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='none', 
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1  = f1_score(labels, predictions, average='weighted')
        return {'accuracy': acc, 'f1': f1}
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    results = trainer.evaluate()
    print("Evaluation Results:", results)
    
    print("Saving model to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! You can now update your backend to use this local_model.")

if __name__ == "__main__":
    main()
