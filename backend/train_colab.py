# ============================================================
#  MENTAL HEALTH CHATBOT — SUICIDE DETECTION TRAINER
#  Run this file on Google Colab (GPU recommended)
# ============================================================
#
#  STEP-BY-STEP INSTRUCTIONS:
#  1. Go to https://colab.research.google.com
#  2. Click File > Upload notebook > Upload .py file (this file)
#     OR: File > New notebook, paste each cell manually
#  3. Change Runtime to GPU: Runtime > Change runtime type > T4 GPU
#  4. Upload Suicide_Detection.csv to your Google Drive
#  5. Run all cells in order
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────
# Paste this in the first Colab cell and run it

"""
!pip install transformers datasets scikit-learn torch accelerate -q
"""


# ── CELL 2: Mount Google Drive ───────────────────────────────
"""
from google.colab import drive
drive.mount('/content/drive')
"""


# ── CELL 3: Load and inspect dataset ────────────────────────
"""
import pandas as pd

CSV_PATH = '/content/drive/MyDrive/Suicide_Detection.csv'  # adjust path if needed

df = pd.read_csv(CSV_PATH)
df = df[['text', 'class']].dropna()
df.columns = ['text', 'label']

# Convert labels to integers
label2id = {'non-suicide': 0, 'suicide': 1}
id2label = {0: 'non-suicide', 1: 'suicide'}
df['label'] = df['label'].map(label2id)

print("Shape:", df.shape)
print(df['label'].value_counts())
print(df.head(3))
"""


# ── CELL 4: Split dataset ────────────────────────────────────
"""
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
print(f"Train: {len(train_df)} | Val: {len(val_df)}")
"""


# ── CELL 5: Tokenize ─────────────────────────────────────────
"""
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = 'distilbert-base-uncased'  # fast + accurate, good for Colab free tier
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset   = val_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print("Tokenization done!")
"""


# ── CELL 6: Load model ───────────────────────────────────────
"""
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
print("Model loaded!")
"""


# ── CELL 7: Training arguments ──────────────────────────────
"""
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

OUTPUT_DIR = '/content/drive/MyDrive/suicide_model'  # saved to your Drive

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=200,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    fp16=True,          # Faster training with mixed precision (GPU only)
    report_to='none',   # Disable wandb
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

print("Trainer ready!")
"""


# ── CELL 8: Train! ───────────────────────────────────────────
"""
trainer.train()
print("Training complete!")
"""


# ── CELL 9: Evaluate ────────────────────────────────────────
"""
results = trainer.evaluate()
print("Evaluation results:")
for k, v in results.items():
    print(f"  {k}: {round(v, 4)}")
"""


# ── CELL 10: Save model + tokenizer ─────────────────────────
"""
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")
print("Download the 'suicide_model' folder from your Google Drive!")
"""


# ── CELL 11 (OPTIONAL): Push to HuggingFace Hub ─────────────
# Only if you want to host on HuggingFace (free)
"""
# First: create account at huggingface.co, get your token from Settings > Tokens
from huggingface_hub import notebook_login
notebook_login()

REPO_NAME = 'your-username/suicide-detection-distilbert'  # change this!

trainer.push_to_hub(REPO_NAME)
tokenizer.push_to_hub(REPO_NAME)
print(f"Model pushed to: https://huggingface.co/{REPO_NAME}")
"""


# ── CELL 12: Quick test ──────────────────────────────────────
"""
from transformers import pipeline

classifier = pipeline('text-classification', model=OUTPUT_DIR)

test_texts = [
    "I want to kill myself, life is meaningless",
    "I'm feeling a bit sad today but I'll be okay",
    "I need help, I can't take this anymore",
    "Had a great day at work today!",
]

for text in test_texts:
    result = classifier(text)[0]
    print(f"[{result['label']}] ({result['score']:.2%}) → {text[:60]}")
"""
