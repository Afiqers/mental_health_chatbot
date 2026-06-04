"""Train a genuinely bilingual (English + Malay) emotion classifier.

Why this exists
---------------
The original model was trained almost entirely on machine-translated Malay
text, so it was confidently wrong on English input — which is why the backend
needed a hand-written lexicon crutch. This script trains DistilmBERT
(distilbert-base-multilingual-cased) on a balanced mix of *both* languages so
the model handles English and Malay natively.

DistilmBERT is chosen because training runs on CPU here; it has half the layers
of mBERT (~2x faster) while keeping multilingual coverage.

Data
----
  - English : data/mental_heath_unbalanced.csv  (column `text`)
  - Malay   : data/malay_dataset_translated.csv (column `text_ms`)

Output
------
  - backend/local_model/        the fine-tuned model (consumed by model.py)
  - backend/eval_output/bilingual_report.txt   overall + per-language metrics

Run:  python train_bilingual.py
"""

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-multilingual-cased"
ENGLISH_PER_CLASS = 600          # sampled from the large English set
MAX_LENGTH = 128
EPOCHS = 3
OUTPUT_DIR = "backend/local_model"
EVAL_DIR = "backend/eval_output"

LABEL2ID = {"Anxiety": 0, "Depression": 1, "Suicidal": 2, "Normal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASSES = list(LABEL2ID.keys())


def load_bilingual():
    frames = []

    # English
    en = pd.read_csv("data/mental_heath_unbalanced.csv")[["text", "status"]].dropna()
    en = en[en["status"].isin(CLASSES)]
    en = (
        en.groupby("status", group_keys=False)
        .apply(lambda x: x.sample(min(ENGLISH_PER_CLASS, len(x)), random_state=42))
    )
    en = en.rename(columns={"text": "text"})
    en["lang"] = "en"
    frames.append(en[["text", "status", "lang"]])

    # Malay (already 500/class)
    ms = pd.read_csv("data/malay_dataset_translated.csv")[["text_ms", "status"]].dropna()
    ms = ms[ms["status"].isin(CLASSES)]
    ms = ms.rename(columns={"text_ms": "text"})
    ms["lang"] = "ms"
    frames.append(ms[["text", "status", "lang"]])

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["status"].map(LABEL2ID)
    print("Class balance:\n", df.groupby(["status", "lang"]).size())
    print("Total examples:", len(df))
    return df


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    df = load_bilingual()

    # 80 / 10 / 10 split, stratified by label.
    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=42, stratify=tmp_df["label"]
    )
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length",
                         max_length=MAX_LENGTH)

    def to_ds(frame):
        ds = Dataset.from_pandas(frame[["text", "label"]].reset_index(drop=True))
        ds = ds.map(tok, batched=True)
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return ds

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=metrics,
    )

    print("Starting training...")
    trainer.train()

    # ── Held-out test evaluation, overall + per language ──
    pred = trainer.predict(test_ds)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    target_names = [ID2LABEL[i] for i in range(4)]

    lines = []
    overall = classification_report(y_true, y_pred, target_names=target_names,
                                    zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    lines.append("=== OVERALL (held-out test) ===")
    lines.append(overall)
    lines.append(f"Accuracy: {acc:.4f}   Macro-F1: {macro:.4f}\n")

    langs = test_df["lang"].to_numpy()
    for lang in ("en", "ms"):
        mask = langs == lang
        if mask.sum() == 0:
            continue
        rep = classification_report(y_true[mask], y_pred[mask],
                                    target_names=target_names, zero_division=0)
        a = accuracy_score(y_true[mask], y_pred[mask])
        lines.append(f"=== {lang.upper()} only (n={int(mask.sum())}) ===")
        lines.append(rep)
        lines.append(f"Accuracy: {a:.4f}\n")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(EVAL_DIR, "bilingual_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("Saving model to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
