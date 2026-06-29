# backend/evaluate.py
"""Evaluate the emotion classifier and produce report-ready artefacts.

Uses the SAME held-out test split as train_bilingual.py / data_utils.py, so the
numbers here are consistent with bilingual_report.txt and never include training
rows (no leakage).

Outputs (into backend/eval_output/):
  - classification_report.txt   precision / recall / F1 per class
  - confusion_matrix.png        labelled heatmap
  - metrics.json                headline accuracy / macro-F1

Run:  python evaluate.py   (from backend/)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_utils import get_splits, ensure_eval_dir, CLASSES, LABEL2ID
from model import emotion_classifier  # reuse the loaded fine-tuned model

DISPLAY_LABELS = [c.upper() for c in CLASSES]  # ["ANXIETY", ...] for the plot


def bert_predict(texts):
    preds = []
    for t in texts:
        scores = emotion_classifier(t, truncation=True)[0]
        preds.append(max(scores, key=lambda x: x["score"])["label"])
    return preds


def plot_confusion(cm, labels, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Bilingual Classifier (held-out test)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[OK] Confusion matrix -> {path}")


def main():
    out_dir = ensure_eval_dir()
    _, _, test_df = get_splits()
    texts = test_df["text"].tolist()
    y_true = test_df["status"].tolist()  # Title-case class names
    print(f"[INFO] Evaluating on held-out test set: {len(texts)} examples")

    y_pred = bert_predict(texts)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
    report = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    print("\n" + report)
    print(f"Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, "n": len(texts)}, f, indent=2)
    plot_confusion(cm, DISPLAY_LABELS, os.path.join(out_dir, "confusion_matrix.png"))
    print(f"\n[DONE] Artefacts written to {out_dir}")


if __name__ == "__main__":
    main()
