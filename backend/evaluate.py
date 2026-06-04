# backend/evaluate.py
"""Evaluate the emotion classifier and produce report-ready artefacts.

Outputs (into backend/eval_output/):
  - classification_report.txt   precision / recall / F1 per class
  - confusion_matrix.png        labelled heatmap
  - metrics.json                headline accuracy / macro-F1

Usage:
    python evaluate.py                      # uses the held-out split of the
                                            # translated dataset if available
    python evaluate.py --data path.csv      # custom CSV (text,status columns)

This is the quantitative evidence for the "NLP / sentiment analysis" part of
the FYP report.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from model import classify_text

LABELS = ["ANXIETY", "DEPRESSION", "SUICIDAL", "NORMAL"]
STATUS_TO_LABEL = {
    "anxiety": "ANXIETY",
    "depression": "DEPRESSION",
    "suicidal": "SUICIDAL",
    "normal": "NORMAL",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "eval_output")


def _default_dataset():
    root = os.path.dirname(os.path.dirname(__file__))
    candidate = os.path.join(root, "data", "malay_dataset_translated.csv")
    return candidate if os.path.exists(candidate) else None


def load_data(path, sample_per_class):
    df = pd.read_csv(path)
    text_col = "text_ms" if "text_ms" in df.columns else "text"
    df = df.dropna(subset=[text_col, "status"])
    df["label"] = df["status"].str.lower().map(STATUS_TO_LABEL)
    df = df.dropna(subset=["label"])

    # Use a held-out split so we never evaluate on training rows.
    _, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    if sample_per_class:
        test_df = (
            test_df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(sample_per_class, len(x)), random_state=42))
        )
    return test_df[text_col].tolist(), test_df["label"].tolist()


def plot_confusion(cm, labels, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Emotion Classifier")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=_default_dataset())
    parser.add_argument("--sample", type=int, default=80,
                        help="max test examples per class (0 = all)")
    args = parser.parse_args()

    if not args.data:
        print("No dataset found. Pass --data path/to/file.csv (text,status).")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] Evaluating on: {args.data}")
    texts, y_true = load_data(args.data, args.sample or None)
    print(f"[INFO] Test examples: {len(texts)}")

    y_pred = []
    for i, t in enumerate(texts, 1):
        emotion, _, _, _ = classify_text(t)
        # Collapse non-core predictions onto the nearest core label.
        if emotion not in LABELS:
            emotion = "NORMAL" if emotion in ("GREETING", "FAREWELL", "NEUTRAL") else "NORMAL"
        y_pred.append(emotion)
        if i % 25 == 0:
            print(f"  ...{i}/{len(texts)}")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\n" + report)
    print(f"Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, "n": len(texts)}, f, indent=2)
    plot_confusion(cm, LABELS, os.path.join(OUT_DIR, "confusion_matrix.png"))
    print(f"\n[DONE] Artefacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
