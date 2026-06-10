# backend/baseline_comparison.py
"""Compare the fine-tuned DistilmBERT against classical ML baselines.

Answers the panel question: "Why deep learning and not something simpler?"
We train TF-IDF + Logistic Regression and TF-IDF + Linear SVM on the SAME
training split, then evaluate all models on the SAME held-out test set.

Outputs (backend/eval_output/):
  - baseline_comparison.txt   per-model accuracy / macro-F1 + full reports
  - baseline_comparison.png   grouped bar chart

Run:  python baseline_comparison.py   (from backend/)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_utils import get_splits, ensure_eval_dir, CLASSES, LABEL2ID
from model import emotion_classifier  # reuse the already-loaded BERT pipeline


def bert_predict(texts):
    preds = []
    for t in texts:
        scores = emotion_classifier(t, truncation=True)[0]
        best = max(scores, key=lambda x: x["score"])["label"]
        preds.append(LABEL2ID.get(best, LABEL2ID["Normal"]))
    return preds


def main():
    out_dir = ensure_eval_dir()
    train_df, _, test_df = get_splits()
    print(f"[INFO] Train {len(train_df)} | Test {len(test_df)}")

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    models = {
        "TF-IDF + LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "TF-IDF + LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
    }

    results = {}      # name -> (accuracy, macro_f1)
    report_blocks = []

    # Classical baselines
    for name, pipe in models.items():
        print(f"[INFO] Training {name} ...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        results[name] = (acc, f1)
        report_blocks.append(
            f"=== {name} ===\n"
            + classification_report(y_test, pred, target_names=CLASSES, zero_division=0)
            + f"\nAccuracy: {acc:.4f}   Macro-F1: {f1:.4f}\n"
        )

    # Fine-tuned DistilmBERT on the identical test set
    print("[INFO] Evaluating DistilmBERT ...")
    bert_pred = bert_predict(X_test)
    acc = accuracy_score(y_test, bert_pred)
    f1 = f1_score(y_test, bert_pred, average="macro")
    results["DistilmBERT (ours)"] = (acc, f1)
    report_blocks.append(
        "=== DistilmBERT (ours) ===\n"
        + classification_report(y_test, bert_pred, target_names=CLASSES, zero_division=0)
        + f"\nAccuracy: {acc:.4f}   Macro-F1: {f1:.4f}\n"
    )

    # ── Summary table ──
    header = f"{'Model':<24}{'Accuracy':>10}{'Macro-F1':>10}"
    table = [header, "-" * len(header)]
    for name, (acc, f1) in results.items():
        table.append(f"{name:<24}{acc:>10.4f}{f1:>10.4f}")
    summary = "\n".join(table)
    print("\n" + summary + "\n")

    with open(os.path.join(out_dir, "baseline_comparison.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n\n" + "\n".join(report_blocks))

    # ── Bar chart ──
    names = list(results.keys())
    accs = [results[n][0] for n in names]
    f1s = [results[n][1] for n in names]
    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], accs, width, label="Accuracy", color="#6c8cff")
    ax.bar([i + width / 2 for i in x], f1s, width, label="Macro-F1", color="#30a46c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classical Baselines vs DistilmBERT")
    for i in x:
        ax.text(i - width / 2, accs[i] + 0.01, f"{accs[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f1s[i] + 0.01, f"{f1s[i]:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "baseline_comparison.png"), dpi=150)
    print(f"[DONE] Artefacts written to {out_dir}")


if __name__ == "__main__":
    main()
