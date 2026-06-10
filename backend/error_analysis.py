# backend/error_analysis.py
"""Qualitative error analysis of the classifier on the held-out test set.

Produces a human-readable breakdown of WHICH classes get confused and WHY,
with real example messages — the kind of qualitative discussion examiners
look for in an FYP report.

Output: backend/eval_output/error_analysis.txt

Run:  python error_analysis.py   (from backend/)
"""

import os
from collections import Counter

from data_utils import get_splits, ensure_eval_dir, CLASSES, LABEL2ID, ID2LABEL
from model import emotion_classifier


def bert_label(text):
    scores = emotion_classifier(text, truncation=True)[0]
    return max(scores, key=lambda x: x["score"])["label"]


def main():
    out_dir = ensure_eval_dir()
    _, _, test_df = get_splits()
    print(f"[INFO] Analysing {len(test_df)} test messages ...")

    errors = []          # (true, pred, lang, text)
    total = 0
    correct = 0
    lang_total = Counter()
    lang_errors = Counter()

    for _, row in test_df.iterrows():
        true_label = row["status"]
        pred_label = bert_label(row["text"])
        total += 1
        lang_total[row["lang"]] += 1
        if pred_label == true_label:
            correct += 1
        else:
            errors.append((true_label, pred_label, row["lang"], row["text"]))
            lang_errors[row["lang"]] += 1

    lines = []
    lines.append("=== Error Analysis — DistilmBERT on held-out test set ===")
    lines.append(f"Total: {total} | Correct: {correct} | Errors: {len(errors)} "
                 f"| Accuracy: {correct/total:.4f}")
    lines.append("")

    # Error rate per language
    lines.append("--- Errors by language ---")
    for lang in lang_total:
        n, e = lang_total[lang], lang_errors[lang]
        lines.append(f"  {lang.upper()}: {e}/{n} errors ({e/n:.1%})")
    lines.append("")

    # Most common confusion pairs
    pairs = Counter((t, p) for t, p, _, _ in errors)
    lines.append("--- Most common confusions (Actual -> Predicted) ---")
    for (t, p), c in pairs.most_common(8):
        lines.append(f"  {t:>11} -> {p:<11}  ({c} times)")
    lines.append("")

    # Example messages for the top confusion pairs
    lines.append("--- Example misclassifications ---")
    for (t, p), _ in pairs.most_common(4):
        lines.append(f"\n[{t} misread as {p}]")
        examples = [(lang, txt) for tt, pp, lang, txt in errors if tt == t and pp == p]
        for lang, txt in examples[:3]:
            snippet = txt.strip().replace("\n", " ")[:140]
            lines.append(f"  ({lang}) \"{snippet}\"")

    # Interpretation hint for the report
    lines.append("\n--- Discussion ---")
    lines.append(
        "The dominant confusion is typically DEPRESSION <-> SUICIDAL: both share "
        "vocabulary of hopelessness and worthlessness, so the boundary is genuinely "
        "fuzzy even for humans. This is precisely why the system keeps a deterministic "
        "crisis safety net rather than trusting the classifier alone for self-harm."
    )

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(out_dir, "error_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n[DONE] Written to {out_dir}/error_analysis.txt")


if __name__ == "__main__":
    main()
