# backend/data_utils.py
"""Shared dataset loading + splitting for the evaluation scripts.

Every evaluation script (baseline_comparison.py, safety_eval.py,
error_analysis.py) uses THIS so they all operate on the exact same held-out
test set that train_bilingual.py held out — guaranteeing fair, leakage-free
comparisons (the fine-tuned model never saw the test rows).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EVAL_DIR = os.path.join(BACKEND_DIR, "eval_output")

# Same label scheme + sampling as train_bilingual.py.
LABEL2ID = {"Anxiety": 0, "Depression": 1, "Suicidal": 2, "Normal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASSES = list(LABEL2ID.keys())
ENGLISH_PER_CLASS = 600


def load_bilingual_df():
    """Load the balanced English + Malay dataset (mirrors train_bilingual.py)."""
    en = pd.read_csv(os.path.join(DATA_DIR, "mental_heath_unbalanced.csv"))[["text", "status"]].dropna()
    en = en[en["status"].isin(CLASSES)]
    en = (
        en.groupby("status", group_keys=False)
        .apply(lambda x: x.sample(min(ENGLISH_PER_CLASS, len(x)), random_state=42))
    )
    en["lang"] = "en"

    ms = pd.read_csv(os.path.join(DATA_DIR, "malay_dataset_translated.csv"))[["text_ms", "status"]].dropna()
    ms = ms[ms["status"].isin(CLASSES)].rename(columns={"text_ms": "text"})
    ms["lang"] = "ms"

    df = pd.concat([en[["text", "status", "lang"]], ms[["text", "status", "lang"]]], ignore_index=True)
    df["label"] = df["status"].map(LABEL2ID)
    return df


def get_splits():
    """Return (train_df, val_df, test_df) — identical to train_bilingual.py."""
    df = load_bilingual_df()
    train_df, tmp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42, stratify=tmp_df["label"])
    return train_df, val_df, test_df


def ensure_eval_dir():
    os.makedirs(EVAL_DIR, exist_ok=True)
    return EVAL_DIR
