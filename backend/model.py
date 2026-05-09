# backend/model.py

import os
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from label_map import LABEL_MAP

# ──────────────────────────────────────────────────────────────
#  MODEL 1: Emotion classifier (local model if available)
# ──────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "local_model")
if os.path.isdir(LOCAL_MODEL_PATH):
    print(f"[INFO] Loading local emotion model from: {LOCAL_MODEL_PATH}")
    emotion_classifier = pipeline(
        "text-classification",
        model=LOCAL_MODEL_PATH,
        top_k=None
    )
else:
    print("[INFO] Loading default emotion model (ourafla)...")
    emotion_classifier = pipeline(
        "text-classification",
        model="ourafla/mental-health-bert-finetuned",
        top_k=None
    )

# ──────────────────────────────────────────────────────────────
#  MODEL 2: Suicide detection (DistilBERT — called directly,
#  NOT via pipeline, to avoid token_type_ids incompatibility)
# ──────────────────────────────────────────────────────────────
SUICIDE_MODEL_PATH = os.environ.get(
    "SUICIDE_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "suicide_model")
)

_suicide_tokenizer = None
_suicide_model     = None

if os.path.isdir(SUICIDE_MODEL_PATH):
    print(f"[INFO] Loading suicide detection model from: {SUICIDE_MODEL_PATH}")
    _suicide_tokenizer = AutoTokenizer.from_pretrained(SUICIDE_MODEL_PATH)
    _suicide_model     = AutoModelForSequenceClassification.from_pretrained(SUICIDE_MODEL_PATH)
    _suicide_model.eval()
    print("[INFO] Suicide detection model loaded successfully.")
else:
    print("[WARN] Suicide model not found. Run training first (train_colab.py).")
    print(f"       Expected path: {SUICIDE_MODEL_PATH}")


def _run_suicide_classifier(text: str):
    """
    Run DistilBERT suicide classifier without pipeline to avoid
    the token_type_ids incompatibility issue.
    Returns (label, score) where label is 'suicide' or 'non-suicide'.
    """
    inputs = _suicide_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    # DistilBERT does NOT use token_type_ids — remove if present
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = _suicide_model(**inputs).logits

    probs      = F.softmax(logits, dim=-1)[0]
    pred_idx   = probs.argmax().item()
    pred_label = _suicide_model.config.id2label[pred_idx]
    pred_score = probs[pred_idx].item()
    return pred_label, round(pred_score, 3)


def classify_text(text: str):
    """
    Classify input text using dual-model pipeline:
      1. Suicide detector (if available) — high priority
      2. Emotion classifier — always runs

    Returns: (emotion, confidence, is_high_risk)
    """
    is_high_risk       = False
    suicide_confidence = 0.0

    # ── Step 1: Check for suicide / self-harm risk ───────────
    if _suicide_model is not None:
        label, score = _run_suicide_classifier(text)
        if label == "suicide" and score >= 0.65:
            is_high_risk       = True
            suicide_confidence = score

    # ── Step 2: Run emotion classifier ───────────────────────
    emotion_results = emotion_classifier(text)[0]
    best = max(emotion_results, key=lambda x: x["score"])
    emotion     = LABEL_MAP.get(best["label"], "UNKNOWN")
    confidence  = round(best["score"], 3)

    # ── Step 3: Override emotion if high risk ─────────────────
    if is_high_risk:
        emotion    = "HIGH_RISK"
        confidence = suicide_confidence

    return emotion, confidence, is_high_risk
