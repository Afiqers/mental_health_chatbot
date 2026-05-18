# backend/model.py

import os
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from label_map import LABEL_MAP

# ──────────────────────────────────────────────────────────────
#  MODEL 1: Emotion classifier (local model if available)
# ──────────────────────────────────────────────────────────────
# local_model files are saved inside local_model/local_model by train_local.py
_LOCAL_BASE = os.path.join(os.path.dirname(__file__), "local_model")
LOCAL_MODEL_PATH = os.path.join(_LOCAL_BASE, "local_model") if os.path.isdir(os.path.join(_LOCAL_BASE, "local_model")) else _LOCAL_BASE
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

_GREETINGS = {
    "hi", "hello", "hey", "hye", "helo", "hai",
    "good morning", "good afternoon", "good evening",
    "selamat pagi", "selamat petang", "selamat malam"
}

_FAREWELLS = {
    "bye", "goodbye", "good bye", "see you", "see ya", "tata", "baibai"
}

# The fine-tuned model struggles with very short 2-3 word phrases (like "i am sad")
# because it was trained on longer, complex sentences. We use a fallback for obvious ones.
_QUICK_EMOTIONS = {
    "i am sad": "DEPRESSION",
    "i'm sad": "DEPRESSION",
    "im sad": "DEPRESSION",
    "saya sedih": "DEPRESSION",
    "aku sedih": "DEPRESSION",
    "i feel sad": "DEPRESSION",
    
    "i am anxious": "ANXIETY",
    "i'm anxious": "ANXIETY",
    "im anxious": "ANXIETY",
    "saya gelisah": "ANXIETY",
    "i feel anxious": "ANXIETY",
    "saya risau": "ANXIETY",
    
    "i want to die": "SUICIDAL",
    "i wanna die": "SUICIDAL",
    "saya nak mati": "SUICIDAL",
    "kill myself": "SUICIDAL",
    "bunuh diri": "SUICIDAL"
}

def classify_text(text: str):
    """
    Classify input text using the primary emotion classifier.
    The local model is a 4-class model (ANXIETY, DEPRESSION, SUICIDAL, NORMAL).
    Short greetings and obvious keywords are caught immediately.

    Returns: (emotion, confidence, is_high_risk)
    """
    normalized_text = text.strip().lower()
    
    # ── Quick checks for greetings/farewells ──
    if normalized_text in _GREETINGS:
        return "GREETING", 1.0, False
    if normalized_text in _FAREWELLS:
        return "FAREWELL", 1.0, False

    # ── Quick checks for obvious short phrases ──
    # If the exact phrase is in our dictionary, or if it's very short and contains a critical keyword
    if normalized_text in _QUICK_EMOTIONS:
        emotion = _QUICK_EMOTIONS[normalized_text]
        return emotion, 0.95, (emotion == "SUICIDAL")
        
    for phrase, emotion in _QUICK_EMOTIONS.items():
        if len(normalized_text.split()) <= 5 and phrase in normalized_text:
            return emotion, 0.90, (emotion == "SUICIDAL")

    emotion_results = emotion_classifier(text)[0]
    best = max(emotion_results, key=lambda x: x["score"])
    
    emotion = LABEL_MAP.get(best["label"], "UNKNOWN")
    confidence = round(best["score"], 3)
    
    is_high_risk = (emotion == "SUICIDAL")

    return emotion, confidence, is_high_risk

