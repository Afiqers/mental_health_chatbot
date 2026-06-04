# backend/model.py
"""Emotion / sentiment classification.

Design (hybrid, ML-primary):
  1. A fine-tuned, genuinely bilingual (English + Malay) DistilmBERT classifier
     is the primary engine. It returns a full probability distribution over the
     emotional states, which downstream code turns into a continuous wellbeing
     score. (Trained by train_bilingual.py; ~80% accuracy on both languages.)
  2. A small, transparent rule layer sits on top for two narrow jobs only:
       - GREETING / FAREWELL small talk (cheap, avoids noisy model output)
       - a CRISIS SAFETY NET that force-flags explicit self-harm language.
     The safety net is a deliberate clinical safeguard: for self-harm we
     prefer to over-trigger than to ever miss.
  3. A lexicon backstop catches the model's occasional false-negative — clear
     distress mislabelled as NORMAL — which is the dangerous direction in a
     mental-health context. The model does the heavy lifting; this only rescues
     the edges.

`classify_text` returns: (emotion, confidence, is_high_risk, distribution)
"""

import os
import re

from transformers import pipeline
from label_map import LABEL_MAP

# ──────────────────────────────────────────────────────────────
#  Load the classifier (local fine-tuned model if present)
# ──────────────────────────────────────────────────────────────
_LOCAL_BASE = os.path.join(os.path.dirname(__file__), "local_model")
LOCAL_MODEL_PATH = (
    os.path.join(_LOCAL_BASE, "local_model")
    if os.path.isdir(os.path.join(_LOCAL_BASE, "local_model"))
    else _LOCAL_BASE
)

if os.path.isdir(LOCAL_MODEL_PATH):
    print(f"[INFO] Loading local emotion model from: {LOCAL_MODEL_PATH}")
    emotion_classifier = pipeline(
        "text-classification", model=LOCAL_MODEL_PATH, top_k=None
    )
else:
    print("[INFO] Loading default emotion model (ourafla)...")
    emotion_classifier = pipeline(
        "text-classification",
        model="ourafla/mental-health-bert-finetuned",
        top_k=None,
    )

# Confidence below which we let the lexicon assist nudge the result.
LOW_CONFIDENCE = 0.55

_GREETINGS = {
    "hi", "hello", "hey", "hye", "helo", "hai",
    "good morning", "good afternoon", "good evening",
    "selamat pagi", "selamat petang", "selamat malam",
}

_FAREWELLS = {
    "bye", "goodbye", "good bye", "see you", "see ya", "tata", "baibai",
    "selamat tinggal", "jumpa lagi",
}

# ── CRISIS SAFETY NET ──────────────────────────────────────────
# Explicit self-harm / suicidal expressions. Matched as whole phrases so
# we don't fire on unrelated substrings, but we intentionally keep the list
# broad. If any of these appear, we force SUICIDAL + high risk regardless of
# what the model thinks.
_CRISIS_PHRASES = [
    # English
    "kill myself", "killing myself", "end my life", "ending my life",
    "want to die", "wanna die", "i want to disappear",
    "don't want to be alive", "dont want to be alive", "not want to be alive",
    "don't want to live", "dont want to live", "no reason to live",
    "life is not worth living", "better off without me", "better off dead",
    "hurt myself", "harm myself", "take my own life", "end it all",
    # Malay
    "nak mati", "bunuh diri", "tak nak hidup", "tidak mahu hidup",
    "nak hilang", "lebih baik saya mati", "mahu mati", "ingin mati",
]
_CRISIS_RE = re.compile("|".join(re.escape(p) for p in _CRISIS_PHRASES))

# ── LEXICON ASSIST (low-confidence nudge only) ─────────────────
# Maps indicative phrases to an emotion. Used ONLY when the model is unsure.
_LEXICON = {
    "DEPRESSION": [
        "i am sad", "i'm sad", "im sad", "i feel sad", "i feel depressed",
        "i feel empty", "feel completely empty", "nothing brings me joy",
        "can't get out of bed", "cant get out of bed", "i feel like a burden",
        "no energy", "no motivation", "everything feels hopeless",
        "i feel numb", "i feel hopeless", "i feel worthless", "i feel lonely",
        "i feel so alone", "i lost interest", "lost interest",
        "saya sedih", "aku sedih", "rasa sedih", "rasa sunyi", "rasa kosong",
        "tiada semangat", "tidak ada semangat", "rasa tertekan", "putus asa",
    ],
    "ANXIETY": [
        "i am anxious", "i'm anxious", "im anxious", "i feel anxious",
        "can't stop worrying", "cant stop worrying", "i feel nervous",
        "i keep overthinking", "can't calm down", "cant calm down",
        "i feel restless", "i feel scared", "i feel panicky",
        "my heart is racing", "panic attack", "i'm worried", "im worried",
        "i'm stressed", "im stressed",
        "saya gelisah", "saya risau", "saya bimbang", "rasa takut",
        "rasa cemas",
    ],
}


# Single-word distress markers (matched on word boundaries) — a coarse second
# layer so phrasing like "I feel so hopeless and empty" is still caught even
# when it doesn't match a full lexicon phrase.
_KEYWORDS = {
    "DEPRESSION": [
        "hopeless", "worthless", "empty", "depressed", "depression",
        "numb", "lonely", "alone", "miserable", "unmotivated",
        "sedih", "sunyi", "kosong", "muram", "tertekan",
    ],
    "ANXIETY": [
        "anxious", "anxiety", "worried", "worrying", "nervous", "panic",
        "panicking", "overthinking", "restless", "scared", "stressed",
        "stress", "gelisah", "risau", "bimbang", "cemas", "takut",
    ],
}
_KEYWORD_RE = {
    emo: re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")
    for emo, words in _KEYWORDS.items()
}


def _build_distribution(raw_results):
    """Turn the pipeline's raw output into {EMOTION: probability}."""
    dist = {}
    for item in raw_results:
        label = LABEL_MAP.get(item["label"], item["label"].upper())
        dist[label] = dist.get(label, 0.0) + float(item["score"])
    return dist


def _lexicon_match(text):
    """Return an emotion if the text matches a lexicon phrase or keyword."""
    # Prefer full-phrase matches (more specific).
    for emotion, phrases in _LEXICON.items():
        for phrase in phrases:
            if phrase in text:
                return emotion
    # Fall back to single-word distress markers.
    for emotion, regex in _KEYWORD_RE.items():
        if regex.search(text):
            return emotion
    return None


def classify_text(text: str):
    """Classify input text.

    Returns: (emotion, confidence, is_high_risk, distribution)
      - emotion: top emotional state (str)
      - confidence: probability of that state (float)
      - is_high_risk: True if the crisis safety net or model flags SUICIDAL
      - distribution: {EMOTION: probability} over all states
    """
    normalized = text.strip().lower()

    # ── Small talk shortcuts ──
    if normalized in _GREETINGS:
        return "GREETING", 1.0, False, {"GREETING": 1.0}
    if normalized in _FAREWELLS:
        return "FAREWELL", 1.0, False, {"FAREWELL": 1.0}

    # ── Crisis safety net (always wins) ──
    if _CRISIS_RE.search(normalized):
        return "SUICIDAL", 0.99, True, {"SUICIDAL": 0.99, "DEPRESSION": 0.01}

    # ── ML classifier (primary) ──
    raw = emotion_classifier(text)[0]
    distribution = _build_distribution(raw)
    emotion = max(distribution, key=distribution.get)
    confidence = round(distribution[emotion], 3)

    # ── Lexicon backstop ──
    # The bilingual model handles most input natively, but can still occasionally
    # mislabel clear distress as NORMAL. We let the lexicon correct two cases:
    #   (a) the model is unsure (low confidence), or
    #   (b) the model says NORMAL/NEUTRAL but a distress phrase is clearly present
    #       (a false-negative we must not miss in a mental-health context).
    if confidence < LOW_CONFIDENCE or emotion in ("NORMAL", "NEUTRAL"):
        hinted = _lexicon_match(normalized)
        if hinted:
            # Demote the (likely wrong) positive mass and promote the hinted
            # distress class, so the wellbeing score agrees with the new label.
            for positive in ("NORMAL", "NEUTRAL"):
                if positive in distribution:
                    distribution[positive] *= 0.2
            distribution[hinted] = max(distribution.get(hinted, 0.0), 0.7)
            emotion = hinted
            confidence = round(distribution[hinted], 3)

    is_high_risk = emotion == "SUICIDAL"
    return emotion, confidence, is_high_risk, distribution
