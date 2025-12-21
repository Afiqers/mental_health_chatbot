# backend/model.py

from transformers import pipeline
from label_map import LABEL_MAP

# Load model once (important for performance)
classifier = pipeline(
    "text-classification",
    model="ourafla/mental-health-bert-finetuned",
    top_k=None
)

def classify_text(text: str):
    """
    Classify input text into mental health emotion.
    Returns: (emotion, confidence)
    """
    results = classifier(text)[0]

    # Select label with highest score
    best_prediction = max(results, key=lambda x: x["score"])

    label = best_prediction["label"]
    emotion = LABEL_MAP.get(label, "UNKNOWN")
    confidence = round(best_prediction["score"], 3)

    return emotion, confidence
