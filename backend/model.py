# backend/model.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from label_map import LABEL_MAP

# 1. Load Model and Tokenizer
MODEL_NAME = "ourafla/mental-health-bert-finetuned"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit? For now just print error.
    tokenizer = None
    model = None

# 2. Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)
    model.eval()

def classify_text(text: str):
    """
    Classify input text using explicit PyTorch inference.
    Returns: (emotion, confidence)
    """
    if not model or not tokenizer:
        return "UNKNOWN", 0.0

    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Calculate probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get best prediction
        confidence, prediction_idx = torch.max(probs, dim=-1)
        
        # Extract values
        prediction_idx = prediction_idx.item()
        confidence = confidence.item()

    # Map to Label
    emotion = LABEL_MAP.get(prediction_idx, "UNKNOWN")

    # --- KEYWORD FALLBACK ---
    # The model sometimes struggles with very short or direct phrases (e.g., "i am sad").
    # We add a safety check here to override "NORMAL" if strong negative keywords are present.
    if emotion == "NORMAL":
        text_lower = text.lower()
        if any(w in text_lower for w in ["sad", "depressed", "hopeless", "unhappy", "cry"]):
            # Override to Depression with high confidence
            emotion = "DEPRESSION"
            confidence = 0.85
        elif any(w in text_lower for w in ["anxious", "scared", "worried", "nervous", "panic"]):
            # Override to Anxiety with high confidence
            emotion = "ANXIETY"
            confidence = 0.85
        elif any(w in text_lower for w in ["kill", "suicide", "end my life", "die"]):
            # Override to Suicidal with high confidence (Safety First)
            emotion = "SUICIDAL"
            confidence = 0.95

    return emotion, confidence
