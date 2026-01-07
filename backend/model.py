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

    return emotion, confidence
