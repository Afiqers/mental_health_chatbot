from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "ourafla/mental-health-bert-finetuned"
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Config id2label:", model.config.id2label)
    print("Config label2id:", model.config.label2id)

    # Test bad sentences again with raw scores
    text = "I want to end my life"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        print(f"\nManual Probabilities for '{text}':", probs[0])

    # Now verify the backend function itself
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
    from backend.model import classify_text
    
    emotion, confidence = classify_text(text)
    print(f"\nBackend classify_text('{text}'): {emotion} ({confidence})")

except Exception as e:
    print(e)
