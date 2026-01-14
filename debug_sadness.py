import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.label_map import LABEL_MAP

MODEL_NAME = "ourafla/mental-health-bert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

text = "i am sad"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]

print(f"Input: '{text}'")
print("-" * 30)
for idx, prob in enumerate(probs):
    label_name = LABEL_MAP.get(idx, f"UNKNOWN({idx})")
    print(f"Index {idx} ({label_name}): {prob.item():.4f}")

best_idx = torch.argmax(probs).item()
print("-" * 30)
print(f"Predicted: Index {best_idx} -> {LABEL_MAP.get(best_idx)}")
