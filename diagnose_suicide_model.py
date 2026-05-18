import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = r'backend\suicide_model'
tok = AutoTokenizer.from_pretrained(path)
mdl = AutoModelForSequenceClassification.from_pretrained(path)
mdl.eval()

tests = [
    'hi', 'bye', 'hello', 'goodbye', 'thanks', 'okay', 'yes', 'no',
    'I want to kill myself', 'I feel anxious', 'saya sedih',
    'I am feeling hopeless', 'good morning'
]

print(f"{'Label':<15} {'Score':>7}  Input")
print("-" * 60)
for t in tests:
    inp = tok(t, return_tensors='pt', truncation=True, max_length=256, padding=True)
    inp.pop('token_type_ids', None)
    with torch.no_grad():
        logits = mdl(**inp).logits
    probs = F.softmax(logits, dim=-1)[0]
    idx = probs.argmax().item()
    label = mdl.config.id2label[idx]
    print(f"[{label:<13}] {probs[idx]:>6.2%}  <- \"{t}\"")
