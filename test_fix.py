from backend.model import classify_text
from backend.label_map import LABEL_MAP

# Test raw
text_raw = "i am sad"
emotion_raw, conf_raw = classify_text(text_raw)
print(f"Raw '{text_raw}': {emotion_raw} ({conf_raw:.4f})")

# Test with appended context (simple)
text_ctx = "I am feeling " + text_raw
emotion_ctx, conf_ctx = classify_text(text_ctx)
print(f"Context '{text_ctx}': {emotion_ctx} ({conf_ctx:.4f})")

# Test with keyword override check simulation
keywords = ["sad", "depressed", "hopeless"]
if any(k in text_raw.lower() for k in keywords) and emotion_raw == "NORMAL":
    print("Keyword Override Triggered: DEPRESSION")
