from transformers import pipeline

try:
    classifier = pipeline(
        "text-classification",
        model="ourafla/mental-health-bert-finetuned",
        top_k=None
    )
    results = classifier("I feel very anxious and scared today.")
    print("Model Output:", results)
except Exception as e:
    print("Error:", e)
