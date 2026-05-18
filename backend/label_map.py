# backend/label_map.py

# This maps the raw output of the model to our standardized backend emotions
LABEL_MAP = {
    # If using the original huggingface model:
    "LABEL_0": "ANXIETY",
    "LABEL_1": "DEPRESSION",
    "LABEL_2": "SUICIDAL",
    "LABEL_3": "NORMAL",

    # If using our local_model (which explicitly sets id2label):
    "Anxiety": "ANXIETY",
    "Depression": "DEPRESSION",
    "Suicidal": "SUICIDAL",
    "Normal": "NORMAL"
}
