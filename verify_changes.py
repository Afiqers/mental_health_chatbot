import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.model import classify_text
from backend.response_generator import generate_response

print("--- Testing Explicit Model Implementation ---")

# Test Cases
test_sentences = [
    "I feel very anxious and scared today.",  # Anxiety
    "I feel hopeless and sad.",               # Depression
    "The weather is nice.",                   # Normal
    "I want to end my life."                  # Suicidal
]

for text in test_sentences:
    emotion, confidence = classify_text(text)
    response = generate_response(emotion, confidence)
    
    print(f"\nInput: {text}")
    print(f"Predicted: {emotion} ({confidence:.2f})")
    print(f"Response: {response[:100]}...") # Truncate for display
