import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.model import classify_text
from backend.response_generator import generate_response

test_sentences = [
    "Hi", 
    "Hello", 
    "Good Morning",
    "Bye", 
    "Thank you", 
    "Thanks",
    "Hi, I am sad",  # Edge case: Greeting + Emotion
    "I am very happy today"
]

print("--- Testing Conversational Intents ---")
for text in test_sentences:
    emotion, confidence = classify_text(text)
    # Mock confidence for response generation if it's 1.0 (rule-based)
    response = generate_response(emotion, confidence)
    
    print(f"\nInput: '{text}'")
    print(f"Predicted: {emotion} ({confidence:.2f})")
    print(f"Response: {response[:80]}...") 
