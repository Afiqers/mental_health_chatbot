import os
import sys

# Add the backend directory to python path for testing
sys.path.append(os.path.abspath("backend"))

from response_generator import generate_response

test_cases = [
    {
        "message": "I'm feeling really anxious about my presentation tomorrow.",
        "emotion": "ANXIETY",
        "confidence": 0.85,
        "is_first": True,
        "context": "95bpm (High)",
        "last_topic": "Work Presentation"
    },
    {
        "message": "I just can't get out of bed, everything feels heavy.",
        "emotion": "DEPRESSION",
        "confidence": 0.90,
        "is_first": False,
        "context": "Sleep Score: Low",
        "last_topic": "None"
    },
    {
        "message": "I hate my life and I want to end it.",
        "emotion": "SUICIDAL",
        "confidence": 0.95,
        "is_first": False,
        "context": None,
        "last_topic": None
    },
    {
        "message": "",
        "emotion": "UNKNOWN",
        "confidence": 0.0,
        "is_first": False,
        "context": "User inactive for 2 days",
        "last_topic": "Job Interview"
    }
]

print("Starting tests...\n")
for i, case in enumerate(test_cases):
    print(f"\n--- Test Case {i+1} ---")
    print(f"Message: {case['message']}")
    print(f"Emotion: {case['emotion']}")
    print("-" * 20)
    response = generate_response(
        user_message=case["message"],
        emotion=case["emotion"], 
        confidence=case["confidence"], 
        is_first_message=case["is_first"],
        context=case.get("context"),
        last_topic=case.get("last_topic")
    )
    print(response)

print("\nTests finished.")
