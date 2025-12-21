# backend/response_generator.py

RESPONSES = {
    "ANXIETY": [
        "It sounds like you're feeling anxious. Try taking a slow breath. I'm here with you.",
        "Feeling anxious can be overwhelming. You’re not alone."
    ],
    "DEPRESSION": [
        "I'm really sorry you're feeling this way. You matter, and your feelings are valid.",
        "It can be heavy carrying these emotions. I'm here to listen."
    ],
    "STRESS": [
        "That sounds stressful. Taking things one step at a time can help.",
        "Stress can build up quickly. Let’s slow things down together."
    ],
    "NEUTRAL": [
        "Thank you for sharing. How can I support you today?",
        "I'm here whenever you'd like to talk."
    ],
    "UNKNOWN": [
        "I'm here to listen. Please tell me more."
    ]
}

def generate_response(emotion: str) -> str:
    """
    Generate empathetic response based on emotion.
    """
    responses = RESPONSES.get(emotion, RESPONSES["UNKNOWN"])
    return responses[0]
