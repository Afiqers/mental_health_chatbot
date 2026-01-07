# backend/response_generator.py

RESPONSES = {
    "ANXIETY": [
        "It sounds like you're feeling anxious. Try taking a slow, deep breath. I'm here with you.",
        "Anxiety can be really tough. Remember that this feeling will pass. You are safe.",
        "I hear you. Focus on the present moment. What is one thing you can see right now?",
        "You're not alone in this. Let's take it one step at a time."
    ],
    "DEPRESSION": [
        "I'm really sorry you're feeling this way. You matter, and your feelings are valid.",
        "It can be so heavy carrying these emotions. Please be gentle with yourself today.",
        "I'm here to listen. You don't have to go through this alone.",
        "Thank you for sharing that with me. It takes courage to open up."
    ],
    "STRESS": [
        "That sounds stressful. Maybe it's time to take a short break?",
        "Stress can build up quickly. Let’s try to slow things down appropriately.",
        "I understand. Is there one small thing you can take off your plate today?",
        " deep breath. You are doing the best you can."
    ],
    "NEUTRAL": [
        "Thank you for sharing. How can I support you today?",
        "I'm here whenever you'd like to talk.",
        "It's good to hear from you. What's on your mind?",
        "I'm listening. Feel free to tell me more."
    ],
    "UNKNOWN": [
        "I'm here to listen. Please tell me more.",
        "I'm not sure I fully understand, but I want to support you. Can you say more?",
        "I'm listening."
    ]
}

def generate_response(emotion: str) -> str:
    """
    Generate empathetic response based on emotion.
    """
    responses = RESPONSES.get(emotion, RESPONSES["UNKNOWN"])
    return responses[0]
