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
    "NORMAL": [
        "I'm glad to hear that! How has your day been otherwise?",
        "That's great! It's always nice to check in. What have you been up to?",
        "Good to hear! I'm here if you ever need to chat.",
        "That sounds positive! Keep it up."
    ],
    "SUICIDAL": [
        "I'm hearing that you're in a lot of pain, and I want you to be safe. Please reach out for help immediately.",
        "You are not alone, and there is help available. Please contact a crisis helpline right now.",
        "It sounds like you are going through a very difficult time. Please talk to a professional who can help you stay safe."
    ],
    "UNKNOWN": [
        "I'm here to listen. Please tell me more.",
        "I'm not sure I fully understand, but I want to support you. Can you say more?",
        "I'm listening."
    ],
    "GREETING": [
        "Hello! It's good to see you. How are you feeling today?",
        "Hi there! I'm here to listen and support you.",
        "Welcome. How can I help you right now?",
        "Hello. I'm ready to listen if you'd like to share."
    ],
    "FAREWELL": [
        "Take care of yourself. I'm here whenever you need to talk.",
        "Goodbye for now. Remember to be kind to yourself.",
        "You're welcome. Wishing you a peaceful day.",
        "See you later. Don't hesitate to return if you need support."
    ]
}

CRITICAL_RESOURCE_MESSAGE = (
    "\n\nIf you are in immediate danger, please call your local emergency number or a crisis hotline:\n"
    "- **Generic**: 999 (Malaysia)\n"
    "- **Befrienders Worldwide**: befrienders.org\n"
    "Please reach out to them—they are trained to help you."
)

def generate_response(emotion: str, confidence: float) -> str:
    """
    Generate empathetic response based on emotion and confidence score.
    If confidence is low (< 0.6), asks for clarification.
    """
    
    # IMMEDIATE SAFETY CHECK
    if emotion == "SUICIDAL":
        import random
        base_msg = random.choice(RESPONSES["SUICIDAL"])
        return base_msg + CRITICAL_RESOURCE_MESSAGE

    if confidence < 0.6:
        return (
            "I'm not completely sure how you're feeling, "
            "but I'm here to listen. Could you tell me more about what's going on?"
        )

    # Get list of possible responses for the emotion
    possible_responses = RESPONSES.get(emotion, RESPONSES["UNKNOWN"])
    
    import random
    selected_response = random.choice(possible_responses)

    # --- LLM REPHRASING ---
    # Only rephrase if safe and confidence is reasonable.
    # Skip for SUICIDAL (already handled above) and low confidence.
    # Also skip if emotion is UNKNOWN to avoid hallucinating meaning.
    if emotion not in ["SUICIDAL", "UNKNOWN"] and confidence >= 0.7:
        from llm_rephraser import rephrase_response
        return rephrase_response(selected_response, emotion)
    
    return selected_response
