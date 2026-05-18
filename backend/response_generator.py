# backend/response_generator.py

RESPONSES = {
    "ANXIETY": [
        "I can hear how overwhelmed you must be feeling right now. 💙 I'm right here with you. Would you like to try a short breathing exercise to help settle your racing mind? (You can say 'yes' or 'just listen')",
        "Anxiety is so hard to carry alone. Let's take a pause together. Try breathing in slowly for 4 seconds, holding for 4, and exhaling for 4. How does your body feel after taking that breath?",
        "It sounds like there's a lot on your plate. To help ground yourself, can you look around and name 3 things you can see right now? I'll wait.",
        "Your mind is racing, but you are safe right here in this moment. Let's break things down. What is the very next tiny step you need to take today?",
        "I hear you. When the physical feelings of stress hit, it can be scary. Can you try unclenching your jaw and dropping your shoulders for just a second?",
        "You're not alone in this. Let's take it one step at a time. Do you want to try a quick grounding exercise, or would you rather I just listen?",
        "Whenever your mind starts running away, try to remind yourself: 'I am safe in this exact moment'. I'm here. What's one thing you can hear right now?",
        "There is no rush to figure everything out. Take your time. Would it help to focus on your breathing for just 30 seconds with me?"
    ],
    "DEPRESSION": [
        "I hear you, and it's completely okay if all you did today was just survive. I'm right here with you. Can we try doing just one small thing right now, like taking a sip of water?",
        "When everything feels heavy, it's so hard to move forward. Let's not look at the whole staircase... just the very next step. What's one tiny thing that would make you feel 1% more comfortable?",
        "On your darkest days, please remember that your presence in this world matters. Have you been able to eat or drink anything in the last few hours?",
        "I'm really sorry you're feeling this way. It's safe to let your guard down here. If you could wave a magic wand, what's one feeling you wish you could have right now?",
        "It can be exhausting carrying these emotions. Please be gentle with yourself today. Would you like to try a simple reflection exercise, or just vent?",
        "I'm here to listen without any judgment. You don't have to carry this alone. What is the hardest part about how you're feeling right now?",
        "Thank you for trusting me with that. It takes courage to open up. Could you try moving to a different room or just stretching your arms for a second? Small movements can sometimes shift our energy a tiny bit.",
        "Your feelings are entirely valid, and I am proud of you for holding on. Is there a safe, comforting space you can retreat to for a little while?"
    ],
    "STRESS": [
        "That sounds like a lot to juggle. Maybe it's time to take a short, 5-minute pause? 🍃 What's one thing you can safely drop for today?",
        "Stress can build up so quickly. Let’s try to slow things down. Could you take a deep belly breath right now?",
        "I understand. You are handling a lot. Is there one small thing you can take off your plate today?",
        "Take a deep breath. You are doing the best you can with the energy you have. Do you want to vent, or do you want to break down your tasks together?",
        "When everything feels like too much, stepping away for just five minutes can reset your mind. Are you in a place where you can close your eyes for a moment?",
        "You are stretched thin, and that's okay. Remember to give yourself some credit. Have you had a chance to rest at all today?",
        "Let's focus only on the very next step. What is one small thing you have absolute control over right now?",
        "It's completely normal to feel overwhelmed when there's so much going on. Please remember to hydrate and be kind to your body today."
    ],
    "SUICIDAL": [
        "Please know that you are not alone and things can get better. Please contact Talian Kasih at 15999 or Befrienders KL at 03-76272929 for immediate support.",
        "Your life is incredibly valuable. Please reach out to someone who can help right now. Call Talian Kasih at 15999."
    ],
    "HIGH_RISK": [
        "Please know that you are not alone and things can get better. Please contact Talian Kasih at 15999 or Befrienders KL at 03-76272929 for immediate support.",
        "Your life is incredibly valuable. Please reach out to someone who can help right now. Call Talian Kasih at 15999."
    ],
    "NORMAL": [
        "Thank you for sharing. How can I support you today?",
        "I'm here whenever you'd like to talk."
    ],
    "NEUTRAL": [
        "Thank you for sharing. How can I support you today?",
        "I'm here whenever you'd like to talk."
    ],
    "UNKNOWN": [
        "I'm not sure I fully understand, but I want to support you. Can you tell me a bit more?",
        "I'm here to listen. Please share as much as you feel comfortable with.",
        "I'm listening. Take all the time you need.",
        "I want to make sure I understand you completely. Could you expand on that a bit?",
        "I am right here with you. What else is going on?",
        "I'm here for you without any judgment. Feel free to talk about whatever is on your mind."
    ],
    "GREETING": [
        "Hello! 🐧 It's good to see you. How are you feeling today in this exact moment?",
        "Hi there! I'm here to listen and support you. What's on your mind?",
        "Welcome! This is a safe space just for you. How can I help you right now?",
        "Hello. I'm ready to listen if you'd like to share. How has your day been?",
        "Hey! I hope you're having a gentle day. Do you want to do a quick mood check-in?",
        "Hi. I'm so glad you decided to reach out. Are you dealing with anything specific today?",
        "Greetings! Feel free to talk about whatever you need to. I'm all ears.",
        "Hello! Remember, I'm here whenever you need a pause. What's on your mind?"
    ],
    "FAREWELL": [
        "Take care of yourself. I'm always just a message away whenever you need a pause.",
        "Goodbye for now. Please remember to be kind to yourself today.",
        "You're welcome. Wishing you a peaceful, gentle rest of your day.",
        "See you later. Don't hesitate to return if you ever feel overwhelmed.",
        "Bye for now. Please remember that your presence matters. 💙",
        "Take it easy. I'll be right here waiting if you need to come back.",
        "Wishing you a safe and calm rest of your day. Goodbye!",
        "Farewell. Remember to take things one small step at a time."
    ]
}

CRITICAL_RESOURCE_MESSAGE = (
    "\n\nIf you are in immediate danger, please call your local emergency number or a crisis hotline:\n"
    "- **Generic**: 999 (Malaysia)\n"
    "- **Befrienders Worldwide**: befrienders.org\n"
    "Please reach out to them—they are trained to help you."
)

def generate_response(user_message: str, emotion: str, confidence: float, is_first_message: bool = False, history: list = None) -> str:
    """
    Generate empathetic response based on emotion and confidence score.
    history: list of {"role": "user"|"bot", "content": "..."} dicts (last N turns).
    If confidence is low (< 0.6), asks for clarification.
    """
    history = history or []

    # IMMEDIATE SAFETY CHECK
    if emotion == "SUICIDAL":
        final_response = "I hear how much pain you are in. You are not alone. Please reach out to these professionals who can stay on the line with you right now.\n" + CRITICAL_RESOURCE_MESSAGE
        if is_first_message:
            final_response = "I am an AI, not a licensed therapist or doctor. If you are in immediate danger, please reach out to emergency services.\n\n" + final_response
        return final_response

    if confidence < 0.6 and user_message:
        return (
            "I'm not completely sure how you're feeling, "
            "but I'm here to listen. Could you tell me more about what's going on?"
        )

    # Get list of possible responses for the emotion
    possible_responses = RESPONSES.get(emotion, RESPONSES["UNKNOWN"])

    import random
    selected_response = random.choice(possible_responses)

    # Rephrase with LLM if confidence is reasonable
    # Skip rephrasing for structural/conversational intents so the LLM doesn't hallucinate "You mentioned feeling GREETING"
    if emotion not in ["UNKNOWN", "GREETING", "FAREWELL"] and confidence >= 0.7:
        from llm_rephraser import rephrase_response
        final_response = rephrase_response(selected_response, emotion, user_message, history=history)
    else:
        final_response = selected_response

    if is_first_message:
        final_response = "I am an AI, not a licensed therapist or doctor. If you are in immediate danger, please reach out to emergency services.\n\n" + final_response

    return final_response
