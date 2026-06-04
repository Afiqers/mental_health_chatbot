import json
import urllib.request
import urllib.error

try:
    # Centralised config (model name, URL, timeout) when run inside the app.
    from config import Config
    OLLAMA_MODEL = Config.OLLAMA_MODEL
    OLLAMA_URL = Config.OLLAMA_URL
    OLLAMA_TIMEOUT = Config.OLLAMA_TIMEOUT
except Exception:
    # Standalone fallback (e.g. running this file directly in a test).
    OLLAMA_MODEL = "llama3.2"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_TIMEOUT = 30

# Common Malay function words and discourse markers for language detection
_MALAY_MARKERS = {
    "saya", "aku", "kamu", "awak", "dia", "kami", "kita", "mereka",
    "tak", "tidak", "tiada", "bukan", "jangan",
    "dan", "atau", "tapi", "tetapi", "sebab", "kerana", "supaya",
    "dengan", "untuk", "kepada", "dari", "daripada", "dalam", "pada",
    "ini", "itu", "yang", "pun", "juga", "sudah", "dah", "akan",
    "boleh", "nak", "mahu", "hendak", "perlu", "ada", "ade",
    "sangat", "amat", "lebih", "agak", "macam", "mcm", "macam",
    "rasa", "rase", "tolong", "please", "terima", "kasih",
    "hai", "helo", "selamat", "baik", "okay", "ok",
    "kenapa", "mengapa", "bagaimana", "mana", "bila", "siapa",
    "lah", "leh", "kan", "lor", "meh", "yer", "ye",
}

def detect_language(text: str) -> str:
    """Returns 'Malay' if the text appears to be Malay, else 'English'."""
    if not text:
        return "English"
    words = set(text.lower().split())
    malay_hits = words & _MALAY_MARKERS
    # If at least 1 Malay marker found OR >25% of words match, treat as Malay
    if malay_hits and (len(malay_hits) >= 1 or len(malay_hits) / len(words) > 0.25):
        return "Malay"
    return "English"

def rephrase_response(text: str, emotion: str, user_message: str = "", history: list = None) -> str:
    """
    Rephrases the input text to be natural and empathetic using a local Ollama model.
    Adapts the language based on the user's message and conversation history.

    history: list of {"role": "user"|"bot", "content": "..."} dicts (last N turns).
    """
    history = history or []

    # Build a readable conversation history string (last 6 messages = 3 turns)
    history_str = ""
    if history:
        recent = history[-6:]
        lines = []
        for turn in recent:
            role = "User" if turn.get("role") == "user" else "Bot"
            lines.append(f"{role}: {turn.get('content', '').strip()}")
        history_str = "\n".join(lines)

    history_section = (
        f"\n\nPrevious conversation:\n{history_str}\n"
        if history_str else ""
    )

    # Map emotion labels to Solace sentiment tiers for prompt guidance
    sentiment_guide = {
        "NORMAL":   "POSITIVE / NEUTRAL — engage supportively, explore what's going well",
        "NEUTRAL":  "POSITIVE / NEUTRAL — engage supportively, explore what's going well",
        "STRESS":   "MILD_DISTRESS — validate feelings, gently explore emotions, offer grounding coping strategies",
        "ANXIETY":  "MILD_DISTRESS — validate feelings, gently explore emotions, offer grounding coping strategies",
        "DEPRESSION": "MODERATE_DISTRESS — prioritize emotional validation, slow down, ask what they need",
        "SUICIDAL": "HIGH_DISTRESS / CRISIS — acknowledge pain directly, share crisis resources, encourage professional help",
        "HIGH_RISK": "HIGH_DISTRESS / CRISIS — acknowledge pain directly, share crisis resources, encourage professional help",
    }
    sentiment_tier = sentiment_guide.get(emotion, "MILD_DISTRESS — validate and listen carefully")

    # Detect the user's language upfront so we can enforce it explicitly
    user_language = detect_language(user_message)

    # System prompt: persona + absolute language constraint (evaluated first by the model)
    system_prompt = (
        f"You are Solace, a compassionate and emotionally intelligent mental health support companion. "
        f"Your role is to provide empathetic, non-judgmental support that helps users process their emotions.\n\n"
        f"ABSOLUTE LANGUAGE RULE: The user is writing in {user_language}. "
        f"You MUST respond entirely in {user_language}. "
        f"Do NOT switch languages, mix languages, or respond in English if the user wrote in Malay. "
        f"This rule overrides everything else."
    )

    # User prompt: the actual rephrasing task
    prompt = (
        f"The user said: '{user_message}'.\n"
        f"Their detected emotional state is: {emotion} — Sentiment tier: {sentiment_tier}.\n"
        f"{history_section}"
        f"Your task is to rephrase the following response to match the Solace persona: '{text}'.\n\n"

        f"## Solace Core Behavior\n"
        f"- Always respond with warmth, empathy, and validation FIRST before offering any advice.\n"
        f"- Mirror the user's emotional tone — calm if they're calm, gentle if they're distressed.\n"
        f"- Use active listening: reflect what you sense, then ask one gentle open-ended follow-up question.\n"
        f"- Never diagnose, prescribe, or claim to replace professional mental health care.\n"
        f"- Keep responses concise (2–4 sentences max) unless the user clearly needs more.\n"
        f"- Avoid clinical jargon — speak like a caring, warm human companion.\n"
        f"- Never be dismissive, toxic-positive (e.g., 'Semua akan baik-baik saja!'), or give unsolicited advice.\n\n"

        f"## Sentiment-Specific Behavior\n"
        f"- POSITIVE / NEUTRAL: Engage supportively and explore what's going well for them.\n"
        f"- MILD_DISTRESS (STRESS, ANXIETY): Validate their feelings, gently explore, offer grounding coping strategies.\n"
        f"- MODERATE_DISTRESS (DEPRESSION): Prioritize emotional validation, slow down the conversation, ask what they need.\n"
        f"- HIGH_DISTRESS / CRISIS (SUICIDAL, HIGH_RISK): Acknowledge their pain directly. "
        f"Include these Malaysian crisis resources naturally in your response: "
        f"Talian Kasih: 15999 | Befrienders KL: 03-76272929. "
        f"Strongly encourage professional help. Do NOT shift to casual conversation.\n\n"

        f"## Critical Rules\n"
        f"1. You are an AI companion, NOT a human. Use 'I' only when referring to yourself as a supportive AI.\n"
        f"2. If the text contains a consent question (e.g., 'Would you like to try...') or a structured "
        f"interactive exercise (e.g., 'name 3 things', 'breathe in for 4 seconds'), YOU MUST preserve "
        f"that structured exercise in your rephrasing — translated into {user_language}.\n"
        f"3. NEVER reference the conversation history explicitly. Never say 'You mentioned', 'You said', "
        f"or 'Earlier you said'. Let context shape your tone naturally without quoting back.\n"
        f"4. End with a gentle open-ended question to keep the conversation going — UNLESS the emotion is "
        f"SUICIDAL or HIGH_RISK, in which case stay focused entirely on their safety.\n\n"

        f"REMINDER: Respond entirely in {user_language}. "
        f"Output the rephrased response directly. No quotation marks, no preamble, no filler phrases."
    )

    data = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,   # system role enforces language + persona at the model level
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", text).strip(' \n"')
    except Exception as e:
        print(f"Ollama API Error: {e} - Ensure Ollama is running and '{OLLAMA_MODEL}' is downloaded.")
        return text
